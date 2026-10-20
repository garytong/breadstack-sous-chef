// Command sous-chef runs the Sous Chef service (README §8):
//
//	sous-chef                                      serve HTTP on $PORT
//	sous-chef -migrate                             apply migrations and exit
//	sous-chef -retention [--days N] [--dry-run]    run the retention job and exit
//
// Serving wires the HTTP layer (internal/api) to the manifest registry, the
// agent loop, the Postgres store, the model route table, the trace recorder
// and the usage recorder/shipper; a TTL reaper and a usage sweep run beside
// the server.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/breadstack/sous-chef/internal/api"
	"github.com/breadstack/sous-chef/internal/auth"
	"github.com/breadstack/sous-chef/internal/config"
	"github.com/breadstack/sous-chef/internal/loop"
	"github.com/breadstack/sous-chef/internal/manifest"
	"github.com/breadstack/sous-chef/internal/model"
	"github.com/breadstack/sous-chef/internal/registry"
	"github.com/breadstack/sous-chef/internal/store"
	"github.com/breadstack/sous-chef/internal/trace"
	"github.com/breadstack/sous-chef/internal/usage"
)

// buildID is set with `-ldflags "-X main.buildID=<sha>"`; BUILD_ID in the
// environment wins when both are present.
var buildID = ""

func main() {
	// -retention takes its own flags (--days, --dry-run), which the main
	// FlagSet would otherwise reject, so both subcommands are dispatched on
	// the first argument; the definitions below only feed -h.
	migrate := flag.Bool("migrate", false, "apply database migrations and exit")
	retention := flag.Bool("retention", false, "run the retention job and exit (its own flags follow: --days N, --dry-run)")
	var retentionArgs []string
	if len(os.Args) > 1 {
		switch strings.TrimLeft(os.Args[1], "-") {
		case "retention":
			*retention = true
			retentionArgs = os.Args[2:]
		case "migrate":
			*migrate = true
		default:
			flag.Parse()
		}
	}

	getenv := os.Getenv
	if buildID != "" && os.Getenv("BUILD_ID") == "" {
		getenv = func(k string) string {
			if k == "BUILD_ID" {
				return buildID
			}
			return os.Getenv(k)
		}
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	switch {
	case *migrate:
		dsn := getenv("DATABASE_URL")
		if dsn == "" {
			fmt.Fprintln(os.Stderr, "sous-chef: -migrate: DATABASE_URL is required")
			os.Exit(1)
		}
		v, err := store.Migrate(dsn)
		if err != nil {
			fmt.Fprintf(os.Stderr, "sous-chef: migrate: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("sous-chef: schema at version %d\n", v)
		return
	case *retention:
		if err := trace.Command(ctx, retentionArgs, getenv, os.Stdout); err != nil {
			fmt.Fprintf(os.Stderr, "sous-chef: retention: %v\n", err)
			os.Exit(1)
		}
		return
	}

	cfg, err := config.Load(getenv)
	if err != nil {
		fmt.Fprintf(os.Stderr, "sous-chef: configuration error:\n%v\n", err)
		os.Exit(1)
	}
	if err := serve(ctx, cfg, getenv); err != nil {
		fmt.Fprintf(os.Stderr, "sous-chef: %v\n", err)
		os.Exit(1)
	}
}

func serve(ctx context.Context, cfg *config.Config, getenv func(string) string) error {
	if cfg.MigrateAtBoot {
		v, err := store.Migrate(cfg.DatabaseURL)
		if err != nil {
			return fmt.Errorf("MIGRATE_AT_BOOT: %w", err)
		}
		log.Printf("sous-chef: migrated to schema version %d", v)
	}
	bootCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	st, err := store.Open(bootCtx, cfg.DatabaseURL)
	cancel()
	if err != nil {
		return err
	}
	defer st.Close()

	models, err := model.NewTable(model.FromConfig(cfg, getenv))
	if err != nil {
		return err
	}
	reg := registry.New(st)
	usageRec := usage.NewRecorder(st, usage.NewShipperFromEnv(getenv))
	runner, err := loop.New(loop.Deps{
		Store: st, Manifests: reg, Models: models, Usage: usageRec,
		TurnTTL: cfg.TurnTTL, MaxConcurrent: cfg.MaxConcurrentTurns,
		// Prompt-cache priming (README §4): acts only on routes with
		// prime:"on" (opt-in; none by default).
		Primer: model.NewPrimer(model.PrimerOptions{}),
	})
	if err != nil {
		return err
	}

	_, defaultOK := cfg.ModelRoutes[cfg.DefaultRoute]
	srv := api.New(auth.NewKeys(cfg.ServiceKeys), api.Options{
		BuildID:                cfg.BuildID,
		Routes:                 cfg.RouteNames(),
		RouteInfo:              cfg.RouteInfo(),
		DefaultRoute:           cfg.DefaultRoute,
		DefaultRouteConfigured: defaultOK,
		RouteAliases:           cfg.RouteAliases,
		ParserErrors:           models.ParserErrors,
		// README §1 limits: 8 MiB manifests, 256 KB for every other body.
		MaxManifestBytes: api.DefaultMaxManifestBytes,
		MaxBodyBytes:     api.DefaultMaxBodyBytes,
	}, api.Deps{
		Manifests: reg,
		// POST /v1/manifests fallback when the registry has no
		// ComputeAndRegister: the same canonical recipe PUT verifies with.
		HashManifest:  manifest.HashBytes,
		Turns:         runner,
		Conversations: loop.Conversations{St: st},
		Feedback:      &loop.Feedback{St: st},
		OptOut:        loop.OptOut{St: st},
		DB:            st,
		Primer:        runner,
	})

	httpSrv := &http.Server{
		Addr:              fmt.Sprintf(":%d", cfg.Port),
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
		// No WriteTimeout: SSE turns stream for up to a model call (60 s)
		// plus heartbeats; Cloud Run's 600 s request timeout is the bound.
		IdleTimeout: 120 * time.Second,
	}

	// Background work: the TTL reaper (README §4 step 8) and, when shipping
	// is on, a periodic re-ship of usage rows the thermometer missed.
	bg, stopBG := context.WithCancel(ctx)
	defer stopBG()
	go runner.RunReaper(bg, time.Minute)
	if usageRec.ShippingEnabled() {
		go sweepUsage(bg, usageRec, 5*time.Minute)
	}

	errc := make(chan error, 1)
	go func() {
		log.Printf("sous-chef %s listening on %s (routes: %v, default: %s, keys: %d, usage shipping: %v)",
			cfg.BuildID, httpSrv.Addr, cfg.RouteNames(), cfg.DefaultRoute, len(cfg.ServiceKeys), usageRec.ShippingEnabled())
		if len(cfg.RouteAliases) > 0 {
			log.Printf("sous-chef: route aliases %v: a turn without a model is answered by %s", cfg.RouteAliases, cfg.AnsweredBy(""))
		}
		if err := httpSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errc <- err
		}
	}()

	select {
	case err := <-errc:
		return fmt.Errorf("listen: %w", err)
	case <-ctx.Done():
	}
	log.Printf("sous-chef: shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := httpSrv.Shutdown(shutdownCtx); err != nil {
		log.Printf("sous-chef: shutdown: %v", err)
	}
	usageRec.Wait()
	return nil
}

// sweepUsage re-ships unshipped usage rows at boot and every interval.
func sweepUsage(ctx context.Context, rec *usage.Recorder, interval time.Duration) {
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		if n, err := rec.Sweep(ctx); err != nil && ctx.Err() == nil {
			log.Printf("sous-chef: usage sweep: %v", err)
		} else if n > 0 {
			log.Printf("sous-chef: usage sweep shipped %d row(s)", n)
		}
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
	}
}
