package trace

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"strconv"
	"strings"
	"time"

	"github.com/breadstack/sous-chef/internal/store"
)

// Retention is the daily job (Cloud Run job `sous-chef-retention`, Cloud
// Scheduler 03:00 UTC; README §6, §9). It applies store.Retention with the
// policy from the environment and logs the counts.

// Getenv is os.Getenv's shape; passed in so tests and main control the env.
type Getenv func(string) string

// PolicyFromEnv builds the policy: TRACE_RETENTION_DAYS (default 90) sets
// the trace, ended-conversation and inactivity windows; shipped usage rows
// always go after 30 days (nothing reads them locally once the thermometer
// has them).
func PolicyFromEnv(getenv Getenv) (store.RetentionPolicy, error) {
	p := store.DefaultRetention
	if v := strings.TrimSpace(getenv("TRACE_RETENTION_DAYS")); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 {
			return p, fmt.Errorf("TRACE_RETENTION_DAYS=%q: want an integer >= 1", v)
		}
		p.TraceDays = n
		p.ConversationDays = n
		p.InactivityDays = n
	}
	return p, nil
}

// Retainer is the slice of *store.Store the job needs.
type Retainer interface {
	Retention(ctx context.Context, p store.RetentionPolicy) (store.RetentionResult, error)
}

// Run applies p once and logs one line per run.
func Run(ctx context.Context, st Retainer, p store.RetentionPolicy, logf func(string, ...any)) (store.RetentionResult, error) {
	if logf == nil {
		logf = log.Printf
	}
	start := time.Now()
	res, err := st.Retention(ctx, p)
	if err != nil {
		logf("retention: FAILED after %s (traces=%d ended_inactive=%d conversations=%d usage_rows=%d): %v",
			time.Since(start).Round(time.Millisecond), res.Traces, res.Ended, res.Conversations, res.UsageRows, err)
		return res, err
	}
	logf("retention: ok in %s: traces=%d ended_inactive=%d conversations=%d usage_rows=%d (policy traces=%dd conversations=%dd shipped_usage=%dd inactivity=%dd)",
		time.Since(start).Round(time.Millisecond), res.Traces, res.Ended, res.Conversations, res.UsageRows,
		p.TraceDays, p.ConversationDays, p.ShippedUsageDays, p.InactivityDays)
	return res, nil
}

// Command is the `sous-chef -retention [--days N] [--dry-run]` hook:
// main.go calls Command(ctx, args, os.Getenv, os.Stdout) and exits with its
// error. It opens its own pool from DATABASE_URL, runs once, prints the
// counts as one line and closes. --dry-run only prints the policy and the
// cutoffs it would use.
func Command(ctx context.Context, args []string, getenv Getenv, out io.Writer) error {
	fs := flag.NewFlagSet("retention", flag.ContinueOnError)
	fs.SetOutput(out)
	days := fs.Int("days", -1, "override TRACE_RETENTION_DAYS for this run (>= 1)")
	dry := fs.Bool("dry-run", false, "print the policy and cutoffs, delete nothing")
	if err := fs.Parse(args); err != nil {
		return err
	}
	p, err := PolicyFromEnv(getenv)
	if err != nil {
		return err
	}
	if *days != -1 {
		if *days < 1 {
			return errors.New("--days must be >= 1")
		}
		p.TraceDays, p.ConversationDays, p.InactivityDays = *days, *days, *days
	}
	if err := p.Validate(); err != nil {
		return err
	}
	if *dry {
		tr, cv, us, idle := p.Cutoffs(time.Now().UTC())
		fmt.Fprintf(out, "retention dry-run: traces before %s, ended conversations before %s, shipped usage before %s, open conversations idle since before %s\n",
			tr.Format(time.RFC3339), cv.Format(time.RFC3339), us.Format(time.RFC3339), idle.Format(time.RFC3339))
		return nil
	}
	dsn := getenv("DATABASE_URL")
	if dsn == "" {
		return errors.New("DATABASE_URL is required")
	}
	st, err := store.Open(ctx, dsn)
	if err != nil {
		return err
	}
	defer st.Close()
	_, err = Run(ctx, st, p, func(format string, a ...any) { fmt.Fprintf(out, format+"\n", a...) })
	return err
}
