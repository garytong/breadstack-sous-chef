// Package api is the HTTP surface of Sous Chef (README §1): a stdlib
// ServeMux router, bearer-key middleware, the SSE event writer with
// heartbeats and its JSON fallback, and the handlers, which delegate to
// the ports in ports.go. Nothing here knows about Postgres or the model.
package api

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"runtime"
	"runtime/debug"
	"strings"
	"time"

	"github.com/breadstack/sous-chef/internal/auth"
)

// Options are the static settings of the HTTP layer.
type Options struct {
	// BuildID is reported by /version (BUILD_ID).
	BuildID string
	// Routes are the configured model route names, for /version.
	Routes []string
	// DefaultRouteConfigured gates /readyz: the default route must exist.
	DefaultRouteConfigured bool
	// Heartbeat is the SSE heartbeat interval; 0 → DefaultHeartbeat, <0 → off.
	Heartbeat time.Duration
	// MaxManifestBytes caps PUT /v1/manifests bodies; 0 → 8 MiB.
	MaxManifestBytes int64
	// MaxBodyBytes caps every other JSON body (turns, tool-results,
	// feedback); 0 → DefaultMaxBodyBytes (256 KB, README §1). Oversized → 413.
	MaxBodyBytes int64
	// ReadyTimeout bounds the /readyz ping; 0 → 2 s.
	ReadyTimeout time.Duration
	// Logf receives one line per request; nil → log.Printf.
	Logf func(format string, a ...any)
}

// DefaultMaxBodyBytes is the spec'd cap on turn / tool-results / feedback
// bodies: 256 KB.
const DefaultMaxBodyBytes = 256 << 10

// DefaultMaxManifestBytes caps PUT/POST manifest bodies: 8 MiB.
const DefaultMaxManifestBytes = 8 << 20

// Deps are the ports. A nil port makes its routes answer 501
// `not_implemented`, so the skeleton runs before the other packages land.
type Deps struct {
	Manifests ManifestRegistry
	// HashManifest backs POST /v1/manifests when Manifests does not
	// implement ComputeRegistrar (main wires manifest.HashBytes).
	HashManifest  ManifestHasher
	Turns         TurnRunner
	Conversations ConversationStore
	Feedback      FeedbackSink
	OptOut        OptOutStore
	DB            Pinger
}

// Server is the HTTP layer.
type Server struct {
	keys *auth.Keys
	opts Options
	deps Deps
}

// New builds a Server. keys must not be nil.
func New(keys *auth.Keys, opts Options, deps Deps) *Server {
	if opts.Heartbeat == 0 {
		opts.Heartbeat = DefaultHeartbeat
	}
	if opts.MaxManifestBytes == 0 {
		opts.MaxManifestBytes = DefaultMaxManifestBytes
	}
	if opts.MaxBodyBytes == 0 {
		opts.MaxBodyBytes = DefaultMaxBodyBytes
	}
	if opts.ReadyTimeout == 0 {
		opts.ReadyTimeout = 2 * time.Second
	}
	if opts.Logf == nil {
		opts.Logf = log.Printf
	}
	return &Server{keys: keys, opts: opts, deps: deps}
}

// Handler returns the full handler tree.
func (s *Server) Handler() http.Handler {
	root := http.NewServeMux()
	// Open routes (README §1): process up, dependencies up, build info.
	root.HandleFunc("GET /healthz", s.healthz)
	root.HandleFunc("GET /readyz", s.readyz)
	root.HandleFunc("GET /version", s.version)

	v1 := http.NewServeMux()
	v1.HandleFunc("POST /v1/manifests", s.postManifest)
	v1.HandleFunc("PUT /v1/manifests/{hash}", s.putManifest)
	v1.HandleFunc("GET /v1/manifests/{hash}", s.getManifest)
	v1.HandleFunc("POST /v1/turns", s.postTurn)
	v1.HandleFunc("POST /v1/turns/{turnId}/tool-results", s.postToolResults)
	v1.HandleFunc("GET /v1/conversations/{siteId}", s.getConversation)
	v1.HandleFunc("DELETE /v1/conversations/{siteId}", s.deleteConversation)
	v1.HandleFunc("POST /v1/feedback", s.postFeedback)
	v1.HandleFunc("PUT /v1/stores/{storeId}/trace-optout", s.putOptOut)
	v1.HandleFunc("DELETE /v1/stores/{storeId}/trace-optout", s.deleteOptOut)
	// Anything else under /v1 is a JSON 404 rather than the mux's text one.
	v1.HandleFunc("/v1/", func(w http.ResponseWriter, r *http.Request) {
		writeError(w, r, ErrNotFound("not_found", "no such route"))
	})

	reject := func(w http.ResponseWriter, r *http.Request) { writeError(w, r, ErrUnauthorized()) }
	root.Handle("/v1/", s.keys.Middleware(reject)(v1))

	return s.recoverer(s.logger(root))
}

// ---- middleware ----

// logger writes one line per request: method path status bytes duration.
func (s *Server) logger(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		sw := &statusWriter{ResponseWriter: w}
		next.ServeHTTP(sw, r)
		if r.URL.Path == "/healthz" || r.URL.Path == "/readyz" {
			return // probes are noise
		}
		s.opts.Logf("api: %s %s %d %dB %s", r.Method, r.URL.Path, sw.status(), sw.bytes, time.Since(start).Round(time.Millisecond))
	})
}

// recoverer turns a handler panic into a 500 when nothing was written yet,
// and otherwise just logs it (an SSE stream in flight cannot be rewound).
func (s *Server) recoverer(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if rec := recover(); rec != nil {
				if rec == http.ErrAbortHandler {
					panic(rec)
				}
				log.Printf("api: panic serving %s %s: %v\n%s", r.Method, r.URL.Path, rec, debug.Stack())
				if sw, ok := w.(*statusWriter); !ok || !sw.wrote {
					writeError(w, r, errors.New("panic"))
				}
			}
		}()
		next.ServeHTTP(w, r)
	})
}

// statusWriter records the status and byte count and keeps http.Flusher
// visible to the SSE writer.
type statusWriter struct {
	http.ResponseWriter
	code  int
	bytes int
	wrote bool
}

func (w *statusWriter) WriteHeader(code int) {
	if !w.wrote {
		w.code = code
		w.wrote = true
	}
	w.ResponseWriter.WriteHeader(code)
}

func (w *statusWriter) Write(b []byte) (int, error) {
	if !w.wrote {
		w.code = http.StatusOK
		w.wrote = true
	}
	n, err := w.ResponseWriter.Write(b)
	w.bytes += n
	return n, err
}

func (w *statusWriter) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// Unwrap lets http.ResponseController reach the underlying writer.
func (w *statusWriter) Unwrap() http.ResponseWriter { return w.ResponseWriter }

func (w *statusWriter) status() int {
	if !w.wrote {
		return http.StatusOK
	}
	return w.code
}

// ---- open routes ----

func (s *Server) healthz(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) readyz(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), s.opts.ReadyTimeout)
	defer cancel()
	checks := map[string]string{}
	ready := true
	switch {
	case s.deps.DB == nil:
		checks["db"] = "unconfigured"
		ready = false
	default:
		if err := s.deps.DB.Ping(ctx); err != nil {
			checks["db"] = "error: " + err.Error()
			ready = false
		} else {
			checks["db"] = "ok"
		}
	}
	if s.opts.DefaultRouteConfigured {
		checks["default_route"] = "ok"
	} else {
		checks["default_route"] = "missing"
		ready = false
	}
	status, code := "ready", http.StatusOK
	if !ready {
		status, code = "not_ready", http.StatusServiceUnavailable
	}
	writeJSON(w, code, map[string]any{"status": status, "checks": checks})
}

func (s *Server) version(w http.ResponseWriter, _ *http.Request) {
	routes := s.opts.Routes
	if routes == nil {
		routes = []string{}
	}
	writeJSON(w, http.StatusOK, map[string]any{"build_id": s.opts.BuildID, "go": runtime.Version(), "routes": routes})
}

// ---- helpers ----

// decodeJSON reads at most limit bytes of r.Body into v. It returns *Error
// bad_request on malformed JSON, trailing data, or an oversized body.
func decodeJSON(w http.ResponseWriter, r *http.Request, limit int64, v any) error {
	if ct := r.Header.Get("Content-Type"); ct != "" && !strings.HasPrefix(strings.ToLower(ct), "application/json") {
		return ErrBadRequest("Content-Type must be application/json")
	}
	body := http.MaxBytesReader(w, r.Body, limit)
	dec := json.NewDecoder(body)
	if err := dec.Decode(v); err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			return &Error{Status: http.StatusRequestEntityTooLarge, Code: "bad_request", Message: "body too large"}
		}
		if errors.Is(err, io.EOF) {
			return ErrBadRequest("empty body")
		}
		return ErrBadRequest("invalid JSON: " + err.Error())
	}
	if dec.More() {
		return ErrBadRequest("trailing data after JSON body")
	}
	return nil
}

// readAll reads at most limit bytes of r.Body (for the raw manifest PUT).
func readAll(w http.ResponseWriter, r *http.Request, limit int64) ([]byte, error) {
	if ct := r.Header.Get("Content-Type"); ct != "" && !strings.HasPrefix(strings.ToLower(ct), "application/json") {
		return nil, ErrBadRequest("Content-Type must be application/json")
	}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, limit))
	if err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			return nil, &Error{Status: http.StatusRequestEntityTooLarge, Code: "bad_request", Message: "body too large"}
		}
		return nil, ErrBadRequest("reading body: " + err.Error())
	}
	if len(body) == 0 {
		return nil, ErrBadRequest("empty body")
	}
	return body, nil
}

// isHexHash reports whether s is a lower-case 64-hex sha256.
func isHexHash(s string) bool {
	if len(s) != 64 {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}
