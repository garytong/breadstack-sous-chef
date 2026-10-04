// Package model is the service's only way to talk to a model (README §5).
//
// Every route speaks the Anthropic Messages API shape through the official
// SDK's beta namespace, as breadstack-thermometer's chat.go does: an
// "anthropic" route is Claude itself (x-api-key, ANTHROPIC_BASE_URL
// override, server-side fallbacks, prompt caching, effort); a
// "messages-compatible" route is an Ollama/vLLM-style /v1/messages endpoint
// that gets plainRequest (no Claude-only fields) and either an API key, a
// Cloud Run ID token (X-Serverless-Authorization) or nothing.
//
// The Table is built once from the config package's route table; a turn
// picks a Backend by route name, allowlist-checked, and calls Message.
package model

import (
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/breadstack/sous-chef/internal/config"
)

// Defaults (README §5, §7).
const (
	DefaultCallTimeout = 60 * time.Second
	DefaultMaxRetries  = 2 // 408/409/429/5xx and connection errors, SDK backoff (0.5 s × 2ⁿ, capped 8 s, Retry-After honoured)
)

// Sentinel errors a caller maps to HTTP codes.
var (
	ErrUnknownRoute    = errors.New("model: unknown route")
	ErrRouteNotAllowed = errors.New("model: route not allowlisted")
)

// Options builds a Table. Routes/Default/Allowlist/APIKey/BaseURL/CallTimeout
// mirror config.Config; FromConfig fills them.
type Options struct {
	Routes      map[string]config.Route
	Default     string   // route used when a turn names none; must be in Routes
	Allowlist   []string // routes a turn may name (README §5 MODEL_ROUTE_ALLOWLIST)
	APIKey      string   // ANTHROPIC_API_KEY, for "anthropic" routes
	BaseURL     string   // ANTHROPIC_BASE_URL, "" = the SDK's default
	CallTimeout time.Duration
	MaxRetries  *int // nil = DefaultMaxRetries

	// RouteAPIKeys holds the key of each messages-compatible route with auth
	// "api-key", by route name (RouteAPIKeysFromEnv reads
	// MODEL_ROUTE_<NAME>_API_KEY). Sent as `Authorization: Bearer` and
	// `x-api-key`.
	RouteAPIKeys map[string]string
	// Tokens mints ID tokens for auth "id-token" routes; nil = the Cloud Run
	// metadata server.
	Tokens TokenSource
	// HTTPClient is used for every route; nil = http.DefaultClient-like
	// client with the SDK's defaults. Tests inject one.
	HTTPClient *http.Client
	// Now is the clock latencies are measured with; nil = time.Now.
	Now func() time.Time
}

// FromConfig derives Options from the parsed environment. getenv supplies
// the per-route keys (MODEL_ROUTE_<NAME>_API_KEY); nil reads none.
func FromConfig(cfg *config.Config, getenv func(string) string) Options {
	o := Options{
		Routes:      cfg.ModelRoutes,
		Default:     cfg.DefaultRoute,
		Allowlist:   cfg.RouteAllowlist,
		APIKey:      cfg.AnthropicAPIKey,
		BaseURL:     cfg.AnthropicBaseURL,
		CallTimeout: cfg.ModelCallTimeout,
	}
	if getenv != nil {
		o.RouteAPIKeys = RouteAPIKeysFromEnv(getenv, cfg.ModelRoutes)
	}
	return o
}

// RouteAPIKeysFromEnv reads MODEL_ROUTE_<NAME>_API_KEY for every route, the
// name upper-cased with `-` and `.` turned into `_`. Missing variables are
// simply absent from the map.
func RouteAPIKeysFromEnv(getenv func(string) string, routes map[string]config.Route) map[string]string {
	keys := map[string]string{}
	for name := range routes {
		if v := getenv(RouteAPIKeyEnv(name)); v != "" {
			keys[name] = v
		}
	}
	return keys
}

// RouteAPIKeyEnv is the environment variable that carries a route's key.
func RouteAPIKeyEnv(name string) string {
	up := strings.ToUpper(name)
	up = strings.NewReplacer("-", "_", ".", "_").Replace(up)
	return "MODEL_ROUTE_" + up + "_API_KEY"
}

// Table is the route table: one Backend per route.
type Table struct {
	backends map[string]*Backend
	def      string
	allow    map[string]bool
}

// Backend is one route, ready to call.
type Backend struct {
	name    string
	route   config.Route
	client  anthropic.Client
	timeout time.Duration
	stream  bool
	now     func() time.Time
	// compatible-route auth; nil/"" = send none
	tokens TokenSource
	apiKey string
}

// NewTable validates the options and builds every route's client. Nothing
// is sent anywhere.
func NewTable(o Options) (*Table, error) {
	if len(o.Routes) == 0 {
		return nil, errors.New("model: no routes configured")
	}
	if _, ok := o.Routes[o.Default]; !ok {
		return nil, fmt.Errorf("model: default route %q is not in the route table", o.Default)
	}
	if o.CallTimeout <= 0 {
		o.CallTimeout = DefaultCallTimeout
	}
	retries := DefaultMaxRetries
	if o.MaxRetries != nil {
		if *o.MaxRetries < 0 {
			return nil, errors.New("model: max retries must be >= 0")
		}
		retries = *o.MaxRetries
	}
	now := o.Now
	if now == nil {
		now = time.Now
	}
	t := &Table{backends: map[string]*Backend{}, def: o.Default, allow: map[string]bool{}}
	for _, name := range o.Allowlist {
		if _, ok := o.Routes[name]; !ok {
			return nil, fmt.Errorf("model: allowlisted route %q is not in the route table", name)
		}
		t.allow[name] = true
	}
	names := make([]string, 0, len(o.Routes))
	for n := range o.Routes {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, name := range names {
		b, err := newBackend(name, o.Routes[name], o, retries, now)
		if err != nil {
			return nil, err
		}
		t.backends[name] = b
	}
	return t, nil
}

func newBackend(name string, r config.Route, o Options, retries int, now func() time.Time) (*Backend, error) {
	if r.Model == "" || r.MaxTokens < 1 {
		return nil, fmt.Errorf("model: route %q: model and max_tokens are required", name)
	}
	b := &Backend{name: name, route: r, timeout: o.CallTimeout, now: now}
	opts := []option.RequestOption{option.WithMaxRetries(retries)}
	if o.HTTPClient != nil {
		opts = append(opts, option.WithHTTPClient(o.HTTPClient))
	}
	switch r.Kind {
	case config.RouteAnthropic:
		if r.Auth != config.AuthAPIKey {
			return nil, fmt.Errorf("model: route %q: anthropic routes use auth api-key, got %q", name, r.Auth)
		}
		if o.APIKey == "" {
			return nil, fmt.Errorf("model: route %q: ANTHROPIC_API_KEY is required", name)
		}
		// No ANTHROPIC_* variable reaches the client behind the operator's
		// back: the key and base URL come from config alone.
		opts = append(opts, option.WithoutEnvironmentDefaults(), option.WithAPIKey(o.APIKey))
		if o.BaseURL != "" {
			if err := checkURL(o.BaseURL); err != nil {
				return nil, fmt.Errorf("model: ANTHROPIC_BASE_URL: %w", err)
			}
			opts = append(opts, option.WithBaseURL(strings.TrimSuffix(o.BaseURL, "/")))
		}
		b.stream = true
	case config.RouteMessagesCompatible:
		if err := checkURL(r.URL); err != nil {
			return nil, fmt.Errorf("model: route %q: url: %w", name, err)
		}
		switch r.Auth {
		case config.AuthAPIKey:
			key := o.RouteAPIKeys[name]
			if key == "" {
				return nil, fmt.Errorf("model: route %q: auth api-key needs %s", name, RouteAPIKeyEnv(name))
			}
			b.apiKey = key
		case config.AuthIDToken:
			b.tokens = o.Tokens
			if b.tokens == nil {
				b.tokens = NewMetadataTokens()
			}
		case config.AuthNone:
		default:
			return nil, fmt.Errorf("model: route %q: auth %q is not api-key|id-token|none", name, r.Auth)
		}
		// As thermometer's inhouseLink: no credential of the client's own,
		// the route's URL, and nothing but our own header on the wire (sign).
		opts = append(opts, option.WithoutEnvironmentDefaults(),
			option.WithBaseURL(strings.TrimSuffix(r.URL, "/")),
			option.WithMiddleware(b.sign))
		b.stream = false
	default:
		return nil, fmt.Errorf("model: route %q: kind %q is not anthropic|messages-compatible", name, r.Kind)
	}
	if r.Stream != nil {
		b.stream = *r.Stream
	}
	b.client = anthropic.NewClient(opts...)
	return b, nil
}

func checkURL(s string) error {
	u, err := url.Parse(s)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return fmt.Errorf("must be an http(s) URL, got %q", s)
	}
	return nil
}

// sign is a messages-compatible backend's middleware (thermometer
// sourdough.go sign/authorize): any credential header the SDK might have
// set is dropped, then the route's own goes on — the ID token in
// X-Serverless-Authorization (Cloud Run checks and strips it before the
// request reaches the model server), or the API key as Bearer + x-api-key.
func (b *Backend) sign(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
	req.Header.Del("X-Api-Key")
	req.Header.Del("Authorization")
	req.Header.Del("X-Serverless-Authorization")
	switch {
	case b.tokens != nil:
		token, err := b.tokens.IDToken(req.Context(), b.route.URL)
		if err != nil {
			return nil, fmt.Errorf("route %s ID token: %w", b.name, err)
		}
		req.Header.Set("X-Serverless-Authorization", "Bearer "+token)
	case b.apiKey != "":
		req.Header.Set("Authorization", "Bearer "+b.apiKey)
		req.Header.Set("X-Api-Key", b.apiKey)
	}
	return next(req)
}

// Route returns the backend for name ("" = the default route). A name that
// is not allowlisted is refused even when it exists.
func (t *Table) Route(name string) (*Backend, error) {
	if name == "" {
		return t.backends[t.def], nil
	}
	b, ok := t.backends[name]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrUnknownRoute, name)
	}
	if !t.allow[name] {
		return nil, fmt.Errorf("%w: %q", ErrRouteNotAllowed, name)
	}
	return b, nil
}

// Default is the default route's name.
func (t *Table) Default() string { return t.def }

// Names lists the routes, sorted (for /version).
func (t *Table) Names() []string {
	out := make([]string, 0, len(t.backends))
	for n := range t.backends {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}

// Name is the route's name; Model the model id it sends; Kind its wire kind;
// Streams whether it reads the model's SSE (text.delta as the model writes).
func (b *Backend) Name() string           { return b.name }
func (b *Backend) Model() string          { return b.route.Model }
func (b *Backend) Kind() config.RouteKind { return b.route.Kind }
func (b *Backend) Streams() bool          { return b.stream }
func (b *Backend) Timeout() time.Duration { return b.timeout }
