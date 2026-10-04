// Package config parses the service's environment (README §7) into a Config.
//
// Load takes a getenv function so tests never touch the process environment.
// Every default in this file is the one the README documents; a missing
// required variable is an error, not a panic, so main can print it and exit.
package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"
)

// RouteKind is the wire protocol a model route speaks. Both kinds use the
// Anthropic Messages API shape; "messages-compatible" is a third-party
// endpoint (Ollama/vLLM) serving /v1/messages.
type RouteKind string

const (
	RouteAnthropic          RouteKind = "anthropic"
	RouteMessagesCompatible RouteKind = "messages-compatible"
)

// RouteAuth is how a route is authenticated.
type RouteAuth string

const (
	AuthAPIKey  RouteAuth = "api-key"
	AuthIDToken RouteAuth = "id-token"
	AuthNone    RouteAuth = "none"
)

// Route is one entry of MODEL_ROUTES (README §5).
type Route struct {
	Kind      RouteKind `json:"kind"`
	URL       string    `json:"url,omitempty"`
	Auth      RouteAuth `json:"auth"`
	Model     string    `json:"model"`
	MaxTokens int       `json:"max_tokens"`
	Effort    string    `json:"effort,omitempty"`
	// Stream is nil when the route did not say; internal/model applies the
	// kind-based default (true for anthropic, false for messages-compatible).
	Stream *bool `json:"stream,omitempty"`
	// Thinking ("disabled"|"enabled") is for messages-compatible routes only
	// and defaults to disabled: the request carries
	// {"thinking":{"type":"disabled"}} (Ollama maps it to think=false, so a
	// qwen3.5-family renderer emits the empty <think> block the SFT rows were
	// trained with) and thinking blocks are dropped from the history the
	// service re-sends. "enabled" sends {"type":"enabled"} and replays the
	// blocks. Anthropic routes reject the field: Claude runs adaptive
	// thinking with the field omitted (README §5).
	Thinking string `json:"thinking,omitempty"`
	// Temperature / TopP are sent as given when set (messages-compatible
	// routes only; nil = the server's default). Anthropic routes reject them:
	// adaptive thinking fixes temperature and disallows top_p.
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
}

// Thinking values a route may name.
const (
	ThinkingDisabled = "disabled"
	ThinkingEnabled  = "enabled"
)

// ThinkingEnabled reports whether this route requests and replays thinking:
// always on an anthropic route (adaptive, preserved thinking); on a
// messages-compatible route only when it says "thinking":"enabled".
func (r Route) ThinkingEnabled() bool {
	if r.Kind == RouteAnthropic {
		return true
	}
	return r.Thinking == ThinkingEnabled
}

// DefaultModelRoutes is the MODEL_ROUTES value used when the variable is
// unset. Model ids follow the claude-api skill as of 2026-10-01 (README §5).
const DefaultModelRoutes = `{"default":{"kind":"anthropic","auth":"api-key","model":"claude-sonnet-5-5","max_tokens":8192,"effort":"medium"},
 "opus":{"kind":"anthropic","auth":"api-key","model":"claude-opus-5-5","max_tokens":8192,"effort":"medium"}}`

// Config is the parsed environment.
type Config struct {
	Port          int
	DatabaseURL   string
	MigrateAtBoot bool
	ServiceKeys   []string

	AnthropicAPIKey  string
	AnthropicBaseURL string
	ModelRoutes      map[string]Route
	DefaultRoute     string
	RouteAllowlist   []string
	ModelCallTimeout time.Duration

	TurnTTL            time.Duration
	MaxConcurrentTurns int
	TraceRetentionDays int

	ThermometerAPIURL string
	ThermometerAPIKey string

	BuildID string
}

// Getenv is the lookup Load reads from; os.Getenv satisfies it.
type Getenv func(string) string

// Load parses the environment. Required: DATABASE_URL, SERVICE_KEYS.
// All validation errors are joined so the operator sees them at once.
func Load(getenv Getenv) (*Config, error) {
	var errs []error
	fail := func(format string, a ...any) { errs = append(errs, fmt.Errorf(format, a...)) }

	c := &Config{
		DatabaseURL:       getenv("DATABASE_URL"),
		AnthropicAPIKey:   getenv("ANTHROPIC_API_KEY"),
		AnthropicBaseURL:  getenv("ANTHROPIC_BASE_URL"),
		DefaultRoute:      def(getenv("DEFAULT_ROUTE"), "default"),
		ThermometerAPIURL: getenv("THERMOMETER_API_URL"),
		ThermometerAPIKey: getenv("THERMOMETER_API_KEY"),
		BuildID:           def(getenv("BUILD_ID"), "dev"),
	}

	var err error
	if c.Port, err = intEnv(getenv, "PORT", 8090); err != nil {
		errs = append(errs, err)
	} else if c.Port < 1 || c.Port > 65535 {
		fail("PORT: %d out of range", c.Port)
	}
	if c.DatabaseURL == "" {
		fail("DATABASE_URL is required")
	}
	switch v := strings.ToLower(getenv("MIGRATE_AT_BOOT")); v {
	case "", "off", "false", "0":
		c.MigrateAtBoot = false
	case "on", "true", "1":
		c.MigrateAtBoot = true
	default:
		fail("MIGRATE_AT_BOOT: %q is not on|off", v)
	}

	c.ServiceKeys = splitCSV(getenv("SERVICE_KEYS"))
	if len(c.ServiceKeys) == 0 {
		fail("SERVICE_KEYS is required (comma-separated bearer keys)")
	}

	routesJSON := def(getenv("MODEL_ROUTES"), DefaultModelRoutes)
	if err := json.Unmarshal([]byte(routesJSON), &c.ModelRoutes); err != nil {
		fail("MODEL_ROUTES: invalid JSON: %v", err)
	} else {
		for name, r := range c.ModelRoutes {
			if e := validateRoute(name, r); e != nil {
				errs = append(errs, e)
			}
		}
		if _, ok := c.ModelRoutes[c.DefaultRoute]; !ok && len(c.ModelRoutes) > 0 {
			fail("DEFAULT_ROUTE: %q is not in MODEL_ROUTES", c.DefaultRoute)
		}
	}
	c.RouteAllowlist = splitCSV(def(getenv("MODEL_ROUTE_ALLOWLIST"), "default,opus"))
	for _, name := range c.RouteAllowlist {
		if _, ok := c.ModelRoutes[name]; !ok && c.ModelRoutes != nil {
			fail("MODEL_ROUTE_ALLOWLIST: %q is not in MODEL_ROUTES", name)
		}
	}

	if c.ModelCallTimeout, err = durEnv(getenv, "MODEL_CALL_TIMEOUT", 60*time.Second); err != nil {
		errs = append(errs, err)
	}
	if c.TurnTTL, err = durEnv(getenv, "TURN_TTL", 10*time.Minute); err != nil {
		errs = append(errs, err)
	}
	if c.MaxConcurrentTurns, err = intEnv(getenv, "MAX_CONCURRENT_TURNS", 8); err != nil {
		errs = append(errs, err)
	} else if c.MaxConcurrentTurns < 1 {
		fail("MAX_CONCURRENT_TURNS must be >= 1")
	}
	if c.TraceRetentionDays, err = intEnv(getenv, "TRACE_RETENTION_DAYS", 90); err != nil {
		errs = append(errs, err)
	} else if c.TraceRetentionDays < 1 {
		fail("TRACE_RETENTION_DAYS must be >= 1")
	}
	if (c.ThermometerAPIURL == "") != (c.ThermometerAPIKey == "") {
		fail("THERMOMETER_API_URL and THERMOMETER_API_KEY must be set together")
	}

	if len(errs) > 0 {
		return nil, errors.Join(errs...)
	}
	return c, nil
}

// RouteNames returns the route table's names, sorted, for /version.
func (c *Config) RouteNames() []string {
	names := make([]string, 0, len(c.ModelRoutes))
	for n := range c.ModelRoutes {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// RouteAllowed reports whether a request may select the named route.
func (c *Config) RouteAllowed(name string) bool {
	for _, n := range c.RouteAllowlist {
		if n == name {
			return true
		}
	}
	return false
}

func validateRoute(name string, r Route) error {
	switch r.Kind {
	case RouteAnthropic:
		if r.URL != "" {
			return fmt.Errorf("MODEL_ROUTES[%s]: url is only valid for messages-compatible routes (use ANTHROPIC_BASE_URL)", name)
		}
	case RouteMessagesCompatible:
		if r.URL == "" {
			return fmt.Errorf("MODEL_ROUTES[%s]: messages-compatible routes need url", name)
		}
	default:
		return fmt.Errorf("MODEL_ROUTES[%s]: kind %q is not anthropic|messages-compatible", name, r.Kind)
	}
	switch r.Auth {
	case AuthAPIKey, AuthIDToken, AuthNone:
	default:
		return fmt.Errorf("MODEL_ROUTES[%s]: auth %q is not api-key|id-token|none", name, r.Auth)
	}
	if r.Model == "" {
		return fmt.Errorf("MODEL_ROUTES[%s]: model is required", name)
	}
	if r.MaxTokens < 1 {
		return fmt.Errorf("MODEL_ROUTES[%s]: max_tokens must be >= 1", name)
	}
	switch r.Effort {
	case "", "low", "medium", "high", "max":
	default:
		return fmt.Errorf("MODEL_ROUTES[%s]: effort %q is not low|medium|high|max", name, r.Effort)
	}
	switch r.Thinking {
	case "", ThinkingDisabled, ThinkingEnabled:
	default:
		return fmt.Errorf("MODEL_ROUTES[%s]: thinking %q is not disabled|enabled", name, r.Thinking)
	}
	if r.Kind == RouteAnthropic {
		if r.Thinking != "" {
			return fmt.Errorf("MODEL_ROUTES[%s]: thinking is only valid for messages-compatible routes (Claude runs adaptive thinking)", name)
		}
		if r.Temperature != nil || r.TopP != nil {
			return fmt.Errorf("MODEL_ROUTES[%s]: temperature/top_p are only valid for messages-compatible routes", name)
		}
	}
	if r.Temperature != nil && (*r.Temperature < 0 || *r.Temperature > 2) {
		return fmt.Errorf("MODEL_ROUTES[%s]: temperature %v is not within 0..2", name, *r.Temperature)
	}
	if r.TopP != nil && (*r.TopP < 0 || *r.TopP > 1) {
		return fmt.Errorf("MODEL_ROUTES[%s]: top_p %v is not within 0..1", name, *r.TopP)
	}
	return nil
}

func def(v, d string) string {
	if strings.TrimSpace(v) == "" {
		return d
	}
	return v
}

func splitCSV(s string) []string {
	var out []string
	for _, p := range strings.Split(s, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

func intEnv(getenv Getenv, key string, d int) (int, error) {
	v := strings.TrimSpace(getenv(key))
	if v == "" {
		return d, nil
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return 0, fmt.Errorf("%s: %q is not an integer", key, v)
	}
	return n, nil
}

func durEnv(getenv Getenv, key string, d time.Duration) (time.Duration, error) {
	v := strings.TrimSpace(getenv(key))
	if v == "" {
		return d, nil
	}
	dur, err := time.ParseDuration(v)
	if err != nil {
		return 0, fmt.Errorf("%s: %q is not a duration (e.g. 60s, 10m)", key, v)
	}
	if dur <= 0 {
		return 0, fmt.Errorf("%s: must be positive", key)
	}
	return dur, nil
}
