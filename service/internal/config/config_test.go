package config

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func env(m map[string]string) Getenv {
	return func(k string) string { return m[k] }
}

func minimal() map[string]string {
	return map[string]string{
		"DATABASE_URL": "postgres://u:p@localhost:5432/souschef",
		"SERVICE_KEYS": "k1",
	}
}

func TestLoadDefaults(t *testing.T) {
	c, err := Load(env(minimal()))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Port != 8090 {
		t.Errorf("Port = %d, want 8090", c.Port)
	}
	if c.MigrateAtBoot {
		t.Error("MigrateAtBoot should default off")
	}
	if c.DefaultRoute != "default" {
		t.Errorf("DefaultRoute = %q", c.DefaultRoute)
	}
	if got := strings.Join(c.RouteAllowlist, ","); got != "default,opus" {
		t.Errorf("RouteAllowlist = %q", got)
	}
	if c.ModelCallTimeout != 60*time.Second || c.TurnTTL != 10*time.Minute {
		t.Errorf("timeouts = %v / %v", c.ModelCallTimeout, c.TurnTTL)
	}
	if c.MaxConcurrentTurns != 8 || c.TraceRetentionDays != 90 {
		t.Errorf("MaxConcurrentTurns=%d TraceRetentionDays=%d", c.MaxConcurrentTurns, c.TraceRetentionDays)
	}
	if c.BuildID != "dev" {
		t.Errorf("BuildID = %q", c.BuildID)
	}
	if got := strings.Join(c.RouteNames(), ","); got != "default,opus" {
		t.Errorf("RouteNames = %q", got)
	}
	r := c.ModelRoutes["default"]
	if r.Kind != RouteAnthropic || r.Model != "claude-sonnet-5-5" || r.MaxTokens != 8192 || r.Effort != "medium" {
		t.Errorf("default route = %+v", r)
	}
	if c.ModelRoutes["opus"].Model != "claude-opus-5-5" {
		t.Errorf("opus route = %+v", c.ModelRoutes["opus"])
	}
	if !c.RouteAllowed("opus") || c.RouteAllowed("haiku") {
		t.Error("RouteAllowed wrong")
	}
}

func TestLoadRequired(t *testing.T) {
	_, err := Load(env(map[string]string{}))
	if err == nil {
		t.Fatal("expected error for empty env")
	}
	for _, want := range []string{"DATABASE_URL is required", "SERVICE_KEYS is required"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q lacks %q", err, want)
		}
	}
}

func TestLoadOverrides(t *testing.T) {
	m := minimal()
	m["PORT"] = "9000"
	m["MIGRATE_AT_BOOT"] = "on"
	m["SERVICE_KEYS"] = " a , b,,c "
	m["MODEL_CALL_TIMEOUT"] = "90s"
	m["TURN_TTL"] = "5m"
	m["MAX_CONCURRENT_TURNS"] = "2"
	m["TRACE_RETENTION_DAYS"] = "30"
	m["THERMOMETER_API_URL"] = "https://t.example/api/ingest/usage"
	m["THERMOMETER_API_KEY"] = "tm_x"
	m["BUILD_ID"] = "abc123"
	m["MODEL_ROUTES"] = `{"default":{"kind":"anthropic","auth":"api-key","model":"claude-sonnet-5-5","max_tokens":4096},
	  "local":{"kind":"messages-compatible","url":"https://gpu.internal/v1/messages","auth":"id-token","model":"qwen","max_tokens":2048,"stream":false},
	  "think":{"kind":"messages-compatible","url":"https://gpu.internal/v1/messages","auth":"none","model":"qwen","max_tokens":2048,"thinking":"enabled","temperature":0.2,"top_p":0.9}}`
	m["MODEL_ROUTE_ALLOWLIST"] = "default,local"
	c, err := Load(env(m))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Port != 9000 || !c.MigrateAtBoot || c.BuildID != "abc123" {
		t.Errorf("basic overrides: %+v", c)
	}
	if got := strings.Join(c.ServiceKeys, "|"); got != "a|b|c" {
		t.Errorf("ServiceKeys = %q (should trim and drop empties)", got)
	}
	if c.ModelCallTimeout != 90*time.Second || c.TurnTTL != 5*time.Minute || c.MaxConcurrentTurns != 2 || c.TraceRetentionDays != 30 {
		t.Errorf("numeric overrides: %+v", c)
	}
	local := c.ModelRoutes["local"]
	if local.Kind != RouteMessagesCompatible || local.Auth != AuthIDToken || local.Stream == nil || *local.Stream {
		t.Errorf("local route = %+v", local)
	}
	if c.ModelRoutes["default"].Stream != nil {
		t.Error("unset stream should stay nil so the model package can apply the kind default")
	}
	// thinking defaults: anthropic always on; messages-compatible off unless said
	if !c.ModelRoutes["default"].ThinkingEnabled() || local.ThinkingEnabled() || local.Thinking != "" || local.Temperature != nil || local.TopP != nil {
		t.Errorf("thinking/temperature/top_p defaults: default=%+v local=%+v", c.ModelRoutes["default"], local)
	}
	think := c.ModelRoutes["think"]
	if !think.ThinkingEnabled() || think.Temperature == nil || *think.Temperature != 0.2 || think.TopP == nil || *think.TopP != 0.9 {
		t.Errorf("think route = %+v", think)
	}
	if !c.RouteAllowed("local") || c.RouteAllowed("opus") {
		t.Error("allowlist override not applied")
	}
}

func TestLoadRejectsBadValues(t *testing.T) {
	cases := map[string]map[string]string{
		"PORT":                             {"PORT": "abc"},
		"PORT: 0 out of range":             {"PORT": "0"},
		"MIGRATE_AT_BOOT":                  {"MIGRATE_AT_BOOT": "maybe"},
		"MODEL_CALL_TIMEOUT":               {"MODEL_CALL_TIMEOUT": "soon"},
		"TURN_TTL: must be":                {"TURN_TTL": "-1s"},
		"MAX_CONCURRENT_TURNS":             {"MAX_CONCURRENT_TURNS": "0"},
		"TRACE_RETENTION_DAYS":             {"TRACE_RETENTION_DAYS": "0"},
		"must be set together":             {"THERMOMETER_API_URL": "https://x"},
		"MODEL_ROUTES: invalid":            {"MODEL_ROUTES": "{"},
		"DEFAULT_ROUTE":                    {"DEFAULT_ROUTE": "nope"},
		"MODEL_ROUTE_ALLOWLIST":            {"MODEL_ROUTE_ALLOWLIST": "default,ghost"},
		"kind \"openai\"":                  {"MODEL_ROUTES": `{"default":{"kind":"openai","auth":"api-key","model":"x","max_tokens":1}}`},
		"auth \"basic\"":                   {"MODEL_ROUTES": `{"default":{"kind":"anthropic","auth":"basic","model":"x","max_tokens":1}}`},
		"model is required":                {"MODEL_ROUTES": `{"default":{"kind":"anthropic","auth":"api-key","max_tokens":1}}`},
		"max_tokens must be":               {"MODEL_ROUTES": `{"default":{"kind":"anthropic","auth":"api-key","model":"x"}}`},
		"routes need url":                  {"MODEL_ROUTES": `{"default":{"kind":"messages-compatible","auth":"none","model":"x","max_tokens":1}}`},
		"url is only valid":                {"MODEL_ROUTES": `{"default":{"kind":"anthropic","url":"https://x","auth":"api-key","model":"x","max_tokens":1}}`},
		"effort \"extreme\"":               {"MODEL_ROUTES": `{"default":{"kind":"anthropic","auth":"api-key","model":"x","max_tokens":1,"effort":"extreme"}}`},
		"thinking \"auto\"":                {"MODEL_ROUTES": `{"default":{"kind":"messages-compatible","url":"https://x","auth":"none","model":"x","max_tokens":1,"thinking":"auto"}}`},
		"thinking is only valid":           {"MODEL_ROUTES": `{"default":{"kind":"anthropic","auth":"api-key","model":"x","max_tokens":1,"thinking":"disabled"}}`},
		"temperature/top_p are only valid": {"MODEL_ROUTES": `{"default":{"kind":"anthropic","auth":"api-key","model":"x","max_tokens":1,"temperature":0.5}}`},
		"temperature 3 is not":             {"MODEL_ROUTES": `{"default":{"kind":"messages-compatible","url":"https://x","auth":"none","model":"x","max_tokens":1,"temperature":3}}`},
		"fields_context \"auto\"":          {"MODEL_ROUTES": `{"default":{"kind":"messages-compatible","url":"https://x","auth":"none","model":"x","max_tokens":1,"fields_context":"auto"}}`},
		"top_p 1.5 is not":                 {"MODEL_ROUTES": `{"default":{"kind":"messages-compatible","url":"https://x","auth":"none","model":"x","max_tokens":1,"top_p":1.5}}`},
	}
	for want, extra := range cases {
		m := minimal()
		for k, v := range extra {
			m[k] = v
		}
		_, err := Load(env(m))
		if err == nil {
			t.Errorf("%s: expected error", want)
			continue
		}
		if !strings.Contains(err.Error(), want) {
			t.Errorf("%s: error %q lacks it", want, err)
		}
	}
}

func TestFieldsContextOption(t *testing.T) {
	m := minimal()
	m["MODEL_ROUTES"] = `{
	 "default":{"kind":"anthropic","auth":"api-key","model":"claude-sonnet-5-5","max_tokens":10},
	 "claude-on":{"kind":"anthropic","auth":"api-key","model":"claude-sonnet-5-5","max_tokens":10,"fields_context":"on"},
	 "claude-off":{"kind":"anthropic","auth":"api-key","model":"claude-sonnet-5-5","max_tokens":10,"fields_context":"off"},
	 "student":{"kind":"messages-compatible","url":"http://gpu/v1/messages","auth":"none","model":"m","max_tokens":10},
	 "student-on":{"kind":"messages-compatible","url":"http://gpu/v1/messages","auth":"none","model":"m","max_tokens":10,"fields_context":"on"},
	 "student-off":{"kind":"messages-compatible","url":"http://gpu/v1/messages","auth":"none","model":"m","max_tokens":10,"fields_context":"off"}}`
	m["MODEL_ROUTE_ALLOWLIST"] = "default"
	c, err := Load(env(m))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	for name, want := range map[string]bool{
		// Unset is on for every kind since 2026-10-10 (plan Q7); "off" opts out.
		"default": true, "claude-on": true, "claude-off": false,
		"student": true, "student-on": true, "student-off": false,
	} {
		if got := c.ModelRoutes[name].FieldsContextEnabled(); got != want {
			t.Errorf("%s: FieldsContextEnabled = %v, want %v", name, got, want)
		}
	}
}

func TestFallbackOption(t *testing.T) {
	const claude = `{"kind":"anthropic","auth":"api-key","model":"claude-sonnet-5-5","max_tokens":10}`
	student := func(extra string) string {
		return `{"kind":"messages-compatible","url":"http://gpu/v1/messages","auth":"none","model":"m","max_tokens":10` + extra + `}`
	}
	load := func(routes string) (*Config, error) {
		m := minimal()
		m["MODEL_ROUTES"] = routes
		m["MODEL_ROUTE_ALLOWLIST"] = "default"
		return Load(env(m))
	}
	// The fallback need not be allowlisted; the route without one has none.
	c, err := load(`{"default":` + student(`,"fallback":"claude"`) + `,"claude":` + claude + `}`)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.ModelRoutes["default"].Fallback != "claude" || c.ModelRoutes["claude"].Fallback != "" {
		t.Errorf("fallbacks = %q / %q", c.ModelRoutes["default"].Fallback, c.ModelRoutes["claude"].Fallback)
	}
	// /version's route_info: the effective fields_context and the fallback.
	info, _ := json.Marshal(c.RouteInfo())
	if string(info) != `{"claude":{"kind":"anthropic","fields_context":true,"fallback":null,"report_as":null},"default":{"kind":"messages-compatible","fields_context":true,"fallback":"claude","report_as":null}}` {
		t.Errorf("RouteInfo = %s", info)
	}
	c, err = load(`{"default":` + student(`,"fields_context":"off"`) + `,"claude":` + strings.TrimSuffix(claude, "}") + `,"fields_context":"on"}}`)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if ri := c.RouteInfo(); ri["default"].FieldsContext || !ri["claude"].FieldsContext || ri["default"].Fallback != nil {
		t.Errorf("RouteInfo with explicit options = %+v", ri)
	}
	for name, tc := range map[string]struct{ routes, want string }{
		"unknown": {`{"default":` + student(`,"fallback":"ghost"`) + `}`, `fallback "ghost" is not in MODEL_ROUTES`},
		"itself":  {`{"default":` + student(`,"fallback":"default"`) + `}`, "fallback names the route itself"},
		"chained": {`{"default":` + student(`,"fallback":"mid"`) + `,"mid":` + student(`,"fallback":"claude"`) + `,"claude":` + claude + `}`, "has a fallback of its own"},
	} {
		if _, err := load(tc.routes); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: error %v lacks %q", name, err, tc.want)
		}
	}
}

func TestReportAsOption(t *testing.T) {
	const claude = `{"kind":"anthropic","auth":"api-key","model":"claude-sonnet-5-5","max_tokens":10`
	const compat = `{"kind":"messages-compatible","url":"http://gpu","auth":"id-token","model":"sous-chef","max_tokens":10,"stream":false,"temperature":0.3,"fallback":"default"`
	load := func(routes string) (*Config, error) {
		m := minimal()
		m["MODEL_ROUTES"] = routes
		m["MODEL_ROUTE_ALLOWLIST"] = "default"
		return Load(env(m))
	}
	// The int Terraform's shape; any kind may set it.
	c, err := load(`{"default":` + claude + `},"blueberry-muffin":` + compat + `,"report_as":"blueberry-muffin"},"named":` + claude + `,"report_as":"claude.house_1"}}`)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	muffin, def := c.ModelRoutes["blueberry-muffin"], c.ModelRoutes["default"]
	if muffin.ReportAs != "blueberry-muffin" || muffin.Model != "sous-chef" || def.ReportAs != "" {
		t.Errorf("report_as = %q (model %q) / %q", muffin.ReportAs, muffin.Model, def.ReportAs)
	}
	// report_as wins over the id the backend echoes and over a failed call;
	// without it the response's id, else the route's model.
	for _, tc := range []struct {
		r          Route
		resp, want string
	}{
		{muffin, "sous-chef", "blueberry-muffin"}, {muffin, "", "blueberry-muffin"},
		{def, "claude-sonnet-5-5-20260901", "claude-sonnet-5-5-20260901"}, {def, "", "claude-sonnet-5-5"},
		{c.ModelRoutes["named"], "claude-sonnet-5-5", "claude.house_1"},
	} {
		if got := tc.r.ReportedModel(tc.resp); got != tc.want {
			t.Errorf("ReportedModel(%q) on %+v = %q, want %q", tc.resp, tc.r.ReportAs, got, tc.want)
		}
	}
	info, _ := json.Marshal(c.RouteInfo())
	if !strings.Contains(string(info), `"blueberry-muffin":{"kind":"messages-compatible","fields_context":true,"fallback":"default","report_as":"blueberry-muffin"}`) ||
		!strings.Contains(string(info), `"default":{"kind":"anthropic","fields_context":true,"fallback":null,"report_as":null}`) {
		t.Errorf("RouteInfo = %s", info)
	}
	// A plain token only.
	for _, bad := range []string{"Blueberry", "blueberry muffin", "-muffin", ".muffin", "muffin/1", "muffin:latest", "müffin", strings.Repeat("a", 65), " "} {
		if _, err := load(`{"default":` + claude + `,"report_as":"` + bad + `"}}`); err == nil || !strings.Contains(err.Error(), "report_as") {
			t.Errorf("report_as %q: error %v", bad, err)
		}
	}
	for _, good := range []string{"a", "9", "blueberry-muffin", "sous-chef.v2_q4", strings.Repeat("a", 64)} {
		if _, err := load(`{"default":` + claude + `,"report_as":"` + good + `"}}`); err != nil {
			t.Errorf("report_as %q refused: %v", good, err)
		}
	}
}

// Priming is opt-in: unset is off on every route kind.
func TestRoutePrime(t *testing.T) {
	load := func(routes string) (*Config, error) {
		m := minimal()
		m["MODEL_ROUTES"] = routes
		m["MODEL_ROUTE_ALLOWLIST"] = "default"
		return Load(env(m))
	}
	c, err := load(`{"default":{"kind":"messages-compatible","url":"http://m","auth":"none","model":"m","max_tokens":10},
		"on":{"kind":"messages-compatible","url":"http://m","auth":"none","model":"m","max_tokens":10,"prime":"on"},
		"off":{"kind":"messages-compatible","url":"http://m","auth":"none","model":"m","max_tokens":10,"prime":"off"},
		"filler":{"kind":"messages-compatible","url":"http://m","auth":"none","model":"m","max_tokens":10,"prime":"on","prime_filler_tokens":500},
		"claude":{"kind":"anthropic","auth":"api-key","model":"claude-sonnet-5-5","max_tokens":10,"prime":"off"}}`)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	for name, want := range map[string]struct {
		on     bool
		filler int
	}{"default": {false, DefaultPrimeFillerTokens}, "on": {true, DefaultPrimeFillerTokens}, "off": {false, DefaultPrimeFillerTokens}, "filler": {true, 500}, "claude": {false, DefaultPrimeFillerTokens}} {
		r := c.ModelRoutes[name]
		if r.PrimeEnabled() != want.on || r.PrimeFiller() != want.filler {
			t.Errorf("%s: PrimeEnabled=%v PrimeFiller=%d, want %v %d", name, r.PrimeEnabled(), r.PrimeFiller(), want.on, want.filler)
		}
	}
	for _, bad := range []string{
		`{"default":{"kind":"messages-compatible","url":"http://m","auth":"none","model":"m","max_tokens":10,"prime":"yes"}}`,
		`{"default":{"kind":"messages-compatible","url":"http://m","auth":"none","model":"m","max_tokens":10,"prime_filler_tokens":-1}}`,
		`{"default":{"kind":"messages-compatible","url":"http://m","auth":"none","model":"m","max_tokens":10,"prime_filler_tokens":9000}}`,
		`{"default":{"kind":"anthropic","auth":"api-key","model":"claude-sonnet-5-5","max_tokens":10,"prime":"on"}}`,
		`{"default":{"kind":"anthropic","auth":"api-key","model":"claude-sonnet-5-5","max_tokens":10,"prime_filler_tokens":1015}}`,
	} {
		if _, err := load(bad); err == nil || !strings.Contains(err.Error(), "prime") {
			t.Errorf("%s: err = %v, want a prime error", bad, err)
		}
	}
}
