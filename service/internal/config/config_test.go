package config

import (
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
	  "local":{"kind":"messages-compatible","url":"https://gpu.internal/v1/messages","auth":"id-token","model":"qwen","max_tokens":2048,"stream":false}}`
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
	if !c.RouteAllowed("local") || c.RouteAllowed("opus") {
		t.Error("allowlist override not applied")
	}
}

func TestLoadRejectsBadValues(t *testing.T) {
	cases := map[string]map[string]string{
		"PORT":                  {"PORT": "abc"},
		"PORT: 0 out of range":  {"PORT": "0"},
		"MIGRATE_AT_BOOT":       {"MIGRATE_AT_BOOT": "maybe"},
		"MODEL_CALL_TIMEOUT":    {"MODEL_CALL_TIMEOUT": "soon"},
		"TURN_TTL: must be":     {"TURN_TTL": "-1s"},
		"MAX_CONCURRENT_TURNS":  {"MAX_CONCURRENT_TURNS": "0"},
		"TRACE_RETENTION_DAYS":  {"TRACE_RETENTION_DAYS": "0"},
		"must be set together":  {"THERMOMETER_API_URL": "https://x"},
		"MODEL_ROUTES: invalid": {"MODEL_ROUTES": "{"},
		"DEFAULT_ROUTE":         {"DEFAULT_ROUTE": "nope"},
		"MODEL_ROUTE_ALLOWLIST": {"MODEL_ROUTE_ALLOWLIST": "default,ghost"},
		"kind \"openai\"":       {"MODEL_ROUTES": `{"default":{"kind":"openai","auth":"api-key","model":"x","max_tokens":1}}`},
		"auth \"basic\"":        {"MODEL_ROUTES": `{"default":{"kind":"anthropic","auth":"basic","model":"x","max_tokens":1}}`},
		"model is required":     {"MODEL_ROUTES": `{"default":{"kind":"anthropic","auth":"api-key","max_tokens":1}}`},
		"max_tokens must be":    {"MODEL_ROUTES": `{"default":{"kind":"anthropic","auth":"api-key","model":"x"}}`},
		"routes need url":       {"MODEL_ROUTES": `{"default":{"kind":"messages-compatible","auth":"none","model":"x","max_tokens":1}}`},
		"url is only valid":     {"MODEL_ROUTES": `{"default":{"kind":"anthropic","url":"https://x","auth":"api-key","model":"x","max_tokens":1}}`},
		"effort \"extreme\"":    {"MODEL_ROUTES": `{"default":{"kind":"anthropic","auth":"api-key","model":"x","max_tokens":1,"effort":"extreme"}}`},
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
