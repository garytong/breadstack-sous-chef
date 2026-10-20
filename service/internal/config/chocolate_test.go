package config

import (
	"encoding/json"
	"strings"
	"testing"
)

// intRoutes is int's MODEL_ROUTES as the live sous-chef revision runs it on
// 2026-10-19 (gcloud run services describe sous-chef), the model service's
// URL shortened.
const intRoutes = `{"blueberry-muffin":{"auth":"id-token","fallback":"default","fast_finish":"on","kind":"messages-compatible","max_tokens":1024,"model":"sous-chef","prime":"on","report_as":"blueberry-muffin","stream":false,"temperature":0.3,"url":"https://sous-chef-model.example.run.app"},` +
	`"default":{"auth":"api-key","effort":"medium","kind":"anthropic","max_tokens":8192,"model":"claude-sonnet-5-5"},` +
	`"opus":{"auth":"api-key","effort":"medium","kind":"anthropic","max_tokens":8192,"model":"claude-opus-5-5"}}`

// chocolateMuffin is the route docs/CHOCOLATE-MUFFIN.md asks Terraform to add
// beside blueberry-muffin: the same model service and Ollama model name, the
// same limits and fast finish, and what Ministral needs — temperature 0.15,
// no prompt-cache priming (calibrated for Qwen3.5's hybrid attention), and
// no silent redraw of a MinistralParser 500.
const chocolateMuffin = `"chocolate-muffin":{"auth":"id-token","fallback":"default","fast_finish":"on","fields_context":"on","kind":"messages-compatible","max_tokens":1024,"model":"sous-chef","parser_retries":"off","prime":"off","report_as":"chocolate-muffin","stream":false,"temperature":0.15,"url":"https://sous-chef-model.example.run.app"}`

func liveIntEnv(extra map[string]string) map[string]string {
	m := minimal()
	m["MODEL_ROUTES"] = intRoutes
	m["DEFAULT_ROUTE"] = "blueberry-muffin"
	m["MODEL_ROUTE_ALLOWLIST"] = "default,opus,blueberry-muffin"
	m["MODEL_CALL_TIMEOUT"] = "60s"
	for k, v := range extra {
		m[k] = v
	}
	return m
}

func withChocolate() string {
	return strings.TrimSuffix(intRoutes, "}") + "," + chocolateMuffin + "}"
}

// int's configuration of today loads as before: no aliases, every route
// retries parser errors (the SDK's 5xx rule), and /version's route_info is
// byte for byte what the previous build printed.
func TestIntConfigToday_Unchanged(t *testing.T) {
	c, err := Load(env(liveIntEnv(nil)))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.RouteAliases != nil || c.AnsweredBy("") != "blueberry-muffin" || c.AnsweredBy("opus") != "opus" {
		t.Errorf("aliases = %v, AnsweredBy(\"\") = %q", c.RouteAliases, c.AnsweredBy(""))
	}
	for name, r := range c.ModelRoutes {
		if !r.ParserRetriesEnabled() {
			t.Errorf("%s: parser retries off without the option", name)
		}
	}
	info, _ := json.Marshal(c.RouteInfo())
	const want = `{"blueberry-muffin":{"kind":"messages-compatible","fields_context":true,"fallback":"default","report_as":"blueberry-muffin"},` +
		`"default":{"kind":"anthropic","fields_context":true,"fallback":null,"report_as":null},` +
		`"opus":{"kind":"anthropic","fields_context":true,"fallback":null,"report_as":null}}`
	if string(info) != want {
		t.Errorf("RouteInfo = %s\nwant      %s", info, want)
	}
}

// The chocolate-muffin route as docs/CHOCOLATE-MUFFIN.md gives it.
func TestChocolateMuffinRoute(t *testing.T) {
	c, err := Load(env(liveIntEnv(map[string]string{"MODEL_ROUTES": withChocolate(),
		"MODEL_ROUTE_ALLOWLIST": "default,opus,blueberry-muffin,chocolate-muffin"})))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	cm := c.ModelRoutes["chocolate-muffin"]
	switch {
	case cm.Kind != RouteMessagesCompatible, !cm.FieldsContextEnabled(), cm.Fallback != "default", cm.ReportAs != "chocolate-muffin":
		t.Errorf("route = %+v", cm)
	case cm.PrimeEnabled(), cm.ParserRetriesEnabled(), !cm.FastFinishEnabled():
		t.Errorf("prime %v, parser retries %v, fast finish %v", cm.PrimeEnabled(), cm.ParserRetriesEnabled(), cm.FastFinishEnabled())
	case cm.Temperature == nil || *cm.Temperature != 0.15 || cm.MaxTokens != 1024 || cm.Model != "sous-chef" || cm.ThinkingEnabled():
		t.Errorf("sampling/limits = %+v", cm)
	}
	if got := cm.ReportedModel("sous-chef"); got != "chocolate-muffin" {
		t.Errorf("reported as %q", got)
	}
	// Available, not answering: DEFAULT_ROUTE still says blueberry-muffin.
	if c.AnsweredBy("") != "blueberry-muffin" {
		t.Errorf("AnsweredBy(\"\") = %q", c.AnsweredBy(""))
	}
	info := c.RouteInfo()
	if info["chocolate-muffin"].ParserRetries != "off" || info["blueberry-muffin"].ParserRetries != "" {
		t.Errorf("route_info parser_retries = %q / %q", info["chocolate-muffin"].ParserRetries, info["blueberry-muffin"].ParserRetries)
	}
	raw, _ := json.Marshal(info["chocolate-muffin"])
	if string(raw) != `{"kind":"messages-compatible","fields_context":true,"fallback":"default","report_as":"chocolate-muffin","parser_retries":"off"}` {
		t.Errorf("route_info[chocolate-muffin] = %s", raw)
	}
}

func TestParserRetriesOption(t *testing.T) {
	load := func(routes string) (*Config, error) {
		m := minimal()
		m["MODEL_ROUTES"] = routes
		m["MODEL_ROUTE_ALLOWLIST"] = "default"
		return Load(env(m))
	}
	c, err := load(`{"default":{"kind":"messages-compatible","url":"http://m","auth":"none","model":"m","max_tokens":10},
		"on":{"kind":"messages-compatible","url":"http://m","auth":"none","model":"m","max_tokens":10,"parser_retries":"on"},
		"off":{"kind":"messages-compatible","url":"http://m","auth":"none","model":"m","max_tokens":10,"parser_retries":"off"},
		"claude":{"kind":"anthropic","auth":"api-key","model":"claude-sonnet-5-5","max_tokens":10,"parser_retries":"on"}}`)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	for name, want := range map[string]bool{"default": true, "on": true, "off": false, "claude": true} {
		if got := c.ModelRoutes[name].ParserRetriesEnabled(); got != want {
			t.Errorf("%s: ParserRetriesEnabled = %v, want %v", name, got, want)
		}
	}
	for _, bad := range []string{
		`{"default":{"kind":"messages-compatible","url":"http://m","auth":"none","model":"m","max_tokens":10,"parser_retries":"never"}}`,
		`{"default":{"kind":"anthropic","auth":"api-key","model":"claude-sonnet-5-5","max_tokens":10,"parser_retries":"off"}}`,
	} {
		if _, err := load(bad); err == nil || !strings.Contains(err.Error(), "parser_retries") {
			t.Errorf("%s: err = %v, want a parser_retries error", bad, err)
		}
	}
}

// MODEL_ROUTE_ALIASES is the switch: blueberry-muffin's traffic, named or as
// the default, is answered by chocolate-muffin while DEFAULT_ROUTE keeps its
// value.
func TestRouteAliases(t *testing.T) {
	allow := "default,opus,blueberry-muffin,chocolate-muffin"
	c, err := Load(env(liveIntEnv(map[string]string{"MODEL_ROUTES": withChocolate(), "MODEL_ROUTE_ALLOWLIST": allow,
		"MODEL_ROUTE_ALIASES": `{"blueberry-muffin":"chocolate-muffin"}`})))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.DefaultRoute != "blueberry-muffin" || c.AnsweredBy("") != "chocolate-muffin" || c.AnsweredBy("blueberry-muffin") != "chocolate-muffin" ||
		c.AnsweredBy("chocolate-muffin") != "chocolate-muffin" || c.AnsweredBy("default") != "default" {
		t.Errorf("default %q, answered by %q / %q", c.DefaultRoute, c.AnsweredBy(""), c.AnsweredBy("blueberry-muffin"))
	}
	// The break-glass (docs/CHOCOLATE-MUFFIN.md §5): every in-house turn to
	// Claude's default route, with no spare L4 needed — int's allowlist
	// already admits "default", with or without chocolate-muffin's route.
	for _, routes := range []string{intRoutes, withChocolate()} {
		c, err := Load(env(liveIntEnv(map[string]string{"MODEL_ROUTES": routes, "MODEL_ROUTE_ALIASES": `{"blueberry-muffin":"default"}`})))
		if err != nil {
			t.Fatalf("break-glass alias: %v", err)
		}
		if c.DefaultRoute != "blueberry-muffin" || c.AnsweredBy("") != "default" || c.AnsweredBy("blueberry-muffin") != "default" ||
			c.ModelRoutes[c.AnsweredBy("")].Kind != RouteAnthropic {
			t.Errorf("break-glass: default %q, answered by %q", c.DefaultRoute, c.AnsweredBy(""))
		}
	}
	for _, blank := range []string{"", "  ", "{}", "null"} {
		c, err := Load(env(liveIntEnv(map[string]string{"MODEL_ROUTES": withChocolate(), "MODEL_ROUTE_ALLOWLIST": allow, "MODEL_ROUTE_ALIASES": blank})))
		if err != nil || c.RouteAliases != nil {
			t.Errorf("MODEL_ROUTE_ALIASES=%q: aliases %v, err %v", blank, c.RouteAliases, err)
		}
	}
	for name, tc := range map[string]struct{ aliases, allow, want string }{
		"bad json":        {`["blueberry-muffin"]`, allow, "MODEL_ROUTE_ALIASES: invalid JSON"},
		"unknown alias":   {`{"ghost":"chocolate-muffin"}`, allow, `MODEL_ROUTE_ALIASES[ghost]: "ghost" is not in MODEL_ROUTES`},
		"unknown target":  {`{"blueberry-muffin":"ghost"}`, allow, `"ghost" is not in MODEL_ROUTES`},
		"empty target":    {`{"blueberry-muffin":""}`, allow, "names no route"},
		"itself":          {`{"blueberry-muffin":"blueberry-muffin"}`, allow, "names the route itself"},
		"chained":         {`{"blueberry-muffin":"chocolate-muffin","chocolate-muffin":"default"}`, allow, "is an alias itself"},
		"not allowlisted": {`{"blueberry-muffin":"chocolate-muffin"}`, "default,opus,blueberry-muffin", "is not in MODEL_ROUTE_ALLOWLIST"},
	} {
		_, err := Load(env(liveIntEnv(map[string]string{"MODEL_ROUTES": withChocolate(), "MODEL_ROUTE_ALLOWLIST": tc.allow, "MODEL_ROUTE_ALIASES": tc.aliases})))
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: err = %v, want %q", name, err, tc.want)
		}
	}
}
