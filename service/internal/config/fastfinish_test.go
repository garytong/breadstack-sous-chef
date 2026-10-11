package config

import (
	"strings"
	"testing"
)

// fast_finish is per route, any kind, off unless "on".
func TestFastFinishOption(t *testing.T) {
	m := minimal()
	m["MODEL_ROUTES"] = `{
	 "default":{"kind":"anthropic","auth":"api-key","model":"claude-sonnet-5-5","max_tokens":10},
	 "claude-on":{"kind":"anthropic","auth":"api-key","model":"claude-sonnet-5-5","max_tokens":10,"fast_finish":"on"},
	 "student":{"kind":"messages-compatible","url":"http://gpu/v1/messages","auth":"none","model":"m","max_tokens":10},
	 "student-on":{"kind":"messages-compatible","url":"http://gpu/v1/messages","auth":"none","model":"m","max_tokens":10,"fast_finish":"on"},
	 "student-off":{"kind":"messages-compatible","url":"http://gpu/v1/messages","auth":"none","model":"m","max_tokens":10,"fast_finish":"off"}}`
	m["MODEL_ROUTE_ALLOWLIST"] = "default"
	c, err := Load(env(m))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	for name, want := range map[string]bool{
		"default": false, "claude-on": true, "student": false, "student-on": true, "student-off": false,
	} {
		if got := c.ModelRoutes[name].FastFinishEnabled(); got != want {
			t.Errorf("%s: FastFinishEnabled = %v, want %v", name, got, want)
		}
	}

	m["MODEL_ROUTES"] = `{"default":{"kind":"anthropic","auth":"api-key","model":"claude-sonnet-5-5","max_tokens":10,"fast_finish":"yes"}}`
	if _, err := Load(env(m)); err == nil || !strings.Contains(err.Error(), `fast_finish "yes" is not on|off`) {
		t.Fatalf("Load = %v", err)
	}
}
