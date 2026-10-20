package api

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/breadstack/sous-chef/internal/config"
)

// /version after this change: route_aliases (MODEL_ROUTE_ALIASES) and
// parser_errors (the routes with parser_retries "off", counted on this
// instance) appear only when there is something to say, so int's /version
// reads exactly as before until the chocolate-muffin route is configured.
func TestVersion_AliasesAndParserErrors(t *testing.T) {
	// Today's configuration: no aliases, no route without parser retries
	// (Table.ParserErrors answers nil) — the body has neither member.
	ts := newTestServer(t, Deps{DB: pinger{}}, func(o *Options) {
		o.DefaultRoute = "default"
		o.ParserErrors = func() map[string]int64 { return nil }
	})
	_, body := do(t, ts, "GET", "/version", "", "")
	if strings.Contains(body, "route_aliases") || strings.Contains(body, "parser_errors") {
		t.Errorf("version body = %s", body)
	}
	muffin, choc, def := "blueberry-muffin", "chocolate-muffin", "default"
	count := int64(0)
	ts = newTestServer(t, Deps{DB: pinger{}}, func(o *Options) {
		o.Routes = []string{"blueberry-muffin", "chocolate-muffin", "default"}
		o.DefaultRoute = "blueberry-muffin"
		o.RouteInfo = map[string]config.RouteInfo{
			"blueberry-muffin": {Kind: config.RouteMessagesCompatible, FieldsContext: true, Fallback: &def, ReportAs: &muffin},
			"chocolate-muffin": {Kind: config.RouteMessagesCompatible, FieldsContext: true, Fallback: &def, ReportAs: &choc, ParserRetries: config.ParserRetriesOff},
			"default":          {Kind: config.RouteAnthropic, FieldsContext: true},
		}
		o.RouteAliases = map[string]string{"blueberry-muffin": "chocolate-muffin"}
		o.ParserErrors = func() map[string]int64 { return map[string]int64{"chocolate-muffin": count} }
	})
	for _, want := range []int64{0, 3} {
		count = want
		_, body = do(t, ts, "GET", "/version", "", "")
		var doc map[string]json.RawMessage
		if err := json.Unmarshal([]byte(body), &doc); err != nil {
			t.Fatal(err)
		}
		if string(doc["default_route"]) != `"blueberry-muffin"` || string(doc["route_aliases"]) != `{"blueberry-muffin":"chocolate-muffin"}` ||
			string(doc["parser_errors"]) != `{"chocolate-muffin":`+map[int64]string{0: "0", 3: "3"}[want]+`}` ||
			!strings.Contains(string(doc["route_info"]), `"chocolate-muffin":{"kind":"messages-compatible","fields_context":true,"fallback":"default","report_as":"chocolate-muffin","parser_retries":"off"}`) ||
			!strings.Contains(string(doc["route_info"]), `"blueberry-muffin":{"kind":"messages-compatible","fields_context":true,"fallback":"default","report_as":"blueberry-muffin"}`) {
			t.Errorf("version body = %s", body)
		}
	}
}
