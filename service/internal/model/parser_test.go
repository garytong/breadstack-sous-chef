package model

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/breadstack/sous-chef/internal/config"
	"github.com/breadstack/sous-chef/internal/model/modeltest"
)

// The bodies Ollama 0.34.4's /v1/messages answers a parser failure with
// (middleware/anthropic.go writeError → anthropic.NewError(500, msg)).
func ollama500(msg string) modeltest.Step {
	return modeltest.Error(http.StatusInternalServerError, "api_error", msg, map[string]string{"Retry-After-Ms": "1"})
}

func TestIsParserMessage(t *testing.T) {
	for msg, want := range map[string]bool{
		// MinistralParser (model/parsers/ministral.go)
		"tool 'set_fieldz' not found": true,
		"tool '' not found":           true,
		"tool 'a'b' not found":        true,
		"invalid character '}' looking for beginning of object key string": true,
		"invalid character 'x' after object key":                           true,
		"unexpected end of JSON input":                                     true,
		"Value looks like object, but can't find closing '}' symbol":       true,
		"Malformed JSON error":                                             true,
		"not a valid UTF-8 string (at position 3): k\xff":                  true,
		// an argument number too large for a float64: what MinistralParser.Add
		// returns for {"value": 1e999} (encoding/json, the value into any)
		"json: cannot unmarshal number 1e999 into Go value of type float64":  true,
		"json: cannot unmarshal number -2e400 into Go value of type float64": true,
		"json: cannot unmarshal string into Go value of type float64":        false,
		"json: cannot unmarshal number 1e999 into Go struct field X.y":       false,
		// qwen3.5's tool calls (qwen3coder.go: encoding/xml)
		"XML syntax error on line 1: unexpected EOF":       true,
		"expected element type <function> but have <tool>": true,
		// anything else a model server may answer 500 with
		"llama runner process has terminated: exit status 2": false,
		"model requires more system memory":                  false,
		"an error was encountered while running the model":   false,
		"EOF":              false,
		"script exhausted": false,
		"":                 false,
		"tool not found":   false,
	} {
		if got := IsParserMessage(msg); got != want {
			t.Errorf("IsParserMessage(%q) = %v, want %v", msg, got, want)
		}
	}
}

func TestParserErrorMessage_Bodies(t *testing.T) {
	for body, want := range map[string]string{
		`{"type":"error","error":{"type":"api_error","message":"tool 'set_fieldz' not found"},"request_id":"req_1"}`: "tool 'set_fieldz' not found",
		`{"error":"tool 'x' not found"}`: "tool 'x' not found", // Ollama's native /api/chat shape
		`{"type":"error","error":{"type":"api_error","message":"llama runner process has terminated"}}`: "",
		`<html>500 Internal Server Error</html>`:                                                        "",
		`{"error":{"message":""}}`:                                                                      "",
		``:                                                                                              "",
	} {
		got, ok := ParserErrorMessage([]byte(body))
		if got != want || ok != (want != "") {
			t.Errorf("ParserErrorMessage(%s) = %q, %v; want %q", body, got, ok, want)
		}
	}
}

// parserTable is one messages-compatible route with the option set as given
// and the SDK's real retry count.
func parserTable(t *testing.T, fake *modeltest.Server, parserRetries string) *Table {
	t.Helper()
	return compatTable(t, fake, config.AuthNone, func(o *Options) {
		r := o.Routes["lab"]
		r.ParserRetries = parserRetries
		r.Stream = boolp(false)
		o.Routes["lab"] = r
	})
}

// The chocolate-muffin case: a MinistralParser 500 is answered once — no
// redraw — and comes back a counted *ParserError that still reads as the
// 500 it is (unavailable, status 500), so the loop handles it like any
// other failed call.
func TestParserRetriesOff_ParserErrorIsNotRetried(t *testing.T) {
	const body = `{"type":"error","error":{"type":"api_error","message":"tool 'set_fieldz' not found"},"request_id":"req_x"}`
	fake := modeltest.New(
		modeltest.Step{Status: 500, Body: body},
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("a redraw nobody should see")}}),
		modeltest.Step{Status: 500, Body: `{"type":"error","error":{"type":"api_error","message":"invalid character 'x' after object key"}}`},
	)
	defer fake.Close()
	tbl := parserTable(t, fake, config.ParserRetriesOff)
	b, _ := tbl.Route("")
	if b.ParserRetries() {
		t.Fatal("ParserRetries() = true")
	}
	_, err := b.Message(context.Background(), sampleRequest(), nil)
	pe, ok := AsParserError(err)
	if !ok {
		t.Fatalf("err = %T %v, want a *ParserError", err, err)
	}
	if pe.Route != "lab" || pe.Message != "tool 'set_fieldz' not found" || pe.Count != 1 {
		t.Errorf("ParserError = %+v", pe)
	}
	if !IsUnavailable(err) || StatusCode(err) != 500 {
		t.Errorf("IsUnavailable %v StatusCode %d: a parser error must read as the 500 it is", IsUnavailable(err), StatusCode(err))
	}
	var apiErr *anthropic.Error
	if !errors.As(err, &apiErr) || apiErr.RawJSON() != body {
		t.Errorf("the SDK's error must carry the server's body unchanged: %v", err)
	}
	if n := len(fake.Requests()); n != 1 {
		t.Fatalf("%d requests: a parser error must not be redrawn", n)
	}
	if got := fake.Requests()[0].Header.Get("X-Stainless-Retry-Count"); got != "0" {
		t.Errorf("retry count header %q", got)
	}
	// The next call is a new call (the redraw step is answered), and the
	// one after it counts as the route's second parser error.
	if resp, err := b.Message(context.Background(), sampleRequest(), nil); err != nil || resp.Text() != "a redraw nobody should see" {
		t.Fatalf("second call: %v", err)
	}
	_, err = b.Message(context.Background(), sampleRequest(), nil)
	if pe, ok := AsParserError(err); !ok || pe.Count != 2 || !strings.HasPrefix(pe.Message, "invalid character") {
		t.Fatalf("third call: %v", err)
	}
	if b.ParserErrors() != 2 || tbl.ParserErrors()["lab"] != 2 || len(fake.Requests()) != 3 {
		t.Errorf("counted %d / %v over %d requests", b.ParserErrors(), tbl.ParserErrors(), len(fake.Requests()))
	}
}

// Every other retry stays the SDK's on the same route: a 500 that is not a
// parser's, a 503, a 429.
func TestParserRetriesOff_OtherErrorsStillRetried(t *testing.T) {
	fake := modeltest.New(
		ollama500("llama runner process has terminated: exit status 2"),
		modeltest.Error(503, "overloaded_error", "busy", map[string]string{"Retry-After-Ms": "1"}),
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("third time")}}),
		modeltest.Error(429, "rate_limit_error", "slow down", map[string]string{"Retry-After-Ms": "1"}),
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("after the 429")}}),
	)
	defer fake.Close()
	tbl := parserTable(t, fake, config.ParserRetriesOff)
	b, _ := tbl.Route("")
	resp, err := b.Message(context.Background(), sampleRequest(), nil)
	if err != nil || resp.Text() != "third time" || len(fake.Requests()) != 3 {
		t.Fatalf("err %v, requests %d", err, len(fake.Requests()))
	}
	resp, err = b.Message(context.Background(), sampleRequest(), nil)
	if err != nil || resp.Text() != "after the 429" || len(fake.Requests()) != 5 {
		t.Fatalf("err %v, requests %d", err, len(fake.Requests()))
	}
	if b.ParserErrors() != 0 {
		t.Errorf("counted %d parser errors", b.ParserErrors())
	}
	// A non-parser 500 to the last attempt is the plain SDK error, as before.
	fake2 := modeltest.New(ollama500("boom"), ollama500("boom"), ollama500("boom"))
	defer fake2.Close()
	b2, _ := parserTable(t, fake2, config.ParserRetriesOff).Route("")
	_, err = b2.Message(context.Background(), sampleRequest(), nil)
	if _, isPE := AsParserError(err); isPE || StatusCode(err) != 500 || len(fake2.Requests()) != 3 {
		t.Errorf("err %T %v, requests %d", err, err, len(fake2.Requests()))
	}
}

// Without the option (every route today, blueberry-muffin included), or
// with it "on", a parser error is retried like any 5xx, and nothing is
// counted.
func TestParserRetriesDefault_RetriedAsBefore(t *testing.T) {
	for _, opt := range []string{"", config.ParserRetriesOn} {
		fake := modeltest.New(
			ollama500("tool 'set_fieldz' not found"),
			ollama500("tool 'set_fieldz' not found"),
			modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("drawn again")}}),
			ollama500("tool 'a' not found"), ollama500("tool 'b' not found"), ollama500("tool 'c' not found"),
		)
		tbl := parserTable(t, fake, opt)
		b, _ := tbl.Route("")
		resp, err := b.Message(context.Background(), sampleRequest(), nil)
		if err != nil || resp.Text() != "drawn again" || len(fake.Requests()) != 3 {
			t.Fatalf("parser_retries %q: %v after %d requests", opt, err, len(fake.Requests()))
		}
		_, err = b.Message(context.Background(), sampleRequest(), nil)
		if _, isPE := AsParserError(err); isPE || !IsUnavailable(err) || StatusCode(err) != 500 || len(fake.Requests()) != 6 {
			t.Errorf("parser_retries %q exhausted: %T %v after %d requests", opt, err, err, len(fake.Requests()))
		}
		if tbl.ParserErrors() != nil || !b.ParserRetries() || b.ParserErrors() != 0 {
			t.Errorf("parser_retries %q: ParserErrors %v", opt, tbl.ParserErrors())
		}
		fake.Close()
	}
}

func TestParserRetriesOff_TimeoutIsUntouched(t *testing.T) {
	fake := modeltest.New(modeltest.Step{Delay: 500 * time.Millisecond, Message: &modeltest.Message{Content: []modeltest.Block{modeltest.Text("late")}}})
	defer fake.Close()
	b, _ := parserTable(t, fake, config.ParserRetriesOff).Route("")
	b.timeout = 100 * time.Millisecond
	_, err := b.Message(context.Background(), sampleRequest(), nil)
	if _, isPE := AsParserError(err); isPE || !IsUnavailable(err) || b.ParserErrors() != 0 {
		t.Errorf("timeout: %T %v", err, err)
	}
}

// MODEL_ROUTE_ALIASES: the switch between two routes on one model server.
func aliasTable(t *testing.T, fake *modeltest.Server, aliases map[string]string, allow []string) (*Table, error) {
	t.Helper()
	return NewTable(Options{
		Routes: map[string]config.Route{
			"default":          {Kind: config.RouteAnthropic, Auth: config.AuthAPIKey, Model: "claude-sonnet-5-5", MaxTokens: 10},
			"blueberry-muffin": {Kind: config.RouteMessagesCompatible, URL: fake.URL, Auth: config.AuthNone, Model: "sous-chef", MaxTokens: 10, Fallback: "default", ReportAs: "blueberry-muffin", Prime: config.PrimeOn},
			"chocolate-muffin": {Kind: config.RouteMessagesCompatible, URL: fake.URL, Auth: config.AuthNone, Model: "sous-chef", MaxTokens: 10, Fallback: "default", ReportAs: "chocolate-muffin", ParserRetries: config.ParserRetriesOff},
		},
		Default: "blueberry-muffin", Allowlist: allow, Aliases: aliases, APIKey: "k", CallTimeout: time.Second,
	})
}

func TestAliases_RouteResolution(t *testing.T) {
	fake := modeltest.New()
	defer fake.Close()
	all := []string{"default", "blueberry-muffin", "chocolate-muffin"}
	// None: every name answers itself, as before.
	tbl, err := aliasTable(t, fake, nil, all)
	if err != nil {
		t.Fatal(err)
	}
	for name, want := range map[string]string{"": "blueberry-muffin", "blueberry-muffin": "blueberry-muffin", "chocolate-muffin": "chocolate-muffin", "default": "default"} {
		if b, err := tbl.Route(name); err != nil || b.Name() != want {
			t.Errorf("no aliases: Route(%q) = %v, %v", name, b, err)
		}
	}
	if tbl.Aliases() != nil {
		t.Errorf("Aliases() = %v", tbl.Aliases())
	}
	if pe := tbl.ParserErrors(); len(pe) != 1 || pe["chocolate-muffin"] != 0 {
		t.Errorf("ParserErrors() = %v", pe)
	}
	// The switch: blueberry-muffin's traffic, named or default, is chocolate-muffin's.
	tbl, err = aliasTable(t, fake, map[string]string{"blueberry-muffin": "chocolate-muffin"}, all)
	if err != nil {
		t.Fatal(err)
	}
	for name, want := range map[string]string{"": "chocolate-muffin", "blueberry-muffin": "chocolate-muffin", "chocolate-muffin": "chocolate-muffin", "default": "default"} {
		b, err := tbl.Route(name)
		if err != nil || b.Name() != want {
			t.Fatalf("aliased: Route(%q) = %v, %v", name, b, err)
		}
		if want == "chocolate-muffin" && (b.ReportedModel("sous-chef") != "chocolate-muffin" || b.PrimeEnabled() || b.ParserRetries()) {
			t.Errorf("Route(%q) answers with the wrong settings", name)
		}
	}
	if tbl.Default() != "blueberry-muffin" || tbl.Aliases()["blueberry-muffin"] != "chocolate-muffin" {
		t.Errorf("Default %q, Aliases %v", tbl.Default(), tbl.Aliases())
	}
	// The allowlist still applies to the name the request says.
	tbl, err = aliasTable(t, fake, map[string]string{"blueberry-muffin": "chocolate-muffin"}, []string{"default", "chocolate-muffin"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := tbl.Route("blueberry-muffin"); !errors.Is(err, ErrRouteNotAllowed) {
		t.Errorf("a name that is not allowlisted: %v", err)
	}
	if b, err := tbl.Route(""); err != nil || b.Name() != "chocolate-muffin" {
		t.Errorf("default: %v %v", b, err)
	}
	// Misconfigurations are refused, as config.Load refuses them.
	for name, tc := range map[string]struct {
		aliases map[string]string
		allow   []string
	}{
		"target not allowlisted": {map[string]string{"blueberry-muffin": "chocolate-muffin"}, []string{"default", "blueberry-muffin"}},
		"chain":                  {map[string]string{"blueberry-muffin": "chocolate-muffin", "chocolate-muffin": "default"}, all},
		"unknown":                {map[string]string{"ghost": "default"}, all},
	} {
		if _, err := aliasTable(t, fake, tc.aliases, tc.allow); err == nil || !strings.Contains(err.Error(), "MODEL_ROUTE_ALIASES") {
			t.Errorf("%s: err = %v", name, err)
		}
	}
}

type brokenBody struct{ sent bool }

func (b *brokenBody) Read(p []byte) (int, error) {
	if !b.sent {
		b.sent = true
		return copy(p, `{"error":{"message":"tool 'x' not`), nil
	}
	return 0, errors.New("connection reset")
}
func (b *brokenBody) Close() error { return nil }

// parserGuard leaves what it cannot read to the SDK, unchanged: a body that
// breaks mid-way breaks the same way, a body past its bound is passed on
// whole, and only a 500 is looked at.
func TestParserGuard_EdgeBodies(t *testing.T) {
	b := &Backend{name: "lab", route: config.Route{ParserRetries: config.ParserRetriesOff}}
	req, _ := http.NewRequest(http.MethodPost, "http://m/v1/messages", nil)
	respond := func(status int, body io.ReadCloser) func(*http.Request) (*http.Response, error) {
		return func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: status, Header: http.Header{}, Body: body}, nil
		}
	}
	res, err := b.parserGuard(req, respond(500, &brokenBody{}))
	if err != nil || res.Header.Get("X-Should-Retry") != "" {
		t.Fatalf("broken body: %v %v", err, res.Header)
	}
	got, rerr := io.ReadAll(res.Body)
	if string(got) != `{"error":{"message":"tool 'x' not` || rerr == nil || rerr.Error() != "connection reset" {
		t.Errorf("broken body re-read: %q, %v", got, rerr)
	}
	big := `{"error":{"message":"tool 'x' not found","pad":"` + strings.Repeat("p", maxErrorBody) + `"}}`
	res, _ = b.parserGuard(req, respond(500, io.NopCloser(strings.NewReader(big))))
	if got, _ := io.ReadAll(res.Body); string(got) != big || res.Header.Get("X-Should-Retry") != "" {
		t.Errorf("oversized body: %d bytes back, header %q", len(got), res.Header.Get("X-Should-Retry"))
	}
	parser := `{"type":"error","error":{"type":"api_error","message":"tool 'x' not found"}}`
	for _, status := range []int{200, 400, 502, 503} {
		res, _ = b.parserGuard(req, respond(status, io.NopCloser(strings.NewReader(parser))))
		if got, _ := io.ReadAll(res.Body); string(got) != parser || res.Header.Get("X-Should-Retry") != "" {
			t.Errorf("status %d: touched (%q, %q)", status, got, res.Header.Get("X-Should-Retry"))
		}
	}
	res, _ = b.parserGuard(req, respond(500, io.NopCloser(strings.NewReader(parser))))
	if got, _ := io.ReadAll(res.Body); string(got) != parser || res.Header.Get("X-Should-Retry") != "false" {
		t.Errorf("parser 500: %q, %q", got, res.Header.Get("X-Should-Retry"))
	}
}
