package loop

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/breadstack/sous-chef/internal/api"
	"github.com/breadstack/sous-chef/internal/config"
	"github.com/breadstack/sous-chef/internal/model"
	"github.com/breadstack/sous-chef/internal/model/modeltest"
)

// muffins is a harness on int's shape after this change: the two in-house
// routes on one model server — blueberry-muffin (Qwen settings: prime on)
// and chocolate-muffin (parser_retries off, prime off) — each falling back
// to Claude's "default", plus "chocolate-solo" (the same without a
// fallback, as the lab's harness runs it). The SDK's real retry count.
type muffins struct {
	*harness
	ollama *modeltest.Server
	claude *modeltest.Server
	logMu  sync.Mutex
	logs   []string
}

func newMuffins(t *testing.T, aliases map[string]string, ollamaSteps, claudeSteps []modeltest.Step) *muffins {
	t.Helper()
	mf := &muffins{ollama: modeltest.New(answering("sous-chef", ollamaSteps)...)}
	t.Cleanup(mf.ollama.Close)
	mf.harness = newHarness(t, func(d *Deps) {
		mf.claude = modeltest.New(answering("claude-sonnet-5-5", claudeSteps)...)
		t.Cleanup(mf.claude.Close)
		inhouse := func(report string) config.Route {
			return config.Route{Kind: config.RouteMessagesCompatible, URL: mf.ollama.URL, Auth: config.AuthNone, Model: "sous-chef",
				MaxTokens: 1024, Stream: boolPtr(false), Fallback: "default", ReportAs: report, FastFinish: config.FastFinishOn}
		}
		bb := inhouse("blueberry-muffin")
		bb.Prime = config.PrimeOn
		cm := inhouse("chocolate-muffin")
		cm.Prime, cm.ParserRetries = config.PrimeOff, config.ParserRetriesOff
		solo := cm
		solo.Fallback = ""
		tbl, err := model.NewTable(model.Options{
			Routes: map[string]config.Route{
				"default":          {Kind: config.RouteAnthropic, Auth: config.AuthAPIKey, Model: "claude-sonnet-5-5", MaxTokens: 1024, Effort: "medium"},
				"blueberry-muffin": bb, "chocolate-muffin": cm, "chocolate-solo": solo,
			},
			Default: "blueberry-muffin", Allowlist: []string{"default", "blueberry-muffin", "chocolate-muffin", "chocolate-solo"},
			Aliases: aliases, APIKey: "sk-test", BaseURL: mf.claude.URL, CallTimeout: 5 * time.Second,
		})
		if err != nil {
			t.Fatal(err)
		}
		d.Models = tbl
		d.Logf = func(f string, a ...any) {
			mf.logMu.Lock()
			mf.logs = append(mf.logs, fmt.Sprintf(f, a...))
			mf.logMu.Unlock()
			t.Logf(f, a...)
		}
	})
	return mf
}

func boolPtr(b bool) *bool { return &b }

func (mf *muffins) req(route, text string) api.TurnRequest {
	r := mf.request(text)
	r.Model = route
	return r
}

func (mf *muffins) logged(prefix string) []string {
	mf.logMu.Lock()
	defer mf.logMu.Unlock()
	var out []string
	for _, l := range mf.logs {
		if strings.HasPrefix(l, prefix) {
			out = append(out, l)
		}
	}
	return out
}

// retryable makes the SDK's backoff 1 ms (Retry-After-Ms), as the other
// retry tests do.
var retryable = map[string]string{"Retry-After-Ms": "1"}

func parser500(tool string) modeltest.Step {
	return modeltest.Error(500, "api_error", "tool '"+tool+"' not found", retryable)
}

// shape is a stream reduced to what "handled like any model error" means:
// the event kinds, the escalation, and turn.done's routes, models and stop.
func shape(t *testing.T, evs []api.CollectedEvent) string {
	t.Helper()
	var parts []string
	for _, e := range evs {
		d := data(t, e)
		switch e.Event {
		case api.EventTurnEscalated:
			parts = append(parts, fmt.Sprintf("escalated %v→%v (%v)", d["from"], d["to"], d["reason"]))
		case api.EventTurnDone:
			raw, _ := json.Marshal(map[string]any{"routes": d["routes"], "models": d["models"], "stop": d["stop"], "route": d["route"]})
			parts = append(parts, "done "+string(raw))
		case api.EventTurnError:
			parts = append(parts, fmt.Sprintf("error %v", d["code"]))
		default:
			parts = append(parts, e.Event)
		}
	}
	return strings.Join(parts, " | ")
}

// A MinistralParser 500 on chocolate-muffin: one request (no redraw), the
// "souschef: model parser error" line, the route's counter, and then the
// turn goes to Claude exactly as after any other failed call — the same
// stream as a call that fails with a plain 500 (retried, then escalated).
func TestParserError_EscalatesToClaudeLikeAnyModelError(t *testing.T) {
	claudeSays := func() []modeltest.Step {
		return []modeltest.Step{reply(modeltest.Text("I changed the headline."))}
	}
	parse := newMuffins(t, nil, []modeltest.Step{parser500("set_fieldz"), reply(modeltest.Text("a redraw must never happen"))}, claudeSays())
	_, parseEvs := parse.start(parse.req("chocolate-muffin", "make the headline say Fresh cookies"))

	plain := newMuffins(t, nil, []modeltest.Step{
		modeltest.Error(500, "api_error", "boom", retryable), modeltest.Error(500, "api_error", "boom", retryable),
		modeltest.Error(500, "api_error", "boom", retryable)}, claudeSays())
	_, plainEvs := plain.start(plain.req("chocolate-muffin", "make the headline say Fresh cookies"))

	if got, want := shape(t, parseEvs), shape(t, plainEvs); got != want {
		t.Fatalf("a parser error is not handled like any model error:\n parser %s\n plain  %s", got, want)
	}
	if !strings.Contains(shape(t, parseEvs), "escalated chocolate-muffin→default (model_error)") ||
		!strings.Contains(shape(t, parseEvs), `"routes":["chocolate-muffin","default"]`) ||
		!strings.Contains(shape(t, parseEvs), `"models":{"chocolate-muffin":"chocolate-muffin","default":"claude-sonnet-5-5"}`) {
		t.Fatalf("stream = %s", shape(t, parseEvs))
	}
	if n := len(parse.ollama.Requests()); n != 1 {
		t.Fatalf("%d requests to the model server: a parser error must not be redrawn", n)
	}
	if n := len(plain.ollama.Requests()); n != 1+model.DefaultMaxRetries {
		t.Fatalf("%d requests: any other 500 keeps the SDK's retries", n)
	}
	// Claude gets the turn's request as it stood: nothing of the failed call.
	if got := msgRoles(parse.claude.Requests()[0]); got != "user" {
		t.Errorf("Claude's request messages = %s", got)
	}
	lines := parse.logged("souschef: model parser error")
	if len(lines) != 1 || !strings.Contains(lines[0], "call 1 on route chocolate-muffin (reported as chocolate-muffin), not retried, 1 on this instance: tool 'set_fieldz' not found") {
		t.Fatalf("log = %q", lines)
	}
	if len(plain.logged("souschef: model parser error")) != 0 {
		t.Errorf("a plain 500 logged as a parser error")
	}
	if pe := parse.runner.d.Models.ParserErrors(); pe["chocolate-muffin"] != 1 || pe["chocolate-solo"] != 0 {
		t.Errorf("ParserErrors = %v", pe)
	}
	// The failed call's usage row is reported under the route's name, as any
	// failed call's.
	parse.st.mu.Lock()
	rows := append([]any(nil), parse.st.usage[0].Model, parse.st.usage[0].Status, parse.st.usage[1].Model, parse.st.usage[1].Status)
	parse.st.mu.Unlock()
	if !reflect.DeepEqual(rows, []any{"chocolate-muffin", "error", "claude-sonnet-5-5", "ok"}) {
		t.Errorf("usage rows = %v", rows)
	}
}

// Without a fallback (the lab's harness: the student alone) the turn ends
// turn.error model_unavailable, as after any 500 — once, not three times.
func TestParserError_NoFallbackIsModelUnavailable(t *testing.T) {
	mf := newMuffins(t, nil, []modeltest.Step{parser500("set_field")}, nil)
	_, evs := mf.start(mf.req("chocolate-solo", "make the headline say Fresh cookies"))
	if got := shape(t, evs); got != "error model_unavailable" {
		t.Fatalf("stream = %s", got)
	}
	if n := len(mf.ollama.Requests()); n != 1 {
		t.Fatalf("%d requests", n)
	}
	if lines := mf.logged("souschef: model parser error"); len(lines) != 1 || !strings.Contains(lines[0], "route chocolate-solo") {
		t.Fatalf("log = %q", lines)
	}
}

// blueberry-muffin (no option): the same 500 is redrawn by the SDK as
// today, and nothing is logged as a parser error.
func TestParserError_BlueberryRetriesAsToday(t *testing.T) {
	mf := newMuffins(t, nil, []modeltest.Step{parser500("set_fieldz"), reply(modeltest.Text("Redrawn."))}, nil)
	_, evs := mf.start(mf.req("", "hello"))
	if got := shape(t, evs); !strings.HasPrefix(got, "text.delta | done ") || !strings.Contains(got, `"name":"blueberry-muffin"`) {
		t.Fatalf("stream = %s", got)
	}
	if n := len(mf.ollama.Requests()); n != 2 {
		t.Fatalf("%d requests, want the 500 and its redraw", n)
	}
	if lines := mf.logged("souschef:"); len(lines) != 0 {
		t.Errorf("log = %q", lines)
	}
}

// MODEL_ROUTE_ALIASES {"blueberry-muffin":"chocolate-muffin"}: int's
// traffic — no model named, or blueberry-muffin named — is chocolate-muffin's
// (its settings, its report_as, its name in turn.done and in the pending doc
// a continuation resumes from), while DEFAULT_ROUTE still says
// blueberry-muffin.
func TestAliases_InHouseTrafficAnsweredByChocolate(t *testing.T) {
	mf := newMuffins(t, map[string]string{"blueberry-muffin": "chocolate-muffin"},
		[]modeltest.Step{
			reply(modeltest.Text("On it."), setFields("s1", "headline")),
			reply(modeltest.Text("Named turn.")),
		}, nil)
	tr, evs := mf.start(mf.req("", "make the headline say Fresh cookies"))
	if got := kinds(evs); got != "text.delta tool.call turn.waiting" {
		t.Fatalf("stream = %s", got)
	}
	evs, err := mf.cont(tr.ID(), okResult("s1", "sections[0].headline"))
	if err != nil {
		t.Fatal(err)
	}
	done := last(evs)
	if d := data(t, done); d["fast_finish"] != true {
		t.Fatalf("turn.done = %v", d)
	}
	route := data(t, done)["route"].(map[string]any)
	if route["name"] != "chocolate-muffin" || route["model"] != "chocolate-muffin" {
		t.Fatalf("route = %v", route)
	}
	_, evs = mf.start(mf.req("blueberry-muffin", "and the tagline?"))
	if route := data(t, last(evs))["route"].(map[string]any); route["name"] != "chocolate-muffin" {
		t.Fatalf("a turn naming blueberry-muffin: route = %v", route)
	}
	reqs := mf.ollama.Requests()
	if len(reqs) != 2 {
		t.Fatalf("%d requests", len(reqs))
	}
	for i, r := range reqs {
		if r.Body["temperature"] != nil || r.Body["max_tokens"] != float64(1024) {
			t.Errorf("request %d: %v %v", i, r.Body["temperature"], r.Body["max_tokens"])
		}
	}
	mf.st.mu.Lock()
	defer mf.st.mu.Unlock()
	for _, row := range mf.st.usage {
		if row.Model != "chocolate-muffin" {
			t.Errorf("usage row reported as %q", row.Model)
		}
	}
	if b, _ := mf.runner.d.Models.Route(""); b.PrimeEnabled() {
		t.Error("the default route's prime is still on: the Primer would prime the Mistral server with Qwen's filler")
	}
}
