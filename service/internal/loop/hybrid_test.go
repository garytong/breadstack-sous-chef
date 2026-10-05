package loop

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/breadstack/sous-chef/internal/api"
	"github.com/breadstack/sous-chef/internal/config"
	"github.com/breadstack/sous-chef/internal/model"
	"github.com/breadstack/sous-chef/internal/model/modeltest"
	"github.com/breadstack/sous-chef/internal/store"
)

// hybrid is a harness with TWO fake models: the student behind the
// messages-compatible route "student" (fallback → "backstop") and Claude
// behind the anthropic routes. "backstop" is NOT allowlisted: only the
// route table can send a turn there.
type hybrid struct {
	*harness
	student *modeltest.Server
	claude  *modeltest.Server
}

func newHybrid(t *testing.T, studentSteps, claudeSteps []modeltest.Step) *hybrid {
	t.Helper()
	student := modeltest.New(answering("sous-chef-student", studentSteps)...)
	t.Cleanup(student.Close)
	hy := &hybrid{student: student}
	hy.harness = newHarness(t, func(d *Deps) {
		// newHarness's own fake is unused: the table below replaces it.
		claude := modeltest.New(answering("claude-sonnet-5-5", claudeSteps)...)
		t.Cleanup(claude.Close)
		hy.claude = claude
		tbl, err := model.NewTable(model.Options{
			Routes: map[string]config.Route{
				"default":  {Kind: config.RouteAnthropic, Auth: config.AuthAPIKey, Model: "claude-sonnet-5-5", MaxTokens: 1024, Effort: "medium"},
				"backstop": {Kind: config.RouteAnthropic, Auth: config.AuthAPIKey, Model: "claude-sonnet-5-5", MaxTokens: 1024, Effort: "medium"},
				"student":  {Kind: config.RouteMessagesCompatible, URL: student.URL, Auth: config.AuthNone, Model: "sous-chef-student", MaxTokens: 512, Fallback: "backstop"},
				"solo":     {Kind: config.RouteMessagesCompatible, URL: student.URL, Auth: config.AuthNone, Model: "sous-chef-student", MaxTokens: 512},
				// The same student under a name of its own (report_as_test.go):
				// the server still echoes "sous-chef-student".
				"muffin":      {Kind: config.RouteMessagesCompatible, URL: student.URL, Auth: config.AuthNone, Model: "sous-chef", MaxTokens: 512, Fallback: "backstop", ReportAs: "blueberry-muffin"},
				"muffin-solo": {Kind: config.RouteMessagesCompatible, URL: student.URL, Auth: config.AuthNone, Model: "sous-chef", MaxTokens: 512, ReportAs: "blueberry-muffin"},
				"named":       {Kind: config.RouteAnthropic, Auth: config.AuthAPIKey, Model: "claude-sonnet-5-5", MaxTokens: 1024, Effort: "medium", ReportAs: "house-claude"},
			},
			Default: "default", Allowlist: []string{"default", "student", "solo", "muffin", "muffin-solo", "named"},
			APIKey: "sk-test", BaseURL: claude.URL, CallTimeout: 5 * time.Second, MaxRetries: intp(0),
		})
		if err != nil {
			t.Fatal(err)
		}
		d.Models = tbl
	})
	return hy
}

// answering stamps the scripted replies with the model id that serves them
// (the response's own model field is what usage rows and traces record).
func answering(modelID string, steps []modeltest.Step) []modeltest.Step {
	for _, st := range steps {
		if st.Message != nil && st.Message.Model == "" {
			st.Message.Model = modelID
		}
	}
	return steps
}

func (hy *hybrid) req(route, text string) api.TurnRequest {
	r := hy.request(text)
	r.Model = route
	return r
}

func reply(blocks ...modeltest.Block) modeltest.Step {
	return modeltest.Reply(modeltest.Message{Content: blocks})
}

func setFields(id, key string) modeltest.Block {
	return modeltest.ToolUse(id, "set_fields", map[string]any{"target": heroTarget, "values": map[string]any{key: "Fresh cookies"}})
}

func refusedResult(id string) api.ToolResult {
	return api.ToolResult{ToolCallID: id, OK: false, Error: &api.ToolError{Code: "E_FIELD", Message: "unknown field itemDescriptions for target", Key: "itemDescriptions"}}
}

func events(evs []api.CollectedEvent, kind string) []api.CollectedEvent {
	var out []api.CollectedEvent
	for _, e := range evs {
		if e.Event == kind {
			out = append(out, e)
		}
	}
	return out
}

func wantEscalated(t *testing.T, evs []api.CollectedEvent, reason string) {
	t.Helper()
	esc := events(evs, api.EventTurnEscalated)
	if len(esc) != 1 {
		t.Fatalf("turn.escalated ×%d in %s", len(esc), kinds(evs))
	}
	if d := data(t, esc[0]); d["from"] != "student" || d["to"] != "backstop" || d["reason"] != reason || d["turnId"] == nil || d["seq"] == nil {
		t.Fatalf("turn.escalated = %v, want student → backstop (%s)", d, reason)
	}
}

func routesOf(t *testing.T, done api.CollectedEvent) string {
	t.Helper()
	if done.Event != api.EventTurnDone {
		t.Fatalf("last event = %s", done.Event)
	}
	raw, _ := json.Marshal(data(t, done)["routes"])
	return string(raw)
}

// msgRoles is the captured request's message roles, e.g. "user assistant user".
func msgRoles(c modeltest.Captured) string {
	var out []string
	for _, m := range c.Body["messages"].([]any) {
		out = append(out, m.(map[string]any)["role"].(string))
	}
	return strings.Join(out, " ")
}

// TestHybrid_RefusedOpEscalates: the student writes an op with an invented
// key, the engine refuses it, and the next call of the same turn is
// Claude's — a valid Anthropic request that carries the student's tool_use
// and its error result — which fixes it. Claude's own refused op later in
// the turn does not escalate again.
func TestHybrid_RefusedOpEscalates(t *testing.T) {
	hy := newHybrid(t,
		[]modeltest.Step{reply(modeltest.Text("On it."), setFields("s1", "itemDescriptions"))},
		[]modeltest.Step{
			reply(setFields("c1", "headlineX")),
			reply(setFields("c2", "headline")),
			reply(modeltest.Text("Done.")),
		})
	turn, evs := hy.start(hy.req("student", "set the hero headline to Fresh cookies"))
	if kinds(evs) != "text.delta tool.call turn.waiting" || text(evs) != "On it." {
		t.Fatalf("stream 1 = %s %q", kinds(evs), text(evs))
	}

	// The engine refuses the student's op → escalation, then Claude's op.
	evs, err := hy.cont(turn.ID(), refusedResult("s1"))
	if err != nil {
		t.Fatal(err)
	}
	if kinds(evs) != "turn.escalated tool.call turn.waiting" {
		t.Fatalf("stream 2 = %s", kinds(evs))
	}
	wantEscalated(t, evs, EscalateToolError)
	if d := data(t, evs[1]); d["id"] != "c1" {
		t.Fatalf("tool.call = %v", d)
	}
	// The pending doc carries the route the turn continues on.
	var pd pendingDoc
	if err := json.Unmarshal(hy.st.turn(turn.ID()).PendingToolCalls, &pd); err != nil {
		t.Fatal(err)
	}
	if pd.Route != "backstop" || pd.Escalated == nil || *pd.Escalated != (escalation{From: "student", To: "backstop", Reason: EscalateToolError}) || strings.Join(pd.Routes, ",") != "student,backstop" {
		t.Fatalf("pending doc = route %q escalated %+v routes %v", pd.Route, pd.Escalated, pd.Routes)
	}

	// Claude's first op is refused too: no second escalation, still Claude.
	evs, err = hy.cont(turn.ID(), refusedResult("c1"))
	if err != nil {
		t.Fatal(err)
	}
	if kinds(evs) != "tool.call turn.waiting" {
		t.Fatalf("stream 3 = %s", kinds(evs))
	}
	evs, err = hy.cont(turn.ID(), api.ToolResult{ToolCallID: "c2", OK: true, Written: []string{"headline"}})
	if err != nil {
		t.Fatal(err)
	}
	if kinds(evs) != "text.delta text.delta turn.done" || text(evs) != "Done." {
		t.Fatalf("stream 4 = %s %q", kinds(evs), text(evs))
	}
	if got := routesOf(t, last(evs)); got != `["student","backstop"]` {
		t.Errorf("turn.done routes = %s", got)
	}

	// One call on the student, three on Claude.
	sreqs, creqs := hy.student.Requests(), hy.claude.Requests()
	if len(sreqs) != 1 || len(creqs) != 3 {
		t.Fatalf("requests: student %d, claude %d", len(sreqs), len(creqs))
	}
	// The student got the plain request; Claude a full Anthropic one: its
	// model, effort, the top-level and system cache breakpoints, the beta
	// fallbacks, strict tools, and no thinking field.
	if s := sreqs[0].Body; s["model"] != "sous-chef-student" || s["cache_control"] != nil || s["output_config"] != nil || s["thinking"].(map[string]any)["type"] != "disabled" {
		t.Errorf("student request = model %v cache_control %v output_config %v thinking %v", s["model"], s["cache_control"], s["output_config"], s["thinking"])
	}
	c := creqs[0].Body
	if c["model"] != "claude-sonnet-5-5" || c["cache_control"] == nil || c["thinking"] != nil || c["fallbacks"] != "default" ||
		c["output_config"].(map[string]any)["effort"] != "medium" || c["max_tokens"] != 1024.0 {
		t.Errorf("claude request fields = model %v cache_control %v thinking %v fallbacks %v output_config %v", c["model"], c["cache_control"], c["thinking"], c["fallbacks"], c["output_config"])
	}
	if !strings.Contains(creqs[0].Header.Get("Anthropic-Beta"), "server-side-fallback") || creqs[0].Header.Get("X-Api-Key") != "sk-test" {
		t.Errorf("claude headers = %v", creqs[0].Header)
	}
	for _, sys := range c["system"].([]any) {
		if sys.(map[string]any)["cache_control"] == nil {
			t.Error("claude system block lacks cache_control")
		}
	}
	strict := 0
	for _, tool := range c["tools"].([]any) {
		if tool.(map[string]any)["strict"] == true {
			strict++
		}
	}
	if strict == 0 {
		t.Error("claude request has no strict tool")
	}
	// Same prefix on both routes, cache_control aside.
	if sreqs[0].SystemText() != creqs[0].SystemText() || len(sreqs[0].Body["tools"].([]any)) != len(c["tools"].([]any)) {
		t.Error("the two routes were sent different system text or tool counts")
	}
	// Claude sees the history as it is: the user turn (with the student's
	// candidate-fields block untouched), the student's tool_use, the error.
	if msgRoles(creqs[0]) != "user assistant user" {
		t.Fatalf("claude call 1 roles = %s", msgRoles(creqs[0]))
	}
	if got, want := userTexts(t, creqs[0]), userTexts(t, sreqs[0]); len(got) != 1 || got[0] != want[0] {
		t.Errorf("claude's user turn differs from the student's:\n%q\n%q", got, want)
	}
	raw := string(creqs[0].Raw)
	if !strings.Contains(raw, `"id":"s1"`) || !strings.Contains(raw, "itemDescriptions") {
		t.Error("claude call 1 lacks the student's tool_use")
	}
	if res := lastToolResults(t, creqs[0]); len(res) != 1 || res[0].ID != "s1" || !res[0].IsError || !strings.Contains(res[0].Content, "E_FIELD") {
		t.Errorf("claude call 1 results = %+v", res)
	}

	// Attribution: usage rows by model, trace calls by route, the row's usage.
	rows := hy.st.usageRows()
	var models []string
	for _, r := range rows {
		models = append(models, r.Model)
	}
	if strings.Join(models, ",") != "sous-chef-student,claude-sonnet-5-5,claude-sonnet-5-5,claude-sonnet-5-5" {
		t.Errorf("usage row models = %v", models)
	}
	tr, ok := hy.st.trace(turn.ID())
	if !ok {
		t.Fatal("no trace")
	}
	var ops struct {
		Route     string
		Escalated *struct {
			From, To, Reason string
			AfterCall        int `json:"after_call"`
		}
		Calls []struct {
			Call  int
			Route string
		}
	}
	if err := json.Unmarshal(tr.Ops, &ops); err != nil {
		t.Fatal(err)
	}
	if ops.Route != "student" || ops.Escalated == nil || ops.Escalated.To != "backstop" || ops.Escalated.Reason != EscalateToolError || ops.Escalated.AfterCall != 1 {
		t.Errorf("trace route %q escalated %+v", ops.Route, ops.Escalated)
	}
	var callRoutes []string
	for _, c := range ops.Calls {
		callRoutes = append(callRoutes, c.Route)
	}
	if strings.Join(callRoutes, ",") != "student,backstop,backstop,backstop" {
		t.Errorf("trace call routes = %v", callRoutes)
	}
	if !strings.Contains(string(tr.Events), `"event":"turn.escalated"`) {
		t.Error("trace events lack turn.escalated")
	}
	row := hy.st.turn(turn.ID())
	if row.Status != store.TurnDone || !strings.Contains(string(row.Usage), `"routes":["student","backstop"]`) || !strings.Contains(string(row.Usage), `"reason":"tool_error"`) || !strings.Contains(string(row.Usage), `"calls":4`) {
		t.Errorf("row = %s %s", row.Status, row.Usage)
	}
}

// TestHybrid_UnsupportedEscalates: the student's report_unsupported is not
// the turn's verdict — it is answered is_error, not shown, its text is
// dropped, and Claude decides. Claude's own report_unsupported stands.
func TestHybrid_UnsupportedEscalates(t *testing.T) {
	unsupported := func(id string) modeltest.Block {
		return modeltest.ToolUse(id, "report_unsupported", map[string]any{"reason": "No such setting."})
	}
	hy := newHybrid(t,
		[]modeltest.Step{reply(modeltest.Text("Sorry, I can't do that."), unsupported("s1"))},
		[]modeltest.Step{reply(unsupported("c1")), reply(modeltest.Text("That one isn't editable here."))})
	turn, evs := hy.start(hy.req("student", "make the checkout page purple"))
	if kinds(evs) != "turn.escalated tool.call text.delta text.delta turn.done" {
		t.Fatalf("events = %s", kinds(evs))
	}
	wantEscalated(t, evs, EscalateUnsupported)
	if text(evs) != "That one isn't editable here." {
		t.Errorf("text = %q (the student's must not be shown)", text(evs))
	}
	// Only Claude's report_unsupported reaches the page.
	if d := data(t, events(evs, api.EventToolCall)[0]); d["id"] != "c1" || d["name"] != "report_unsupported" {
		t.Errorf("tool.call = %v", d)
	}
	if got := routesOf(t, last(evs)); got != `["student","backstop"]` {
		t.Errorf("routes = %s", got)
	}
	creqs := hy.claude.Requests()
	if len(hy.student.Requests()) != 1 || len(creqs) != 2 {
		t.Fatalf("requests: student %d, claude %d", len(hy.student.Requests()), len(creqs))
	}
	if res := lastToolResults(t, creqs[0]); len(res) != 1 || res[0].ID != "s1" || !res[0].IsError || !strings.Contains(res[0].Content, "E_ESCALATED") {
		t.Errorf("student's report_unsupported answered %+v", res)
	}
	if res := lastToolResults(t, creqs[1]); len(res) != 1 || res[0].ID != "c1" || res[0].IsError {
		t.Errorf("claude's report_unsupported answered %+v", res)
	}
	if hy.st.turn(turn.ID()).Status != store.TurnDone {
		t.Errorf("status = %s", hy.st.turn(turn.ID()).Status)
	}
}

// TestHybrid_ModelErrorEscalates: the student's call fails; the SAME
// request goes to Claude and the turn completes.
func TestHybrid_ModelErrorEscalates(t *testing.T) {
	hy := newHybrid(t,
		[]modeltest.Step{modeltest.Error(500, "api_error", "gpu on fire", nil)},
		[]modeltest.Step{reply(modeltest.Text("Hello."))})
	turn, evs := hy.start(hy.req("student", "hello"))
	if kinds(evs) != "turn.escalated text.delta text.delta turn.done" || text(evs) != "Hello." {
		t.Fatalf("events = %s %q", kinds(evs), text(evs))
	}
	wantEscalated(t, evs, EscalateModelError)
	sreqs, creqs := hy.student.Requests(), hy.claude.Requests()
	if len(sreqs) != 1 || len(creqs) != 1 || msgRoles(creqs[0]) != "user" || userTexts(t, creqs[0])[0] != userTexts(t, sreqs[0])[0] {
		t.Fatalf("requests: student %d, claude %d", len(sreqs), len(creqs))
	}
	rows := hy.st.usageRows()
	if len(rows) != 2 || rows[0].Status != "error" || rows[0].Model != "sous-chef-student" || rows[1].Status != "ok" || rows[1].Model != "claude-sonnet-5-5" {
		t.Errorf("usage rows = %+v", rows)
	}
	msgs, _ := hy.st.ListMessages(context.Background(), hy.st.turn(turn.ID()).ConversationID, 0, 0)
	if len(msgs) != 2 || msgs[1].Role != "assistant" {
		t.Errorf("stored messages = %d", len(msgs))
	}
}

// TestHybrid_MaxTokensEscalates: a cut-off student reply is discarded — not
// shown, not stored — and Claude answers the same request.
func TestHybrid_MaxTokensEscalates(t *testing.T) {
	hy := newHybrid(t,
		[]modeltest.Step{modeltest.Reply(modeltest.Message{StopReason: "max_tokens", Content: []modeltest.Block{modeltest.Text("Sure, I will sta")}})},
		[]modeltest.Step{reply(modeltest.Text("All set."))})
	turn, evs := hy.start(hy.req("student", "hello"))
	if kinds(evs) != "turn.escalated text.delta text.delta turn.done" || text(evs) != "All set." {
		t.Fatalf("events = %s %q", kinds(evs), text(evs))
	}
	wantEscalated(t, evs, EscalateMaxTokens)
	if d := data(t, last(evs)); d["stop"] != "end_turn" || d["usage"].(map[string]any)["calls"] != 2.0 {
		t.Errorf("done = %v", d)
	}
	if creqs := hy.claude.Requests(); len(creqs) != 1 || msgRoles(creqs[0]) != "user" {
		t.Fatalf("claude requests = %d", len(creqs))
	}
	msgs, _ := hy.st.ListMessages(context.Background(), hy.st.turn(turn.ID()).ConversationID, 0, 0)
	if len(msgs) != 2 || strings.Contains(string(msgs[1].Content), "I will sta") {
		t.Errorf("stored messages = %d: %s", len(msgs), msgs[len(msgs)-1].Content)
	}
}

// TestHybrid_FallbackFailureIsTheTurns: once on the fallback nothing
// escalates again — its error ends the turn as any model error does.
func TestHybrid_FallbackFailureIsTheTurns(t *testing.T) {
	hy := newHybrid(t,
		[]modeltest.Step{modeltest.Error(500, "api_error", "gpu on fire", nil)},
		[]modeltest.Step{modeltest.Error(529, "overloaded_error", "busy", nil)})
	turn, evs := hy.start(hy.req("student", "hello"))
	if kinds(evs) != "turn.escalated turn.error" {
		t.Fatalf("events = %s", kinds(evs))
	}
	if d := data(t, last(evs)); d["code"] != "model_unavailable" {
		t.Errorf("error = %v", d)
	}
	if len(hy.student.Requests()) != 1 || len(hy.claude.Requests()) != 1 {
		t.Errorf("requests: student %d, claude %d", len(hy.student.Requests()), len(hy.claude.Requests()))
	}
	if row := hy.st.turn(turn.ID()); row.Status != store.TurnError || !strings.Contains(string(row.Usage), `"reason":"model_error"`) {
		t.Errorf("row = %s %s", row.Status, row.Usage)
	}
}

// TestHybrid_HealthyTurnStaysOnTheStudent: a turn the student gets right
// never touches the fallback; turn.done still names the route.
func TestHybrid_HealthyTurnStaysOnTheStudent(t *testing.T) {
	hy := newHybrid(t,
		[]modeltest.Step{reply(modeltest.Text("On it."), setFields("s1", "headline")), reply(modeltest.Text("Done."))},
		nil)
	turn, evs := hy.start(hy.req("student", "set the hero headline"))
	if kinds(evs) != "text.delta tool.call turn.waiting" {
		t.Fatalf("stream 1 = %s", kinds(evs))
	}
	evs, err := hy.cont(turn.ID(), api.ToolResult{ToolCallID: "s1", OK: true, Written: []string{"headline"}})
	if err != nil {
		t.Fatal(err)
	}
	if kinds(evs) != "text.delta turn.done" || text(evs) != "Done." {
		t.Fatalf("stream 2 = %s %q", kinds(evs), text(evs))
	}
	if got := routesOf(t, last(evs)); got != `["student"]` {
		t.Errorf("routes = %s", got)
	}
	if len(hy.student.Requests()) != 2 || len(hy.claude.Requests()) != 0 {
		t.Errorf("requests: student %d, claude %d", len(hy.student.Requests()), len(hy.claude.Requests()))
	}
	if u := string(hy.st.turn(turn.ID()).Usage); !strings.Contains(u, `"routes":["student"]`) || strings.Contains(u, "escalated") {
		t.Errorf("row usage = %s", u)
	}
}

// TestHybrid_RoutesWithoutFallbackUnchanged: the same failures on routes
// that name no fallback behave as before — no event, no `routes`, the
// refused op goes back to the same model, the failed call ends the turn.
func TestHybrid_RoutesWithoutFallbackUnchanged(t *testing.T) {
	// A compatible route with no fallback: the refused op returns to it.
	hy := newHybrid(t,
		[]modeltest.Step{reply(setFields("s1", "itemDescriptions")), reply(modeltest.Text("Sorry."))},
		[]modeltest.Step{reply(setFields("c1", "headlineX")), reply(modeltest.Text("Hm."))})
	turn, _ := hy.start(hy.req("solo", "set the hero headline"))
	evs, err := hy.cont(turn.ID(), refusedResult("s1"))
	if err != nil {
		t.Fatal(err)
	}
	if kinds(evs) != "text.delta turn.done" {
		t.Fatalf("solo: %s", kinds(evs))
	}
	if d := data(t, last(evs)); d["routes"] != nil {
		t.Errorf("solo turn.done carries routes: %v", d)
	}
	if len(hy.student.Requests()) != 2 || len(hy.claude.Requests()) != 0 {
		t.Fatalf("solo requests: student %d, claude %d", len(hy.student.Requests()), len(hy.claude.Requests()))
	}
	if u := string(hy.st.turn(turn.ID()).Usage); strings.Contains(u, "routes") || strings.Contains(u, "escalated") {
		t.Errorf("solo row usage = %s", u)
	}
	// An anthropic route: likewise.
	turn, _ = hy.start(hy.req("default", "set the hero headline"))
	evs, err = hy.cont(turn.ID(), refusedResult("c1"))
	if err != nil {
		t.Fatal(err)
	}
	if kinds(evs) != "text.delta text.delta turn.done" || data(t, last(evs))["routes"] != nil {
		t.Fatalf("default: %s %v", kinds(evs), data(t, last(evs)))
	}
	// A failed call on a route with no fallback ends the turn (script spent → 500).
	_, evs = hy.start(hy.req("solo", "hello"))
	if kinds(evs) != "turn.error" {
		t.Fatalf("solo error: %s", kinds(evs))
	}
	if len(hy.claude.Requests()) != 2 {
		t.Errorf("claude requests = %d", len(hy.claude.Requests()))
	}
}

// TestHybrid_ResumeAfterSuspensionKeepsRoute: an escalated turn that waits
// for the browser is continued on ANOTHER instance (rebuilt from the row):
// it resumes on the fallback — reached through its own route, the fallback
// itself not being allowlisted — does not escalate a second time, and
// still reports both routes.
func TestHybrid_ResumeAfterSuspensionKeepsRoute(t *testing.T) {
	hy := newHybrid(t,
		[]modeltest.Step{reply(setFields("s1", "itemDescriptions"))},
		[]modeltest.Step{
			reply(modeltest.ToolUse("p1", "pick_asset", map[string]any{"purpose": "hero image"})),
			reply(setFields("c2", "headlineX")),
			reply(modeltest.Text("Done.")),
		})
	turn, _ := hy.start(hy.req("student", "set the hero image"))
	evs, err := hy.cont(turn.ID(), refusedResult("s1"))
	if err != nil {
		t.Fatal(err)
	}
	if kinds(evs) != "turn.escalated turn.suspended" {
		t.Fatalf("stream 2 = %s", kinds(evs))
	}
	if hy.st.turn(turn.ID()).Status != store.TurnSuspended {
		t.Fatalf("status = %s", hy.st.turn(turn.ID()).Status)
	}
	// "Another instance": the parked entry is gone, the row is all there is.
	if hy.runner.unpark(turn.ID()) == nil {
		t.Fatal("turn was not parked")
	}
	evs, err = hy.cont(turn.ID(), api.ToolResult{ToolCallID: "p1", OK: true, Asset: &api.Asset{PublicURL: "https://cdn/x.jpg"}})
	if err != nil {
		t.Fatal(err)
	}
	if kinds(evs) != "tool.call turn.waiting" {
		t.Fatalf("stream 3 = %s", kinds(evs))
	}
	// And once more from the row, with the fallback's own op refused.
	hy.runner.unpark(turn.ID())
	evs, err = hy.cont(turn.ID(), refusedResult("c2"))
	if err != nil {
		t.Fatal(err)
	}
	if kinds(evs) != "text.delta text.delta turn.done" {
		t.Fatalf("stream 4 = %s", kinds(evs))
	}
	if got := routesOf(t, last(evs)); got != `["student","backstop"]` {
		t.Errorf("routes = %s", got)
	}
	if len(hy.student.Requests()) != 1 || len(hy.claude.Requests()) != 3 {
		t.Errorf("requests: student %d, claude %d", len(hy.student.Requests()), len(hy.claude.Requests()))
	}
	if u := string(hy.st.turn(turn.ID()).Usage); !strings.Contains(u, `"routes":["student","backstop"]`) || !strings.Contains(u, `"from":"student"`) {
		t.Errorf("row usage = %s", u)
	}
}

// TestHybrid_ServiceRefusedOpEscalates: an op the service itself refuses
// (an unknown tool here) never reaches the browser, and escalates too.
func TestHybrid_ServiceRefusedOpEscalates(t *testing.T) {
	hy := newHybrid(t,
		[]modeltest.Step{reply(modeltest.ToolUse("s1", "set_field", map[string]any{"key": "headline"}))},
		[]modeltest.Step{reply(modeltest.Text("Fixed."))})
	_, evs := hy.start(hy.req("student", "set the hero headline"))
	if kinds(evs) != "turn.escalated text.delta text.delta turn.done" {
		t.Fatalf("events = %s", kinds(evs))
	}
	wantEscalated(t, evs, EscalateToolError)
	if res := lastToolResults(t, hy.claude.Requests()[0]); len(res) != 1 || !res[0].IsError || !strings.Contains(res[0].Content, "unknown tool") {
		t.Errorf("results = %+v", res)
	}
}
