package loop

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/breadstack/sous-chef/internal/api"
	"github.com/breadstack/sous-chef/internal/config"
	"github.com/breadstack/sous-chef/internal/model"
	"github.com/breadstack/sous-chef/internal/model/modeltest"
)

// newFast is newHybrid's two fakes behind a table with fast_finish routes:
// "student" (messages-compatible, fast_finish on, fallback "backstop"),
// "solo" (messages-compatible, fast_finish on, no fallback), "slow" (the
// same without fast_finish) and "claude-fast" (anthropic, fast_finish on).
func newFast(t *testing.T, studentSteps, claudeSteps []modeltest.Step) *hybrid {
	t.Helper()
	student := modeltest.New(answering("sous-chef-student", studentSteps)...)
	t.Cleanup(student.Close)
	hy := &hybrid{student: student}
	hy.harness = newHarness(t, func(d *Deps) {
		claude := modeltest.New(answering("claude-sonnet-5-5", claudeSteps)...)
		t.Cleanup(claude.Close)
		hy.claude = claude
		tbl, err := model.NewTable(model.Options{
			Routes: map[string]config.Route{
				"default":     {Kind: config.RouteAnthropic, Auth: config.AuthAPIKey, Model: "claude-sonnet-5-5", MaxTokens: 1024, Effort: "medium"},
				"backstop":    {Kind: config.RouteAnthropic, Auth: config.AuthAPIKey, Model: "claude-sonnet-5-5", MaxTokens: 1024, Effort: "medium"},
				"claude-fast": {Kind: config.RouteAnthropic, Auth: config.AuthAPIKey, Model: "claude-sonnet-5-5", MaxTokens: 1024, Effort: "medium", FastFinish: config.FastFinishOn},
				"student":     {Kind: config.RouteMessagesCompatible, URL: student.URL, Auth: config.AuthNone, Model: "sous-chef-student", MaxTokens: 512, Fallback: "backstop", FastFinish: config.FastFinishOn},
				"solo":        {Kind: config.RouteMessagesCompatible, URL: student.URL, Auth: config.AuthNone, Model: "sous-chef-student", MaxTokens: 512, FastFinish: config.FastFinishOn},
				"slow":        {Kind: config.RouteMessagesCompatible, URL: student.URL, Auth: config.AuthNone, Model: "sous-chef-student", MaxTokens: 512},
			},
			Default: "default", Allowlist: []string{"default", "claude-fast", "student", "solo", "slow"},
			APIKey: "sk-test", BaseURL: claude.URL, CallTimeout: 5 * time.Second, MaxRetries: intp(0),
		})
		if err != nil {
			t.Fatal(err)
		}
		d.Models = tbl
	})
	return hy
}

func okResult(id string, written ...string) api.ToolResult {
	return api.ToolResult{ToolCallID: id, OK: true, Written: written}
}

// wantFastFinish checks a stream that ended without a closing call: the
// service's own line marked source:"service", then turn.done with
// fast_finish:true and stop end_turn.
func wantFastFinish(t *testing.T, evs []api.CollectedEvent, ops int) {
	t.Helper()
	if kinds(evs) != "text.delta turn.done" {
		t.Fatalf("stream = %s", kinds(evs))
	}
	if d := data(t, evs[0]); d["text"] != fastFinishReply(ops) || d["source"] != api.TextSourceService {
		t.Fatalf("text.delta = %v", d)
	}
	d := data(t, evs[1])
	if d["fast_finish"] != true || d["stop"] != StopEndTurn {
		t.Fatalf("turn.done = %v", d)
	}
	if s0, s1 := data(t, evs[0])["seq"].(float64), d["seq"].(float64); s1 <= s0 {
		t.Fatalf("seqs %v then %v", s0, s1)
	}
}

// wantClosingCall checks a stream whose turn made its closing call.
func wantClosingCall(t *testing.T, evs []api.CollectedEvent, want string) {
	t.Helper()
	if len(evs) == 0 || last(evs).Event != api.EventTurnDone || text(evs) != want {
		t.Fatalf("stream = %s %q", kinds(evs), text(evs))
	}
	d := data(t, last(evs))
	if _, ok := d["fast_finish"]; ok {
		t.Fatalf("turn.done carries fast_finish: %v", d)
	}
	for _, e := range events(evs, api.EventTextDelta) {
		if _, ok := data(t, e)["source"]; ok {
			t.Fatalf("a model text.delta carries source: %s", e.Data)
		}
	}
}

func storedUsageOf(t *testing.T, h *harness, turnID string) map[string]any {
	t.Helper()
	var u map[string]any
	if err := json.Unmarshal(h.st.turn(turnID).Usage, &u); err != nil {
		t.Fatal(err)
	}
	return u
}

// userBlocks lists a captured request's message `i` content block types and
// its text blocks' text.
func userBlocks(t *testing.T, c modeltest.Captured, i int) (types []string, texts []string) {
	t.Helper()
	msgs := c.Body["messages"].([]any)
	if i < 0 {
		i += len(msgs)
	}
	m := msgs[i].(map[string]any)
	switch content := m["content"].(type) {
	case string:
		return []string{"text"}, []string{content}
	case []any:
		for _, b := range content {
			bm := b.(map[string]any)
			types = append(types, bm["type"].(string))
			if bm["type"] == "text" {
				texts = append(texts, bm["text"].(string))
			}
		}
	}
	return types, texts
}

// A clean batch on a fast_finish route ends the turn with ONE model call:
// the service says the fixed line, turn.done says fast_finish, the stored
// usage and the trace record it, and the conversation ends on the results.
// The next turn's request alternates roles — the results and the new text
// in one user message, results first — and the history stays stable.
func TestFastFinish_CleanBatchEndsWithoutClosingCall(t *testing.T) {
	hy := newFast(t, []modeltest.Step{
		reply(modeltest.Text("Updating the headline."), setFields("s1", "headline")),
		reply(modeltest.Text("Hello again.")),
	}, nil)
	turn, evs := hy.start(hy.req("solo", "set the hero headline to Fresh cookies"))
	if kinds(evs) != "text.delta tool.call turn.waiting" {
		t.Fatalf("stream 1 = %s", kinds(evs))
	}
	evs, err := hy.cont(turn.ID(), okResult("s1", "headline"))
	if err != nil {
		t.Fatal(err)
	}
	wantFastFinish(t, evs, 1)
	if d := data(t, last(evs)); d["usage"].(map[string]any)["calls"] != 1.0 {
		t.Errorf("turn.done usage = %v", d["usage"])
	}
	if n := len(hy.student.Requests()); n != 1 {
		t.Fatalf("model calls = %d, want 1", n)
	}
	row := hy.st.turn(turn.ID())
	if row.Status != "done" {
		t.Errorf("row status = %s", row.Status)
	}
	if u := storedUsageOf(t, hy.harness, turn.ID()); u["fast_finish"] != true || u["calls"] != 1.0 {
		t.Errorf("turns.usage = %v", u)
	}
	tr, ok := hy.st.trace(turn.ID())
	if !ok {
		t.Fatal("no trace")
	}
	var tevs []struct {
		Seq  int             `json:"seq"`
		Kind string          `json:"event"`
		Data json.RawMessage `json:"data"`
	}
	if err := json.Unmarshal(tr.Events, &tevs); err != nil {
		t.Fatal(err)
	}
	var tk []string
	for _, e := range tevs {
		tk = append(tk, e.Kind)
	}
	if got := strings.Join(tk, " "); got != "text.delta tool.call turn.waiting tool.result turn.fast_finish text.delta turn.done" {
		t.Fatalf("trace events = %s", got)
	}
	if ff := tevs[4]; !strings.Contains(string(ff.Data), `"ops":1`) || !strings.Contains(string(tevs[5].Data), `"source":"service"`) {
		t.Errorf("trace fast_finish = %s, text = %s", ff.Data, tevs[5].Data)
	}
	if got := strings.Join(roles(t, hy.harness, row.ConversationID), " "); got != "user assistant user" {
		t.Fatalf("stored roles = %s", got)
	}

	// Next turn: stored as its own row, sent merged.
	_, evs = hy.start(hy.req("solo", "thanks"))
	wantClosingCall(t, evs, "Hello again.")
	if got := strings.Join(roles(t, hy.harness, row.ConversationID), " "); got != "user assistant user user assistant" {
		t.Fatalf("stored roles = %s", got)
	}
	req := hy.student.Requests()[1]
	if got := msgRoles(req); got != "user assistant user" {
		t.Fatalf("turn 2 request roles = %s", got)
	}
	types, texts := userBlocks(t, req, -1)
	if strings.Join(types, ",") != "tool_result,text" || len(texts) != 1 || !strings.HasPrefix(texts[0], "<context>\n") || !strings.HasSuffix(texts[0], "thanks") {
		t.Fatalf("turn 2 last user message = %v %q", types, texts)
	}
	if res := lastToolResults(t, req); len(res) != 1 || res[0].ID != "s1" || res[0].IsError || res[0].Content != `{"ok":true,"written":["headline"]}` {
		t.Fatalf("turn 2 replays results %+v", res)
	}
	// Stable history: the earlier turn's user message lost its context.
	if _, texts := userBlocks(t, req, 0); len(texts) != 1 || texts[0] != "set the hero headline to Fresh cookies" {
		t.Fatalf("turn 2 first user message = %q", texts)
	}
}

func TestFastFinish_SeveralOps(t *testing.T) {
	hy := newFast(t, []modeltest.Step{reply(setFields("s1", "headline"), setFields("s2", "subheadline"))}, nil)
	turn, _ := hy.start(hy.req("solo", "set both"))
	evs, err := hy.cont(turn.ID(), okResult("s1", "headline"), okResult("s2", "subheadline"))
	if err != nil {
		t.Fatal(err)
	}
	wantFastFinish(t, evs, 2)
	if !strings.Contains(text(evs), "those 2 changes") {
		t.Errorf("text = %q", text(evs))
	}
}

// A batch the model wrote after a clean lookup is judged on its own: the
// lookup's results message carried no error.
func TestFastFinish_AfterCleanLookup(t *testing.T) {
	hy := newFast(t, []modeltest.Step{
		reply(modeltest.ToolUse("lk1", "lookup_fields", map[string]any{"target": heroTarget, "query": "headline"})),
		reply(setFields("s1", "headline")),
	}, nil)
	turn, evs := hy.start(hy.req("solo", "set the hero headline"))
	if last(evs).Event != api.EventTurnWaiting {
		t.Fatalf("stream 1 = %s", kinds(evs))
	}
	if res := lastToolResults(t, hy.student.Requests()[1]); len(res) != 1 || res[0].IsError {
		t.Fatalf("lookup result = %+v", res)
	}
	evs, err := hy.cont(turn.ID(), okResult("s1", "headline"))
	if err != nil {
		t.Fatal(err)
	}
	wantFastFinish(t, evs, 1)
	if n := len(hy.student.Requests()); n != 2 {
		t.Fatalf("model calls = %d", n)
	}
}

// The launcher echoes a section-targeted op's own target.sectionId as the
// result's sectionId (cookie-cutter e7e90106 turns.js): not a new section,
// so a clean section edit still finishes fast — set_fields and an
// edit_list update alike.
func TestFastFinish_LauncherEchoedSectionID(t *testing.T) {
	hy := newFast(t, []modeltest.Step{
		reply(setFields("s1", "headline"),
			modeltest.ToolUse("u1", "edit_list", map[string]any{"target": heroTarget, "list": "slides", "action": "update", "index": 0, "item": map[string]any{"bgColor": "#fff"}})),
	}, nil)
	turn, evs := hy.start(hy.req("solo", "set the hero headline and background"))
	if last(evs).Event != api.EventTurnWaiting {
		t.Fatalf("stream 1 = %s", kinds(evs))
	}
	evs, err := hy.cont(turn.ID(),
		api.ToolResult{ToolCallID: "s1", OK: true, Written: []string{"headline"}, SectionID: "hero-1"},
		api.ToolResult{ToolCallID: "u1", OK: true, SectionID: "hero-1"})
	if err != nil {
		t.Fatal(err)
	}
	wantFastFinish(t, evs, 2)
	if n := len(hy.student.Requests()); n != 1 {
		t.Fatalf("model calls = %d", n)
	}
}

// Every other batch keeps the closing call.
func TestFastFinish_ClosingCallStillRuns(t *testing.T) {
	op := reply(setFields("s1", "headline"))
	closing := reply(modeltest.Text("Done."))
	cases := []struct {
		name    string
		route   string
		steps   []modeltest.Step
		results []api.ToolResult
		calls   int
	}{
		{"route option off", "slow", []modeltest.Step{op, closing}, []api.ToolResult{okResult("s1", "headline")}, 2},
		{"warning", "solo", []modeltest.Step{op, closing},
			[]api.ToolResult{{ToolCallID: "s1", OK: true, Written: []string{"headline"}, Warnings: []string{"W_CLAMPED:headline"}}}, 2},
		{"engine refused the op", "solo", []modeltest.Step{op, closing}, []api.ToolResult{refusedResult("s1")}, 2},
		{"hidden member", "solo", []modeltest.Step{op, closing},
			[]api.ToolResult{{ToolCallID: "s1", OK: true, Hidden: json.RawMessage(`[{"key":"headline","by":"layout","byValue":"split"}]`)}}, 2},
		{"malformed hidden member", "solo", []modeltest.Step{op, closing},
			[]api.ToolResult{{ToolCallID: "s1", OK: true, Hidden: json.RawMessage(`{"headline":1}`)}}, 2},
		{"a sectionId that is not the op's own", "solo", []modeltest.Step{op, closing},
			[]api.ToolResult{{ToolCallID: "s1", OK: true, SectionID: "faq-2"}}, 2},
		// The deployed launcher (cookie-cutter e7e90106 turns.js) answers an
		// add with no sectionId at all: the op type alone keeps the call.
		{"section add, answered as the launcher does", "solo", []modeltest.Step{
			reply(modeltest.ToolUse("a1", "edit_list", map[string]any{"target": map[string]any{"kind": "page", "page": "home"}, "list": "sections", "action": "add", "item": map[string]any{"type": "faq"}})), closing},
			[]api.ToolResult{okResult("a1")}, 2},
		{"section duplicate", "solo", []modeltest.Step{
			reply(modeltest.ToolUse("d1", "edit_list", map[string]any{"target": map[string]any{"kind": "page", "page": "home"}, "list": "sections", "action": "duplicate", "sectionId": "hero-1"})), closing},
			[]api.ToolResult{{ToolCallID: "d1", OK: true, SectionID: "hero-1"}}, 2},
		{"list row add", "solo", []modeltest.Step{
			reply(modeltest.ToolUse("a2", "edit_list", map[string]any{"target": heroTarget, "list": "slides", "action": "add", "item": map[string]any{"bgColor": "#fff"}})), closing},
			[]api.ToolResult{{ToolCallID: "a2", OK: true, SectionID: "hero-1"}}, 2},
		{"missing result", "solo", []modeltest.Step{op, closing}, nil, 2},
		{"one of two ops warned", "solo", []modeltest.Step{reply(setFields("s1", "headline"), setFields("s2", "subheadline")), closing},
			[]api.ToolResult{okResult("s1", "headline"), {ToolCallID: "s2", OK: true, Warnings: []string{"W_HIDDEN:subheadline"}}}, 2},
		{"server tool in the batch", "solo", []modeltest.Step{
			reply(modeltest.ToolUse("lk1", "lookup_fields", map[string]any{"target": heroTarget, "query": "headline"}), setFields("s1", "headline")), closing},
			[]api.ToolResult{okResult("s1", "headline")}, 2},
		{"op the service refused in the batch", "solo", []modeltest.Step{
			reply(modeltest.ToolUse("r1", "reset", map[string]any{"target": heroTarget}), setFields("s1", "headline")), closing},
			[]api.ToolResult{okResult("s1", "headline")}, 2},
		{"report_unsupported in the batch", "solo", []modeltest.Step{
			reply(modeltest.ToolUse("u1", "report_unsupported", map[string]any{"request": "a chat widget", "reason": "no such feature"}), setFields("s1", "headline")), closing},
			[]api.ToolResult{okResult("s1", "headline")}, 2},
		{"pick_asset", "solo", []modeltest.Step{
			reply(modeltest.ToolUse("p1", "pick_asset", map[string]any{"purpose": "hero image"})), closing},
			[]api.ToolResult{{ToolCallID: "p1", OK: true, Asset: &api.Asset{PublicURL: "https://cdn/x.jpg"}}}, 2},
		{"earlier server-side error", "solo", []modeltest.Step{
			reply(modeltest.ToolUse("x1", "frobnicate", map[string]any{})), op, closing},
			[]api.ToolResult{okResult("s1", "headline")}, 3},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			hy := newFast(t, tc.steps, nil)
			turn, evs := hy.start(hy.req(tc.route, "set the hero headline"))
			if e := last(evs).Event; e != api.EventTurnWaiting && e != api.EventTurnSuspended {
				t.Fatalf("stream 1 = %s", kinds(evs))
			}
			evs, err := hy.cont(turn.ID(), tc.results...)
			if err != nil {
				t.Fatal(err)
			}
			wantClosingCall(t, evs, "Done.")
			if n := len(hy.student.Requests()); n != tc.calls {
				t.Fatalf("model calls = %d, want %d", n, tc.calls)
			}
			if u := storedUsageOf(t, hy.harness, turn.ID()); u["fast_finish"] != nil {
				t.Errorf("turns.usage = %v", u)
			}
		})
	}
}

// An error in an earlier batch of the turn keeps the closing call after a
// later clean one — on this instance, and after a rebuild from the row
// (pending doc `had_error`). A clean batch after a rebuild still finishes
// fast.
func TestFastFinish_EarlierErrorAndRebuild(t *testing.T) {
	for _, rebuilt := range []bool{false, true} {
		name := "same instance"
		if rebuilt {
			name = "rebuilt"
		}
		t.Run(name, func(t *testing.T) {
			hy := newFast(t, []modeltest.Step{
				reply(setFields("s1", "itemDescriptions")),
				reply(setFields("s2", "headline")),
				reply(modeltest.Text("Fixed: I used the headline.")),
			}, nil)
			turn, _ := hy.start(hy.req("solo", "set the hero headline"))
			evs, err := hy.cont(turn.ID(), refusedResult("s1"))
			if err != nil || kinds(evs) != "tool.call turn.waiting" {
				t.Fatalf("stream 2 = %s %v", kinds(evs), err)
			}
			var pd pendingDoc
			if err := json.Unmarshal(hy.st.turn(turn.ID()).PendingToolCalls, &pd); err != nil || !pd.HadError {
				t.Fatalf("pending doc had_error = %v (%v)", pd.HadError, err)
			}
			if rebuilt {
				hy.runner.unpark(turn.ID())
			}
			evs, err = hy.cont(turn.ID(), okResult("s2", "headline"))
			if err != nil {
				t.Fatal(err)
			}
			wantClosingCall(t, evs, "Fixed: I used the headline.")
			if n := len(hy.student.Requests()); n != 3 {
				t.Fatalf("model calls = %d", n)
			}

			hy = newFast(t, []modeltest.Step{reply(setFields("s1", "headline"))}, nil)
			turn, _ = hy.start(hy.req("solo", "set the hero headline"))
			if rebuilt {
				hy.runner.unpark(turn.ID())
			}
			evs, err = hy.cont(turn.ID(), okResult("s1", "headline"))
			if err != nil {
				t.Fatal(err)
			}
			wantFastFinish(t, evs, 1)
		})
	}
}

// Hybrid escalation is unchanged on a fast_finish route: a clean batch
// finishes fast without the fallback; a refused op or a hidden write still
// escalates, and an escalated turn keeps its closing call on the fallback.
func TestFastFinish_Hybrid(t *testing.T) {
	t.Run("clean", func(t *testing.T) {
		hy := newFast(t, []modeltest.Step{reply(setFields("s1", "headline"))}, nil)
		turn, _ := hy.start(hy.req("student", "set the hero headline"))
		evs, err := hy.cont(turn.ID(), okResult("s1", "headline"))
		if err != nil {
			t.Fatal(err)
		}
		wantFastFinish(t, evs, 1)
		if got := routesOf(t, last(evs)); got != `["student"]` {
			t.Errorf("routes = %s", got)
		}
		if n := len(hy.claude.Requests()); n != 0 {
			t.Errorf("claude calls = %d", n)
		}
	})
	t.Run("refused op escalates", func(t *testing.T) {
		hy := newFast(t,
			[]modeltest.Step{reply(setFields("s1", "itemDescriptions"))},
			[]modeltest.Step{reply(setFields("c1", "headline")), reply(modeltest.Text("Fixed."))})
		turn, _ := hy.start(hy.req("student", "set the hero headline"))
		evs, err := hy.cont(turn.ID(), refusedResult("s1"))
		if err != nil {
			t.Fatal(err)
		}
		wantEscalated(t, evs, EscalateToolError)
		evs, err = hy.cont(turn.ID(), okResult("c1", "headline"))
		if err != nil {
			t.Fatal(err)
		}
		wantClosingCall(t, evs, "Fixed.")
		if got := routesOf(t, last(evs)); got != `["student","backstop"]` {
			t.Errorf("routes = %s", got)
		}
		if n := len(hy.claude.Requests()); n != 2 {
			t.Errorf("claude calls = %d", n)
		}
	})
	t.Run("hidden write escalates", func(t *testing.T) {
		hy := newFast(t,
			[]modeltest.Step{reply(setMaxWidth("s1"))},
			[]modeltest.Step{reply(modeltest.Text("The width only shows with the Classic layout."))})
		turn, _ := hy.start(hy.req("student", "make the hero narrower"))
		evs, err := hy.cont(turn.ID(), hiddenResult("s1", "split"))
		if err != nil {
			t.Fatal(err)
		}
		wantEscalated(t, evs, EscalateHiddenWrite)
		wantClosingCall(t, evs, "The width only shows with the Classic layout.")
	})
}

// On an anthropic route the next turn's request is valid as Claude reads
// it: roles alternate, the results and the new text share one user message
// (tool_result blocks first), the ops message's thinking block goes back
// exactly as Claude wrote it, and the earlier user message is replayed as
// stored (with its context: no stable-history rewrite on Claude).
func TestFastFinish_AnthropicNextTurn(t *testing.T) {
	hy := newFast(t, nil, []modeltest.Step{
		reply(modeltest.Thinking("claude thoughts"), modeltest.Text("Updating."), setFields("c1", "headline")),
		reply(modeltest.Text("You're welcome.")),
	})
	turn, evs := hy.start(hy.req("claude-fast", "set the hero headline"))
	if last(evs).Event != api.EventTurnWaiting {
		t.Fatalf("stream 1 = %s", kinds(evs))
	}
	evs, err := hy.cont(turn.ID(), okResult("c1", "headline"))
	if err != nil {
		t.Fatal(err)
	}
	wantFastFinish(t, evs, 1)

	_, evs = hy.start(hy.req("claude-fast", "thanks"))
	wantClosingCall(t, evs, "You're welcome.")
	reqs := hy.claude.Requests()
	if len(reqs) != 2 {
		t.Fatalf("claude calls = %d", len(reqs))
	}
	req := reqs[1]
	if got := msgRoles(req); got != "user assistant user" {
		t.Fatalf("roles = %s", got)
	}
	var body struct {
		Messages []struct {
			Role    string            `json:"role"`
			Content []json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(req.Raw, &body); err != nil {
		t.Fatal(err)
	}
	var think struct{ Type, Thinking, Signature string }
	if err := json.Unmarshal(body.Messages[1].Content[0], &think); err != nil || think.Type != "thinking" ||
		think.Thinking != "claude thoughts" || think.Signature != "sig" {
		t.Fatalf("assistant block 0 = %s", body.Messages[1].Content[0])
	}
	types, texts := userBlocks(t, req, -1)
	if strings.Join(types, ",") != "tool_result,text" || len(texts) != 1 || !strings.HasSuffix(texts[0], "thanks") {
		t.Fatalf("last user message = %v %q", types, texts)
	}
	if _, texts := userBlocks(t, req, 0); len(texts) != 1 || !strings.HasPrefix(texts[0], "<context>\n") {
		t.Fatalf("first user message = %q", texts)
	}
	// The assistant ops message is byte-identical across later turns.
	_, _ = hy.start(hy.req("claude-fast", "and again"))
	var body3 struct {
		Messages []json.RawMessage `json:"messages"`
	}
	reqs = hy.claude.Requests()
	if len(reqs) != 3 {
		t.Fatalf("claude calls = %d", len(reqs))
	}
	var body2 struct {
		Messages []json.RawMessage `json:"messages"`
	}
	_ = json.Unmarshal(reqs[1].Raw, &body2)
	_ = json.Unmarshal(reqs[2].Raw, &body3)
	if string(body2.Messages[1]) != string(body3.Messages[1]) || string(body2.Messages[2]) != string(body3.Messages[2]) {
		t.Fatalf("history moved:\n%s\n%s", body2.Messages[1], body3.Messages[1])
	}
}

// wantResultsThenText checks a captured request alternates roles and ends
// with ONE user message of tool_result blocks followed by the turn's text.
func wantResultsThenText(t *testing.T, c modeltest.Captured) {
	t.Helper()
	roles := strings.Fields(msgRoles(c))
	for i := 1; i < len(roles); i++ {
		if roles[i] == roles[i-1] {
			t.Fatalf("roles do not alternate: %v", roles)
		}
	}
	types, texts := userBlocks(t, c, -1)
	n := len(types)
	if n < 2 || types[n-1] != "text" || len(texts) != 1 {
		t.Fatalf("last user message = %v", types)
	}
	for _, ty := range types[:n-1] {
		if ty != "tool_result" {
			t.Fatalf("last user message = %v", types)
		}
	}
}
