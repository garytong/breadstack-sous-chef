package loop

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/breadstack/sous-chef/internal/api"
	"github.com/breadstack/sous-chef/internal/config"
	"github.com/breadstack/sous-chef/internal/model"
	"github.com/breadstack/sous-chef/internal/model/modeltest"
	"github.com/breadstack/sous-chef/internal/registry"
	"github.com/breadstack/sous-chef/internal/store"
	"github.com/breadstack/sous-chef/internal/usage"
)

// --- fixtures ---

var (
	compiledOnce sync.Once
	compiled     *registry.Compiled
	compiledErr  error
)

func snapshotCompiled(t *testing.T) *registry.Compiled {
	t.Helper()
	compiledOnce.Do(func() {
		var body []byte
		body, compiledErr = os.ReadFile("../../testdata/manifest.snapshot.json")
		if compiledErr != nil {
			return
		}
		compiled, compiledErr = registry.Compile(body)
	})
	if compiledErr != nil {
		t.Fatalf("snapshot: %v", compiledErr)
	}
	return compiled
}

type memLoader struct{ byHash map[string]*registry.Compiled }

func (l memLoader) Load(_ context.Context, hash string) (*registry.Compiled, error) {
	if c, ok := l.byHash[hash]; ok {
		return c, nil
	}
	return nil, api.ErrUnknownManifest(hash)
}

type harness struct {
	t      *testing.T
	st     *memStore
	fake   *modeltest.Server
	runner *Runner
	hash   string
	clock  time.Time
}

func newHarness(t *testing.T, mutate func(*Deps), steps ...modeltest.Step) *harness {
	t.Helper()
	c := snapshotCompiled(t)
	fake := modeltest.New(steps...)
	t.Cleanup(fake.Close)
	tbl, err := model.NewTable(model.Options{
		Routes: map[string]config.Route{
			"default": {Kind: config.RouteAnthropic, Auth: config.AuthAPIKey, Model: "claude-sonnet-5-5", MaxTokens: 1024, Effort: "medium"},
			"opus":    {Kind: config.RouteAnthropic, Auth: config.AuthAPIKey, Model: "claude-opus-5-5", MaxTokens: 1024, Effort: "medium"},
		},
		Default: "default", Allowlist: []string{"default", "opus"},
		APIKey: "sk-test", BaseURL: fake.URL, CallTimeout: 5 * time.Second, MaxRetries: intp(0),
	})
	if err != nil {
		t.Fatal(err)
	}
	h := &harness{t: t, st: newMemStore(), fake: fake, hash: c.Hash, clock: time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)}
	h.st.now = func() time.Time { return h.clock }
	d := Deps{
		Store: h.st, Manifests: memLoader{byHash: map[string]*registry.Compiled{c.Hash: c}}, Models: tbl,
		Usage: usage.NewRecorder(h.st, nil), Logf: t.Logf, Now: func() time.Time { return h.clock },
	}
	if mutate != nil {
		mutate(&d)
	}
	h.runner, err = New(d)
	if err != nil {
		t.Fatal(err)
	}
	return h
}

func intp(n int) *int { return &n }

func (h *harness) request(text string) api.TurnRequest {
	return api.TurnRequest{SiteID: "site-1", StoreID: "store-1", ManifestHash: h.hash, BuildID: "abc123",
		Context: json.RawMessage(`{"page":"home","sections":[{"id":"hero-1","type":"hero"}]}`), Text: text, UserEmail: "owner@shop.com"}
}

// run starts (or continues) and runs a turn, collecting its events.
func (h *harness) start(req api.TurnRequest) (api.Turn, []api.CollectedEvent) {
	h.t.Helper()
	turn, err := h.runner.Start(context.Background(), req)
	if err != nil {
		h.t.Fatalf("Start: %v", err)
	}
	col := &api.Collector{}
	if err := turn.Run(context.Background(), col); err != nil {
		h.t.Logf("Run: %v", err)
	}
	return turn, col.Events()
}

func (h *harness) cont(turnID string, results ...api.ToolResult) ([]api.CollectedEvent, error) {
	h.t.Helper()
	turn, err := h.runner.Continue(context.Background(), turnID, api.ToolResultsRequest{Results: results})
	if err != nil {
		return nil, err
	}
	col := &api.Collector{}
	if err := turn.Run(context.Background(), col); err != nil {
		h.t.Logf("Run: %v", err)
	}
	return col.Events(), nil
}

func kinds(evs []api.CollectedEvent) string {
	var parts []string
	for _, e := range evs {
		parts = append(parts, e.Event)
	}
	return strings.Join(parts, " ")
}

func data(t *testing.T, ev api.CollectedEvent) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(ev.Data, &m); err != nil {
		t.Fatal(err)
	}
	return m
}

func text(evs []api.CollectedEvent) string {
	var sb strings.Builder
	for _, e := range evs {
		if e.Event == api.EventTextDelta {
			var d struct{ Text string }
			_ = json.Unmarshal(e.Data, &d)
			sb.WriteString(d.Text)
		}
	}
	return sb.String()
}

func last(evs []api.CollectedEvent) api.CollectedEvent { return evs[len(evs)-1] }

// toolResults extracts the tool_result blocks of the last user message of a
// captured request, in order: id → (content, is_error).
type capturedResult struct {
	ID      string
	Content string
	IsError bool
}

func lastToolResults(t *testing.T, c modeltest.Captured) []capturedResult {
	t.Helper()
	msgs := c.Body["messages"].([]any)
	lastMsg := msgs[len(msgs)-1].(map[string]any)
	if lastMsg["role"] != "user" {
		t.Fatalf("last message role = %v", lastMsg["role"])
	}
	var out []capturedResult
	for _, b := range lastMsg["content"].([]any) {
		m := b.(map[string]any)
		if m["type"] != "tool_result" {
			continue
		}
		content := ""
		switch cv := m["content"].(type) {
		case string:
			content = cv
		case []any:
			for _, x := range cv {
				content += x.(map[string]any)["text"].(string)
			}
		}
		isErr, _ := m["is_error"].(bool)
		out = append(out, capturedResult{ID: m["tool_use_id"].(string), Content: content, IsError: isErr})
	}
	return out
}

var heroTarget = map[string]any{"kind": "section", "page": "home", "sectionId": "hero-1"}

// --- tests ---

func TestStart_PreChecks(t *testing.T) {
	h := newHarness(t, func(d *Deps) { d.MaxConcurrent = 1 })
	ctx := context.Background()

	req := h.request("hi")
	req.ManifestHash = strings.Repeat("a", 64)
	_, err := h.runner.Start(ctx, req)
	var apiErr *api.Error
	if !errors.As(err, &apiErr) || apiErr.Status != 409 || apiErr.Code != "unknown_manifest" || apiErr.Extra["hash"] != req.ManifestHash {
		t.Errorf("unknown hash: %v", err)
	}

	req = h.request("hi")
	req.Model = "nope"
	if _, err := h.runner.Start(ctx, req); !errors.As(err, &apiErr) || apiErr.Status != 400 {
		t.Errorf("bad route: %v", err)
	}

	req = h.request("hi")
	req.Context = json.RawMessage(`[1,2]`)
	if _, err := h.runner.Start(ctx, req); !errors.As(err, &apiErr) || apiErr.Status != 400 {
		t.Errorf("bad context: %v", err)
	}

	// Busy: one slot, held by a started-but-not-run turn.
	first, err := h.runner.Start(ctx, h.request("one"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := h.runner.Start(ctx, h.request("two")); !errors.As(err, &apiErr) || apiErr.Code != "busy" || apiErr.Status != 503 {
		t.Errorf("busy: %v", err)
	}
	if h.runner.Live() != 1 {
		t.Errorf("live = %d", h.runner.Live())
	}
	// Running the first (script empty → model 500 → turn.error) releases the slot.
	col := &api.Collector{}
	_ = first.Run(ctx, col)
	if h.runner.Live() != 0 || kinds(col.Events()) != "turn.error" {
		t.Errorf("after run: live=%d events=%s", h.runner.Live(), kinds(col.Events()))
	}
	if d := data(t, last(col.Events())); d["code"] != "model_unavailable" {
		t.Errorf("error code = %v", d["code"])
	}
	if h.st.turn(first.ID()).Status != store.TurnError {
		t.Errorf("turn status = %s", h.st.turn(first.ID()).Status)
	}
	rows := h.st.usageRows()
	if len(rows) != 1 || rows[0].Status != "error" || rows[0].Operation != usage.OpChat {
		t.Errorf("usage rows = %+v", rows)
	}
}

func TestTurn_TextOnly(t *testing.T) {
	h := newHarness(t, nil, modeltest.Reply(modeltest.Message{
		Content: []modeltest.Block{modeltest.Text("Hello there.")},
		Usage:   modeltest.Usage{InputTokens: 10, OutputTokens: 4, CacheReadInputTokens: 900},
	}))
	turn, evs := h.start(h.request("hello"))
	if kinds(evs) != "text.delta text.delta turn.done" || text(evs) != "Hello there." {
		t.Fatalf("events = %s text=%q", kinds(evs), text(evs))
	}
	d := data(t, last(evs))
	if d["stop"] != "end_turn" || d["conversationId"] == "" || d["conversationRotated"] != nil {
		t.Errorf("done = %v", d)
	}
	u := d["usage"].(map[string]any)
	if u["calls"] != 1.0 || u["input_tokens"] != 10.0 || u["output_tokens"] != 4.0 || u["cache_read_input_tokens"] != 900.0 {
		t.Errorf("usage = %v", u)
	}
	// Row: done, usage persisted; conversation has user + assistant messages.
	row := h.st.turn(turn.ID())
	if row.Status != store.TurnDone || !strings.Contains(string(row.Usage), `"calls":1`) {
		t.Errorf("row = %+v", row)
	}
	msgs, _ := h.st.ListMessages(context.Background(), row.ConversationID, 0, 0)
	var userBlocks []struct{ Text string }
	_ = json.Unmarshal(msgs[0].Content, &userBlocks)
	if len(msgs) != 2 || msgs[0].Role != "user" || msgs[1].Role != "assistant" || len(userBlocks) != 1 || !strings.HasPrefix(userBlocks[0].Text, "<context>\n{\"page\":\"home\"") || !strings.HasSuffix(userBlocks[0].Text, "</context>\nhello") {
		t.Errorf("messages = %d, user text = %q", len(msgs), userBlocks)
	}
	// The request carried the prefix: 10 tools, 2 system blocks with cache_control, tool_choice auto.
	reqs := h.fake.Requests()
	if len(reqs) != 1 || len(reqs[0].Body["tools"].([]any)) != 10 || len(reqs[0].Body["system"].([]any)) != 2 {
		t.Fatalf("request shape: %d requests", len(reqs))
	}
	if tc := reqs[0].Body["tool_choice"].(map[string]any); tc["type"] != "auto" {
		t.Errorf("tool_choice = %v", tc)
	}
	if !strings.Contains(reqs[0].SystemText(), "s:heroBanner") || !strings.Contains(reqs[0].SystemText(), "Sous Chef") {
		t.Error("system blocks lack rules/index")
	}
	// Trace recorded with the merchant text and one call.
	tr, ok := h.st.trace(turn.ID())
	if !ok || !strings.Contains(string(tr.Ops), `"text":"hello"`) || !strings.Contains(string(tr.Ops), `"status":"done"`) {
		t.Errorf("trace = %v %s", ok, tr.Ops)
	}
	if !strings.Contains(string(tr.Events), `"event":"turn.done"`) {
		t.Errorf("trace events = %s", tr.Events)
	}
	// Usage row.
	rows := h.st.usageRows()
	if len(rows) != 1 || rows[0].Status != "ok" || rows[0].InputTokens != 10 || rows[0].CacheReadTokens != 900 || rows[0].TurnID != turn.ID() {
		t.Errorf("usage rows = %+v", rows)
	}
}

func TestTurn_ServerToolsAnsweredInline(t *testing.T) {
	h := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{
			modeltest.Text("Let me check."),
			modeltest.ToolUse("tu1", "lookup_fields", map[string]any{"target": heroTarget, "query": "headline"}),
			modeltest.ToolUse("tu2", "describe_target", map[string]any{"target": heroTarget}),
		}}),
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("The hero has a headline field.")}}),
	)
	turn, evs := h.start(h.request("what can I change on the hero?"))
	if !strings.HasSuffix(kinds(evs), "turn.done") || strings.Contains(kinds(evs), "tool.call") || strings.Contains(kinds(evs), "turn.waiting") {
		t.Fatalf("events = %s", kinds(evs))
	}
	if text(evs) != "Let me check.The hero has a headline field." {
		t.Errorf("text = %q", text(evs))
	}
	reqs := h.fake.Requests()
	if len(reqs) != 2 {
		t.Fatalf("model calls = %d", len(reqs))
	}
	res := lastToolResults(t, reqs[1])
	if len(res) != 2 || res[0].ID != "tu1" || res[1].ID != "tu2" || res[0].IsError || res[1].IsError {
		t.Fatalf("tool results = %+v", res)
	}
	if !strings.Contains(res[0].Content, `"headline"`) || !strings.Contains(res[1].Content, `"lists"`) {
		t.Errorf("server results: %.200s | %.200s", res[0].Content, res[1].Content)
	}
	// Prefix identical on both calls.
	if reqs[0].SystemText() != reqs[1].SystemText() {
		t.Error("system differs between calls")
	}
	t0, _ := json.Marshal(reqs[0].Body["tools"])
	t1, _ := json.Marshal(reqs[1].Body["tools"])
	if string(t0) != string(t1) {
		t.Error("tools differ between calls")
	}
	d := data(t, last(evs))
	if d["usage"].(map[string]any)["calls"] != 2.0 {
		t.Errorf("calls = %v", d["usage"])
	}
	if len(h.st.usageRows()) != 2 || h.st.usageRows()[1].Operation != usage.OpToolTurn {
		t.Errorf("usage rows = %+v", h.st.usageRows())
	}
	tr, _ := h.st.trace(turn.ID())
	if !strings.Contains(string(tr.Ops), `"server":true`) {
		t.Error("trace ops must flag server tools")
	}
}

func TestTurn_MixedBatchWaitsThenContinues(t *testing.T) {
	h := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{
			modeltest.ToolUse("op1", "set_fields", map[string]any{"target": heroTarget, "values": map[string]any{"headline": "Fresh cookies"}}),
			modeltest.ToolUse("lk1", "lookup_fields", map[string]any{"target": heroTarget, "query": "headline"}),
			modeltest.ToolUse("op2", "edit_list", map[string]any{"target": heroTarget, "list": "slides", "action": "update", "index": 0, "item": map[string]any{"bgColor": "#fff"}}),
		}}),
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("Done — headline updated.")}}),
	)
	turn, evs := h.start(h.request("change the hero headline"))
	if kinds(evs) != "tool.call tool.call turn.waiting" {
		t.Fatalf("events = %s", kinds(evs))
	}
	c1 := data(t, evs[0])
	if c1["name"] != "set_fields" || c1["id"] != "op1" || c1["seq"] != 1.0 || c1["turnId"] != turn.ID() {
		t.Errorf("tool.call = %v", c1)
	}
	if in := c1["input"].(map[string]any); in["values"].(map[string]any)["headline"] != "Fresh cookies" {
		t.Errorf("input = %v", in)
	}
	w := data(t, evs[2])
	ids, _ := json.Marshal(w["toolCallIds"])
	if string(ids) != `["op1","op2"]` {
		t.Errorf("toolCallIds = %s", ids)
	}
	row := h.st.turn(turn.ID())
	if row.Status != store.TurnWaitingTools || !strings.Contains(string(row.PendingToolCalls), `"server_results"`) {
		t.Errorf("row = %s %s", row.Status, row.PendingToolCalls)
	}
	if h.runner.Parked() != 1 || h.runner.Live() != 0 {
		t.Errorf("parked=%d live=%d", h.runner.Parked(), h.runner.Live())
	}

	// Unknown id → 400, still waiting.
	_, err := h.cont(turn.ID(), api.ToolResult{ToolCallID: "zzz", OK: true})
	var apiErr *api.Error
	if !errors.As(err, &apiErr) || apiErr.Status != 400 || h.st.turn(turn.ID()).Status != store.TurnWaitingTools || h.runner.Parked() != 1 {
		t.Errorf("unknown id: %v status=%s parked=%d", err, h.st.turn(turn.ID()).Status, h.runner.Parked())
	}

	// op2 missing → is_error for it; op1 ok; lk1 from the stash; order = model order.
	evs2, err := h.cont(turn.ID(), api.ToolResult{ToolCallID: "op1", OK: true, Written: []string{"headline"}, SectionID: "hero-1", Warnings: []string{"W_HIDDEN:headline"}})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(kinds(evs2), "turn.done") || text(evs2) != "Done — headline updated." {
		t.Fatalf("events = %s", kinds(evs2))
	}
	reqs := h.fake.Requests()
	res := lastToolResults(t, reqs[1])
	if len(res) != 3 || res[0].ID != "op1" || res[1].ID != "lk1" || res[2].ID != "op2" {
		t.Fatalf("results order = %+v", res)
	}
	if res[0].IsError || !strings.Contains(res[0].Content, `"written":["headline"]`) || !strings.Contains(res[0].Content, `"sectionId":"hero-1"`) {
		t.Errorf("op1 = %+v", res[0])
	}
	if res[1].IsError || !strings.Contains(res[1].Content, "headline") {
		t.Errorf("lk1 = %+v", res[1])
	}
	if !res[2].IsError || !strings.Contains(res[2].Content, "no result returned") {
		t.Errorf("op2 = %+v", res[2])
	}
	// seq continues across the two streams.
	if d := data(t, last(evs2)); d["seq"].(float64) <= 3 {
		t.Errorf("seq did not continue: %v", d["seq"])
	}
	if h.runner.Parked() != 0 || h.st.turn(turn.ID()).Status != store.TurnDone {
		t.Errorf("parked=%d status=%s", h.runner.Parked(), h.st.turn(turn.ID()).Status)
	}
	// The trace keeps browser outcomes apart from the model's ops.
	tr, _ := h.st.trace(turn.ID())
	if !strings.Contains(string(tr.Events), `"event":"tool.result"`) || !strings.Contains(string(tr.Events), `"for_seq":1`) {
		t.Errorf("trace events = %s", tr.Events)
	}
	var ops struct {
		Ops []struct {
			Name   string `json:"name"`
			Server bool   `json:"server"`
		} `json:"ops"`
	}
	_ = json.Unmarshal(tr.Ops, &ops)
	if len(ops.Ops) != 3 || ops.Ops[0].Name != "set_fields" || ops.Ops[0].Server || !ops.Ops[1].Server {
		t.Errorf("trace ops = %+v", ops.Ops)
	}
	// 409 once done.
	_, err = h.cont(turn.ID(), api.ToolResult{ToolCallID: "op1", OK: true})
	if !errors.As(err, &apiErr) || apiErr.Code != "turn_not_waiting" || apiErr.Extra["status"] != "done" {
		t.Errorf("after done: %v", err)
	}
	_, err = h.cont("turn-404", api.ToolResult{ToolCallID: "op1", OK: true})
	if !errors.As(err, &apiErr) || apiErr.Code != "turn_not_found" {
		t.Errorf("missing: %v", err)
	}
}

func TestTurn_PickAssetSuspendsAndResumes(t *testing.T) {
	h := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{
			modeltest.Text("Pick an image."),
			modeltest.ToolUse("pa1", "pick_asset", map[string]any{"purpose": "Hero background", "asset_type": "image"}),
		}}),
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{
			modeltest.ToolUse("sf1", "set_fields", map[string]any{"target": heroTarget,
				"values": map[string]any{"bgImage": "https://cdn/x.jpg"},
				"assets": map[string]any{"bgImage": map[string]any{"alt": "a cookie", "srcset": "https://cdn/x.jpg 1x"}}}),
		}}),
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("Background set.")}}),
	)
	turn, evs := h.start(h.request("put a new background on the hero"))
	if kinds(evs) != "text.delta text.delta turn.suspended" {
		t.Fatalf("events = %s", kinds(evs))
	}
	s := data(t, last(evs))
	if s["reason"] != "pick_asset" || s["toolCallId"] != "pa1" || s["purpose"] != "Hero background" || s["assetType"] != "image" {
		t.Errorf("suspended = %v", s)
	}
	if h.st.turn(turn.ID()).Status != store.TurnSuspended {
		t.Errorf("status = %s", h.st.turn(turn.ID()).Status)
	}
	evs2, err := h.cont(turn.ID(), api.ToolResult{ToolCallID: "pa1", OK: true, Asset: &api.Asset{PublicURL: "https://cdn/x.jpg", Alt: "a cookie", Srcset: json.RawMessage(`{"480":"https://cdn/x-480.jpg","960":"https://cdn/x.jpg"}`)}})
	if err != nil {
		t.Fatal(err)
	}
	if kinds(evs2) != "tool.call turn.waiting" {
		t.Fatalf("events = %s", kinds(evs2))
	}
	in := data(t, evs2[0])["input"].(map[string]any)
	if in["values"].(map[string]any)["bgImage"] != "https://cdn/x.jpg" || in["assets"].(map[string]any)["bgImage"].(map[string]any)["alt"] != "a cookie" {
		t.Errorf("set_fields input = %v", in)
	}
	res := lastToolResults(t, h.fake.Requests()[1])
	if len(res) != 1 || res[0].IsError || !strings.Contains(res[0].Content, `"public_url":"https://cdn/x.jpg"`) || !strings.Contains(res[0].Content, `"alt":"a cookie"`) {
		t.Errorf("pick_asset result = %+v", res)
	}
	evs3, err := h.cont(turn.ID(), api.ToolResult{ToolCallID: "sf1", OK: true, Written: []string{"bgImage"}})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(kinds(evs3), "turn.done") || text(evs3) != "Background set." {
		t.Fatalf("events = %s", kinds(evs3))
	}
	if d := data(t, last(evs3)); d["usage"].(map[string]any)["calls"] != 3.0 {
		t.Errorf("calls = %v", d["usage"])
	}

	// A declined picker is an error the model reads.
	h2 := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.ToolUse("pa1", "pick_asset", map[string]any{"purpose": "x", "asset_type": "image"})}}),
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("No image chosen.")}}),
	)
	turn2, _ := h2.start(h2.request("image please"))
	if _, err := h2.cont(turn2.ID(), api.ToolResult{ToolCallID: "pa1", OK: true}); err != nil {
		t.Fatal(err)
	}
	res = lastToolResults(t, h2.fake.Requests()[1])
	if len(res) != 1 || !res[0].IsError || !strings.Contains(res[0].Content, "E_ASSET") {
		t.Errorf("declined picker = %+v", res)
	}
}

func TestTurn_BrowserErrorReachesModel(t *testing.T) {
	h := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.ToolUse("op1", "set_fields", map[string]any{"target": heroTarget, "values": map[string]any{"nope": 1}})}}),
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("That key does not exist.")}}),
	)
	turn, _ := h.start(h.request("set nope"))
	if _, err := h.cont(turn.ID(), api.ToolResult{ToolCallID: "op1", OK: false, Error: &api.ToolError{Code: "E_FIELD", Message: "unknown key", Key: "nope"}}); err != nil {
		t.Fatal(err)
	}
	res := lastToolResults(t, h.fake.Requests()[1])
	if len(res) != 1 || !res[0].IsError || res[0].Content != `{"code":"E_FIELD","message":"unknown key","key":"nope"}` {
		t.Errorf("error result = %+v", res)
	}
}

func TestTurn_ReportUnsupportedEchoesAndEnds(t *testing.T) {
	h := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.ToolUse("ru1", "report_unsupported", map[string]any{"reason": "no payment settings here"})}}),
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("Payment settings are not editable here; please contact support.")}}),
	)
	_, evs := h.start(h.request("change the payment gateway"))
	if kinds(evs) != "tool.call text.delta text.delta turn.done" {
		t.Fatalf("events = %s", kinds(evs))
	}
	if d := data(t, evs[0]); d["name"] != "report_unsupported" {
		t.Errorf("echo = %v", d)
	}
	res := lastToolResults(t, h.fake.Requests()[1])
	if len(res) != 1 || res[0].IsError || !strings.Contains(res[0].Content, "shown_to_merchant") {
		t.Errorf("result = %+v", res)
	}
}

func TestTurn_UnknownToolIsAnError(t *testing.T) {
	h := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.ToolUse("x1", "launch_rocket", map[string]any{})}}),
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("I cannot do that.")}}),
	)
	_, evs := h.start(h.request("launch"))
	if !strings.HasSuffix(kinds(evs), "turn.done") || strings.Contains(kinds(evs), "tool.call") {
		t.Fatalf("events = %s", kinds(evs))
	}
	res := lastToolResults(t, h.fake.Requests()[1])
	if len(res) != 1 || !res[0].IsError || !strings.Contains(res[0].Content, "unknown tool") {
		t.Errorf("result = %+v", res)
	}
}

func TestTurn_MaxCalls(t *testing.T) {
	var steps []modeltest.Step
	for i := 0; i < 10; i++ {
		steps = append(steps, modeltest.Reply(modeltest.Message{Content: []modeltest.Block{
			modeltest.ToolUse("lk", "lookup_fields", map[string]any{"target": heroTarget, "query": "x"}),
		}}))
	}
	h := newHarness(t, func(d *Deps) { d.MaxCalls = 4 }, steps...)
	turn, evs := h.start(h.request("loop forever"))
	if kinds(evs) != "text.delta turn.done" || text(evs) != noConclusion {
		t.Fatalf("events = %s text=%q", kinds(evs), text(evs))
	}
	d := data(t, last(evs))
	if d["stop"] != "max_calls" || d["usage"].(map[string]any)["calls"] != 4.0 {
		t.Errorf("done = %v", d)
	}
	if n := len(h.fake.Requests()); n != 4 {
		t.Errorf("model calls = %d, want 4", n)
	}
	if h.st.turn(turn.ID()).Status != store.TurnDone {
		t.Errorf("status = %s", h.st.turn(turn.ID()).Status)
	}
	// The default is 8.
	if r, _ := New(Deps{Store: h.st, Manifests: memLoader{}, Models: h.runner.d.Models}); r.d.MaxCalls != 8 || r.d.TurnTTL != 10*time.Minute || r.d.MaxConcurrent != 8 || r.d.RotateAt != 40 {
		t.Errorf("defaults = %+v", r.d)
	}
}

func TestTurn_MaxTokensAndRefusal(t *testing.T) {
	h := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{StopReason: "max_tokens", Content: []modeltest.Block{modeltest.Text("Here is a very long")}}),
		modeltest.Reply(modeltest.Message{StopReason: "refusal", StopCategory: "cyber", Content: []modeltest.Block{}}),
		modeltest.Reply(modeltest.Message{StopReason: "max_tokens", Content: []modeltest.Block{}}),
	)
	_, evs := h.start(h.request("write"))
	if data(t, last(evs))["stop"] != "max_tokens" || !strings.HasSuffix(text(evs), cutShort) || !strings.HasPrefix(text(evs), "Here is a very long") {
		t.Errorf("max_tokens: %s %q", kinds(evs), text(evs))
	}
	_, evs = h.start(h.request("hack"))
	if data(t, last(evs))["stop"] != "refusal" || text(evs) != refusalReply {
		t.Errorf("refusal: %s %q", kinds(evs), text(evs))
	}
	rows := h.st.usageRows()
	if rows[1].Status != "refusal" {
		t.Errorf("refusal usage row status = %s", rows[1].Status)
	}
	_, evs = h.start(h.request("empty"))
	if data(t, last(evs))["stop"] != "max_tokens" || text(evs) != "" {
		t.Errorf("empty max_tokens: %s %q", kinds(evs), text(evs))
	}
}

func TestTurn_ModelErrors(t *testing.T) {
	h := newHarness(t, nil,
		modeltest.Error(400, "invalid_request_error", "bad", nil),
		modeltest.Error(429, "rate_limit_error", "slow down", nil),
	)
	turn, evs := h.start(h.request("x"))
	if kinds(evs) != "turn.error" || data(t, evs[0])["code"] != "model_error" || !strings.Contains(data(t, evs[0])["message"].(string), "400") {
		t.Errorf("400: %s %v", kinds(evs), data(t, evs[0]))
	}
	if h.st.turn(turn.ID()).Status != store.TurnError {
		t.Error("turn not marked error")
	}
	tr, ok := h.st.trace(turn.ID())
	if !ok || !strings.Contains(string(tr.Ops), `"status":"error"`) {
		t.Errorf("trace = %v %s", ok, tr.Ops)
	}
	_, evs = h.start(h.request("y"))
	if data(t, evs[0])["code"] != "model_unavailable" {
		t.Errorf("429: %v", data(t, evs[0]))
	}
}

func TestTurn_RebuiltFromStorage(t *testing.T) {
	h := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{
			modeltest.ToolUse("lk1", "describe_target", map[string]any{"target": heroTarget}),
			modeltest.ToolUse("op1", "reset", map[string]any{"target": heroTarget, "keys": []string{"headline"}}),
		}, Usage: modeltest.Usage{InputTokens: 7, OutputTokens: 3}}),
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("Reset.")}, Usage: modeltest.Usage{InputTokens: 1, OutputTokens: 1}}),
	)
	turn, evs := h.start(h.request("reset the headline"))
	if kinds(evs) != "tool.call turn.waiting" {
		t.Fatalf("events = %s", kinds(evs))
	}
	// Simulate another instance: drop the in-memory continuation.
	h.runner.mu.Lock()
	delete(h.runner.parked, turn.ID())
	h.runner.mu.Unlock()
	evs2, err := h.cont(turn.ID(), api.ToolResult{ToolCallID: "op1", OK: true, Written: []string{"headline"}})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(kinds(evs2), "turn.done") || text(evs2) != "Reset." {
		t.Fatalf("events = %s", kinds(evs2))
	}
	d := data(t, last(evs2))
	if d["usage"].(map[string]any)["calls"] != 2.0 || d["usage"].(map[string]any)["input_tokens"] != 8.0 || d["seq"].(float64) <= 2 {
		t.Errorf("rebuilt usage/seq = %v", d)
	}
	reqs := h.fake.Requests()
	res := lastToolResults(t, reqs[1])
	if len(res) != 2 || res[0].ID != "lk1" || res[0].IsError || res[1].ID != "op1" || !strings.Contains(res[1].Content, `"written":["headline"]`) {
		t.Errorf("rebuilt results = %+v", res)
	}
	// The rebuilt request replays the stored history: user, assistant(tool_use), user(tool_result).
	msgs := reqs[1].Body["messages"].([]any)
	if len(msgs) != 3 || msgs[1].(map[string]any)["role"] != "assistant" {
		t.Errorf("replayed messages = %d", len(msgs))
	}
	if reqs[0].SystemText() != reqs[1].SystemText() {
		t.Error("prefix changed across the rebuild")
	}
}

func TestTurn_TTLExpiry(t *testing.T) {
	h := newHarness(t, func(d *Deps) { d.TurnTTL = time.Minute },
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.ToolUse("op1", "set_fields", map[string]any{"target": heroTarget, "values": map[string]any{"headline": "x"}})}}),
	)
	turn, _ := h.start(h.request("x"))
	if n, _ := h.runner.Reap(context.Background()); n != 0 {
		t.Errorf("fresh turn reaped")
	}
	h.clock = h.clock.Add(2 * time.Minute)
	n, err := h.runner.Reap(context.Background())
	if err != nil || n != 1 || h.runner.Parked() != 0 || h.st.turn(turn.ID()).Status != store.TurnExpired {
		t.Fatalf("reap = %d %v parked=%d status=%s", n, err, h.runner.Parked(), h.st.turn(turn.ID()).Status)
	}
	tr, ok := h.st.trace(turn.ID())
	if !ok || !strings.Contains(string(tr.Ops), `"status":"expired"`) {
		t.Errorf("trace = %v %s", ok, tr.Ops)
	}
	_, err = h.cont(turn.ID(), api.ToolResult{ToolCallID: "op1", OK: true})
	var apiErr *api.Error
	if !errors.As(err, &apiErr) || apiErr.Code != "turn_not_waiting" || apiErr.Extra["status"] != "expired" {
		t.Errorf("continue after expiry: %v", err)
	}
}

func TestTurn_ConversationHistoryAndRotation(t *testing.T) {
	var steps []modeltest.Step
	for i := 0; i < 4; i++ {
		steps = append(steps, modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("ok")}}))
	}
	h := newHarness(t, func(d *Deps) { d.RotateAt = 4 }, steps...)
	t1, evs1 := h.start(h.request("one"))
	conv1 := data(t, last(evs1))["conversationId"].(string)
	_, evs2 := h.start(h.request("two"))
	if data(t, last(evs2))["conversationId"] != conv1 {
		t.Error("second turn opened a new conversation")
	}
	reqs := h.fake.Requests()
	if n := len(reqs[1].Body["messages"].([]any)); n != 3 {
		t.Errorf("second turn replayed %d messages, want 3", n)
	}
	// 4 messages stored → third turn rotates.
	_, evs3 := h.start(h.request("three"))
	d := data(t, last(evs3))
	if d["conversationId"] == conv1 || d["conversationRotated"] != true {
		t.Errorf("rotation: %v", d)
	}
	if n := len(h.fake.Requests()[2].Body["messages"].([]any)); n != 1 {
		t.Errorf("rotated turn replayed %d messages, want 1", n)
	}
	// Explicit conversationId: wrong site / ended / unknown → 400.
	req := h.request("four")
	req.ConversationID = conv1 // ended by the rotation
	var apiErr *api.Error
	if _, err := h.runner.Start(context.Background(), req); !errors.As(err, &apiErr) || apiErr.Status != 400 {
		t.Errorf("ended conversation: %v", err)
	}
	req.ConversationID = "conv-nope"
	if _, err := h.runner.Start(context.Background(), req); !errors.As(err, &apiErr) || apiErr.Status != 400 {
		t.Errorf("unknown conversation: %v", err)
	}
	req.ConversationID = d["conversationId"].(string)
	req.SiteID = "other-site"
	if _, err := h.runner.Start(context.Background(), req); !errors.As(err, &apiErr) || apiErr.Status != 400 {
		t.Errorf("other site: %v", err)
	}
	_ = t1
}

func TestTurn_OptOutAndNoTraces(t *testing.T) {
	h := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("ok")}}),
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("ok")}}),
	)
	_ = h.st.SetOptOut(context.Background(), "store-1", "owner@shop.com")
	turn, evs := h.start(h.request("x"))
	if !strings.HasSuffix(kinds(evs), "turn.done") {
		t.Fatal(kinds(evs))
	}
	if _, ok := h.st.trace(turn.ID()); ok {
		t.Error("opted-out store was traced")
	}
	if len(h.st.usageRows()) != 1 {
		t.Error("usage must be recorded regardless of the opt-out")
	}
	h2 := newHarness(t, func(d *Deps) { d.NoTraces = true }, modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("ok")}}))
	turn2, _ := h2.start(h2.request("x"))
	if _, ok := h2.st.trace(turn2.ID()); ok {
		t.Error("NoTraces still traced")
	}
}

func TestTurn_ClientGoneKeepsState(t *testing.T) {
	h := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("hello"), modeltest.ToolUse("op1", "set_fields", map[string]any{"target": heroTarget, "values": map[string]any{"headline": "x"}})}}),
	)
	turn, err := h.runner.Start(context.Background(), h.request("x"))
	if err != nil {
		t.Fatal(err)
	}
	em := failingEmitter{}
	if err := turn.Run(context.Background(), em); err != nil {
		t.Errorf("Run returned %v; a gone client is not a turn failure", err)
	}
	if h.st.turn(turn.ID()).Status != store.TurnWaitingTools {
		t.Errorf("status = %s; the turn must still be waiting for the browser", h.st.turn(turn.ID()).Status)
	}
	if h.runner.Live() != 0 {
		t.Error("slot not released")
	}
}

type failingEmitter struct{}

func (failingEmitter) Emit(api.Event) error { return api.ErrClientGone }

func TestAdapters(t *testing.T) {
	h := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("ok")}}),
	)
	ctx := context.Background()
	convs := Conversations{St: h.st}
	_, err := convs.Get(ctx, "site-1", 0)
	var apiErr *api.Error
	if !errors.As(err, &apiErr) || apiErr.Status != 404 || apiErr.Code != "conversation_not_found" {
		t.Errorf("no conversation: %v", err)
	}
	turn, _ := h.start(h.request("hello"))
	conv, err := convs.Get(ctx, "site-1", 0)
	if err != nil || len(conv.Messages) != 2 || conv.Messages[0].Seq != 1 || conv.Messages[0].Role != "user" || len(conv.Turns) != 1 || conv.StoreID != "store-1" {
		t.Errorf("get = %+v %v", conv, err)
	}
	if !strings.Contains(string(conv.Turns[0]), `"status":"done"`) || !strings.Contains(string(conv.Turns[0]), turn.ID()) {
		t.Errorf("turn row = %s", conv.Turns[0])
	}
	conv, _ = convs.Get(ctx, "site-1", 2)
	if len(conv.Messages) != 1 {
		t.Errorf("before=2 → %d messages", len(conv.Messages))
	}
	if err := convs.End(ctx, "site-1"); err != nil {
		t.Fatal(err)
	}
	if _, err := convs.Get(ctx, "site-1", 0); err == nil {
		t.Error("ended conversation still open")
	}
	if err := convs.End(ctx, "site-1"); err != nil {
		t.Errorf("ending nothing must be a no-op: %v", err)
	}

	fb := &Feedback{St: h.st, Logf: t.Logf}
	saved := true
	if err := fb.Record(ctx, api.Feedback{TurnID: turn.ID(), KeptPaths: []string{"a.b"}, Saved: &saved, Note: "nice"}); err != nil {
		t.Fatal(err)
	}
	undone := false
	if err := fb.Record(ctx, api.Feedback{TurnID: turn.ID(), Undone: &undone}); err != nil {
		t.Fatal(err)
	}
	tr, _ := h.st.trace(turn.ID())
	if s := string(tr.Feedback); !strings.Contains(s, `"kept_paths":["a.b"]`) || !strings.Contains(s, `"saved":true`) || !strings.Contains(s, `"undone":false`) || !strings.Contains(s, `"note":"nice"`) {
		t.Errorf("feedback = %s", s)
	}
	if err := fb.Record(ctx, api.Feedback{TurnID: "turn-missing", Note: "x"}); err != nil {
		t.Errorf("feedback for an untraced turn must be a no-op: %v", err)
	}
	if err := fb.Record(ctx, api.Feedback{TurnID: turn.ID()}); err != nil {
		t.Errorf("empty feedback: %v", err)
	}
	// Oversize feedback is a 400 before touching the store; a store cap
	// error maps to 400 too.
	if err := fb.Record(ctx, api.Feedback{TurnID: turn.ID(), Note: strings.Repeat("n", store.MaxFeedbackBytes)}); !errors.As(err, &apiErr) || apiErr.Status != 400 {
		t.Errorf("oversize note: %v", err)
	}
	if err := (&Feedback{St: cappedFeedbackStore{}}).Record(ctx, api.Feedback{TurnID: turn.ID(), Note: "x"}); !errors.As(err, &apiErr) || apiErr.Status != 400 {
		t.Errorf("store cap error not mapped to 400: %v", err)
	}

	oo := OptOut{St: h.st}
	if err := oo.Set(ctx, "", "x"); !errors.As(err, &apiErr) || apiErr.Status != 400 {
		t.Errorf("empty store id: %v", err)
	}
	if err := oo.Set(ctx, "store-9", "owner@shop.com"); err != nil {
		t.Fatal(err)
	}
	if out, _ := h.st.IsOptedOut(ctx, "store-9"); !out {
		t.Error("not set")
	}
	if err := oo.Clear(ctx, "store-9"); err != nil {
		t.Fatal(err)
	}
	if out, _ := h.st.IsOptedOut(ctx, "store-9"); out {
		t.Error("not cleared")
	}
	if err := oo.Clear(ctx, "store-9"); err != nil {
		t.Errorf("clearing twice: %v", err)
	}
}

func TestPendingDoc_RoundTrip(t *testing.T) {
	pd := pendingDoc{Route: "default", CallCount: 2, Seq: 5, SuspendedID: "pa1",
		Calls: []pendingCall{
			{ID: "lk1", Name: "lookup_fields", Input: json.RawMessage(`{}`), Server: true},
			{ID: "op1", Name: "set_fields", Input: json.RawMessage(`{"values":{}}`), Seq: 1},
			{ID: "pa1", Name: "pick_asset", Input: json.RawMessage(`{"purpose":"Hero","asset_type":"image"}`), Seq: 2},
		},
		ServerResults: []serverResult{{ID: "lk1", Content: `{"fields":[]}`}},
	}
	raw, _ := json.Marshal(pd)
	var back pendingDoc
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatal(err)
	}
	ids, _ := json.Marshal(back.browserIDs())
	if string(ids) != `["op1","pa1"]` {
		t.Errorf("browserIDs = %s", ids)
	}
	if _, ok := back.browser("lk1"); ok {
		t.Error("server call reported as browser")
	}
	if sr, ok := back.serverResult("lk1"); !ok || sr.Content != `{"fields":[]}` {
		t.Errorf("serverResult = %+v %v", sr, ok)
	}
	ev := back.suspendedEvent()
	if ev["purpose"] != "Hero" || ev["toolCallId"] != "pa1" || ev["assetType"] != "image" {
		t.Errorf("suspended = %v", ev)
	}
	if more, _ := json.Marshal(ev["toolCallIds"]); string(more) != `["op1","pa1"]` {
		t.Errorf("toolCallIds = %s", more)
	}
}

// cappedFeedbackStore always reports the merged-size cap.
type cappedFeedbackStore struct{ FeedbackStore }

func (cappedFeedbackStore) MergeFeedback(context.Context, string, json.RawMessage) error {
	return store.ErrFeedbackTooLarge
}
