package trace

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/breadstack/sous-chef/internal/store"
)

// fakeStore is an in-memory Store for the pure tests.
type fakeStore struct {
	mu      sync.Mutex
	optout  map[string]bool
	lookErr error
	insErr  error
	traces  []store.Trace
}

func (f *fakeStore) IsOptedOut(_ context.Context, storeID string) (bool, error) {
	if f.lookErr != nil {
		return false, f.lookErr
	}
	return f.optout[storeID], nil
}

func (f *fakeStore) InsertTrace(_ context.Context, t store.Trace) error {
	if f.insErr != nil {
		return f.insErr
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.traces = append(f.traces, t)
	return nil
}

var fixed = time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)

func begin() Begin {
	return Begin{TurnID: "t-1", SiteID: "site-1", StoreID: "store-1", ManifestHash: "abc", BuildID: "v3.0.91",
		Route: "default", UserEmail: "m@x.test", Text: "make the hero bigger",
		Context: json.RawMessage(`{"page":"home"}`), StartedAt: fixed}
}

func TestNilRecorderIsNoOp(t *testing.T) {
	var r *Recorder
	if r.Enabled() {
		t.Fatal("nil recorder enabled")
	}
	r.ModelCall(ModelCall{}, nil)
	r.MarkServer("x")
	r.Event(1, "text.delta", nil)
	r.ToolResult(1, nil)
	r.SetClock(nil)
	if err := r.Finish(context.Background(), "done", "end_turn", ""); err != nil {
		t.Fatal(err)
	}
	if ops, evs := r.Snapshot(); ops != nil || evs != nil {
		t.Fatal("nil snapshot must be nil")
	}
}

func TestNewHonoursOptOut(t *testing.T) {
	fs := &fakeStore{optout: map[string]bool{"store-1": true}}
	if r := New(context.Background(), fs, begin()); r != nil {
		t.Fatal("opted-out store got a recorder")
	}
	fs = &fakeStore{}
	if r := New(context.Background(), fs, begin()); r == nil {
		t.Fatal("non-opted-out store got nil")
	}
	if r := New(context.Background(), nil, begin()); r != nil {
		t.Fatal("nil store must give nil recorder")
	}
	// A failed lookup FAILS CLOSED: no recorder, nothing written (S3).
	fs = &fakeStore{lookErr: errors.New("db down")}
	r := New(context.Background(), fs, begin())
	if r != nil {
		t.Fatal("lookup error must disable tracing (fail closed)")
	}
	r.ModelCall(ModelCall{Model: "m"}, json.RawMessage(`[{"type":"text","text":"x"}]`))
	if err := r.Finish(context.Background(), "done", "end_turn", ""); err != nil {
		t.Fatal(err)
	}
	if len(fs.traces) != 0 {
		t.Fatalf("trace written despite failed opt-out lookup: %d", len(fs.traces))
	}
}

func TestRecorderFullTurn(t *testing.T) {
	fs := &fakeStore{}
	r := New(context.Background(), fs, begin())
	r.SetClock(func() time.Time { return fixed.Add(time.Second) })

	// Call 1: thinking + text + two tool calls (one server-side).
	r.ModelCall(ModelCall{Model: "claude-sonnet-5-5", StopReason: "tool_use", LatencyMS: 800, RequestID: "msg_1",
		Usage: Usage{InputTokens: 10, OutputTokens: 5, CacheReadInputTokens: 3000}},
		json.RawMessage(`[
			{"type":"thinking","thinking":"SECRET PLAN","signature":"sig"},
			{"type":"text","text":"Let me look that up."},
			{"type":"tool_use","id":"tu_1","name":"lookup_fields","input":{"query":"hero title"}},
			{"type":"tool_use","id":"tu_2","name":"set_fields","input":{"target":{"kind":"section","page":"home","sectionId":"hero-1"},"values":{"titleSize":"xl"}}}
		]`))
	r.MarkServer("tu_1")
	r.Event(1, "text.delta", json.RawMessage(`{"text":"Let me "}`))
	r.Event(2, "text.delta", json.RawMessage(`{"text":"look that up."}`))
	r.Event(3, "tool.call", json.RawMessage(`{"id":"tu_2","name":"set_fields","input":{}}`))
	r.Event(4, "turn.waiting", json.RawMessage(`{"toolCallIds":["tu_2"]}`))
	r.ToolResult(3, json.RawMessage(`{"toolCallId":"tu_2","ok":true,"written":["settings.titleSize"]}`))
	// Call 2: text only, end_turn.
	r.ModelCall(ModelCall{Model: "claude-sonnet-5-5", StopReason: "end_turn", LatencyMS: 400,
		Usage: Usage{InputTokens: 2, OutputTokens: 7, CacheReadInputTokens: 3100}},
		json.RawMessage(`[{"type":"text","text":"Done — the hero title is now extra large."}]`))
	r.Event(5, "text.delta", json.RawMessage(`{"text":"Done — the hero title is now extra large."}`))
	r.Event(6, "turn.done", json.RawMessage(`{"stop":"end_turn"}`))

	if err := r.Finish(context.Background(), "done", "end_turn", ""); err != nil {
		t.Fatal(err)
	}
	if err := r.Finish(context.Background(), "done", "end_turn", ""); err != nil {
		t.Fatal("second Finish must be a no-op")
	}
	if len(fs.traces) != 1 {
		t.Fatalf("traces written = %d", len(fs.traces))
	}
	tr := fs.traces[0]
	if tr.TurnID != "t-1" || tr.SiteID != "site-1" || tr.StoreID != "store-1" || !tr.Created.Equal(fixed) {
		t.Errorf("trace header = %+v", tr)
	}
	if string(tr.Context) != `{"page":"home"}` {
		t.Errorf("context = %s", tr.Context)
	}
	if strings.Contains(string(tr.Ops), "SECRET PLAN") || strings.Contains(string(tr.Events), "SECRET PLAN") {
		t.Fatal("thinking text leaked into the trace")
	}
	var ops opsDoc
	if err := json.Unmarshal(tr.Ops, &ops); err != nil {
		t.Fatal(err)
	}
	if ops.ManifestHash != "abc" || ops.Route != "default" || ops.Text != "make the hero bigger" || ops.UserEmail != "m@x.test" {
		t.Errorf("ops header = %+v", ops)
	}
	if len(ops.Calls) != 2 || ops.Calls[0].Call != 1 || ops.Calls[1].Call != 2 {
		t.Fatalf("calls = %+v", ops.Calls)
	}
	if ops.Calls[0].Text != "Let me look that up." || ops.Calls[0].StopReason != "tool_use" || ops.Calls[0].RequestID != "msg_1" {
		t.Errorf("call 1 = %+v", ops.Calls[0])
	}
	if len(ops.Ops) != 2 || ops.Ops[0].ToolUseID != "tu_1" || !ops.Ops[0].Server || ops.Ops[1].Server || ops.Ops[1].Name != "set_fields" || ops.Ops[1].Call != 1 {
		t.Errorf("ops = %+v", ops.Ops)
	}
	if !strings.Contains(string(ops.Ops[1].Input), `"titleSize":"xl"`) {
		t.Errorf("op input not raw: %s", ops.Ops[1].Input)
	}
	if ops.Outcome.Status != "done" || ops.Outcome.Stop != "end_turn" || ops.Outcome.Calls != 2 ||
		ops.Outcome.Usage.InputTokens != 12 || ops.Outcome.Usage.OutputTokens != 12 || ops.Outcome.Usage.CacheReadInputTokens != 6100 {
		t.Errorf("outcome = %+v", ops.Outcome)
	}
	if !ops.EndedAt.Equal(fixed.Add(time.Second)) || !ops.StartedAt.Equal(fixed) {
		t.Errorf("timestamps = %v / %v", ops.StartedAt, ops.EndedAt)
	}
	// Events: deltas 1+2 coalesced, then tool.call, waiting, tool.result, delta, done = 6.
	var evs []Event
	if err := json.Unmarshal(tr.Events, &evs); err != nil {
		t.Fatal(err)
	}
	kinds := []string{}
	for _, e := range evs {
		kinds = append(kinds, e.Kind)
	}
	want := "text.delta tool.call turn.waiting tool.result text.delta turn.done"
	if got := strings.Join(kinds, " "); got != want {
		t.Fatalf("event kinds = %q, want %q", got, want)
	}
	if string(evs[0].Data) != `{"text":"Let me look that up."}` || evs[0].Seq != 1 {
		t.Errorf("coalesced delta = %+v", evs[0])
	}
	if evs[3].ForSeq != 3 || !strings.Contains(string(evs[3].Data), `"ok":true`) {
		t.Errorf("tool.result = %+v", evs[3])
	}
	if strings.Contains(string(tr.Ops), `<`) {
		t.Error("HTML escaping must be off")
	}
}

func TestFinishErrorsAndInsertFailure(t *testing.T) {
	fs := &fakeStore{insErr: errors.New("pg down")}
	r := New(context.Background(), fs, begin())
	r.ModelCall(ModelCall{StopReason: "end_turn"}, json.RawMessage(`[{"type":"text","text":"x"}]`))
	if err := r.Finish(context.Background(), "error", "", "model_unavailable"); err == nil {
		t.Fatal("insert failure must be returned (and logged), never hidden")
	}
	ops, _ := r.Snapshot()
	if !strings.Contains(string(ops), `"status":"error"`) || !strings.Contains(string(ops), `"error":"model_unavailable"`) {
		t.Errorf("outcome not recorded: %s", ops)
	}
}

func TestModelCallDefaults(t *testing.T) {
	r := New(context.Background(), &fakeStore{}, begin())
	// Explicit text wins over block text; missing input becomes {}; string
	// content (no blocks) records nothing but the call.
	r.ModelCall(ModelCall{Text: "override"}, json.RawMessage(`[{"type":"text","text":"ignored"},{"type":"tool_use","id":"tu","name":"reset"}]`))
	r.ModelCall(ModelCall{}, json.RawMessage(`"plain string"`))
	ops, _ := r.Snapshot()
	var doc opsDoc
	_ = json.Unmarshal(ops, &doc)
	if len(doc.Calls) != 2 || doc.Calls[0].Text != "override" || doc.Calls[1].Call != 2 || doc.Calls[1].Text != "" {
		t.Errorf("calls = %+v", doc.Calls)
	}
	if len(doc.Ops) != 1 || string(doc.Ops[0].Input) != `{}` {
		t.Errorf("ops = %+v", doc.Ops)
	}
}

func TestConcurrentEvents(t *testing.T) {
	r := New(context.Background(), &fakeStore{}, begin())
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			r.Event(i, "tool.call", json.RawMessage(`{}`))
			r.ModelCall(ModelCall{}, nil)
		}(i)
	}
	wg.Wait()
	_, evs := r.Snapshot()
	var list []Event
	_ = json.Unmarshal(evs, &list)
	if len(list) != 50 {
		t.Errorf("events = %d", len(list))
	}
}

func TestStripThinking(t *testing.T) {
	in := json.RawMessage(`[{"type":"thinking","thinking":"hmm","signature":"s"},{"type":"text","text":"a"},{"type":"redacted_thinking","data":"x"},{"type":"tool_use","id":"1","name":"n","input":{"b":1,"a":2}},{"type":"future_block","foo":"bar"}]`)
	out := StripThinking(in)
	want := `[{"type":"text","text":"a"},{"type":"tool_use","id":"1","name":"n","input":{"b":1,"a":2}},{"type":"future_block","foo":"bar"}]`
	if string(out) != want {
		t.Errorf("StripThinking =\n%s\nwant\n%s", out, want)
	}
	if !HasThinking(in) || HasThinking(out) {
		t.Error("HasThinking wrong")
	}
	// Untouched when nothing to strip (same bytes, not re-encoded).
	plain := json.RawMessage(`[ {"type":"text", "text":"spaced"} ]`)
	if string(StripThinking(plain)) != string(plain) {
		t.Error("content without thinking must be returned byte-for-byte")
	}
	// Non-array content passes through.
	str := json.RawMessage(`"hello"`)
	if string(StripThinking(str)) != `"hello"` || StripThinking(nil) != nil {
		t.Error("non-array content must pass through")
	}
	// Everything stripped → empty array.
	if got := StripThinking(json.RawMessage(`[{"type":"thinking","thinking":"x"}]`)); string(got) != `[]` {
		t.Errorf("all-thinking → %s", got)
	}
}

func TestMergeTextDelta(t *testing.T) {
	out, ok := mergeTextDelta(json.RawMessage(`{"text":"a<b"}`), json.RawMessage(`{"text":"&c"}`))
	var got struct {
		Text string `json:"text"`
	}
	if !ok || json.Unmarshal(out, &got) != nil || got.Text != "a<b&c" {
		t.Errorf("merge = %s %v", out, ok)
	}
	if _, ok := mergeTextDelta(json.RawMessage(`nope`), json.RawMessage(`{}`)); ok {
		t.Error("invalid JSON merged")
	}
}
