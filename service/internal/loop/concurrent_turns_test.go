package loop

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/breadstack/sous-chef/internal/api"
	"github.com/breadstack/sous-chef/internal/model/modeltest"
	"github.com/breadstack/sous-chef/internal/store"
)

// Round-4 regressions (refuter N4, N2-r, B1, site check): two turns of the
// same conversation, the reaper, and tenant assertions on continuations.

// N4 (refuter B2): an older turn of the SAME conversation expires while a
// younger, healthy turn is waiting_tools. The older turn was running (no
// pending doc: the instance died mid-call); the reaper must not treat the
// conversation's tail — the younger turn's batch — as the expired turn's
// and answer it is_error, or the browser's real results hit the CAS and
// the healthy turn dies. Start supersedes running turns now (site-wide), so
// the stale row is built directly, after the young turn parked — a
// mixed-version fleet, or an instance that died before it could supersede.
func TestReaper_LeavesYoungerTurnTailRunning(t *testing.T) {
	h := newHarness(t, func(d *Deps) { d.TurnTTL = 10 * time.Minute },
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{opUse("op2", "headline", "x")}}),
		textReply("done"),
		textReply("next"),
	)
	ctx := context.Background()
	t2, evs := h.start(h.request("retry"))
	if !strings.HasSuffix(kinds(evs), "turn.waiting") {
		t.Fatalf("turn 2: %s", kinds(evs))
	}
	conv, _ := h.st.GetConversation(ctx, h.st.turn(t2.ID()).ConversationID)
	// Turn 1: an instance died 9 minutes earlier after creating the row (row
	// running, no assistant message, no pending doc).
	h.clock = h.clock.Add(-9 * time.Minute)
	row1, _ := h.st.CreateTurn(ctx, store.NewTurn{ConversationID: conv.ID, ManifestHash: h.hash, Model: "m"})
	h.clock = h.clock.Add(9 * time.Minute)
	before := roles(t, h, conv.ID)
	// 2 minutes later (turn 1 at 11 min, turn 2 at 2 min) the reaper runs.
	h.clock = h.clock.Add(2 * time.Minute)
	n, err := h.runner.Reap(ctx)
	if err != nil || n != 1 || h.st.turn(row1.ID).Status != store.TurnExpired || h.st.turn(t2.ID()).Status != store.TurnWaitingTools {
		t.Fatalf("reap=%d %v row1=%s row2=%s", n, err, h.st.turn(row1.ID).Status, h.st.turn(t2.ID()).Status)
	}
	if after := roles(t, h, conv.ID); strings.Join(after, " ") != strings.Join(before, " ") {
		_, res := storedTail(t, h, conv.ID)
		t.Fatalf("reaping turn 1 answered turn 2's batch: roles %v → %v, tail results %+v", before, after, res)
	}
	// The browser answers op2 on the healthy turn.
	evs2, err := h.cont(t2.ID(), api.ToolResult{ToolCallID: "op2", OK: true})
	if err != nil {
		t.Fatalf("healthy turn 2's continuation refused: %v (row2=%s)", err, h.st.turn(t2.ID()).Status)
	}
	if !strings.HasSuffix(kinds(evs2), "turn.done") || h.st.turn(t2.ID()).Status != store.TurnDone {
		t.Errorf("turn 2 continuation: %s row2=%s", kinds(evs2), h.st.turn(t2.ID()).Status)
	}
	reqs := h.fake.Requests()
	assertReplayable(t, reqs[len(reqs)-1])
	if res := lastToolResults(t, reqs[len(reqs)-1]); len(res) != 1 || res[0].ID != "op2" || res[0].IsError {
		t.Errorf("model saw %+v", res)
	}
	h.start(h.request("y"))
	reqs = h.fake.Requests()
	assertReplayable(t, reqs[len(reqs)-1])
}

// N4 (refuter B2b): the expired turn WAS waiting (pending doc with pa1), but
// a later turn of the conversation owns the tail now (op3). The reaper may
// answer only ids of its own pending doc; a tail sharing none of them is
// left alone. Start supersedes waiting turns now (B1), so this row layout is
// built directly — a mixed-version fleet or a turn parked by an instance
// that never got to supersede must not be able to kill a healthy turn.
func TestReaper_LeavesYoungerTurnTailPendingDoc(t *testing.T) {
	h := newHarness(t, func(d *Deps) { d.TurnTTL = 10 * time.Minute },
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.ToolUse("pa1", "pick_asset", map[string]any{"purpose": "bg"})}}),
		textReply("done"),
		textReply("next"),
	)
	ctx := context.Background()
	t1, evs := h.start(h.request("pick"))
	if kinds(evs) != "turn.suspended" {
		t.Fatalf("turn 1: %s", kinds(evs))
	}
	convID := h.st.turn(t1.ID()).ConversationID
	// Drop the parked entry: the reaper works from the stored pending doc.
	h.runner.mu.Lock()
	delete(h.runner.parked, t1.ID())
	h.runner.mu.Unlock()
	// 9 minutes later a newer turn (written straight to the store) answered
	// pa1 and went waiting on op3: [u a(pa1) u(pa1 repair) u a(op3)].
	h.clock = h.clock.Add(9 * time.Minute)
	for _, m := range []struct{ role, content string }{
		{"user", `[{"type":"tool_result","tool_use_id":"pa1","is_error":true,"content":"not executed"}]`},
		{"user", `[{"type":"text","text":"<context>\n{}\n</context>\nthird"}]`},
		{"assistant", `[{"type":"tool_use","id":"op3","name":"set_fields","input":{"target":{"kind":"section","page":"home","sectionId":"hero-1"},"values":{"headline":"x"}}}]`},
	} {
		if _, err := h.st.AppendMessage(ctx, convID, m.role, json.RawMessage(m.content)); err != nil {
			t.Fatal(err)
		}
	}
	row3, _ := h.st.CreateTurn(ctx, store.NewTurn{ConversationID: convID, ManifestHash: h.hash, Model: "m"})
	pd3, _ := json.Marshal(pendingDoc{Route: "default", CallCount: 1, Seq: 2, Calls: []pendingCall{{ID: "op3", Name: "set_fields", Input: json.RawMessage(`{}`), Seq: 1}}})
	if err := h.st.SetTurnWaiting(ctx, row3.ID, store.TurnWaitingTools, pd3); err != nil {
		t.Fatal(err)
	}
	before := roles(t, h, convID)
	h.clock = h.clock.Add(2 * time.Minute) // turn 1 at 11 min, turn 3 at 2 min
	n, err := h.runner.Reap(ctx)
	if err != nil || n != 1 || h.st.turn(t1.ID()).Status != store.TurnExpired || h.st.turn(row3.ID).Status != store.TurnWaitingTools {
		t.Fatalf("reap=%d %v row1=%s row3=%s", n, err, h.st.turn(t1.ID()).Status, h.st.turn(row3.ID).Status)
	}
	if after := roles(t, h, convID); strings.Join(after, " ") != strings.Join(before, " ") {
		_, res := storedTail(t, h, convID)
		t.Fatalf("reaping turn 1 (pa1) answered turn 3's op3: roles %v → %v, tail %+v", before, after, res)
	}
	evs3, err := h.cont(row3.ID, api.ToolResult{ToolCallID: "op3", OK: true})
	if err != nil {
		t.Fatalf("healthy turn 3 refused: %v (row3=%s)", err, h.st.turn(row3.ID).Status)
	}
	if !strings.HasSuffix(kinds(evs3), "turn.done") {
		t.Errorf("turn 3 continuation: %s", kinds(evs3))
	}
	reqs := h.fake.Requests()
	assertReplayable(t, reqs[len(reqs)-1])
	if res := lastToolResults(t, reqs[len(reqs)-1]); len(res) != 1 || res[0].ID != "op3" || res[0].IsError {
		t.Errorf("model saw %+v", res)
	}
}

// N2-r (refuter A): the reaper expires the row right after an all-server
// batch's results were appended — the tail is clean, so the reaper appends
// nothing — and the loop must still notice: no further model call, no
// turn.done on an expired row.
func TestRun_StopsWhenRowExpiredBetweenCalls(t *testing.T) {
	h := newHarness(t, func(d *Deps) { d.TurnTTL = time.Minute },
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{
			modeltest.ToolUse("lk1", "lookup_fields", map[string]any{"target": heroTarget, "query": "headline"}),
		}}),
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{
			modeltest.ToolUse("lk2", "lookup_fields", map[string]any{"target": heroTarget, "query": "subtitle"}),
		}}),
		textReply("done"),
		textReply("next"),
	)
	ctx := context.Background()
	turn, err := h.runner.Start(ctx, h.request("x"))
	if err != nil {
		t.Fatal(err)
	}
	userAppends, fired := 0, false
	h.st.mu.Lock()
	h.st.afterAppend = func(role string) {
		if role != "user" {
			return
		}
		userAppends++
		if userAppends != 1 || fired { // the hook is installed after Start: #1 is lk1's results
			return
		}
		fired = true
		h.clock = h.clock.Add(2 * time.Minute)
		if n, err := h.runner.Reap(ctx); n != 1 || err != nil {
			t.Errorf("reap in hook = %d %v", n, err)
		}
	}
	h.st.mu.Unlock()
	col := &api.Collector{}
	_ = turn.Run(ctx, col)
	h.st.mu.Lock()
	h.st.afterAppend = nil
	h.st.mu.Unlock()
	if !fired {
		t.Fatal("hook never fired")
	}
	row := h.st.turn(turn.ID())
	if last := data(t, last(col.Events())); last["code"] != "turn_expired" || !strings.HasSuffix(kinds(col.Events()), "turn.error") {
		t.Errorf("events = %s (%v): an expired row must end turn_expired, not turn.done", kinds(col.Events()), last)
	}
	if n := len(h.fake.Requests()); n != 1 {
		t.Errorf("model called %d times; the loop must make no call after expiry", n)
	}
	if row.Status != store.TurnExpired || h.runner.Live() != 0 || h.runner.Parked() != 0 {
		t.Errorf("row=%s live=%d parked=%d", row.Status, h.runner.Live(), h.runner.Parked())
	}
	if tr, ok := h.st.trace(turn.ID()); !ok || !strings.Contains(string(tr.Ops), `"status":"expired"`) {
		t.Errorf("trace = %v %s (the trace carries the row's terminal status)", ok, tr.Ops)
	}
	if got := roles(t, h, row.ConversationID); strings.Join(got, " ") != "user assistant user" {
		t.Errorf("stored roles = %v", got)
	}
	h.start(h.request("y"))
	reqs := h.fake.Requests()
	assertReplayable(t, reqs[len(reqs)-1])
}

// B1: a Start while a previous turn of the same conversation is suspended
// (pick_asset, the user is browsing assets, then types instead) supersedes
// that turn: its row ends superseded, its parked entry and trace are
// closed, its batch is answered in storage with the superseded reason, and
// its late Continue is 409 turn_not_waiting status=superseded with nothing
// appended.
func TestStart_SupersedesSuspendedTurn(t *testing.T) {
	h := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{
			modeltest.ToolUse("pa1", "pick_asset", map[string]any{"purpose": "bg"}),
		}}),
		textReply("second turn reply"),
		textReply("third"),
	)
	t1, evs := h.start(h.request("pick an image"))
	if kinds(evs) != "turn.suspended" || h.runner.Parked() != 1 {
		t.Fatalf("events=%s parked=%d", kinds(evs), h.runner.Parked())
	}
	convID := h.st.turn(t1.ID()).ConversationID
	var warned bool
	h.runner.d.Logf = func(f string, a ...any) {
		if strings.Contains(f, "WARNING") {
			warned = true
		}
		t.Logf(f, a...)
	}
	t2, evs2 := h.start(h.request("actually change the headline"))
	if !strings.HasSuffix(kinds(evs2), "turn.done") {
		t.Fatalf("second turn: %s", kinds(evs2))
	}
	assertReplayable(t, h.fake.Requests()[1])
	if warned {
		t.Error("a superseded tail is the expected outcome, not a repair WARNING")
	}
	row1 := h.st.turn(t1.ID())
	if row1.Status != store.TurnSuperseded || row1.Ended == nil || row1.PendingToolCalls != nil || h.runner.Parked() != 0 {
		t.Errorf("row1=%s ended=%v pending=%s parked=%d", row1.Status, row1.Ended != nil, row1.PendingToolCalls, h.runner.Parked())
	}
	if tr, ok := h.st.trace(t1.ID()); !ok || !strings.Contains(string(tr.Ops), `"status":"superseded"`) {
		t.Errorf("turn 1 trace = %v %s", ok, tr.Ops)
	}
	// Stored: user, assistant(pa1), user(pa1 superseded), user(text), assistant.
	msgs, _ := h.st.ListMessages(context.Background(), convID, 0, 0)
	if got := roles(t, h, convID); strings.Join(got, " ") != "user assistant user user assistant" {
		t.Fatalf("stored roles = %v", got)
	}
	if !strings.Contains(string(msgs[2].Content), `"tool_use_id":"pa1"`) || !strings.Contains(string(msgs[2].Content), reasonSuperseded) {
		t.Errorf("tail repair = %s", msgs[2].Content)
	}
	before := roles(t, h, convID)
	// The user now picks the asset: too late.
	_, err := h.cont(t1.ID(), api.ToolResult{ToolCallID: "pa1", OK: true, Asset: &api.Asset{PublicURL: "https://cdn/a.jpg"}})
	var apiErr *api.Error
	if !errors.As(err, &apiErr) || apiErr.Code != "turn_not_waiting" || apiErr.Status != 409 || apiErr.Extra["status"] != store.TurnSuperseded {
		t.Fatalf("late pick = %v", err)
	}
	if after := roles(t, h, convID); strings.Join(after, " ") != strings.Join(before, " ") {
		t.Errorf("late pick appended: %v → %v", before, after)
	}
	if h.runner.Parked() != 0 || h.runner.Live() != 0 {
		t.Errorf("parked=%d live=%d", h.runner.Parked(), h.runner.Live())
	}
	if h.st.turn(t2.ID()).Status != store.TurnDone {
		t.Errorf("row2 = %s", h.st.turn(t2.ID()).Status)
	}
	h.start(h.request("z"))
	reqs := h.fake.Requests()
	assertReplayable(t, reqs[len(reqs)-1])
	if n := len(reqs[len(reqs)-1].Body["messages"].([]any)); n != 6 {
		t.Errorf("third request replayed %d messages, want 6", n)
	}
}

// B1: superseding a waiting_tools turn keeps the server results it had
// stashed (the work was done) and answers only the browser ids with the
// superseded reason; the superseded turn's Continue is 409 status=superseded
// also when its parked entry was dropped (another instance).
func TestStart_SupersedesWaitingTurnKeepsServerResults(t *testing.T) {
	h := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{
			modeltest.ToolUse("lk1", "lookup_fields", map[string]any{"target": heroTarget, "query": "headline"}),
			opUse("op1", "headline", "x"),
		}}),
		textReply("second"),
	)
	t1, evs := h.start(h.request("x"))
	if !strings.HasSuffix(kinds(evs), "turn.waiting") {
		t.Fatalf("turn 1: %s", kinds(evs))
	}
	convID := h.st.turn(t1.ID()).ConversationID
	h.runner.mu.Lock()
	delete(h.runner.parked, t1.ID())
	h.runner.mu.Unlock()
	_, evs2 := h.start(h.request("y"))
	if !strings.HasSuffix(kinds(evs2), "turn.done") {
		t.Fatalf("turn 2: %s", kinds(evs2))
	}
	assertReplayable(t, h.fake.Requests()[1])
	msgs, _ := h.st.ListMessages(context.Background(), convID, 0, 0)
	if got := roles(t, h, convID); strings.Join(got, " ") != "user assistant user user assistant" {
		t.Fatalf("stored roles = %v", got)
	}
	var blocks []map[string]any
	_ = json.Unmarshal(msgs[2].Content, &blocks)
	if len(blocks) != 2 || blocks[0]["tool_use_id"] != "lk1" || blocks[0]["is_error"] == true || blocks[1]["tool_use_id"] != "op1" || blocks[1]["is_error"] != true {
		t.Errorf("superseded repair = %s", msgs[2].Content)
	}
	if !strings.Contains(string(msgs[2].Content), "headline") || !strings.Contains(string(msgs[2].Content), reasonSuperseded) {
		t.Errorf("superseded repair lost the lookup result or the reason: %s", msgs[2].Content)
	}
	_, err := h.cont(t1.ID(), api.ToolResult{ToolCallID: "op1", OK: true})
	var apiErr *api.Error
	if !errors.As(err, &apiErr) || apiErr.Code != "turn_not_waiting" || apiErr.Extra["status"] != store.TurnSuperseded {
		t.Fatalf("late Continue = %v", err)
	}
	if h.st.turn(t1.ID()).Status != store.TurnSuperseded {
		t.Errorf("row1 = %s", h.st.turn(t1.ID()).Status)
	}
}

// Site check: a tool-results body naming a siteId (and storeId) must match
// the turn's conversation; a mismatch is 404 turn_not_found (not 409 — the
// turn's existence is not revealed), the turn stays waiting and parked, and
// a body without siteId is accepted and logged once per turn.
func TestContinue_VerifiesSiteAndStore(t *testing.T) {
	h := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{opUse("op1", "headline", "x")}}),
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{opUse("op2", "subtitle", "y")}}),
		textReply("done"),
	)
	ctx := context.Background()
	turn, evs := h.start(h.request("x"))
	if !strings.HasSuffix(kinds(evs), "turn.waiting") {
		t.Fatalf("events = %s", kinds(evs))
	}
	var apiErr *api.Error
	for name, body := range map[string]api.ToolResultsRequest{
		"wrong site":  {SiteID: "site-2", Results: []api.ToolResult{{ToolCallID: "op1", OK: true}}},
		"wrong store": {SiteID: "site-1", StoreID: "store-2", Results: []api.ToolResult{{ToolCallID: "op1", OK: true}}},
	} {
		_, err := h.runner.Continue(ctx, turn.ID(), body)
		if !errors.As(err, &apiErr) || apiErr.Code != "turn_not_found" || apiErr.Status != 404 {
			t.Errorf("%s: Continue = %v, want 404 turn_not_found", name, err)
		}
		if h.st.turn(turn.ID()).Status != store.TurnWaitingTools || h.runner.Parked() != 1 || h.runner.Live() != 0 {
			t.Errorf("%s: row=%s parked=%d live=%d", name, h.st.turn(turn.ID()).Status, h.runner.Parked(), h.runner.Live())
		}
	}
	// No siteId: accepted, logged once for this turn across two continuations.
	noSite := 0
	h.runner.d.Logf = func(f string, a ...any) {
		if strings.Contains(f, "no siteId") {
			noSite++
		}
		t.Logf(f, a...)
	}
	evs, err := h.cont(turn.ID(), api.ToolResult{ToolCallID: "op1", OK: true})
	if err != nil || !strings.HasSuffix(kinds(evs), "turn.waiting") {
		t.Fatalf("unverified Continue = %v %s", err, kinds(evs))
	}
	// The right site and store: accepted.
	t2, err := h.runner.Continue(ctx, turn.ID(), api.ToolResultsRequest{SiteID: "site-1", StoreID: "store-1", Results: []api.ToolResult{{ToolCallID: "op2", OK: true}}})
	if err != nil {
		t.Fatalf("matching Continue = %v", err)
	}
	col := &api.Collector{}
	_ = t2.Run(ctx, col)
	if !strings.HasSuffix(kinds(col.Events()), "turn.done") {
		t.Errorf("events = %s", kinds(col.Events()))
	}
	if noSite != 1 {
		t.Errorf("siteId-less body logged %d times, want once per turn", noSite)
	}
	// A siteId on an unknown turn is the same 404.
	_, err = h.runner.Continue(ctx, "turn-missing", api.ToolResultsRequest{SiteID: "site-1", Results: []api.ToolResult{{ToolCallID: "op1", OK: true}}})
	if !errors.As(err, &apiErr) || apiErr.Code != "turn_not_found" {
		t.Errorf("unknown turn = %v", err)
	}
}

// Site check on feedback: siteId/storeId must match the turn's conversation
// (404 turn_not_found otherwise, also for an unknown turn); no siteId is
// accepted as before.
func TestFeedback_VerifiesSiteAndStore(t *testing.T) {
	h := newHarness(t, nil, textReply("ok"))
	ctx := context.Background()
	turn, _ := h.start(h.request("hello"))
	fb := &Feedback{St: h.st, Logf: t.Logf}
	saved := true
	var apiErr *api.Error
	for name, body := range map[string]api.Feedback{
		"wrong site":   {TurnID: turn.ID(), SiteID: "site-2", Saved: &saved},
		"wrong store":  {TurnID: turn.ID(), SiteID: "site-1", StoreID: "store-2", Saved: &saved},
		"unknown turn": {TurnID: "turn-missing", SiteID: "site-1", Saved: &saved},
	} {
		if err := fb.Record(ctx, body); !errors.As(err, &apiErr) || apiErr.Code != "turn_not_found" || apiErr.Status != 404 {
			t.Errorf("%s: Record = %v, want 404 turn_not_found", name, err)
		}
	}
	if tr, _ := h.st.trace(turn.ID()); len(tr.Feedback) != 0 {
		t.Errorf("feedback merged despite the mismatch: %s", tr.Feedback)
	}
	if err := fb.Record(ctx, api.Feedback{TurnID: turn.ID(), SiteID: "site-1", StoreID: "store-1", Saved: &saved}); err != nil {
		t.Fatalf("matching feedback = %v", err)
	}
	if err := fb.Record(ctx, api.Feedback{TurnID: turn.ID(), Note: "no site"}); err != nil {
		t.Fatalf("siteId-less feedback = %v", err)
	}
	tr, _ := h.st.trace(turn.ID())
	if !strings.Contains(string(tr.Feedback), `"saved":true`) || !strings.Contains(string(tr.Feedback), `"note":"no site"`) || strings.Contains(string(tr.Feedback), "siteId") {
		t.Errorf("feedback = %s", tr.Feedback)
	}
}
