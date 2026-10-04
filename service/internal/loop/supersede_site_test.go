package loop

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/breadstack/sous-chef/internal/api"
	"github.com/breadstack/sous-chef/internal/model"
	"github.com/breadstack/sous-chef/internal/model/modeltest"
	"github.com/breadstack/sous-chef/internal/store"
)

// Round-5 regressions (refuter N5, RUNNING-WINDOW, G6, G7, pd.Seq): the
// supersede is site-wide and covers running turns; a turn whose row is no
// longer running never ends turn.done; tenant ids are each verified when
// present; the superseded tail owner is matched by tool_use id.

func pickUse(id string) modeltest.Block {
	return modeltest.ToolUse(id, "pick_asset", map[string]any{"purpose": "bg"})
}

// expectSuperseded409 asserts the launcher's drop-silently shape: 409
// turn_not_waiting with status superseded.
func expectSuperseded409(t *testing.T, err error) {
	t.Helper()
	var apiErr *api.Error
	if !errors.As(err, &apiErr) || apiErr.Code != "turn_not_waiting" || apiErr.Status != 409 || apiErr.Extra["status"] != store.TurnSuperseded {
		t.Fatalf("late Continue = %v, want 409 turn_not_waiting status=superseded", err)
	}
}

// watchWarnings replaces the runner's Logf with one that records whether a
// history-repair WARNING was logged.
func watchWarnings(t *testing.T, h *harness) *bool {
	t.Helper()
	warned := new(bool)
	h.runner.d.Logf = func(f string, a ...any) {
		if strings.Contains(f, "WARNING") {
			*warned = true
		}
		t.Logf(f, a...)
	}
	return warned
}

// N5 (refuter G5): turn 1 is suspended on conversation C1; the next Start
// reaches RotateAt and rotates to C2. The Start must supersede turn 1 even
// though it sits on the previous conversation: its late pick is 409
// superseded, nothing is written into the ended C1, and the site check
// still answers 404 for a wrong siteId.
func TestStart_SupersedesTurnOnRotatedConversation(t *testing.T) {
	h := newHarness(t, func(d *Deps) { d.RotateAt = 2 },
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{pickUse("pa1")}}),
		textReply("turn 2 on the fresh conversation"),
		textReply("turn 3"),
	)
	ctx := context.Background()
	t1, evs := h.start(h.request("pick"))
	if kinds(evs) != "turn.suspended" || h.runner.Parked() != 1 {
		t.Fatalf("turn 1: %s parked=%d", kinds(evs), h.runner.Parked())
	}
	c1 := h.st.turn(t1.ID()).ConversationID
	warned := watchWarnings(t, h)
	// [u a] = 2 messages → the next Start rotates.
	t2, evs2 := h.start(h.request("new topic"))
	c2 := h.st.turn(t2.ID()).ConversationID
	if c1 == c2 || !strings.HasSuffix(kinds(evs2), "turn.done") || data(t, last(evs2))["conversationRotated"] != true {
		t.Fatalf("rotation: c1=%s c2=%s events=%s", c1, c2, kinds(evs2))
	}
	if conv1, _ := h.st.GetConversation(ctx, c1); conv1.EndedAt == nil {
		t.Fatal("c1 not ended")
	}
	row1 := h.st.turn(t1.ID())
	if row1.Status != store.TurnSuperseded || row1.Ended == nil || row1.PendingToolCalls != nil || h.runner.Parked() != 0 {
		t.Fatalf("row1=%s ended=%v pending=%s parked=%d: the Start that rotated must supersede the old conversation's turn", row1.Status, row1.Ended != nil, row1.PendingToolCalls, h.runner.Parked())
	}
	if tr, ok := h.st.trace(t1.ID()); !ok || !strings.Contains(string(tr.Ops), `"status":"superseded"`) {
		t.Errorf("turn 1 trace = %v %s", ok, tr.Ops)
	}
	before := roles(t, h, c1)
	pick := api.ToolResult{ToolCallID: "pa1", OK: true, Asset: &api.Asset{PublicURL: "https://cdn/a.jpg"}}
	var apiErr *api.Error
	_, err := h.runner.Continue(ctx, t1.ID(), api.ToolResultsRequest{SiteID: "site-2", Results: []api.ToolResult{pick}})
	if !errors.As(err, &apiErr) || apiErr.Code != "turn_not_found" {
		t.Errorf("wrong site on the rotated-away turn = %v, want 404", err)
	}
	_, err = h.runner.Continue(ctx, t1.ID(), api.ToolResultsRequest{SiteID: "site-1", StoreID: "store-1", Results: []api.ToolResult{pick}})
	expectSuperseded409(t, err)
	if after := roles(t, h, c1); strings.Join(after, " ") != strings.Join(before, " ") {
		t.Errorf("late pick wrote into the ended conversation: %v → %v", before, after)
	}
	if *warned {
		t.Error("WARNING logged: the superseded turn's tail is not a repair")
	}
	if h.runner.Parked() != 0 || h.runner.Live() != 0 {
		t.Errorf("parked=%d live=%d", h.runner.Parked(), h.runner.Live())
	}
	h.start(h.request("z"))
	for _, r := range h.fake.Requests() {
		assertReplayable(t, r)
	}
}

// N5 (refuter G5b): the conversation was ended under the suspended turn by
// a store-level EndConversation (an instance without the superseding End
// adapter); the next Start opens a fresh conversation and must still
// supersede the turn on the ended one.
func TestStart_SupersedesTurnOnEndedConversation(t *testing.T) {
	h := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{pickUse("pa1")}}),
		textReply("fresh"),
	)
	ctx := context.Background()
	t1, _ := h.start(h.request("pick"))
	c1 := h.st.turn(t1.ID()).ConversationID
	if ok, err := h.st.EndConversation(ctx, "site-1"); err != nil || !ok {
		t.Fatalf("end = %v %v", ok, err)
	}
	t2, evs2 := h.start(h.request("fresh start"))
	if h.st.turn(t2.ID()).ConversationID == c1 || !strings.HasSuffix(kinds(evs2), "turn.done") {
		t.Fatalf("expected a new conversation: %s", kinds(evs2))
	}
	if row1 := h.st.turn(t1.ID()); row1.Status != store.TurnSuperseded || h.runner.Parked() != 0 {
		t.Fatalf("row1=%s parked=%d: the ended conversation's suspended turn must be superseded", row1.Status, h.runner.Parked())
	}
	before := roles(t, h, c1)
	_, err := h.cont(t1.ID(), api.ToolResult{ToolCallID: "pa1", OK: true, Asset: &api.Asset{PublicURL: "https://cdn/a.jpg"}})
	expectSuperseded409(t, err)
	if after := roles(t, h, c1); strings.Join(after, " ") != strings.Join(before, " ") {
		t.Errorf("late pick appended: %v → %v", before, after)
	}
	assertReplayable(t, h.fake.Requests()[1])
}

// DELETE /v1/conversations (Conversations.End) supersedes the site's live
// turns itself, so a late Continue is 409 superseded even before any new
// Start, and a turn still running in the ended conversation stops at its
// next status-guarded write with turn.error turn_expired.
func TestConversationsEnd_SupersedesLiveTurns(t *testing.T) {
	h := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{opUse("op1", "headline", "x")}}),
		textReply("fresh"),
	)
	ctx := context.Background()
	t1, evs := h.start(h.request("x"))
	if !strings.HasSuffix(kinds(evs), "turn.waiting") {
		t.Fatalf("turn 1: %s", kinds(evs))
	}
	c1 := h.st.turn(t1.ID()).ConversationID
	var logs []string
	convs := Conversations{St: h.st, Logf: func(f string, a ...any) { logs = append(logs, f); t.Logf(f, a...) }}
	if err := convs.End(ctx, "site-1"); err != nil {
		t.Fatal(err)
	}
	if conv1, _ := h.st.GetConversation(ctx, c1); conv1.EndedAt == nil {
		t.Fatal("c1 not ended")
	}
	if row1 := h.st.turn(t1.ID()); row1.Status != store.TurnSuperseded || row1.Ended == nil || row1.PendingToolCalls != nil {
		t.Fatalf("row1 after DELETE = %+v", row1)
	}
	if n := len(logs); n != 1 || !strings.Contains(logs[0], "DELETE") {
		t.Errorf("End logged %d lines: %q", n, logs)
	}
	// The parked entry is stale until the late Continue (or the reaper): it
	// goes with the 409.
	before := roles(t, h, c1)
	_, err := h.runner.Continue(ctx, t1.ID(), api.ToolResultsRequest{SiteID: "site-1", Results: []api.ToolResult{{ToolCallID: "op1", OK: true}}})
	expectSuperseded409(t, err)
	if h.runner.Parked() != 0 {
		t.Errorf("parked = %d", h.runner.Parked())
	}
	if tr, ok := h.st.trace(t1.ID()); !ok || !strings.Contains(string(tr.Ops), `"status":"superseded"`) {
		t.Errorf("turn 1 trace = %v %s", ok, tr.Ops)
	}
	if after := roles(t, h, c1); strings.Join(after, " ") != strings.Join(before, " ") {
		t.Errorf("DELETE'd conversation written to: %v → %v", before, after)
	}

	// A turn mid-model-call (Start done, Run not yet) when a second DELETE
	// lands: its pre-call row check finds the row superseded, no model call.
	t2, err := h.runner.Start(ctx, h.request("y"))
	if err != nil {
		t.Fatal(err)
	}
	c2 := h.st.turn(t2.ID()).ConversationID
	logs = nil
	if err := convs.End(ctx, "site-1"); err != nil {
		t.Fatal(err)
	}
	if h.st.turn(t2.ID()).Status != store.TurnSuperseded || len(logs) != 1 {
		t.Fatalf("row2 after DELETE = %s logs=%q", h.st.turn(t2.ID()).Status, logs)
	}
	before = roles(t, h, c2)
	col := &api.Collector{}
	_ = t2.Run(ctx, col)
	if kinds(col.Events()) != "turn.error" || data(t, last(col.Events()))["code"] != "turn_expired" || !strings.Contains(data(t, last(col.Events()))["message"].(string), "superseded") {
		t.Errorf("running turn after DELETE: %s %v", kinds(col.Events()), data(t, last(col.Events())))
	}
	if n := len(h.fake.Requests()); n != 1 {
		t.Errorf("model called %d times; the superseded turn must make no call", n)
	}
	if after := roles(t, h, c2); strings.Join(after, " ") != strings.Join(before, " ") {
		t.Errorf("DELETE'd conversation written to: %v → %v", before, after)
	}
	if h.runner.Live() != 0 || h.runner.Parked() != 0 {
		t.Errorf("live=%d parked=%d", h.runner.Live(), h.runner.Parked())
	}
	// The next turn starts fresh.
	t3, evs3 := h.start(h.request("fresh"))
	if c3 := h.st.turn(t3.ID()).ConversationID; c3 == c1 || c3 == c2 || !strings.HasSuffix(kinds(evs3), "turn.done") {
		t.Errorf("turn 3: %s", kinds(evs3))
	}
}

// runningWindow drives the RUNNING-WINDOW ordering: turn A has passed its
// pre-call row check and is "in the model" when Start B lands (hooked on
// A's GetTurn); runB says whether B also runs to completion inside the
// window (ordering [u(A) u(B) a(B) a(A)]) or only starts (B's reply lands
// after A's: [u(A) u(B) a(A) a(B)]). Returns A's events and id, B's turn
// and the conversation id.
func runningWindow(t *testing.T, h *harness, runB bool) ([]api.CollectedEvent, string, api.Turn, string) {
	t.Helper()
	ctx := context.Background()
	tA, err := h.runner.Start(ctx, h.request("A text"))
	if err != nil {
		t.Fatal(err)
	}
	convID := h.st.turn(tA.ID()).ConversationID
	var tB api.Turn
	fired := false
	h.st.mu.Lock()
	h.st.afterGetTurn = func(id string) {
		if fired || id != tA.ID() {
			return
		}
		fired = true
		tB, err = h.runner.Start(ctx, h.request("B text"))
		if err != nil {
			t.Errorf("Start B = %v", err)
			return
		}
		if runB {
			_ = tB.Run(ctx, &api.Collector{})
		}
	}
	h.st.mu.Unlock()
	colA := &api.Collector{}
	_ = tA.Run(ctx, colA)
	h.st.mu.Lock()
	h.st.afterGetTurn = nil
	h.st.mu.Unlock()
	if !fired || tB == nil {
		t.Fatal("hook never fired")
	}
	if !runB {
		_ = tB.Run(ctx, &api.Collector{})
	}
	return colA.Events(), tA.ID(), tB, convID
}

// assertRunningWindowClosed holds the invariants of both orderings: A's
// row is superseded, A ended turn.error turn_expired (the superseded
// message), its late Continue is 409 superseded, B is done, the stored
// history has no dangling tool_use (no WARNING on the next Start), and
// every request was replayable.
func assertRunningWindowClosed(t *testing.T, h *harness, evsA []api.CollectedEvent, tA string, tB api.Turn, convID string, wantRoles string) {
	t.Helper()
	if !strings.HasSuffix(kinds(evsA), "turn.error") || data(t, last(evsA))["code"] != "turn_expired" || !strings.Contains(data(t, last(evsA))["message"].(string), "superseded") {
		t.Errorf("A events = %s %v", kinds(evsA), data(t, last(evsA)))
	}
	if h.st.turn(tA).Status != store.TurnSuperseded || h.st.turn(tB.ID()).Status != store.TurnDone {
		t.Errorf("rowA=%s rowB=%s", h.st.turn(tA).Status, h.st.turn(tB.ID()).Status)
	}
	if tr, ok := h.st.trace(tA); !ok || !strings.Contains(string(tr.Ops), `"status":"superseded"`) {
		t.Errorf("A trace = %v %s", ok, tr.Ops)
	}
	if got := roles(t, h, convID); strings.Join(got, " ") != wantRoles {
		t.Errorf("stored roles = %v, want %s", got, wantRoles)
	}
	before := roles(t, h, convID)
	_, err := h.cont(tA, api.ToolResult{ToolCallID: "opA", OK: true})
	expectSuperseded409(t, err)
	if after := roles(t, h, convID); strings.Join(after, " ") != strings.Join(before, " ") {
		t.Errorf("late results appended: %v → %v", before, after)
	}
	if h.runner.Parked() != 0 || h.runner.Live() != 0 {
		t.Errorf("parked=%d live=%d", h.runner.Parked(), h.runner.Live())
	}
	warned := watchWarnings(t, h)
	h.start(h.request("C text"))
	if *warned {
		t.Error("WARNING on the next Start: a mid-history dangle survived the running window")
	}
	for _, r := range h.fake.Requests() {
		assertReplayable(t, r)
	}
}

// RUNNING-WINDOW (refuter G8b, ordering [u(A) u(B) a(A) a(B)]): Start B
// supersedes the running turn A and stores its user text; A's tool_use
// batch lands next, but its compare-and-set append finds u(B) on the tail
// instead of its own u(A), so nothing of A is stored, no tool.call is
// emitted for ops the page must not apply, and A ends turn_expired; B's
// reply lands on a clean tail (round 6: before the CAS, A's batch was
// stored and answered by A itself, u(A) u(B) a(A) u(answered) a(B)).
func TestStart_SupersedesRunningTurn_BatchLandsFirst(t *testing.T) {
	h := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{opUse("opA", "headline", "a")}}),
		textReply("B reply"),
		textReply("C reply"),
	)
	evsA, tA, tB, convID := runningWindow(t, h, false)
	if kinds(evsA) != "turn.error" {
		t.Errorf("A events = %s", kinds(evsA))
	}
	assertRunningWindowClosed(t, h, evsA, tA, tB, convID, "user user assistant")
}

// RUNNING-WINDOW (refuter G8, ordering [u(A) u(B) a(B) a(A)]): B starts
// and finishes while A is in the model; A's batch then arrives behind B's
// reply, its compare-and-set append fails, and A ends turn_expired with
// nothing stored — instead of going waiting_tools behind B (where its
// Continue was 409 status=error and the batch dangled mid-history until
// rotation) or appending a(A) u(answered) after B's reply.
func TestStart_SupersedesRunningTurn_ReplyLandsFirst(t *testing.T) {
	h := newHarness(t, nil,
		textReply("B reply"), // B's request reaches the model first
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{opUse("opA", "headline", "a")}}),
		textReply("C reply"),
	)
	evsA, tA, tB, convID := runningWindow(t, h, true)
	if kinds(evsA) != "turn.error" {
		t.Errorf("A events = %s", kinds(evsA))
	}
	assertRunningWindowClosed(t, h, evsA, tA, tB, convID, "user user assistant")
}

// RUNNING-WINDOW, text reply: the superseded running turn's final reply
// (already streamed as text.delta) is not stored — its compare-and-set
// append fails on B's user text — and the turn ends turn.error
// turn_expired, never turn.done.
func TestStart_SupersedesRunningTurn_TextReply(t *testing.T) {
	h := newHarness(t, nil,
		textReply("A reply"),
		textReply("B reply"),
		textReply("C reply"),
	)
	evsA, tA, tB, convID := runningWindow(t, h, false)
	if !strings.HasPrefix(kinds(evsA), "text.delta") || !strings.HasSuffix(kinds(evsA), "turn.error") || data(t, last(evsA))["code"] != "turn_expired" || !strings.Contains(data(t, last(evsA))["message"].(string), "superseded") {
		t.Errorf("A events = %s %v", kinds(evsA), data(t, last(evsA)))
	}
	if h.st.turn(tA).Status != store.TurnSuperseded || h.st.turn(tB.ID()).Status != store.TurnDone {
		t.Errorf("rowA=%s rowB=%s", h.st.turn(tA).Status, h.st.turn(tB.ID()).Status)
	}
	if got := roles(t, h, convID); strings.Join(got, " ") != "user user assistant" {
		t.Errorf("stored roles = %v", got)
	}
	warned := watchWarnings(t, h)
	h.start(h.request("C text"))
	if *warned {
		t.Error("WARNING on the next Start")
	}
	for _, r := range h.fake.Requests() {
		assertReplayable(t, r)
	}
}

// G6: the TTL fires DURING the final (text-only) model call — after the
// pre-call row check — so FinishTurn finds the row expired. The turn must
// end turn.error turn_expired, not turn.done on an expired row.
func TestRun_ExpiryDuringFinalModelCall(t *testing.T) {
	h := newHarness(t, func(d *Deps) { d.TurnTTL = time.Minute }, textReply("done"), textReply("next"))
	ctx := context.Background()
	turn, err := h.runner.Start(ctx, h.request("x"))
	if err != nil {
		t.Fatal(err)
	}
	fired := false
	h.st.mu.Lock()
	h.st.afterGetTurn = func(id string) {
		if fired || id != turn.ID() {
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
	h.st.afterGetTurn = nil
	h.st.mu.Unlock()
	if !fired {
		t.Fatal("hook never fired")
	}
	if !strings.HasPrefix(kinds(col.Events()), "text.delta") || !strings.HasSuffix(kinds(col.Events()), "turn.error") || data(t, last(col.Events()))["code"] != "turn_expired" {
		t.Errorf("events = %s %v: an expired row must end turn_expired, not turn.done", kinds(col.Events()), data(t, last(col.Events())))
	}
	row := h.st.turn(turn.ID())
	if row.Status != store.TurnExpired || row.Usage != nil || h.runner.Live() != 0 {
		t.Errorf("row=%s usage=%s live=%d", row.Status, row.Usage, h.runner.Live())
	}
	if tr, ok := h.st.trace(turn.ID()); !ok || !strings.Contains(string(tr.Ops), `"status":"expired"`) || strings.Contains(string(tr.Ops), `"status":"done"`) {
		t.Errorf("trace = %v %s", ok, tr.Ops)
	}
	if n := len(h.fake.Requests()); n != 1 {
		t.Errorf("model called %d times", n)
	}
	h.start(h.request("y"))
	reqs := h.fake.Requests()
	assertReplayable(t, reqs[len(reqs)-1])
}

// G7: feedback logs the siteId-less note once per turn (as tool-results
// does), and a body naming only a storeId is verified on that storeId —
// on feedback and on tool-results alike.
func TestTenantCheck_StoreOnlyBodyAndLogCadence(t *testing.T) {
	h := newHarness(t, nil, textReply("ok"), textReply("ok 2"))
	ctx := context.Background()
	turn, _ := h.start(h.request("hello"))
	turn2, _ := h.start(h.request("hello again"))
	noSite, storeOnly := 0, 0
	fb := &Feedback{St: h.st, Logf: func(f string, a ...any) {
		switch {
		case strings.Contains(f, "only its storeId"):
			storeOnly++
		case strings.Contains(f, "no siteId"):
			noSite++
		}
		t.Logf(f, a...)
	}}
	saved := true
	var apiErr *api.Error
	// storeId alone, wrong store → 404, nothing merged.
	err := fb.Record(ctx, api.Feedback{TurnID: turn.ID(), StoreID: "store-2", Saved: &saved})
	if !errors.As(err, &apiErr) || apiErr.Code != "turn_not_found" || apiErr.Status != 404 {
		t.Errorf("storeId-only wrong store = %v, want 404", err)
	}
	if tr, _ := h.st.trace(turn.ID()); len(tr.Feedback) != 0 {
		t.Errorf("merged despite the mismatch: %s", tr.Feedback)
	}
	// storeId alone, right store → accepted; unknown turn with storeId → 404.
	if err := fb.Record(ctx, api.Feedback{TurnID: turn.ID(), StoreID: "store-1", Saved: &saved}); err != nil {
		t.Errorf("storeId-only right store = %v", err)
	}
	if err := fb.Record(ctx, api.Feedback{TurnID: "turn-missing", StoreID: "store-1", Saved: &saved}); !errors.As(err, &apiErr) || apiErr.Code != "turn_not_found" {
		t.Errorf("storeId-only unknown turn = %v", err)
	}
	// Four siteId-less bodies on the first turn (already noted once above,
	// as storeId-only) and one on the second: one line per turn in all.
	for i := 0; i < 4; i++ {
		if err := fb.Record(ctx, api.Feedback{TurnID: turn.ID(), Note: "n"}); err != nil {
			t.Fatal(err)
		}
	}
	if err := fb.Record(ctx, api.Feedback{TurnID: turn2.ID(), Note: "n"}); err != nil {
		t.Fatal(err)
	}
	if storeOnly != 1 || noSite != 1 {
		t.Errorf("storeId-only note logged %d times (want 1: turn 1), siteId-less note %d (want 1: turn 2) — once per turn in all", storeOnly, noSite)
	}
	if tr, _ := h.st.trace(turn.ID()); !strings.Contains(string(tr.Feedback), `"saved":true`) || !strings.Contains(string(tr.Feedback), `"note":"n"`) {
		t.Errorf("feedback = %s", tr.Feedback)
	}

	// tool-results: storeId alone is verified too.
	h2 := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{opUse("op1", "headline", "x")}}),
		textReply("done"),
	)
	t3, _ := h2.start(h2.request("x"))
	_, err = h2.runner.Continue(ctx, t3.ID(), api.ToolResultsRequest{StoreID: "store-2", Results: []api.ToolResult{{ToolCallID: "op1", OK: true}}})
	if !errors.As(err, &apiErr) || apiErr.Code != "turn_not_found" || apiErr.Status != 404 {
		t.Errorf("tool-results storeId-only wrong store = %v, want 404", err)
	}
	if h2.st.turn(t3.ID()).Status != store.TurnWaitingTools || h2.runner.Parked() != 1 {
		t.Errorf("refused continuation changed the turn: row=%s parked=%d", h2.st.turn(t3.ID()).Status, h2.runner.Parked())
	}
	tt, err := h2.runner.Continue(ctx, t3.ID(), api.ToolResultsRequest{StoreID: "store-1", Results: []api.ToolResult{{ToolCallID: "op1", OK: true}}})
	if err != nil {
		t.Fatalf("tool-results storeId-only right store = %v", err)
	}
	col := &api.Collector{}
	_ = tt.Run(ctx, col)
	if !strings.HasSuffix(kinds(col.Events()), "turn.done") {
		t.Errorf("events = %s", kinds(col.Events()))
	}
}

// pd.Seq nit: the superseded turn that owns a dangling tail is found by
// matching the tail's tool_use ids against each pending doc — not by the
// docs' Seq, a per-turn event counter that says nothing across turns. A
// superseded running turn (no doc) explains a tail no doc covers.
func TestRepairHistory_SupersededOwnerMatchedByIDs(t *testing.T) {
	h := newHarness(t, nil)
	var logs []string
	h.runner.d.Logf = func(f string, a ...any) { logs = append(logs, f) }
	toolUse := func(id string) anthropic.BetaMessageParam {
		return anthropic.BetaMessageParam{Role: anthropic.BetaMessageParamRoleAssistant,
			Content: []anthropic.BetaContentBlockParamUnion{anthropic.NewBetaToolUseBlock(id, map[string]any{}, "set_fields")}}
	}
	msgs := []anthropic.BetaMessageParam{model.UserTextTurn("a"), toolUse("op7")}
	// The doc with the HIGHER seq does not own the tail; the lower one does,
	// and it carries a stashed server result for a sibling call.
	high := &pendingDoc{Seq: 9, CallCount: 3, Calls: []pendingCall{{ID: "op1", Name: "set_fields"}}}
	low := &pendingDoc{Seq: 2, CallCount: 1, Calls: []pendingCall{{ID: "op7", Name: "set_fields"}}}
	out, tail := h.runner.repairHistory(context.Background(), "conv", "", msgs, &supersededTurns{docs: []*pendingDoc{high, low}})
	if len(out) != 3 || tail == nil || len(toolResultIDs(*tail)) != 1 || toolResultIDs(*tail)[0] != "op7" {
		t.Fatalf("repaired = %d messages, tail=%v", len(out), tail)
	}
	raw, _ := json.Marshal(tail.Content)
	if !strings.Contains(string(raw), reasonSuperseded) || len(logs) != 0 {
		t.Errorf("tail = %s logs=%q: the owning doc must be matched by id (no WARNING)", raw, logs)
	}
	// A superseded running turn with no doc explains the tail too.
	out, tail = h.runner.repairHistory(context.Background(), "conv", "", msgs, &supersededTurns{running: true})
	raw, _ = json.Marshal(tail.Content)
	if len(out) != 3 || !strings.Contains(string(raw), reasonSuperseded) || len(logs) != 0 {
		t.Errorf("running owner: tail = %s logs=%q", raw, logs)
	}
	// Nothing superseded explains it → the WARNING repair.
	out, tail = h.runner.repairHistory(context.Background(), "conv", "", msgs, &supersededTurns{docs: []*pendingDoc{high}})
	raw, _ = json.Marshal(tail.Content)
	if len(out) != 3 || !strings.Contains(string(raw), reasonHistoryRepair) || len(logs) != 1 || !strings.Contains(logs[0], "WARNING") {
		t.Errorf("foreign tail: tail = %s logs=%q", raw, logs)
	}
	// nil is the no-supersede case.
	if _, ok := (*supersededTurns)(nil).owner([]toolUseRef{{ID: "op7"}}); ok {
		t.Error("nil supersededTurns owns a tail")
	}
}
