package loop

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/breadstack/sous-chef/internal/api"
	"github.com/breadstack/sous-chef/internal/model/modeltest"
	"github.com/breadstack/sous-chef/internal/store"
)

// Round-6 regressions (refuter H4b, H6): every message a turn writes is a
// compare-and-set on the tail it owns, so a superseded turn's late write
// never lands on top of the newer turn's; and a Start's storage block (row
// + supersede) is one serialized transaction, so two Starts racing for one
// site leave exactly one live turn — the one that committed last.

func countKind(evs []api.CollectedEvent, kind string) int {
	n := 0
	for _, e := range evs {
		if e.Event == kind {
			n++
		}
	}
	return n
}

// H4b: A is in the model when B starts, supersedes A and goes waiting_tools
// on its own batch. A's late batch must NOT be stored on top of B's (it
// would be answered by A's closeDangling, and B's own results would then
// conflict: B's Continue 409 status=error, rowB error, a(opB) dangling
// mid-history until rotation). A writes nothing, emits no tool.call, and
// ends turn_expired; B's results land and B finishes.
func TestStart_LateBatchOfSupersededTurnNeverBuriesWaitingTurn(t *testing.T) {
	h := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{opUse("opB", "headline", "b")}}), // B's call (first to reach the fake)
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{opUse("opA", "headline", "a")}}), // A's late reply
		textReply("B done"),
		textReply("C reply"),
	)
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
		col := &api.Collector{}
		_ = tB.Run(ctx, col)
		if kinds(col.Events()) != "tool.call turn.waiting" {
			t.Errorf("B events = %s", kinds(col.Events()))
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
	evsA := colA.Events()
	if kinds(evsA) != "turn.error" || data(t, last(evsA))["code"] != "turn_expired" || !strings.Contains(data(t, last(evsA))["message"].(string), "superseded") {
		t.Errorf("A events = %s %v; want turn.error turn_expired (superseded) and no tool.call for the unstored batch", kinds(evsA), data(t, last(evsA)))
	}
	if h.st.turn(tA.ID()).Status != store.TurnSuperseded || h.st.turn(tB.ID()).Status != store.TurnWaitingTools {
		t.Errorf("rowA=%s rowB=%s", h.st.turn(tA.ID()).Status, h.st.turn(tB.ID()).Status)
	}
	if got := roles(t, h, convID); strings.Join(got, " ") != "user user assistant" {
		t.Errorf("stored roles after A's late batch = %v, want user user assistant (B's batch is the tail)", got)
	}
	_, err = h.cont(tA.ID(), api.ToolResult{ToolCallID: "opA", OK: true})
	expectSuperseded409(t, err)
	// The browser applied B's op and reports it: B's own results must land.
	evsB, err := h.cont(tB.ID(), api.ToolResult{ToolCallID: "opB", OK: true})
	if err != nil {
		t.Fatalf("the newer turn B's own results refused: %v (rowB=%s)", err, h.st.turn(tB.ID()).Status)
	}
	if !strings.HasSuffix(kinds(evsB), "turn.done") || h.st.turn(tB.ID()).Status != store.TurnDone {
		t.Errorf("B resume = %s rowB=%s", kinds(evsB), h.st.turn(tB.ID()).Status)
	}
	warned := watchWarnings(t, h)
	h.start(h.request("C text"))
	if got := roles(t, h, convID); strings.Join(got, " ") != "user user assistant user assistant user assistant" {
		t.Errorf("final roles = %v", got)
	}
	if *warned {
		t.Error("WARNING on the next Start: B's batch dangles mid-history")
	}
	if h.runner.Parked() != 0 || h.runner.Live() != 0 {
		t.Errorf("parked=%d live=%d", h.runner.Parked(), h.runner.Live())
	}
	for _, r := range h.fake.Requests() {
		assertReplayable(t, r)
	}
}

// racingStartScript is the model script of the racing-Start tests: the
// survivor's batch, its reply after the results, and turn C.
func racingStartScript() []modeltest.Step {
	return []modeltest.Step{
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{opUse("op1", "headline", "one")}}),
		textReply("after op1"),
		textReply("C reply"),
	}
}

// assertNewestSurvives holds the invariants of two Starts (A first, B
// inside A's window) racing for one site: A's row is superseded and B's
// running; A ends turn.error turn_expired (superseded) without a model call
// and nothing of it stored; B runs, waits and continues normally; A's late
// Continue is 409 superseded; the next Start replays without a repair.
func assertNewestSurvives(t *testing.T, h *harness, tA, tB api.Turn) {
	t.Helper()
	ctx := context.Background()
	convID := h.st.turn(tB.ID()).ConversationID
	if h.st.turn(tA.ID()).Status != store.TurnSuperseded || h.st.turn(tB.ID()).Status != store.TurnRunning {
		t.Fatalf("after both Starts: rowA=%s rowB=%s; want the older superseded, the newer running", h.st.turn(tA.ID()).Status, h.st.turn(tB.ID()).Status)
	}
	if h.st.turn(tA.ID()).ConversationID != convID {
		t.Errorf("A and B on different conversations: %s / %s", h.st.turn(tA.ID()).ConversationID, convID)
	}
	colA := &api.Collector{}
	_ = tA.Run(ctx, colA)
	evsA := colA.Events()
	if kinds(evsA) != "turn.error" || data(t, last(evsA))["code"] != "turn_expired" || !strings.Contains(data(t, last(evsA))["message"].(string), "superseded") {
		t.Errorf("A events = %s %v", kinds(evsA), data(t, last(evsA)))
	}
	if n := len(h.fake.Requests()); n != 0 {
		t.Errorf("the superseded Start made %d model call(s)", n)
	}
	if got := roles(t, h, convID); strings.Join(got, " ") != "user" {
		t.Errorf("stored roles before B runs = %v, want only B's text", got)
	}
	colB := &api.Collector{}
	_ = tB.Run(ctx, colB)
	if kinds(colB.Events()) != "tool.call turn.waiting" {
		t.Errorf("B events = %s", kinds(colB.Events()))
	}
	_, err := h.cont(tA.ID(), api.ToolResult{ToolCallID: "op1", OK: true})
	expectSuperseded409(t, err)
	evsB, err := h.cont(tB.ID(), api.ToolResult{ToolCallID: "op1", OK: true})
	if err != nil || !strings.HasSuffix(kinds(evsB), "turn.done") {
		t.Errorf("B continue = %s %v", kinds(evsB), err)
	}
	if tr, ok := h.st.trace(tA.ID()); !ok || !strings.Contains(string(tr.Ops), `"status":"superseded"`) {
		t.Errorf("A trace = %v %s", ok, tr.Ops)
	}
	warned := watchWarnings(t, h)
	h.start(h.request("C text"))
	if got := roles(t, h, convID); strings.Join(got, " ") != "user assistant user assistant user assistant" {
		t.Errorf("final roles = %v", got)
	}
	if *warned {
		t.Error("WARNING on the next Start: the racing-Start window left a dangle")
	}
	if h.runner.Parked() != 0 || h.runner.Live() != 0 {
		t.Errorf("parked=%d live=%d", h.runner.Parked(), h.runner.Live())
	}
	for _, r := range h.fake.Requests() {
		assertReplayable(t, r)
	}
}

// H6: B's Start runs entirely inside A's StartTurn→history window. Before
// round 6 the row was created after the supersede, so neither Start saw the
// other's row and both survived running (A's Continue 409 status=error,
// dangle until rotation). Now the Starts' storage blocks are serialized: B's
// commits after A's and supersedes A's row, so A is the one closed.
func TestStart_RacingStarts_StartWindow(t *testing.T) {
	h := newHarness(t, nil, racingStartScript()...)
	ctx := context.Background()
	var tB api.Turn
	var errB error
	fired := false
	h.st.mu.Lock()
	h.st.afterStartTurn = func(string) {
		if fired {
			return
		}
		fired = true
		tB, errB = h.runner.Start(ctx, h.request("B text"))
	}
	h.st.mu.Unlock()
	tA, err := h.runner.Start(ctx, h.request("A text"))
	h.st.mu.Lock()
	h.st.afterStartTurn = nil
	h.st.mu.Unlock()
	if err != nil || errB != nil || !fired {
		t.Fatalf("Start A=%v B=%v fired=%v", err, errB, fired)
	}
	assertNewestSurvives(t, h, tA, tB)
}

// B's Start runs between A's pre-append row check (still running) and A's
// user-text append: the append is a compare-and-set on the tail A read, so
// it fails on B's text, A finds its row superseded and stores nothing.
func TestStart_RacingStarts_UserTextWindow(t *testing.T) {
	h := newHarness(t, nil, racingStartScript()...)
	ctx := context.Background()
	var tB api.Turn
	var errB error
	fired := false
	h.st.mu.Lock()
	h.st.afterGetTurn = func(string) { // A's first GetTurn is the check before its user text
		if fired {
			return
		}
		fired = true
		tB, errB = h.runner.Start(ctx, h.request("B text"))
	}
	h.st.mu.Unlock()
	tA, err := h.runner.Start(ctx, h.request("A text"))
	h.st.mu.Lock()
	h.st.afterGetTurn = nil
	h.st.mu.Unlock()
	if err != nil || errB != nil || !fired {
		t.Fatalf("Start A=%v B=%v fired=%v", err, errB, fired)
	}
	assertNewestSurvives(t, h, tA, tB)
}

// A message that lands between a Start's history read and its user-text
// append, while the turn is still the site's live one (here injected
// directly), makes the compare-and-set fail: the Start re-reads the tail
// and its request carries that message — never built over.
func TestStart_UserTextRetriesWhenTailMoves(t *testing.T) {
	h := newHarness(t, nil, textReply("reply"))
	ctx := context.Background()
	fired := false
	h.st.mu.Lock()
	h.st.afterGetTurn = func(id string) {
		if fired {
			return
		}
		fired = true
		convID := h.st.turn(id).ConversationID
		if _, err := h.st.AppendMessage(ctx, convID, "user", json.RawMessage(`[{"type":"text","text":"slipped in"}]`)); err != nil {
			t.Error(err)
		}
	}
	h.st.mu.Unlock()
	turn, evs := h.start(h.request("mine"))
	h.st.mu.Lock()
	h.st.afterGetTurn = nil
	h.st.mu.Unlock()
	if !fired || !strings.HasSuffix(kinds(evs), "turn.done") {
		t.Fatalf("fired=%v events=%s", fired, kinds(evs))
	}
	convID := h.st.turn(turn.ID()).ConversationID
	if got := roles(t, h, convID); strings.Join(got, " ") != "user user assistant" {
		t.Errorf("stored roles = %v", got)
	}
	reqs := h.fake.Requests()
	if len(reqs) != 1 {
		t.Fatalf("model called %d times", len(reqs))
	}
	msgs, _ := json.Marshal(reqs[0].Body["messages"])
	if !strings.Contains(string(msgs), "slipped in") || !strings.Contains(string(msgs), "mine") || strings.LastIndex(string(msgs), "slipped in") > strings.LastIndex(string(msgs), "mine") {
		t.Errorf("request messages = %s; want the slipped-in text before this turn's", msgs)
	}
	assertReplayable(t, reqs[0])
}

// A Start that fails after its row exists (here: the user-text append)
// closes the row as error instead of leaving it running until the reaper.
func TestStart_FailedStartClosesRow(t *testing.T) {
	h := newHarness(t, nil, textReply("never"))
	h.st.mu.Lock()
	h.st.failAppends = 1
	h.st.mu.Unlock()
	if _, err := h.runner.Start(context.Background(), h.request("x")); err == nil || !strings.Contains(err.Error(), "injected") {
		t.Fatalf("Start = %v, want the injected append failure", err)
	}
	h.st.mu.Lock()
	rows := make([]store.Turn, 0, len(h.st.turns))
	for _, row := range h.st.turns {
		rows = append(rows, *row)
	}
	h.st.mu.Unlock()
	if len(rows) != 1 || rows[0].Status != store.TurnError || rows[0].Ended == nil {
		t.Errorf("turn rows after a failed start = %+v; want one, error", rows)
	}
	if h.runner.Live() != 0 {
		t.Errorf("live=%d", h.runner.Live())
	}
}
