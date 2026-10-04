package loop

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/breadstack/sous-chef/internal/api"
	"github.com/breadstack/sous-chef/internal/model/modeltest"
	"github.com/breadstack/sous-chef/internal/store"
)

// Round-7 regressions (refuter I1–I7): a Start's storage block is one
// serialized transaction per site (store.StartTurn), so the most recently
// committed Start is the site's only live turn — whatever the instances'
// clocks say; a superseded turn's batch is never streamed before its row is
// stored waiting; nothing is written into an ended conversation; a turn that
// loses its tail while still running ends as a plain error.

func liveRows(h *harness) []string {
	h.st.mu.Lock()
	defer h.st.mu.Unlock()
	var out []string
	for id, t := range h.st.turns {
		if t.Status == store.TurnRunning || t.Status == store.TurnWaitingTools || t.Status == store.TurnSuspended {
			out = append(out, id)
		}
	}
	sort.Strings(out)
	return out
}

func ordinal(h *harness, id string) int {
	h.st.mu.Lock()
	defer h.st.mu.Unlock()
	return h.st.order[id]
}

// assertLastCommittedSurvives holds the invariants of several Starts racing
// for one site: exactly one row is live afterwards and it is the Start that
// committed last (the memstore's creation ordinal); the losers end
// turn.error turn_expired (superseded) with no model call; the survivor
// runs, waits, continues and finishes; the losers' late Continues are 409
// superseded; the next Start replays with no WARNING.
func assertLastCommittedSurvives(t *testing.T, h *harness, turns []api.Turn) {
	t.Helper()
	ctx := context.Background()
	live := liveRows(h)
	if len(live) != 1 {
		t.Fatalf("live rows after %d Starts = %v (want exactly one)", len(turns), live)
	}
	best := turns[0]
	for _, tt := range turns[1:] {
		if ordinal(h, tt.ID()) > ordinal(h, best.ID()) {
			best = tt
		}
	}
	if live[0] != best.ID() {
		t.Errorf("survivor %s is not the last committed Start %s (ordinals %d vs %d)", live[0], best.ID(), ordinal(h, live[0]), ordinal(h, best.ID()))
	}
	convID := h.st.turn(best.ID()).ConversationID
	for _, tt := range turns {
		if tt.ID() == best.ID() {
			continue
		}
		col := &api.Collector{}
		_ = tt.Run(ctx, col)
		evs := col.Events()
		if kinds(evs) != "turn.error" || data(t, last(evs))["code"] != "turn_expired" || !strings.Contains(data(t, last(evs))["message"].(string), "superseded") {
			t.Errorf("loser %s events = %s %v", tt.ID(), kinds(evs), data(t, last(evs)))
		}
		if h.st.turn(tt.ID()).Status != store.TurnSuperseded {
			t.Errorf("loser %s row = %s", tt.ID(), h.st.turn(tt.ID()).Status)
		}
	}
	if n := len(h.fake.Requests()); n != 0 {
		t.Errorf("losers made %d model call(s)", n)
	}
	pre := roles(t, h, convID)
	for _, r := range pre {
		if r != "user" {
			t.Errorf("stored before the survivor runs = %v, want only user text(s)", pre)
		}
	}
	col := &api.Collector{}
	_ = best.Run(ctx, col)
	if kinds(col.Events()) != "tool.call turn.waiting" {
		t.Errorf("survivor events = %s", kinds(col.Events()))
	}
	for _, tt := range turns {
		if tt.ID() == best.ID() {
			continue
		}
		_, err := h.cont(tt.ID(), api.ToolResult{ToolCallID: "op1", OK: true})
		expectSuperseded409(t, err)
	}
	evs, err := h.cont(best.ID(), api.ToolResult{ToolCallID: "op1", OK: true})
	if err != nil || !strings.HasSuffix(kinds(evs), "turn.done") {
		t.Errorf("survivor continue = %s %v", kinds(evs), err)
	}
	warned := watchWarnings(t, h)
	h.start(h.request("D text"))
	if got := roles(t, h, convID); strings.Join(got, " ") != strings.Join(pre, " ")+" assistant user assistant user assistant" {
		t.Errorf("final roles = %v", got)
	}
	if *warned {
		t.Error("WARNING on the next Start")
	}
	if h.runner.Parked() != 0 || h.runner.Live() != 0 {
		t.Errorf("parked=%d live=%d", h.runner.Parked(), h.runner.Live())
	}
	for _, r := range h.fake.Requests() {
		assertReplayable(t, r)
	}
}

// I1: three Starts nested inside each other's StartTurn→history window: the
// innermost (last committed) survives, the two outer ones are superseded.
func TestStart_RacingStarts_ThreeNested(t *testing.T) {
	h := newHarness(t, nil, racingStartScript()...)
	ctx := context.Background()
	var inner []api.Turn
	var errs []error
	depth := 0
	h.st.mu.Lock()
	h.st.afterStartTurn = func(string) {
		if depth >= 2 {
			return
		}
		depth++
		tt, err := h.runner.Start(ctx, h.request(fmt.Sprintf("inner %d", depth)))
		inner = append(inner, tt)
		errs = append(errs, err)
	}
	h.st.mu.Unlock()
	tA, err := h.runner.Start(ctx, h.request("A text"))
	h.st.mu.Lock()
	h.st.afterStartTurn = nil
	h.st.mu.Unlock()
	if err != nil || len(inner) != 2 || errs[0] != nil || errs[1] != nil {
		t.Fatalf("Starts: A=%v inner=%d errs=%v", err, len(inner), errs)
	}
	assertLastCommittedSurvives(t, h, []api.Turn{tA, inner[0], inner[1]})
}

// I2: three goroutines Start at once, repeatedly: exactly one live row, the
// last committed; the losers end without a model call; the survivor
// completes.
func TestStart_RacingStarts_ThreeConcurrent(t *testing.T) {
	for iter := 0; iter < 25; iter++ {
		h := newHarness(t, nil, racingStartScript()...)
		h.runner.d.Logf = func(string, ...any) {}
		ctx := context.Background()
		var wg sync.WaitGroup
		turns := make([]api.Turn, 3)
		errs := make([]error, 3)
		for i := 0; i < 3; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				turns[i], errs[i] = h.runner.Start(ctx, h.request(fmt.Sprintf("text %d", i)))
			}(i)
		}
		wg.Wait()
		for i, err := range errs {
			if err != nil {
				t.Fatalf("iter %d: Start %d = %v", iter, i, err)
			}
		}
		h.runner.d.Logf = t.Logf
		assertLastCommittedSurvives(t, h, turns)
		if t.Failed() {
			t.Fatalf("iter %d failed", iter)
		}
	}
}

// I6: the second Start's instance has a clock 50ms BEHIND, so its row gets
// an earlier started than the first's. Under the (started, id) rule of
// round 6 the second was "older" and both survived; the serialized
// StartTurn supersedes whatever is live regardless of started.
func TestStart_RacingStarts_ClockSkewIrrelevant(t *testing.T) {
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
		h.clock = h.clock.Add(-50 * time.Millisecond) // instance 2 is 50ms behind
		tB, errB = h.runner.Start(ctx, h.request("B text"))
		h.clock = h.clock.Add(50 * time.Millisecond)
	}
	h.st.mu.Unlock()
	tA, err := h.runner.Start(ctx, h.request("A text"))
	h.st.mu.Lock()
	h.st.afterStartTurn = nil
	h.st.mu.Unlock()
	if err != nil || errB != nil || !fired {
		t.Fatalf("A=%v B=%v fired=%v", err, errB, fired)
	}
	if !h.st.turn(tB.ID()).Started.Before(h.st.turn(tA.ID()).Started) {
		t.Fatalf("B.started %v is not before A.started %v: the skew was not applied", h.st.turn(tB.ID()).Started, h.st.turn(tA.ID()).Started)
	}
	assertNewestSurvives(t, h, tA, tB)
}

// I7: A runs on conversation C1 (RotateAt reached) when B's Start rotates
// to C2 and supersedes A. A's late batch's compare-and-set runs against C1,
// whose tail is still A's own text — the ended conversation refuses the
// write anyway: nothing is written into C1, no tool.call is streamed for
// the superseded batch, A ends turn.error turn_expired (superseded).
func TestStart_SupersededRunningTurn_RotatedConversationRefusesLateBatch(t *testing.T) {
	h := newHarness(t, func(d *Deps) { d.RotateAt = 1 },
		textReply("B reply"),
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{opUse("opA", "headline", "a")}}),
		textReply("C reply"),
	)
	ctx := context.Background()
	tA, err := h.runner.Start(ctx, h.request("A text"))
	if err != nil {
		t.Fatal(err)
	}
	c1 := h.st.turn(tA.ID()).ConversationID
	fired := false
	var tB api.Turn
	h.st.mu.Lock()
	h.st.afterGetTurn = func(id string) {
		if fired || id != tA.ID() {
			return
		}
		fired = true
		var evs []api.CollectedEvent
		tB, evs = h.start(h.request("B text"))
		if !strings.HasSuffix(kinds(evs), "turn.done") {
			t.Errorf("B = %s", kinds(evs))
		}
	}
	h.st.mu.Unlock()
	colA := &api.Collector{}
	_ = tA.Run(ctx, colA)
	h.st.mu.Lock()
	h.st.afterGetTurn = nil
	h.st.mu.Unlock()
	if !fired {
		t.Fatal("hook never fired")
	}
	c2 := h.st.turn(tB.ID()).ConversationID
	if c1 == c2 {
		t.Fatal("no rotation")
	}
	if conv1, _ := h.st.GetConversation(ctx, c1); conv1.EndedAt == nil {
		t.Fatal("C1 not ended")
	}
	evsA := colA.Events()
	if countKind(evsA, "tool.call") != 0 {
		t.Errorf("tool.call streamed for a superseded turn's batch (rotated case): %s", kinds(evsA))
	}
	if kinds(evsA) != "turn.error" || data(t, last(evsA))["code"] != "turn_expired" || !strings.Contains(data(t, last(evsA))["message"].(string), "superseded") {
		t.Errorf("A events = %s %v", kinds(evsA), data(t, last(evsA)))
	}
	if got := roles(t, h, c1); strings.Join(got, " ") != "user" {
		t.Errorf("the superseded turn wrote into the ended conversation c1: %v", got)
	}
	if got := roles(t, h, c2); strings.Join(got, " ") != "user assistant" {
		t.Errorf("c2 roles = %v", got)
	}
	if h.st.turn(tA.ID()).Status != store.TurnSuperseded {
		t.Errorf("rowA = %s", h.st.turn(tA.ID()).Status)
	}
	if tr, ok := h.st.trace(tA.ID()); !ok || !strings.Contains(string(tr.Ops), `"status":"superseded"`) {
		t.Errorf("A trace = %v %s", ok, tr.Ops)
	}
	_, err = h.cont(tA.ID(), api.ToolResult{ToolCallID: "opA", OK: true})
	expectSuperseded409(t, err)
	warned := watchWarnings(t, h)
	h.start(h.request("C text"))
	if *warned {
		t.Error("WARNING")
	}
	if h.runner.Parked() != 0 || h.runner.Live() != 0 {
		t.Errorf("parked=%d live=%d", h.runner.Parked(), h.runner.Live())
	}
	for _, r := range h.fake.Requests() {
		assertReplayable(t, r)
	}
}

// The status guard precedes the stream: a turn whose row stops being
// running between its assistant append and SetTurnWaiting (here the row is
// superseded by a store-level supersede in that window, as a DELETE on
// another instance would) streams no tool.call for the batch, answers its
// own stored tool_use blocks is_error, and ends turn.error (superseded).
func TestRun_NoToolCallWhenRowLostBeforeSetTurnWaiting(t *testing.T) {
	h := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{opUse("opA", "headline", "a")}}),
		textReply("next"),
	)
	ctx := context.Background()
	tA, err := h.runner.Start(ctx, h.request("A text"))
	if err != nil {
		t.Fatal(err)
	}
	convID := h.st.turn(tA.ID()).ConversationID
	fired := false
	h.st.mu.Lock()
	h.st.afterAppend = func(role string) {
		if fired || role != "assistant" {
			return
		}
		fired = true
		if _, err := h.st.SupersedeTurnsForSite(ctx, "site-1"); err != nil {
			t.Error(err)
		}
	}
	h.st.mu.Unlock()
	col := &api.Collector{}
	_ = tA.Run(ctx, col)
	h.st.mu.Lock()
	h.st.afterAppend = nil
	h.st.mu.Unlock()
	if !fired {
		t.Fatal("hook never fired")
	}
	evs := col.Events()
	if kinds(evs) != "turn.error" || data(t, last(evs))["code"] != "turn_expired" || !strings.Contains(data(t, last(evs))["message"].(string), "superseded") {
		t.Errorf("events = %s %v; want turn.error only (no tool.call for a batch whose row was lost)", kinds(evs), data(t, last(evs)))
	}
	if got := roles(t, h, convID); strings.Join(got, " ") != "user assistant user" {
		t.Errorf("roles = %v; the stored batch must be answered is_error", got)
	}
	msgs, _ := h.st.ListMessages(ctx, convID, 0, 0)
	if tail := string(msgs[len(msgs)-1].Content); !strings.Contains(tail, "opA") || !strings.Contains(tail, reasonSuperseded) {
		t.Errorf("tail = %s", tail)
	}
	if h.st.turn(tA.ID()).Status != store.TurnSuperseded || h.runner.Parked() != 0 {
		t.Errorf("row=%s parked=%d", h.st.turn(tA.ID()).Status, h.runner.Parked())
	}
	warned := watchWarnings(t, h)
	h.start(h.request("B"))
	if *warned {
		t.Error("WARNING")
	}
	for _, r := range h.fake.Requests() {
		assertReplayable(t, r)
	}
}

// I5: the tail moves under a turn that is STILL running (a foreign write
// between its pre-call row check and its assistant append). The
// compare-and-set refuses the reply; the turn ends as a plain error — the
// row error, the trace outcome.status error, the message naming the lost
// tail, not "expired".
func TestRun_LostTailWhileStillRunningEndsAsError(t *testing.T) {
	h := newHarness(t, nil, modeltest.Reply(modeltest.Message{Content: []modeltest.Block{opUse("opA", "headline", "a")}}), textReply("next"))
	ctx := context.Background()
	turn, err := h.runner.Start(ctx, h.request("A text"))
	if err != nil {
		t.Fatal(err)
	}
	convID := h.st.turn(turn.ID()).ConversationID
	fired := false
	h.st.mu.Lock()
	h.st.afterGetTurn = func(id string) {
		if fired || id != turn.ID() {
			return
		}
		fired = true
		if _, err := h.st.AppendMessage(ctx, convID, "user", json.RawMessage(`[{"type":"text","text":"foreign"}]`)); err != nil {
			t.Error(err)
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
	evs := col.Events()
	row := h.st.turn(turn.ID())
	tr, _ := h.st.trace(turn.ID())
	var ops struct {
		Outcome struct {
			Status string `json:"status"`
			Error  string `json:"error"`
		} `json:"outcome"`
	}
	_ = json.Unmarshal(tr.Ops, &ops)
	if countKind(evs, "tool.call") != 0 {
		t.Error("tool.call emitted for an unstored batch")
	}
	if got := roles(t, h, convID); strings.Join(got, " ") != "user user" {
		t.Errorf("roles = %v (A's reply must not be stored)", got)
	}
	if row.Status != store.TurnError {
		t.Errorf("row = %s; a still-running turn that lost its tail ends error", row.Status)
	}
	if ops.Outcome.Status != store.TurnError {
		t.Errorf("trace outcome.status = %q, want error (row = %s)", ops.Outcome.Status, row.Status)
	}
	msg := data(t, last(evs))["message"].(string)
	if kinds(evs) != "turn.error" || strings.Contains(msg, "expired") || !strings.Contains(msg, "lost") {
		t.Errorf("events = %s message = %q; want a lost-tail message, not expired", kinds(evs), msg)
	}
	warned := watchWarnings(t, h)
	h.start(h.request("B"))
	if *warned {
		t.Error("WARNING")
	}
}

// DELETE /v1/conversations racing a Start on the same site: whichever
// commits second wins — a Start after the End opens a fresh conversation;
// an End after the Start supersedes the new turn (its loop ends at its
// first row check). Either way exactly the right thing is live.
func TestStart_RacesConversationsEnd(t *testing.T) {
	h := newHarness(t, nil, racingStartScript()...)
	ctx := context.Background()
	convs := Conversations{St: h.st, Logf: t.Logf}
	// End lands inside the Start's StartTurn→history window.
	fired := false
	h.st.mu.Lock()
	h.st.afterStartTurn = func(string) {
		if fired {
			return
		}
		fired = true
		if err := convs.End(ctx, "site-1"); err != nil {
			t.Error(err)
		}
	}
	h.st.mu.Unlock()
	tA, err := h.runner.Start(ctx, h.request("A text"))
	h.st.mu.Lock()
	h.st.afterStartTurn = nil
	h.st.mu.Unlock()
	if err != nil || !fired {
		t.Fatalf("Start = %v fired=%v", err, fired)
	}
	rowA := h.st.turn(tA.ID())
	if rowA.Status != store.TurnSuperseded {
		t.Fatalf("rowA = %s; the DELETE after the Start must supersede it", rowA.Status)
	}
	col := &api.Collector{}
	_ = tA.Run(ctx, col)
	if kinds(col.Events()) != "turn.error" || len(h.fake.Requests()) != 0 {
		t.Errorf("A = %s, model calls = %d", kinds(col.Events()), len(h.fake.Requests()))
	}
	if got := roles(t, h, rowA.ConversationID); len(got) != 0 {
		t.Errorf("the ended conversation holds %v", got)
	}
	// A Start after the End: a fresh conversation, nothing superseded.
	tB, evs := h.start(h.request("B text"))
	if h.st.turn(tB.ID()).ConversationID == rowA.ConversationID || kinds(evs) != "tool.call turn.waiting" {
		t.Errorf("B = %s on %s", kinds(evs), h.st.turn(tB.ID()).ConversationID)
	}
	if live := liveRows(h); len(live) != 1 || live[0] != tB.ID() {
		t.Errorf("live = %v", live)
	}
}

// Round-8 regression (refuter I8): a Start that supersedes a WAITING turn
// (the dangling a(opW) tail on C1 is its to repair) and is itself superseded
// by a DELETE between its StartTurn and its history read. The repair is
// appended into the ENDED C1 (refused: store.ErrConversationEnded) — the
// Start must end like every other lost Start (turn.error turn_expired
// naming the supersede, no model call, nothing stored in C1), never return
// an error (which the handler would render as a 500).
func TestStart_SupersedingWaitingTurnRacesConversationsEnd(t *testing.T) {
	h := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{opUse("opW", "headline", "w")}}),
		textReply("S reply"),
	)
	ctx := context.Background()
	tW, evs := h.start(h.request("W text"))
	if kinds(evs) != "tool.call turn.waiting" {
		t.Fatalf("W = %s", kinds(evs))
	}
	c1 := h.st.turn(tW.ID()).ConversationID
	convs := Conversations{St: h.st, Logf: t.Logf}
	var logs []string
	h.runner.d.Logf = func(f string, a ...any) {
		logs = append(logs, fmt.Sprintf(f, a...))
		t.Logf(f, a...)
	}
	fired := false
	h.st.mu.Lock()
	h.st.afterStartTurn = func(string) {
		if fired {
			return
		}
		fired = true
		if err := convs.End(ctx, "site-1"); err != nil {
			t.Error(err)
		}
	}
	h.st.mu.Unlock()
	tS, err := h.runner.Start(ctx, h.request("S text"))
	h.st.mu.Lock()
	h.st.afterStartTurn = nil
	h.st.mu.Unlock()
	if !fired {
		t.Fatal("hook not fired")
	}
	if err != nil {
		t.Fatalf("Start superseded by a DELETE while repairing a dangling tail failed: %v", err)
	}
	if h.st.turn(tW.ID()).Status != store.TurnSuperseded || h.st.turn(tS.ID()).Status != store.TurnSuperseded {
		t.Fatalf("rowW=%s rowS=%s", h.st.turn(tW.ID()).Status, h.st.turn(tS.ID()).Status)
	}
	col := &api.Collector{}
	_ = tS.Run(ctx, col)
	sEvs := col.Events()
	if kinds(sEvs) != "turn.error" || len(h.fake.Requests()) != 1 {
		t.Errorf("S = %s, model calls = %d (want turn.error, 1)", kinds(sEvs), len(h.fake.Requests()))
	}
	var e struct{ Code, Message string }
	_ = json.Unmarshal(sEvs[len(sEvs)-1].Data, &e)
	if e.Code != "turn_expired" || !strings.Contains(e.Message, "superseded") {
		t.Errorf("S error = %+v", e)
	}
	// Nothing landed in the ended conversation: neither the repair nor S's
	// text; W's batch stays open there (the conversation is over).
	if got := roles(t, h, c1); strings.Join(got, " ") != "user assistant" {
		t.Errorf("roles(c1) = %v", got)
	}
	for _, l := range logs {
		if strings.Contains(l, "already answered by another writer") || strings.Contains(l, "WARNING") {
			t.Errorf("false log line: %q", l)
		}
	}
	if h.runner.Live() != 0 || h.runner.Parked() != 0 {
		t.Errorf("live=%d parked=%d", h.runner.Live(), h.runner.Parked())
	}
	// The site starts fresh afterwards.
	tX, evs := h.start(h.request("X text"))
	if h.st.turn(tX.ID()).ConversationID == c1 || !strings.HasSuffix(kinds(evs), "turn.done") {
		t.Errorf("X = %s on %s", kinds(evs), h.st.turn(tX.ID()).ConversationID)
	}
}

// Round-8 regression (refuter F4, log noise): a Start racing the reaper on
// another instance. The reaper expired the waiting turn W first (so W is
// not in the Start's Superseded set) but its closing append has not landed
// when the Start reads the history: the dangling a(opW) tail is the expected
// outcome of that race, not a write path that failed — the Start repairs it
// with the reaper's own reason and logs no WARNING. Here the reaper's store
// step runs without its append (ExpireStaleTurnsWithPending directly).
func TestStart_DanglingTailOfExpiredTurnIsNotAWarning(t *testing.T) {
	h := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{opUse("opW", "headline", "w")}}),
		textReply("S reply"),
	)
	ctx := context.Background()
	tW, evs := h.start(h.request("W text"))
	if kinds(evs) != "tool.call turn.waiting" {
		t.Fatalf("W = %s", kinds(evs))
	}
	c1 := h.st.turn(tW.ID()).ConversationID
	h.clock = h.clock.Add(11 * time.Minute)
	expired, err := h.st.ExpireStaleTurnsWithPending(ctx, 10*time.Minute)
	if err != nil || len(expired) != 1 || expired[0].ID != tW.ID() {
		t.Fatalf("expire = %v %v", expired, err)
	}
	var logs []string
	h.runner.d.Logf = func(f string, a ...any) {
		logs = append(logs, fmt.Sprintf(f, a...))
		t.Logf(f, a...)
	}
	tS, evs := h.start(h.request("S text"))
	if !strings.HasSuffix(kinds(evs), "turn.done") || h.st.turn(tS.ID()).ConversationID != c1 {
		t.Fatalf("S = %s on %s", kinds(evs), h.st.turn(tS.ID()).ConversationID)
	}
	got := roles(t, h, c1)
	if strings.Join(got, " ") != "user assistant user user assistant" {
		t.Fatalf("roles(c1) = %v", got)
	}
	msgs, _ := h.st.ListMessages(ctx, c1, 0, 0)
	if !strings.Contains(string(msgs[2].Content), reasonExpired) || !strings.Contains(string(msgs[2].Content), `"opW"`) {
		t.Errorf("stored repair = %s, want opW answered with %q", msgs[2].Content, reasonExpired)
	}
	informed := false
	for _, l := range logs {
		if strings.Contains(l, "WARNING") {
			t.Errorf("WARNING logged for the reaper race: %q", l)
		}
		if strings.Contains(l, tW.ID()) && strings.Contains(l, store.TurnExpired) && strings.Contains(l, "unanswered") {
			informed = true
		}
	}
	if !informed {
		t.Errorf("no information line naming turn %s (%s); logs=%q", tW.ID(), store.TurnExpired, logs)
	}
	reqs := h.fake.Requests()
	assertReplayable(t, reqs[len(reqs)-1])
	// The reaper's own closing append, landing now, finds the tail closed.
	if n, err := h.runner.closeDangling(ctx, c1, nil, reasonExpired); n != 0 || err != nil {
		t.Errorf("late reaper close answered %d id(s), err=%v", n, err)
	}
}
