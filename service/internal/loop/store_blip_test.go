package loop

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/breadstack/sous-chef/internal/api"
	"github.com/breadstack/sous-chef/internal/model/modeltest"
	"github.com/breadstack/sous-chef/internal/store"
)

// Round-9 regressions (refuter T1): a transient GetTurn failure at a site
// where the loop re-reads its row status is a storage blip, not a lost row.
// A Start answers 503 store_unavailable (its row finished error, nothing in
// the conversation, no model call); a running turn ends turn.error
// store_unavailable with no further model call. Each test fails GetTurn
// exactly once, at the consulted site, with the memstore's failGetTurns.

func expectStoreUnavailable503(t *testing.T, err error) {
	t.Helper()
	var apiErr *api.Error
	if !errors.As(err, &apiErr) || apiErr.Status != 503 || apiErr.Code != "store_unavailable" {
		t.Fatalf("Start err = %v; want 503 store_unavailable", err)
	}
}

func assertStoreUnavailableTurnError(t *testing.T, evs []api.CollectedEvent) {
	t.Helper()
	if len(evs) == 0 || last(evs).Event != "turn.error" {
		t.Fatalf("events = %s; want a trailing turn.error", kinds(evs))
	}
	if countKind(evs, "tool.call") != 0 {
		t.Errorf("tool.call emitted: %s", kinds(evs))
	}
	var e struct{ Code, Message string }
	_ = json.Unmarshal(last(evs).Data, &e)
	if e.Code != "store_unavailable" || !strings.Contains(e.Message, "storage") {
		t.Errorf("turn.error = %+v; want store_unavailable", e)
	}
}

// Start site 1: the repair of a dangling tail is refused (a DELETE ended the
// conversation between StartTurn and the history read) and the row re-read
// that would name the supersede fails. Before the fix the blip read as
// "expired" and a lostStart turn was handed out; now the Start is 503.
func TestStart_StoreBlipAtRepairStatusCheckIs503(t *testing.T) {
	h := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{opUse("opW", "headline", "w")}}),
		textReply("never sent"),
	)
	ctx := context.Background()
	tW, evs := h.start(h.request("W text"))
	if kinds(evs) != "tool.call turn.waiting" {
		t.Fatalf("W = %s", kinds(evs))
	}
	c1 := h.st.turn(tW.ID()).ConversationID
	convs := Conversations{St: h.st, Logf: t.Logf}
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
		h.st.mu.Lock()
		h.st.failGetTurns = 1
		h.st.mu.Unlock()
	}
	h.st.mu.Unlock()
	_, err := h.runner.Start(ctx, h.request("S text"))
	h.st.mu.Lock()
	h.st.afterStartTurn = nil
	left := h.st.failGetTurns
	h.st.mu.Unlock()
	if !fired || left != 0 {
		t.Fatalf("fired=%v failGetTurns left=%d", fired, left)
	}
	expectStoreUnavailable503(t, err)
	if n := len(h.fake.Requests()); n != 1 {
		t.Errorf("model calls = %d, want 1 (W's only)", n)
	}
	if got := roles(t, h, c1); strings.Join(got, " ") != "user assistant" {
		t.Errorf("roles(c1) = %v; nothing of S may land", got)
	}
	// The refused Start's row is finished error (it was superseded by the
	// DELETE already, so the finish is a no-op there) and nothing is live.
	if live := liveRows(h); len(live) != 0 || h.runner.Live() != 0 || h.runner.Parked() != 0 {
		t.Errorf("live rows=%v live=%d parked=%d", live, h.runner.Live(), h.runner.Parked())
	}
}

// Start site 2: the row check right before the user text fails.
func TestStart_StoreBlipAtSelfCheckIs503(t *testing.T) {
	h := newHarness(t, nil, textReply("never sent"))
	ctx := context.Background()
	var rowID string
	h.st.mu.Lock()
	h.st.afterStartTurn = func(id string) {
		rowID = id
		h.st.mu.Lock()
		h.st.failGetTurns = 1
		h.st.mu.Unlock()
	}
	h.st.mu.Unlock()
	_, err := h.runner.Start(ctx, h.request("A text"))
	h.st.mu.Lock()
	h.st.afterStartTurn = nil
	h.st.mu.Unlock()
	expectStoreUnavailable503(t, err)
	if rowID == "" {
		t.Fatal("hook not fired")
	}
	row := h.st.turn(rowID)
	if row.Status != store.TurnError {
		t.Errorf("row = %s; a refused Start finishes its row error", row.Status)
	}
	if got := roles(t, h, row.ConversationID); len(got) != 0 {
		t.Errorf("conversation holds %v; nothing of the refused Start may be stored", got)
	}
	if n := len(h.fake.Requests()); n != 0 {
		t.Errorf("model calls = %d", n)
	}
	if h.runner.Live() != 0 {
		t.Errorf("live = %d (the slot must be released)", h.runner.Live())
	}
	// The site is usable right after the blip.
	if _, evs := h.start(h.request("B text")); !strings.HasSuffix(kinds(evs), "turn.done") {
		t.Errorf("next Start = %s", kinds(evs))
	}
}

// Start site 3: the user-text append loses the tail to a foreign write that
// landed after the row check (row still running) and the status re-read
// fails. Before the fix the blip read as "expired" → lostStart.
func TestStart_StoreBlipAtUserTextStatusCheckIs503(t *testing.T) {
	h := newHarness(t, nil, textReply("never sent"))
	ctx := context.Background()
	var rowID string
	h.st.mu.Lock()
	h.st.afterGetTurn = func(id string) {
		if rowID != "" {
			return
		}
		rowID = id
		convID := h.st.turn(id).ConversationID
		if _, err := h.st.AppendMessage(ctx, convID, "user", json.RawMessage(`[{"type":"text","text":"foreign"}]`)); err != nil {
			t.Error(err)
		}
		h.st.mu.Lock()
		h.st.failGetTurns = 1
		h.st.mu.Unlock()
	}
	h.st.mu.Unlock()
	_, err := h.runner.Start(ctx, h.request("A text"))
	h.st.mu.Lock()
	h.st.afterGetTurn = nil
	left := h.st.failGetTurns
	h.st.mu.Unlock()
	if rowID == "" || left != 0 {
		t.Fatalf("fired=%v failGetTurns left=%d", rowID != "", left)
	}
	expectStoreUnavailable503(t, err)
	row := h.st.turn(rowID)
	if row.Status != store.TurnError {
		t.Errorf("row = %s", row.Status)
	}
	if got := roles(t, h, row.ConversationID); strings.Join(got, " ") != "user" {
		t.Errorf("roles = %v; only the foreign text may be stored", got)
	}
	if n := len(h.fake.Requests()); n != 0 {
		t.Errorf("model calls = %d", n)
	}
}

// Run site 1: SetTurnWaiting finds the row no longer running (superseded
// while the model call was in flight) and the re-read fails. No tool.call,
// no reason guessed: the stored batch is left open and the NEXT Start
// attributes it to the superseded row (INFO, reasonSuperseded, no WARNING).
func TestRun_StoreBlipAtSetTurnWaitingStatusCheck(t *testing.T) {
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
		h.st.mu.Lock()
		h.st.failGetTurns = 1
		h.st.mu.Unlock()
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
	assertStoreUnavailableTurnError(t, col.Events())
	if n := len(h.fake.Requests()); n != 1 {
		t.Errorf("model calls = %d, want 1", n)
	}
	if got := roles(t, h, convID); strings.Join(got, " ") != "user assistant" {
		t.Errorf("roles = %v; the batch is left for the next Start, not answered on a guess", got)
	}
	if h.st.turn(tA.ID()).Status != store.TurnSuperseded || h.runner.Parked() != 0 || h.runner.Live() != 0 {
		t.Errorf("row=%s parked=%d live=%d", h.st.turn(tA.ID()).Status, h.runner.Parked(), h.runner.Live())
	}
	warned := watchWarnings(t, h)
	_, evs := h.start(h.request("B"))
	if !strings.HasSuffix(kinds(evs), "turn.done") {
		t.Errorf("B = %s", kinds(evs))
	}
	if *warned {
		t.Error("WARNING: the superseded row owns the tail, that is information")
	}
	msgs, _ := h.st.ListMessages(ctx, convID, 0, 0)
	if got := roles(t, h, convID); strings.Join(got, " ") != "user assistant user user assistant" {
		t.Errorf("roles after B = %v", got)
	}
	if !strings.Contains(string(msgs[2].Content), reasonSuperseded) || !strings.Contains(string(msgs[2].Content), `"opA"`) {
		t.Errorf("repair = %s; want opA answered with %q", msgs[2].Content, reasonSuperseded)
	}
	for _, r := range h.fake.Requests() {
		assertReplayable(t, r)
	}
}

// Run site 2: FinishTurn finds the row no longer running after the final
// reply and the re-read fails: turn.error store_unavailable, not turn.done
// and not a guessed turn_expired.
func TestRun_StoreBlipAtFinishStatusCheck(t *testing.T) {
	h := newHarness(t, nil, textReply("final"))
	ctx := context.Background()
	tA, err := h.runner.Start(ctx, h.request("A text"))
	if err != nil {
		t.Fatal(err)
	}
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
		h.st.mu.Lock()
		h.st.failGetTurns = 1
		h.st.mu.Unlock()
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
	assertStoreUnavailableTurnError(t, evs)
	if countKind(evs, "turn.done") != 0 {
		t.Errorf("turn.done on a row storage says did not complete: %s", kinds(evs))
	}
	if n := len(h.fake.Requests()); n != 1 {
		t.Errorf("model calls = %d, want 1", n)
	}
	tr, _ := h.st.trace(tA.ID())
	var ops struct {
		Outcome struct{ Status string } `json:"outcome"`
	}
	_ = json.Unmarshal(tr.Ops, &ops)
	if ops.Outcome.Status != store.TurnError {
		t.Errorf("trace outcome.status = %q, want error (the exit could not be named)", ops.Outcome.Status)
	}
}

// Run site 3: the assistant append loses the tail (a foreign write between
// the pre-call row check and the append) and the re-read in fail() fails.
// The reply is not stored, no second model call is made, and the row — still
// running — is finished error.
func TestRun_StoreBlipAtLostTailStatusCheck(t *testing.T) {
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
		h.st.mu.Lock()
		h.st.failGetTurns = 1
		h.st.mu.Unlock()
	}
	h.st.mu.Unlock()
	col := &api.Collector{}
	_ = turn.Run(ctx, col)
	h.st.mu.Lock()
	h.st.afterGetTurn = nil
	left := h.st.failGetTurns
	h.st.mu.Unlock()
	if !fired || left != 0 {
		t.Fatalf("fired=%v failGetTurns left=%d", fired, left)
	}
	assertStoreUnavailableTurnError(t, col.Events())
	if n := len(h.fake.Requests()); n != 1 {
		t.Errorf("model calls = %d, want 1", n)
	}
	if got := roles(t, h, convID); strings.Join(got, " ") != "user user" {
		t.Errorf("roles = %v (A's reply must not be stored)", got)
	}
	if h.st.turn(turn.ID()).Status != store.TurnError {
		t.Errorf("row = %s; a still-running turn is finished error", h.st.turn(turn.ID()).Status)
	}
	warned := watchWarnings(t, h)
	h.start(h.request("B"))
	if *warned {
		t.Error("WARNING")
	}
}
