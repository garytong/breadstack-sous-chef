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

// assertReplayable fails when a captured request's messages would be
// rejected by the Messages API: a tool_use id with no tool_result after
// it, a tool_result answering nothing, or an id answered twice.
func assertReplayable(t *testing.T, c modeltest.Captured) {
	t.Helper()
	open := map[string]bool{}
	answered := map[string]int{}
	for i, m := range c.Body["messages"].([]any) {
		msg := m.(map[string]any)
		blocks, _ := msg["content"].([]any)
		for _, b := range blocks {
			blk, ok := b.(map[string]any)
			if !ok {
				continue
			}
			switch blk["type"] {
			case "tool_use":
				open[blk["id"].(string)] = true
			case "tool_result":
				id := blk["tool_use_id"].(string)
				if !open[id] {
					t.Errorf("messages[%d]: tool_result %q answers no open tool_use", i, id)
				}
				delete(open, id)
				answered[id]++
			}
		}
	}
	for id := range open {
		t.Errorf("request leaves tool_use %q unanswered", id)
	}
	for id, n := range answered {
		if n > 1 {
			t.Errorf("tool_use %q answered %d times", id, n)
		}
	}
}

// storedTail returns the conversation's messages and the tool_result blocks
// of its last message.
func storedTail(t *testing.T, h *harness, convID string) ([]store.Message, []capturedResult) {
	t.Helper()
	msgs, err := h.st.ListMessages(context.Background(), convID, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) == 0 {
		return msgs, nil
	}
	var blocks []map[string]any
	_ = json.Unmarshal(msgs[len(msgs)-1].Content, &blocks)
	var out []capturedResult
	for _, b := range blocks {
		if b["type"] != "tool_result" {
			continue
		}
		content := ""
		switch cv := b["content"].(type) {
		case string:
			content = cv
		case []any:
			for _, x := range cv {
				content += x.(map[string]any)["text"].(string)
			}
		}
		isErr, _ := b["is_error"].(bool)
		out = append(out, capturedResult{ID: b["tool_use_id"].(string), Content: content, IsError: isErr})
	}
	return msgs, out
}

func textReply(s string) modeltest.Step {
	return modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text(s)}})
}

// P1: max_calls persists the last assistant tool_use turn; its ids must be
// answered is_error so the next turn's request is valid.
func TestDangling_MaxCalls(t *testing.T) {
	// Two all-server batches loop; the third (last allowed) call wants a
	// browser op too and is never dispatched.
	var steps []modeltest.Step
	for _, id := range []string{"lk1", "lk2"} {
		steps = append(steps, modeltest.Reply(modeltest.Message{Content: []modeltest.Block{
			modeltest.ToolUse(id, "lookup_fields", map[string]any{"target": heroTarget, "query": "x"}),
		}}))
	}
	steps = append(steps, modeltest.Reply(modeltest.Message{Content: []modeltest.Block{
		modeltest.ToolUse("lk3", "lookup_fields", map[string]any{"target": heroTarget, "query": "x"}),
		modeltest.ToolUse("op", "set_fields", map[string]any{"target": heroTarget, "values": map[string]any{"headline": "x"}}),
	}}))
	steps = append(steps, textReply("next turn"))
	h := newHarness(t, func(d *Deps) { d.MaxCalls = 3 }, steps...)
	turn, evs := h.start(h.request("loop"))
	if data(t, last(evs))["stop"] != "max_calls" {
		t.Fatalf("events = %s", kinds(evs))
	}
	msgs, res := storedTail(t, h, h.st.turn(turn.ID()).ConversationID)
	// user, (assistant, user-results)×2 answered normally, assistant, user-abandon.
	if len(msgs) != 7 || msgs[5].Role != "assistant" || msgs[6].Role != "user" {
		t.Fatalf("stored %d messages; tail roles %s %s", len(msgs), msgs[len(msgs)-2].Role, msgs[len(msgs)-1].Role)
	}
	if len(res) != 2 || res[0].ID != "lk3" || res[1].ID != "op" || !res[0].IsError || !res[1].IsError || !strings.Contains(res[0].Content, reasonMaxCalls) {
		t.Errorf("abandon results = %+v", res)
	}
	_, evs2 := h.start(h.request("again"))
	if !strings.HasSuffix(kinds(evs2), "turn.done") {
		t.Fatalf("next turn: %s", kinds(evs2))
	}
	reqs := h.fake.Requests()
	assertReplayable(t, reqs[len(reqs)-1])
	// 8 stored messages; the abandon results and the new text go as one
	// user message (prompt.MergeUserRuns).
	if n := len(reqs[len(reqs)-1].Body["messages"].([]any)); n != 7 {
		t.Errorf("next request replayed %d messages, want 7", n)
	}
	wantResultsThenText(t, reqs[len(reqs)-1])
}

// P1: a stop other than tool_use (end_turn here, max_tokens below) that
// still carries tool_use blocks runs nothing — the blocks are answered
// is_error and the turn ends normally.
func TestDangling_EndTurnWithToolUse(t *testing.T) {
	h := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{StopReason: "end_turn", Content: []modeltest.Block{
			modeltest.Text("Changing it."),
			modeltest.ToolUse("op1", "set_fields", map[string]any{"target": heroTarget, "values": map[string]any{"headline": "x"}}),
		}}),
		textReply("second"),
		modeltest.Reply(modeltest.Message{StopReason: "max_tokens", Content: []modeltest.Block{
			modeltest.ToolUse("pa1", "pick_asset", map[string]any{"purpose": "x"}),
		}}),
		textReply("fourth"),
	)
	turn, evs := h.start(h.request("x"))
	if strings.Contains(kinds(evs), "tool.call") || strings.Contains(kinds(evs), "turn.waiting") || data(t, last(evs))["stop"] != "end_turn" {
		t.Fatalf("events = %s", kinds(evs))
	}
	if h.st.turn(turn.ID()).Status != store.TurnDone || h.runner.Parked() != 0 {
		t.Errorf("status=%s parked=%d", h.st.turn(turn.ID()).Status, h.runner.Parked())
	}
	msgs, res := storedTail(t, h, h.st.turn(turn.ID()).ConversationID)
	if len(msgs) != 3 || len(res) != 1 || res[0].ID != "op1" || !res[0].IsError || !strings.Contains(res[0].Content, reasonNoToolStop) {
		t.Errorf("stored=%d tail=%+v", len(msgs), res)
	}
	h.start(h.request("y"))
	assertReplayable(t, h.fake.Requests()[1])

	_, evs = h.start(h.request("z"))
	if data(t, last(evs))["stop"] != "max_tokens" || strings.Contains(kinds(evs), "turn.suspended") {
		t.Fatalf("max_tokens with pick_asset: %s", kinds(evs))
	}
	h.start(h.request("w"))
	assertReplayable(t, h.fake.Requests()[3])
}

// P1: TTL expiry closes the tail from the stored pending doc on whichever
// instance reaps (here: the parked entry was dropped, as on another
// instance). Stashed server results are kept; browser ids are is_error.
func TestDangling_TTLExpiryFromPendingDoc(t *testing.T) {
	h := newHarness(t, func(d *Deps) { d.TurnTTL = time.Minute },
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{
			modeltest.ToolUse("lk1", "lookup_fields", map[string]any{"target": heroTarget, "query": "headline"}),
			modeltest.ToolUse("op1", "set_fields", map[string]any{"target": heroTarget, "values": map[string]any{"headline": "x"}}),
			modeltest.ToolUse("pa1", "pick_asset", map[string]any{"purpose": "bg"}),
		}}),
		textReply("after"),
	)
	turn, evs := h.start(h.request("x"))
	if !strings.HasSuffix(kinds(evs), "turn.suspended") {
		t.Fatalf("events = %s", kinds(evs))
	}
	h.runner.mu.Lock()
	delete(h.runner.parked, turn.ID())
	h.runner.mu.Unlock()

	h.clock = h.clock.Add(2 * time.Minute)
	n, err := h.runner.Reap(context.Background())
	if err != nil || n != 1 || h.st.turn(turn.ID()).Status != store.TurnExpired {
		t.Fatalf("reap = %d %v status=%s", n, err, h.st.turn(turn.ID()).Status)
	}
	msgs, res := storedTail(t, h, h.st.turn(turn.ID()).ConversationID)
	if len(msgs) != 3 || len(res) != 3 || res[0].ID != "lk1" || res[1].ID != "op1" || res[2].ID != "pa1" {
		t.Fatalf("stored=%d tail=%+v", len(msgs), res)
	}
	if res[0].IsError || !strings.Contains(res[0].Content, "headline") {
		t.Errorf("stashed server result not kept: %+v", res[0])
	}
	if !res[1].IsError || !res[2].IsError || !strings.Contains(res[1].Content, reasonExpired) {
		t.Errorf("browser ids not is_error: %+v %+v", res[1], res[2])
	}
	// A second reap appends nothing more.
	if n, _ := h.runner.Reap(context.Background()); n != 0 {
		t.Errorf("second reap = %d", n)
	}
	if msgs, _ := storedTail(t, h, h.st.turn(turn.ID()).ConversationID); len(msgs) != 3 {
		t.Errorf("second reap appended: %d messages", len(msgs))
	}
	h.start(h.request("y"))
	assertReplayable(t, h.fake.Requests()[1])
}

// P1: a running turn (no pending doc: the instance died after persisting
// the assistant turn) expires too; the reaper reads the tail instead.
func TestDangling_TTLExpiryRunningTurn(t *testing.T) {
	h := newHarness(t, func(d *Deps) { d.TurnTTL = time.Minute }, textReply("after"))
	ctx := context.Background()
	conv, _ := h.st.OpenConversation(ctx, "site-1", "store-1")
	row, _ := h.st.CreateTurn(ctx, store.NewTurn{ConversationID: conv.ID, ManifestHash: h.hash, Model: "m"})
	_, _ = h.st.AppendMessage(ctx, conv.ID, "user", json.RawMessage(`[{"type":"text","text":"<context>\n{}\n</context>\nhi"}]`))
	_, _ = h.st.AppendMessage(ctx, conv.ID, "assistant", json.RawMessage(`[{"type":"tool_use","id":"op9","name":"set_fields","input":{}}]`))

	h.clock = h.clock.Add(2 * time.Minute)
	if n, err := h.runner.Reap(ctx); err != nil || n != 1 || h.st.turn(row.ID).Status != store.TurnExpired {
		t.Fatalf("reap = %d %v", n, err)
	}
	msgs, res := storedTail(t, h, conv.ID)
	if len(msgs) != 3 || len(res) != 1 || res[0].ID != "op9" || !res[0].IsError {
		t.Fatalf("stored=%d tail=%+v", len(msgs), res)
	}
	h.start(h.request("y"))
	assertReplayable(t, h.fake.Requests()[0])
}

// P1: the reaper expires the row while the loop is mid-call; SetTurnWaiting
// then fails and the loop must close its own tail (the reaper saw no
// assistant turn yet). Exactly one is_error message results.
func TestDangling_ReaperBeatsSetTurnWaiting(t *testing.T) {
	h := newHarness(t, func(d *Deps) { d.TurnTTL = time.Minute },
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{
			modeltest.ToolUse("lk1", "lookup_fields", map[string]any{"target": heroTarget, "query": "headline"}),
			modeltest.ToolUse("op1", "set_fields", map[string]any{"target": heroTarget, "values": map[string]any{"headline": "x"}}),
		}}),
		textReply("after"),
	)
	turn, err := h.runner.Start(context.Background(), h.request("x"))
	if err != nil {
		t.Fatal(err)
	}
	// Expire the row during the model call — after the loop's pre-call row
	// check, before it persists anything: a reaper on another instance
	// racing a slow model.
	fired := false
	h.st.mu.Lock()
	h.st.afterGetTurn = func(id string) {
		if fired || id != turn.ID() {
			return
		}
		fired = true
		h.clock = h.clock.Add(2 * time.Minute)
		if n, _ := h.runner.Reap(context.Background()); n != 1 {
			t.Errorf("reap = %d", n)
		}
	}
	h.st.mu.Unlock()
	col := &api.Collector{}
	_ = turn.Run(context.Background(), col)
	h.st.mu.Lock()
	h.st.afterGetTurn = nil
	h.st.mu.Unlock()
	if !fired {
		t.Fatal("the loop never read its row before calling the model")
	}
	// No tool.call: the row is stored waiting BEFORE the batch is streamed,
	// and the status guard failed, so the page never sees op1.
	if kinds(col.Events()) != "turn.error" || data(t, last(col.Events()))["code"] != "turn_expired" {
		t.Fatalf("events = %s %v", kinds(col.Events()), data(t, last(col.Events())))
	}
	msgs, res := storedTail(t, h, h.st.turn(turn.ID()).ConversationID)
	if len(msgs) != 3 || len(res) != 2 || res[0].ID != "lk1" || res[0].IsError || res[1].ID != "op1" || !res[1].IsError {
		t.Fatalf("stored=%d tail=%+v", len(msgs), res)
	}
	if h.runner.Parked() != 0 {
		t.Errorf("parked = %d", h.runner.Parked())
	}
	h.start(h.request("y"))
	assertReplayable(t, h.fake.Requests()[1])
}

// P1: Continue fails after ResumeTurn (the results write fails): the turn
// must not stay running, and the batch is answered is_error.
func TestDangling_ContinueFailsAfterResume(t *testing.T) {
	h := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{
			modeltest.ToolUse("lk1", "lookup_fields", map[string]any{"target": heroTarget, "query": "headline"}),
			modeltest.ToolUse("op1", "set_fields", map[string]any{"target": heroTarget, "values": map[string]any{"headline": "x"}}),
		}}),
		textReply("after"),
	)
	turn, _ := h.start(h.request("x"))
	h.st.mu.Lock()
	h.st.failAppends = 1
	h.st.mu.Unlock()
	_, err := h.cont(turn.ID(), api.ToolResult{ToolCallID: "op1", OK: true})
	if err == nil || !strings.Contains(err.Error(), "injected") {
		t.Fatalf("Continue = %v", err)
	}
	row := h.st.turn(turn.ID())
	if row.Status != store.TurnError || row.PendingToolCalls != nil || h.runner.Live() != 0 || h.runner.Parked() != 0 {
		t.Fatalf("row=%s pending=%s live=%d parked=%d", row.Status, row.PendingToolCalls, h.runner.Live(), h.runner.Parked())
	}
	msgs, res := storedTail(t, h, row.ConversationID)
	if len(msgs) != 3 || len(res) != 2 || res[0].ID != "lk1" || res[0].IsError || res[1].ID != "op1" || !res[1].IsError || !strings.Contains(res[1].Content, reasonContinueFail) {
		t.Fatalf("stored=%d tail=%+v", len(msgs), res)
	}
	tr, ok := h.st.trace(turn.ID())
	if !ok || !strings.Contains(string(tr.Ops), `"status":"error"`) {
		t.Errorf("trace = %v %s", ok, tr.Ops)
	}
	h.start(h.request("y"))
	assertReplayable(t, h.fake.Requests()[1])

	// Both writes fail: the stored tail stays dangling and the read-side
	// guard still produces a valid next request.
	h2 := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{
			modeltest.ToolUse("op1", "set_fields", map[string]any{"target": heroTarget, "values": map[string]any{"headline": "x"}}),
		}}),
		textReply("after"),
	)
	turn2, _ := h2.start(h2.request("x"))
	h2.st.mu.Lock()
	h2.st.failAppends = 2
	h2.st.mu.Unlock()
	if _, err := h2.cont(turn2.ID(), api.ToolResult{ToolCallID: "op1", OK: true}); err == nil {
		t.Fatal("Continue succeeded with a failing store")
	}
	if msgs, _ := storedTail(t, h2, h2.st.turn(turn2.ID()).ConversationID); len(msgs) != 2 || msgs[1].Role != "assistant" {
		t.Fatalf("expected a dangling stored tail, got %d messages", len(msgs))
	}
	var logged bool
	h2.runner.d.Logf = func(f string, a ...any) {
		if strings.Contains(f, "unanswered tool_use") {
			logged = true
		}
		t.Logf(f, a...)
	}
	_, evs := h2.start(h2.request("y"))
	if !strings.HasSuffix(kinds(evs), "turn.done") || !logged {
		t.Fatalf("guarded turn: %s logged=%v", kinds(evs), logged)
	}
	reqs := h2.fake.Requests()
	assertReplayable(t, reqs[len(reqs)-1])
	if n := len(reqs[len(reqs)-1].Body["messages"].([]any)); n != 3 {
		t.Errorf("guarded request has %d messages, want 3 (user, assistant, synthetic results + user text)", n)
	}
	wantResultsThenText(t, reqs[len(reqs)-1])
}

// P4: a parked turn that another instance continued and finished is
// evicted by the reaper; so are entries past TurnTTL when the database
// cannot be asked.
func TestReap_EvictsParkedTurns(t *testing.T) {
	h := newHarness(t, func(d *Deps) { d.TurnTTL = time.Minute },
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{
			modeltest.ToolUse("op1", "set_fields", map[string]any{"target": heroTarget, "values": map[string]any{"headline": "x"}}),
		}}),
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{
			modeltest.ToolUse("op2", "set_fields", map[string]any{"target": heroTarget, "values": map[string]any{"headline": "y"}}),
		}}),
	)
	ctx := context.Background()
	a, _ := h.start(h.request("x"))
	// The second turn is another site's (a Start supersedes every live turn
	// of its own site, README §4), so both stay parked.
	reqB := h.request("x")
	reqB.SiteID, reqB.StoreID = "site-2", "store-2"
	b, _ := h.start(reqB)
	if h.runner.Parked() != 2 {
		t.Fatalf("parked = %d", h.runner.Parked())
	}
	// Instance B continues and finishes turn a.
	if _, err := h.st.ResumeTurn(ctx, a.ID()); err != nil {
		t.Fatal(err)
	}
	_ = h.st.FinishTurn(ctx, a.ID(), store.TurnDone, json.RawMessage(`{}`))
	n, err := h.runner.Reap(ctx)
	if err != nil || n != 1 || h.runner.Parked() != 1 {
		t.Fatalf("reap = %d %v parked=%d", n, err, h.runner.Parked())
	}
	if tr, ok := h.st.trace(a.ID()); !ok || !strings.Contains(string(tr.Ops), `"status":"done"`) || !strings.Contains(string(tr.Ops), "another instance") {
		t.Errorf("trace a = %v %s", ok, tr.Ops)
	}
	if _, err := h.cont(a.ID(), api.ToolResult{ToolCallID: "op1", OK: true}); err == nil {
		t.Error("continuing a finished turn succeeded")
	}
	// Running elsewhere: still parked (ours again if it comes back waiting).
	if _, err := h.st.ResumeTurn(ctx, b.ID()); err != nil {
		t.Fatal(err)
	}
	if n, _ := h.runner.Reap(ctx); n != 0 || h.runner.Parked() != 1 {
		t.Errorf("running elsewhere: reap=%d parked=%d", n, h.runner.Parked())
	}
	// Database down + TTL passed: aged out regardless.
	h.st.mu.Lock()
	h.st.failDB = true
	h.st.mu.Unlock()
	h.clock = h.clock.Add(2 * time.Minute)
	n, err = h.runner.Reap(ctx)
	if err == nil || n != 1 || h.runner.Parked() != 0 {
		t.Fatalf("db down: reap=%d err=%v parked=%d", n, err, h.runner.Parked())
	}
	if tr, ok := h.st.trace(b.ID()); !ok || !strings.Contains(string(tr.Ops), `"status":"expired"`) {
		t.Errorf("trace b = %v %s", ok, tr.Ops)
	}
	// A parked row that vanished is evicted too (b is finished first so the
	// database has nothing of its own to expire).
	h.st.mu.Lock()
	h.st.failDB = false
	h.st.mu.Unlock()
	_ = h.st.FinishTurn(ctx, b.ID(), store.TurnError, nil)
	h.fake.Append(modeltest.Reply(modeltest.Message{Content: []modeltest.Block{
		modeltest.ToolUse("op3", "set_fields", map[string]any{"target": heroTarget, "values": map[string]any{"headline": "z"}}),
	}}))
	_, _ = h.st.EndConversation(ctx, "site-1")
	c, _ := h.start(h.request("x"))
	h.st.mu.Lock()
	delete(h.st.turns, c.ID())
	h.st.mu.Unlock()
	if n, err := h.runner.Reap(ctx); err != nil || n != 1 || h.runner.Parked() != 0 {
		t.Errorf("vanished row: reap=%d err=%v parked=%d", n, err, h.runner.Parked())
	}
}

// P6: two pick_asset calls in one batch — the first suspends, the extra is
// answered server-side is_error; pick_asset's pending seq is the
// turn.suspended event's seq even when it is not the last call.
func TestPickAsset_MultipleInOneBatch(t *testing.T) {
	h := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{
			modeltest.ToolUse("pa1", "pick_asset", map[string]any{"purpose": "Hero background", "asset_type": "image"}),
			modeltest.ToolUse("op1", "set_fields", map[string]any{"target": heroTarget, "values": map[string]any{"headline": "x"}}),
			modeltest.ToolUse("pa2", "pick_asset", map[string]any{"purpose": "Logo", "asset_type": "image"}),
		}}),
		textReply("Done."),
	)
	turn, evs := h.start(h.request("x"))
	if kinds(evs) != "tool.call turn.suspended" {
		t.Fatalf("events = %s", kinds(evs))
	}
	s := data(t, last(evs))
	if s["toolCallId"] != "pa1" || s["purpose"] != "Hero background" || s["seq"] != 2.0 {
		t.Errorf("suspended = %v", s)
	}
	if ids, _ := json.Marshal(s["toolCallIds"]); string(ids) != `["pa1","op1"]` {
		t.Errorf("toolCallIds = %s", ids)
	}
	var pd pendingDoc
	if err := json.Unmarshal(h.st.turn(turn.ID()).PendingToolCalls, &pd); err != nil {
		t.Fatal(err)
	}
	if pd.SuspendedID != "pa1" || len(pd.Calls) != 3 || pd.Calls[0].Seq != 2 || pd.Calls[1].Seq != 1 || !pd.Calls[2].Server || pd.Calls[2].Seq != 0 {
		t.Errorf("pending = %+v", pd.Calls)
	}
	if sr, ok := pd.serverResult("pa2"); !ok || !sr.IsError || !strings.Contains(sr.Content, "one pick_asset") {
		t.Errorf("pa2 stash = %+v %v", sr, ok)
	}
	// pa2 is not a browser id: answering it is a 400.
	if _, err := h.cont(turn.ID(), api.ToolResult{ToolCallID: "pa2", OK: true}); err == nil {
		t.Error("pa2 accepted as a browser result")
	}
	evs2, err := h.cont(turn.ID(),
		api.ToolResult{ToolCallID: "pa1", OK: true, Asset: &api.Asset{PublicURL: "https://cdn/x.jpg", Alt: "a"}},
		api.ToolResult{ToolCallID: "op1", OK: true, Written: []string{"headline"}},
	)
	if err != nil || !strings.HasSuffix(kinds(evs2), "turn.done") {
		t.Fatalf("continue: %v %s", err, kinds(evs2))
	}
	reqs := h.fake.Requests()
	res := lastToolResults(t, reqs[1])
	if len(res) != 3 || res[0].ID != "pa1" || res[0].IsError || res[1].ID != "op1" || res[1].IsError || res[2].ID != "pa2" || !res[2].IsError || !strings.Contains(res[2].Content, reasonExtraPick) {
		t.Errorf("results = %+v", res)
	}
	assertReplayable(t, reqs[1])
	tr, _ := h.st.trace(turn.ID())
	var evsDoc []struct {
		Seq  int             `json:"seq"`
		Kind string          `json:"event"`
		Data json.RawMessage `json:"data"`
	}
	if err := json.Unmarshal(tr.Events, &evsDoc); err != nil {
		t.Fatal(err)
	}
	var forSeqs []int
	for _, e := range evsDoc {
		if e.Kind == "tool.result" {
			var d struct {
				ForSeq int `json:"for_seq"`
			}
			_ = json.Unmarshal(e.Data, &d)
			forSeqs = append(forSeqs, d.ForSeq)
		}
	}
	// pa1 answers the suspended event (seq 2), op1 its tool.call (seq 1).
	if len(forSeqs) != 2 || forSeqs[0] != 2 || forSeqs[1] != 1 {
		t.Errorf("tool.result for_seq = %v (events %s)", forSeqs, tr.Events)
	}
	var ops struct {
		Ops []struct {
			ID     string `json:"tool_use_id"`
			Server bool   `json:"server"`
		} `json:"ops"`
	}
	_ = json.Unmarshal(tr.Ops, &ops)
	if len(ops.Ops) != 3 || ops.Ops[2].ID != "pa2" || !ops.Ops[2].Server || ops.Ops[0].Server {
		t.Errorf("trace ops = %+v", ops.Ops)
	}
}

// P6: trace tool.result events continue the turn's seq instead of
// restarting from the recorder's event count (which collides once
// text.delta events are coalesced).
func TestTrace_ToolResultSeqContinues(t *testing.T) {
	h := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{
			modeltest.Text("Let "), modeltest.Text("me "), modeltest.Text("change that."),
			modeltest.ToolUse("op1", "set_fields", map[string]any{"target": heroTarget, "values": map[string]any{"headline": "x"}}),
			modeltest.ToolUse("op2", "set_fields", map[string]any{"target": heroTarget, "values": map[string]any{"subtitle": "y"}}),
		}}),
		textReply("Done."),
	)
	turn, evs := h.start(h.request("x"))
	waiting := data(t, last(evs))["seq"].(float64)
	evs2, err := h.cont(turn.ID(), api.ToolResult{ToolCallID: "op1", OK: true}, api.ToolResult{ToolCallID: "op2", OK: true})
	if err != nil {
		t.Fatal(err)
	}
	// The continuation's first event comes after the two tool.result seqs.
	if first := data(t, evs2[0])["seq"].(float64); first != waiting+3 {
		t.Errorf("first continuation seq = %v, want %v", first, waiting+3)
	}
	tr, _ := h.st.trace(turn.ID())
	var evsDoc []struct {
		Seq  int    `json:"seq"`
		Kind string `json:"event"`
	}
	if err := json.Unmarshal(tr.Events, &evsDoc); err != nil {
		t.Fatal(err)
	}
	seen := map[int]string{}
	prev := 0
	for _, e := range evsDoc {
		if other, dup := seen[e.Seq]; dup {
			t.Errorf("seq %d used by %s and %s", e.Seq, other, e.Kind)
		}
		seen[e.Seq] = e.Kind
		if e.Seq <= prev {
			t.Errorf("seq %d (%s) does not follow %d", e.Seq, e.Kind, prev)
		}
		prev = e.Seq
	}
	if seen[int(waiting)+1] != "tool.result" || seen[int(waiting)+2] != "tool.result" {
		t.Errorf("tool.result seqs: %v", seen)
	}
}

// --- round-3 regressions (refuter N1/N2/N3) ---

func opUse(id, key, val string) modeltest.Block {
	return modeltest.ToolUse(id, "set_fields", map[string]any{"target": heroTarget, "values": map[string]any{key: val}})
}

// roles lists the stored message roles of a conversation.
func roles(t *testing.T, h *harness, convID string) []string {
	t.Helper()
	msgs, err := h.st.ListMessages(context.Background(), convID, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]string, 0, len(msgs))
	for _, m := range msgs {
		out = append(out, m.Role)
	}
	return out
}

// N1: a stored tail that dangles (both the results write and the abandon
// write failed) is repaired IN STORAGE when the next turn starts, not only
// in that turn's request — otherwise the turn after it replays
// [user, assistant(tool_use), user(text), assistant, user] and the Messages
// API rejects every later request until rotation.
func TestDangling_RepairPersistedForNextTurn(t *testing.T) {
	h := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{opUse("op1", "headline", "x")}}),
		textReply("guarded reply"),
		textReply("third reply"),
	)
	turn, _ := h.start(h.request("x"))
	h.st.mu.Lock()
	h.st.failAppends = 2 // results write AND abandon write fail → stored tail dangles
	h.st.mu.Unlock()
	if _, err := h.cont(turn.ID(), api.ToolResult{ToolCallID: "op1", OK: true}); err == nil {
		t.Fatal("Continue succeeded with a failing store")
	}
	convID := h.st.turn(turn.ID()).ConversationID
	if got := roles(t, h, convID); len(got) != 2 || got[1] != "assistant" {
		t.Fatalf("expected a dangling stored tail, got %v", got)
	}
	_, evs := h.start(h.request("y"))
	if !strings.HasSuffix(kinds(evs), "turn.done") {
		t.Fatalf("guarded turn: %s", kinds(evs))
	}
	assertReplayable(t, h.fake.Requests()[1])
	// The repair is now stored, between the dangling tail and the user text.
	got := roles(t, h, convID)
	want := []string{"user", "assistant", "user", "user", "assistant"}
	if strings.Join(got, " ") != strings.Join(want, " ") {
		t.Fatalf("stored roles after the guarded turn = %v, want %v", got, want)
	}
	msgs, _ := h.st.ListMessages(context.Background(), convID, 0, 0)
	var blocks []map[string]any
	_ = json.Unmarshal(msgs[2].Content, &blocks)
	if len(blocks) != 1 || blocks[0]["type"] != "tool_result" || blocks[0]["tool_use_id"] != "op1" || blocks[0]["is_error"] != true {
		t.Fatalf("stored repair = %s", msgs[2].Content)
	}
	// The turn AFTER the guarded one replays a valid history with no repair.
	var warned bool
	h.runner.d.Logf = func(f string, a ...any) {
		if strings.Contains(f, "unanswered tool_use") {
			warned = true
		}
		t.Logf(f, a...)
	}
	_, evs = h.start(h.request("z"))
	if !strings.HasSuffix(kinds(evs), "turn.done") {
		t.Fatalf("third turn: %s", kinds(evs))
	}
	reqs := h.fake.Requests()
	assertReplayable(t, reqs[len(reqs)-1])
	if warned {
		t.Error("the third turn still needed the read-side repair: the fix was not persisted")
	}
	// 6 stored; the persisted repair and the second turn's text go as one
	// user message (prompt.MergeUserRuns).
	if n := len(reqs[len(reqs)-1].Body["messages"].([]any)); n != 5 {
		t.Errorf("third request has %d messages, want 5", n)
	}
}

// N1 (read side): repairHistory answers an unanswered tool_use anywhere in
// the history, not only at the tail, and reports the tail repair alone.
func TestRepairHistory_WholeHistory(t *testing.T) {
	h := newHarness(t, nil)
	toolUse := func(id string) anthropic.BetaMessageParam {
		return anthropic.BetaMessageParam{Role: anthropic.BetaMessageParamRoleAssistant,
			Content: []anthropic.BetaContentBlockParamUnion{anthropic.NewBetaToolUseBlock(id, map[string]any{}, "set_fields")}}
	}
	mid := []anthropic.BetaMessageParam{
		model.UserTextTurn("a"),
		toolUse("t1"),
		model.UserTextTurn("b"), // t1 never answered
		toolUse("t2"),
		model.ToolResultsTurn([]model.ToolResult{{ID: "t2", Content: "ok"}}),
		toolUse("t3"),
	}
	out, tail := h.runner.repairHistory(context.Background(), "conv", "", mid, nil)
	if len(out) != 8 || tail == nil {
		t.Fatalf("repaired = %d messages, tail=%v", len(out), tail != nil)
	}
	if ids := toolResultIDs(out[2]); len(ids) != 1 || ids[0] != "t1" {
		t.Errorf("mid repair = %v", ids)
	}
	if ids := toolResultIDs(out[7]); len(ids) != 1 || ids[0] != "t3" || len(toolResultIDs(*tail)) != 1 {
		t.Errorf("tail repair = %v", ids)
	}
	// Nothing to repair → the input is returned as is, no tail.
	clean := mid[3:5]
	if out, tail := h.runner.repairHistory(context.Background(), "conv", "", clean, nil); len(out) != 2 || tail != nil {
		t.Errorf("clean history repaired: %d messages, tail=%v", len(out), tail != nil)
	}
}

func toolResultIDs(m anthropic.BetaMessageParam) []string {
	var ids []string
	for _, c := range m.Content {
		if c.OfToolResult != nil {
			ids = append(ids, c.OfToolResult.ToolUseID)
		}
	}
	return ids
}

// N2: the reaper expires the turn between the loop persisting an all-server
// assistant batch and persisting its results. The reaper closes the tail
// is_error; the loop must notice (compare-and-set on the tail) and stop,
// not answer the same ids a second time and run on an expired row.
func TestDangling_ReaperBetweenAssistantAndServerResults(t *testing.T) {
	h := newHarness(t, func(d *Deps) { d.TurnTTL = time.Minute },
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{
			modeltest.ToolUse("lk1", "lookup_fields", map[string]any{"target": heroTarget, "query": "headline"}),
		}}),
		textReply("done"),
		textReply("next"),
	)
	ctx := context.Background()
	turn, err := h.runner.Start(ctx, h.request("x"))
	if err != nil {
		t.Fatal(err)
	}
	fired := false
	h.st.mu.Lock()
	h.st.afterAppend = func(role string) {
		if role != "assistant" || fired {
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
	evs := col.Events()
	if !strings.HasSuffix(kinds(evs), "turn.error") || data(t, last(evs))["code"] != "turn_expired" {
		t.Fatalf("events = %s (last %s)", kinds(evs), last(evs).Data)
	}
	row := h.st.turn(turn.ID())
	if row.Status != store.TurnExpired {
		t.Fatalf("row status = %s, want expired (the loop must not finish a reaped turn)", row.Status)
	}
	got := roles(t, h, row.ConversationID)
	if strings.Join(got, " ") != "user assistant user" {
		t.Fatalf("stored roles = %v, want [user assistant user] (one answer for lk1)", got)
	}
	_, res := storedTail(t, h, row.ConversationID)
	if len(res) != 1 || res[0].ID != "lk1" || !res[0].IsError || !strings.Contains(res[0].Content, reasonExpired) {
		t.Fatalf("stored tail = %+v", res)
	}
	if len(h.fake.Requests()) != 1 {
		t.Fatalf("model called %d times, want 1 (no request after the reaper took the turn)", len(h.fake.Requests()))
	}
	if h.runner.Live() != 0 || h.runner.Parked() != 0 {
		t.Errorf("live=%d parked=%d", h.runner.Live(), h.runner.Parked())
	}
	h.start(h.request("y"))
	reqs := h.fake.Requests()
	assertReplayable(t, reqs[len(reqs)-1])
}

// N2 (continuation side): the reaper expires the turn between ResumeTurn
// and the results write. The write is refused (compare-and-set), the
// browser gets 409 turn_not_waiting, and the batch is answered exactly once
// (by the reaper).
func TestDangling_ReaperBetweenResumeAndResults(t *testing.T) {
	h := newHarness(t, func(d *Deps) { d.TurnTTL = time.Minute },
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{opUse("op1", "headline", "x")}}),
		textReply("done"),
		textReply("next"),
	)
	ctx := context.Background()
	turn, evs := h.start(h.request("x"))
	if !strings.HasSuffix(kinds(evs), "turn.waiting") {
		t.Fatalf("events = %s", kinds(evs))
	}
	row := h.st.turn(turn.ID())
	fired := false
	h.st.mu.Lock()
	h.st.afterResume = func() {
		if fired {
			return
		}
		fired = true
		h.clock = h.clock.Add(2 * time.Minute)
		if n, err := h.runner.Reap(ctx); n != 1 || err != nil {
			t.Errorf("reap in hook = %d %v", n, err)
		}
	}
	h.st.mu.Unlock()
	_, err := h.cont(turn.ID(), api.ToolResult{ToolCallID: "op1", OK: true})
	h.st.mu.Lock()
	h.st.afterResume = nil
	h.st.mu.Unlock()
	if !fired {
		t.Fatal("hook never fired")
	}
	var apiErr *api.Error
	if !errors.As(err, &apiErr) || apiErr.Code != "turn_not_waiting" || apiErr.Extra["status"] != store.TurnExpired {
		t.Fatalf("Continue = %v, want 409 turn_not_waiting status=expired", err)
	}
	got := roles(t, h, row.ConversationID)
	if strings.Join(got, " ") != "user assistant user" {
		t.Fatalf("stored roles = %v, want [user assistant user] (op1 answered once)", got)
	}
	_, res := storedTail(t, h, row.ConversationID)
	if len(res) != 1 || res[0].ID != "op1" || !res[0].IsError || !strings.Contains(res[0].Content, reasonExpired) {
		t.Fatalf("stored tail = %+v (the reaper's answer, not the browser's)", res)
	}
	if st := h.st.turn(turn.ID()).Status; st != store.TurnExpired {
		t.Errorf("row status = %s, want expired", st)
	}
	if len(h.fake.Requests()) != 1 || h.runner.Live() != 0 || h.runner.Parked() != 0 {
		t.Errorf("requests=%d live=%d parked=%d", len(h.fake.Requests()), h.runner.Live(), h.runner.Parked())
	}
	h.start(h.request("y"))
	reqs := h.fake.Requests()
	assertReplayable(t, reqs[len(reqs)-1])
}

// N3: A's parked entry is stale because the turn was continued on B and is
// now waiting AGAIN with a new batch. A continuation that lands on A with
// the new ids must be served from the row, not refused on the stale doc;
// and A's reaper evicts such an entry on its own.
func TestContinue_StaleParkedEntryRebuildsFromRow(t *testing.T) {
	h := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{opUse("op1", "headline", "x")}}),
		textReply("done"),
	)
	ctx := context.Background()
	turn, evs := h.start(h.request("x"))
	if !strings.HasSuffix(kinds(evs), "turn.waiting") || h.runner.Parked() != 1 {
		t.Fatalf("events = %s parked=%d", kinds(evs), h.runner.Parked())
	}
	row := h.st.turn(turn.ID())
	// Instance B: resume, write op1's result, a second assistant batch (op2), park again.
	if _, err := h.st.ResumeTurn(ctx, turn.ID()); err != nil {
		t.Fatal(err)
	}
	_, _ = h.st.AppendMessage(ctx, row.ConversationID, "user", json.RawMessage(`[{"type":"tool_result","tool_use_id":"op1","content":"{\"ok\":true}"}]`))
	_, _ = h.st.AppendMessage(ctx, row.ConversationID, "assistant", json.RawMessage(`[{"type":"tool_use","id":"op2","name":"set_fields","input":{"target":{"kind":"section","page":"home","sectionId":"hero-1"},"values":{"subtitle":"y"}}}]`))
	pd := pendingDoc{Route: "default", Calls: []pendingCall{{ID: "op2", Name: "set_fields", Input: json.RawMessage(`{}`), Seq: 3}}, CallCount: 2, Seq: 4}
	raw, _ := json.Marshal(pd)
	if err := h.st.SetTurnWaiting(ctx, turn.ID(), store.TurnWaitingTools, raw); err != nil {
		t.Fatal(err)
	}
	// A stale result for the OLD batch is refused on the row, not accepted on the stale entry.
	if _, err := h.cont(turn.ID(), api.ToolResult{ToolCallID: "op1", OK: true}); err == nil || !strings.Contains(err.Error(), `"op1" is not a pending call`) {
		t.Fatalf("stale id accepted or wrong error: %v", err)
	}
	// The browser answers op2 and lands on A: served from the row.
	evs2, err := h.cont(turn.ID(), api.ToolResult{ToolCallID: "op2", OK: true})
	if err != nil {
		t.Fatalf("continuation with the NEW batch id on the stale instance was refused: %v (row status=%s)", err, h.st.turn(turn.ID()).Status)
	}
	if !strings.HasSuffix(kinds(evs2), "turn.done") {
		t.Errorf("events = %s", kinds(evs2))
	}
	reqs := h.fake.Requests()
	assertReplayable(t, reqs[len(reqs)-1])
	if res := lastToolResults(t, reqs[len(reqs)-1]); len(res) != 1 || res[0].ID != "op2" || res[0].IsError {
		t.Errorf("model saw %+v", res)
	}
	if h.runner.Parked() != 0 || h.runner.Live() != 0 || h.st.turn(turn.ID()).Status != store.TurnDone {
		t.Errorf("parked=%d live=%d status=%s", h.runner.Parked(), h.runner.Live(), h.st.turn(turn.ID()).Status)
	}
	// The stale entry's partial trace was closed as superseded.
	if tr, ok := h.st.trace(turn.ID()); !ok || !strings.Contains(string(tr.Ops), `"status":"superseded"`) {
		t.Errorf("trace = %v %s", ok, tr.Ops)
	}

	// Reap alone evicts a superseded entry (row waiting on a batch it does not know).
	h2 := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{opUse("op1", "headline", "x")}}),
	)
	t2, _ := h2.start(h2.request("x"))
	if _, err := h2.st.ResumeTurn(ctx, t2.ID()); err != nil {
		t.Fatal(err)
	}
	if err := h2.st.SetTurnWaiting(ctx, t2.ID(), store.TurnWaitingTools, raw); err != nil {
		t.Fatal(err)
	}
	if n, err := h2.runner.Reap(ctx); err != nil || n != 1 || h2.runner.Parked() != 0 {
		t.Fatalf("reap of a superseded entry = %d %v parked=%d", n, err, h2.runner.Parked())
	}
	if tr, ok := h2.st.trace(t2.ID()); !ok || !strings.Contains(string(tr.Ops), `"status":"superseded"`) {
		t.Errorf("trace = %v %s", ok, tr.Ops)
	}
	// An entry whose doc IS the row's stays parked.
	h3 := newHarness(t, nil,
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{opUse("op1", "headline", "x")}}),
	)
	h3.start(h3.request("x"))
	if n, err := h3.runner.Reap(ctx); err != nil || n != 0 || h3.runner.Parked() != 1 {
		t.Fatalf("reap of an owned entry = %d %v parked=%d", n, err, h3.runner.Parked())
	}
}

func TestSamePending(t *testing.T) {
	a := &pendingDoc{Seq: 4, CallCount: 2, Calls: []pendingCall{{ID: "x"}, {ID: "y", Server: true}}}
	b := &pendingDoc{Seq: 4, CallCount: 2, Calls: []pendingCall{{ID: "x", Input: json.RawMessage(`{"k":1}`)}, {ID: "y"}}}
	if !samePending(a, b) {
		t.Error("docs differing only in input/server flags must match")
	}
	for _, c := range []*pendingDoc{
		nil,
		{Seq: 5, CallCount: 2, Calls: a.Calls},
		{Seq: 4, CallCount: 3, Calls: a.Calls},
		{Seq: 4, CallCount: 2, Calls: a.Calls[:1]},
		{Seq: 4, CallCount: 2, Calls: []pendingCall{{ID: "y"}, {ID: "x"}}},
		{Seq: 4, CallCount: 2, Calls: a.Calls, SuspendedID: "x"},
	} {
		if samePending(a, c) {
			t.Errorf("%+v matched %+v", c, a)
		}
	}
}
