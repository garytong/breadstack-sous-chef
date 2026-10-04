package loop

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/breadstack/sous-chef/internal/api"
	"github.com/breadstack/sous-chef/internal/model"
	"github.com/breadstack/sous-chef/internal/prompt"
	"github.com/breadstack/sous-chef/internal/registry"
	"github.com/breadstack/sous-chef/internal/store"
	"github.com/breadstack/sous-chef/internal/toolgen"
	"github.com/breadstack/sous-chef/internal/trace"
	"github.com/breadstack/sous-chef/internal/usage"
)

// Fixed replies (README §4 step 7).
const (
	// noConclusion is said when the model still wanted tools on its last
	// allowed call (thermometer chat.go noConclusion, reworded for editing).
	noConclusion = "I ran out of steps before finishing this request. Tell me what still needs changing and I'll continue from here."
	// refusalReply is said on stop_reason refusal.
	refusalReply = "I can't help with that request."
	// cutShort is appended when the reply hit max_tokens.
	cutShort = "\n\n(The reply was cut short — ask for the rest.)"
)

// Turn stop labels for turn.done.
const (
	StopEndTurn   = "end_turn"
	StopMaxCalls  = "max_calls"
	StopMaxTokens = "max_tokens"
	StopRefusal   = "refusal"
)

// turn is one turn's state, alive on this instance from Start/Continue
// through Run; parked while it waits for the browser.
type turn struct {
	r        *Runner
	id       string
	conv     *store.Conversation
	rotated  bool
	compiled *registry.Compiled
	backend  *model.Backend
	req      api.TurnRequest

	history   []anthropic.BetaMessageParam // every message of the conversation, this turn's included
	next      model.Request                // the request Run sends first
	operation string                       // usage operation of that call
	calls     int
	usage     model.Usage
	seq       int
	tailSeq   int // seq of the last stored message this turn knows; CAS base for answering a batch
	started   time.Time
	rec       *trace.Recorder // nil = not traced
	release   func()
	gone      bool // the client went away; keep working, stop emitting
	pending   *pendingDoc
	finished  bool
	lostAs    string // row status that ended the turn (expired/superseded); the trace's status
}

// ID implements api.Turn.
func (t *turn) ID() string { return t.id }

// Run implements api.Turn.Run: drive the model until this request's stream
// is complete. It emits turn.error itself; a returned error is only logged.
func (t *turn) Run(ctx context.Context, em api.Emitter) error {
	defer func() {
		if t.release != nil {
			t.release()
			t.release = nil
		}
	}()
	for {
		// The row is re-read before every model call: the TTL reaper (or a
		// newer turn on the conversation) may have closed this turn while the
		// loop was between writes with nothing dangling for the compare-and-
		// set appends to catch. A turn whose row is no longer running makes
		// no further call and ends as turn_expired.
		if err := t.checkRunning(ctx); err != nil {
			return t.fail(ctx, em, err)
		}
		t.calls++
		last := t.calls >= t.r.d.MaxCalls
		var streamedText int
		onText := func(s string) {
			if s == "" {
				return
			}
			streamedText += len(s)
			t.emit(em, api.EventTextDelta, map[string]string{"text": s})
		}
		callStart := t.r.d.Now()
		resp, err := t.backend.Message(ctx, t.next, onText)
		t.recordUsage(ctx, resp, err, t.r.d.Now().Sub(callStart))
		if err != nil {
			return t.fail(ctx, em, err)
		}
		t.usage = addUsage(t.usage, resp.Usage)
		t.traceCall(resp)

		// The assistant turn goes into the history before its results
		// (thermometer); thinking blocks ride along unchanged — the API wants
		// them back on the same model (preserved thinking) and their text is
		// empty under the default display. The append is a compare-and-set
		// on the tail this turn last wrote (its user text, or its last
		// results message): a healthy turn always owns the tail, so it is a
		// no-op check; a turn superseded while this call was in flight finds
		// the newer turn's messages there instead and writes nothing (its
		// stale batch would otherwise land on top of the newer turn's and
		// bury it), ending turn.error turn_expired below. No tool.call is
		// emitted for a batch that was never stored.
		assistant := resp.AssistantTurn()
		if err := t.persistAfter(ctx, assistant); err != nil {
			return t.fail(ctx, em, err)
		}
		t.history = append(t.history, assistant)

		uses := resp.ToolUses()
		wantsTools := resp.StopReason == string(anthropic.BetaStopReasonToolUse) && len(uses) > 0
		if !wantsTools {
			// end_turn/max_tokens/refusal with tool_use blocks still in the
			// content: nothing runs them, so answer them is_error now or the
			// next request fails on the unanswered ids.
			if err := t.abandon(ctx, uses, reasonNoToolStop); err != nil {
				return t.fail(ctx, em, err)
			}
			return t.finishFinal(ctx, em, resp)
		}
		if last {
			// No further request will be made, so no tool runs for nothing
			// (thermometer lastTurn): nothing is emitted for the calls, but
			// the persisted tool_use blocks are answered is_error.
			t.emit(em, api.EventTextDelta, map[string]string{"text": noConclusion})
			if err := t.abandon(ctx, uses, reasonMaxCalls); err != nil {
				return t.fail(ctx, em, err)
			}
			return t.done(ctx, em, StopMaxCalls)
		}

		results, pending, calls := t.dispatch(uses)
		if pending == nil {
			// Every call was a server tool: tell the browser about the echoed
			// ones, answer and go round again.
			t.emitCalls(em, calls)
			if err := t.appendResults(ctx, results); err != nil {
				return t.fail(ctx, em, err)
			}
			continue
		}
		pending.Route = t.backend.Name()
		pending.CallCount = t.calls
		pending.Usage = t.usage
		pending.Rotated = t.rotated
		status := store.TurnWaitingTools
		if pending.SuspendedID != "" {
			status = store.TurnSuspended
		}
		// seq must include the tool.call events emitted below and the
		// waiting/suspended event after them.
		pending.Seq = t.seq + len(calls) + 1
		if pending.SuspendedID != "" {
			// turn.suspended is the next event emitted after this batch's
			// tool.call events, whatever position pick_asset had in the batch.
			for i := range pending.Calls {
				if pending.Calls[i].ID == pending.SuspendedID {
					pending.Calls[i].Seq = pending.Seq
				}
			}
		}
		raw, _ := json.Marshal(pending)
		// The row goes waiting/suspended BEFORE any tool.call is streamed: the
		// status guard is the last check that this turn is still the site's
		// live one. A batch whose row is no longer running (the reaper, a
		// newer Start, a DELETE got there after the compare-and-set above
		// passed — possible when the newer turn lives on another
		// conversation, so this one's tail was still its own) is never shown
		// to the page: the page must not apply ops of a turn the user has
		// moved on from. Its stored tool_use blocks are answered is_error
		// with the reason the row's status gives, and the turn ends
		// turn.error turn_expired.
		if err := t.r.d.Store.SetTurnWaiting(ctx, t.id, status, raw); err != nil {
			if errors.Is(err, store.ErrNotFound) {
				status, serr := t.rowStatus(ctx)
				if serr != nil {
					// The guard said the row is no longer running; storage
					// cannot say why. The stored batch is left for the next
					// Start to answer with its owner's real status.
					return t.storeUnavailable(ctx, em, fmt.Errorf("loop: turn %s: row no longer running at SetTurnWaiting, status unreadable: %w", t.id, serr))
				}
				reason := reasonExpired
				if status == store.TurnSuperseded {
					reason = reasonSuperseded
				}
				if _, cerr := t.r.closeDangling(context.WithoutCancel(ctx), t.conv.ID, pending, reason); cerr != nil {
					t.r.d.Logf("loop: turn %s: %v", t.id, cerr)
				}
				return t.lostRow(ctx, em, status, fmt.Errorf("loop: turn %s: row is %s at SetTurnWaiting: %w", t.id, status, err))
			}
			return t.fail(ctx, em, err)
		}
		t.pending = pending
		t.r.park(t)
		t.emitCalls(em, calls)
		if status == store.TurnSuspended {
			t.emit(em, api.EventTurnSuspended, pending.suspendedEvent())
		} else {
			t.emit(em, api.EventTurnWaiting, map[string]any{"toolCallIds": pending.browserIDs()})
		}
		return nil
	}
}

// checkRunning returns errLostTail (→ turn_expired) when the turn's row is
// no longer running: expired by the reaper, or finished elsewhere. A read
// that fails is logged and the call proceeds — the next write fails on its
// own if the database is really gone.
func (t *turn) checkRunning(ctx context.Context) error {
	row, err := t.r.d.Store.GetTurn(ctx, t.id)
	switch {
	case errors.Is(err, store.ErrNotFound):
		return fmt.Errorf("loop: turn %s: row gone before model call %d: %w", t.id, t.calls+1, errLostTail)
	case err != nil:
		t.r.d.Logf("loop: turn %s: row check before model call %d: %v", t.id, t.calls+1, err)
		return nil
	case row.Status == store.TurnSuperseded:
		return fmt.Errorf("loop: turn %s: row is %s before model call %d: %w", t.id, row.Status, t.calls+1, errSuperseded)
	case row.Status != store.TurnRunning:
		return fmt.Errorf("loop: turn %s: row is %s before model call %d: %w", t.id, row.Status, t.calls+1, errLostTail)
	}
	return nil
}

// rowStatus re-reads the turn's row status after a status-guarded write or
// compare-and-set append found the row no longer its own. A row that is
// gone (store.ErrNotFound — rows are retained, so this is academic) is
// store.TurnExpired, the generic "no longer ours". A read that FAILS is
// returned as the error, never guessed at: a database blip is not a lost
// row — the callers end the turn store_unavailable (Run) or answer 503
// store_unavailable (Start) instead of treating the turn as expired on
// evidence they do not have (README §4 "Storage failures").
func (t *turn) rowStatus(ctx context.Context) (string, error) {
	row, err := t.r.d.Store.GetTurn(context.WithoutCancel(ctx), t.id)
	switch {
	case errors.Is(err, store.ErrNotFound):
		return store.TurnExpired, nil
	case err != nil:
		return "", fmt.Errorf("loop: turn %s: row status: %w", t.id, err)
	}
	return row.Status, nil
}

// storeUnavailable ends a turn whose row status could not be read when the
// loop needed it to name the turn's exit: turn.error store_unavailable, no
// further model call. The row is finished error when storage takes the
// write (a no-op on a row that is no longer running); the trace says error.
// Nothing is written into the conversation: a batch this turn stored stays
// open for the next Start's repair, which attributes it by the row's real
// status once storage is back (a reason guessed here could name the wrong
// actor).
func (t *turn) storeUnavailable(ctx context.Context, em api.Emitter, cause error) error {
	return t.errorOut(ctx, em, "store_unavailable", "storage is temporarily unavailable; the turn ended without finishing", cause)
}

// lostRow ends a turn whose row is no longer running — expired by the
// reaper, or superseded by a newer turn for the site / a DELETE of its
// conversation — as turn.error turn_expired, the message naming which. The
// row itself is left as it is (FinishTurn would be a no-op on it), and the
// trace's outcome.status is the row's terminal status. A turn that lost its
// conversation tail while its row is STILL running (a foreign write landed
// between its row check and its append) has no terminal status to borrow:
// it ends as a plain error — the row is finished error, the trace says
// error, and the message says the tail was lost.
func (t *turn) lostRow(ctx context.Context, em api.Emitter, status string, cause error) error {
	var msg string
	switch status {
	case store.TurnSuperseded:
		msg = "the turn was superseded: a newer turn started for the site, or the conversation was ended, before this one finished"
		t.lostAs = status
	case store.TurnExpired, store.TurnDone, store.TurnError:
		msg = "the turn expired before it finished"
		t.lostAs = status
	default:
		msg = "the turn lost its conversation: another writer appended to the conversation while this turn was running"
	}
	return t.errorOut(ctx, em, "turn_expired", msg, cause)
}

// dispatch answers the server tools in a batch and lays out the pending
// doc; nothing is emitted here. calls lists, in order, the tool_use blocks
// the browser is told about as tool.call events (the browser ops, and the
// server tools that are echoed) — emitCalls sends them once the row is
// stored waiting, and their event seqs are assigned here on that
// assumption (t.seq + 1, + 2, …). pending is nil when nothing needs the
// browser.
func (t *turn) dispatch(uses []model.Block) ([]model.ToolResult, *pendingDoc, []model.Block) {
	var results []model.ToolResult
	var calls []model.Block
	pd := &pendingDoc{}
	for _, u := range uses {
		runs, known := toolgen.RunsWhere(u.Name)
		call := pendingCall{ID: u.ID, Name: u.Name, Input: u.Input}
		switch {
		case !known:
			call.Server = true
			res := model.ToolResult{ID: u.ID, IsError: true, Content: errJSON("E_OP", fmt.Sprintf("unknown tool %q", u.Name))}
			results = append(results, res)
			pd.ServerResults = append(pd.ServerResults, serverResult{ID: u.ID, Content: res.Content, IsError: true})
		case runs == toolgen.RunsServer:
			call.Server = true
			t.rec.MarkServer(u.ID)
			out, _ := toolgen.Dispatch(t.compiled.Manifest, u.Name, u.Input)
			res := model.ToolResult{ID: u.ID, Content: out.Content, IsError: out.IsError}
			results = append(results, res)
			pd.ServerResults = append(pd.ServerResults, serverResult{ID: u.ID, Content: res.Content, IsError: res.IsError})
			if toolgen.EchoedToBrowser(u.Name) {
				calls = append(calls, u)
			}
		case runs == toolgen.RunsSuspend && pd.SuspendedID != "":
			// A second pick_asset in one batch: one picker per turn
			// (README §4 step 5: the model is told to call it alone), so
			// the extras are answered here, is_error, and never shown.
			call.Server = true
			t.rec.MarkServer(u.ID)
			res := model.ToolResult{ID: u.ID, IsError: true, Content: errJSON("E_OP", reasonExtraPick)}
			results = append(results, res)
			pd.ServerResults = append(pd.ServerResults, serverResult{ID: u.ID, Content: res.Content, IsError: true})
		case runs == toolgen.RunsSuspend:
			// pick_asset: the turn.suspended event carries the call; the
			// browser answers it like any pending call, with `asset`. Its
			// Seq is fixed by Run, once every tool.call of the batch is
			// counted.
			pd.SuspendedID = u.ID
		default: // browser op
			call.Seq = t.seq + len(calls) + 1
			calls = append(calls, u)
		}
		pd.Calls = append(pd.Calls, call)
	}
	if len(pd.browserIDs()) == 0 {
		return results, nil, calls
	}
	return results, pd, calls
}

// emitCalls streams the batch's tool.call events in dispatch order; emit
// assigns exactly the seqs dispatch recorded in the pending doc.
func (t *turn) emitCalls(em api.Emitter, calls []model.Block) {
	for _, u := range calls {
		t.emit(em, api.EventToolCall, toolCallEvent(u))
	}
}

// resume (Continue) turns the browser's results plus the stashed server
// results into one user message in call order.
func (t *turn) resume(ctx context.Context, body api.ToolResultsRequest) error {
	answered := map[string]api.ToolResult{}
	for _, res := range body.Results {
		answered[res.ToolCallID] = res
	}
	var results []model.ToolResult
	for _, c := range t.pending.Calls {
		if c.Server {
			sr, ok := t.pending.serverResult(c.ID)
			if !ok {
				sr = serverResult{ID: c.ID, Content: errJSON("E_OP", reasonServerLost), IsError: true}
			}
			results = append(results, model.ToolResult{ID: c.ID, Content: sr.Content, IsError: sr.IsError})
			continue
		}
		res, ok := answered[c.ID]
		if !ok {
			results = append(results, model.ToolResult{ID: c.ID, IsError: true, Content: errJSON("E_OP", "no result returned by the browser for this call")})
			t.traceToolResult(c.Seq, json.RawMessage(fmt.Sprintf(`{"toolCallId":%q,"ok":false,"error":{"code":"E_OP","message":"no result returned"}}`, c.ID)))
			continue
		}
		raw, _ := json.Marshal(res)
		t.traceToolResult(c.Seq, raw)
		results = append(results, browserResult(c.Name, res))
	}
	if err := t.appendResults(ctx, results); err != nil {
		return err // t.pending kept: failResume answers the batch from it
	}
	t.pending = nil
	return nil
}

// traceToolResult records one browser result in the trace as a tool.result
// event that continues the turn's seq (the recorder's own ToolResult numbers
// from its event count, which collides with stream seqs once text.delta
// events have been coalesced). The answered tool.call's seq rides in the
// payload as for_seq.
func (t *turn) traceToolResult(forSeq int, result json.RawMessage) {
	t.seq++ // even untraced (a rebuilt turn), so seq numbering does not depend on the instance
	if t.rec == nil {
		return
	}
	var doc map[string]any
	if json.Unmarshal(result, &doc) != nil || doc == nil {
		doc = map[string]any{"result": result}
	}
	doc["for_seq"] = forSeq
	raw, _ := json.Marshal(doc)
	t.rec.Event(t.seq, "tool.result", raw)
}

// failResume closes a turn whose continuation failed after ResumeTurn: the
// row would otherwise stay running with no pending doc until the reaper.
// The batch is answered is_error (best effort — the failure was probably a
// write; when the tail was lost someone else already answered it) and the
// turn ends as error.
func (t *turn) failResume(ctx context.Context, cause error) {
	t.finished = true
	if t.pending != nil && !errors.Is(cause, errLostTail) {
		uses := make([]model.Block, 0, len(t.pending.Calls))
		for _, c := range t.pending.Calls {
			uses = append(uses, model.Block{Type: "tool_use", ID: c.ID, Name: c.Name})
		}
		if err := t.abandon(ctx, uses, reasonContinueFail); err != nil {
			t.r.d.Logf("loop: turn %s: %v", t.id, err)
		}
	}
	t.pending = nil
	usageJSON, _ := json.Marshal(t.usageDoc())
	if ferr := t.r.d.Store.FinishTurn(ctx, t.id, store.TurnError, usageJSON); ferr != nil && !errors.Is(ferr, store.ErrNotFound) {
		t.r.d.Logf("loop: turn %s: finish error: %v", t.id, ferr)
	}
	_ = t.rec.Finish(ctx, "error", "", cause.Error())
	t.r.d.Logf("loop: turn %s: continuation failed: %v", t.id, cause)
}

// appendResults adds the tool_result user message to the history and the
// store and prepares the next request. The store write is a compare-and-set
// on the assistant message it answers: if the reaper expired this turn in
// between and closed that tail is_error, the write fails with errLostTail
// instead of answering the same ids a second time.
func (t *turn) appendResults(ctx context.Context, results []model.ToolResult) error {
	next, err := prompt.Continue(t.compiled.Prefix, t.history, results)
	if err != nil {
		return err
	}
	if err := t.persistAfter(ctx, next.Messages[len(next.Messages)-1]); err != nil {
		return err
	}
	t.history = next.Messages
	t.next = next
	t.operation = usage.OpToolTurn
	return nil
}

// finishFinal ends the turn after a reply that asked for no browser work.
func (t *turn) finishFinal(ctx context.Context, em api.Emitter, resp *model.Response) error {
	stop := resp.StopReason
	switch stop {
	case string(anthropic.BetaStopReasonRefusal):
		t.r.d.Logf("loop: turn %s: refusal (%s)", t.id, resp.StopCategory)
		t.emit(em, api.EventTextDelta, map[string]string{"text": refusalReply})
		stop = StopRefusal
	case string(anthropic.BetaStopReasonMaxTokens):
		if resp.Text() != "" {
			t.emit(em, api.EventTextDelta, map[string]string{"text": cutShort})
		}
		stop = StopMaxTokens
	case string(anthropic.BetaStopReasonToolUse), "":
		// A tool_use stop with no tool_use block (an open model behind a
		// compatible server does this) is the answer.
		stop = StopEndTurn
	}
	return t.done(ctx, em, stop)
}

// done closes the turn as done and emits turn.done. When the row is no
// longer ours (FinishTurn finds it not running: the reaper expired it, or a
// newer turn superseded it, while the final model call was in flight) the
// turn ends turn.error turn_expired instead — a turn.done on an expired row
// would tell the launcher a turn completed that storage says did not.
func (t *turn) done(ctx context.Context, em api.Emitter, stop string) error {
	t.finished = true
	usageJSON, _ := json.Marshal(t.usageDoc())
	if err := t.r.d.Store.FinishTurn(ctx, t.id, store.TurnDone, usageJSON); err != nil {
		if errors.Is(err, store.ErrNotFound) {
			status, serr := t.rowStatus(ctx)
			if serr != nil {
				return t.storeUnavailable(ctx, em, fmt.Errorf("loop: turn %s: row no longer running at finish (%s), status unreadable: %w", t.id, stop, serr))
			}
			return t.lostRow(ctx, em, status, fmt.Errorf("loop: turn %s: row is %s at finish (%s): %w", t.id, status, stop, errLostTail))
		}
		t.r.d.Logf("loop: turn %s: finish: %v", t.id, err)
	}
	data := map[string]any{"usage": t.usageDoc(), "stop": stop, "conversationId": t.conv.ID}
	if t.rotated {
		data["conversationRotated"] = true
	}
	t.emit(em, api.EventTurnDone, data)
	if err := t.rec.Finish(context.WithoutCancel(ctx), "done", stop, ""); err != nil {
		t.r.d.Logf("loop: turn %s: trace: %v", t.id, err)
	}
	return nil
}

// fail ends the turn on a model or storage error.
func (t *turn) fail(ctx context.Context, em api.Emitter, err error) error {
	var apiErr *api.Error
	switch {
	case errors.As(err, &apiErr):
		return t.errorOut(ctx, em, apiErr.Code, apiErr.Message, err)
	case errors.Is(err, errSuperseded):
		return t.lostRow(ctx, em, store.TurnSuperseded, err)
	case errors.Is(err, errLostTail):
		// The conversation moved on without this turn — the reaper expired
		// it and answered its batch, or a newer turn superseded it and wrote
		// its own messages, while the loop was between writes or in the
		// model. The row is no longer ours (FinishTurn is a no-op on it).
		// When the row cannot be read the turn is still over (its tail is
		// gone) but its exit cannot be named: store_unavailable.
		status, serr := t.rowStatus(ctx)
		if serr != nil {
			return t.storeUnavailable(ctx, em, fmt.Errorf("%w (status unreadable: %v)", err, serr))
		}
		return t.lostRow(ctx, em, status, err)
	case model.IsUnavailable(err):
		return t.errorOut(ctx, em, "model_unavailable", "the model is temporarily unavailable; try again in a moment", err)
	case model.StatusCode(err) != 0:
		return t.errorOut(ctx, em, "model_error", fmt.Sprintf("the model rejected the request (HTTP %d)", model.StatusCode(err)), err)
	case ctx.Err() != nil:
		return t.errorOut(ctx, em, "cancelled", "the request was cancelled", err)
	default:
		return t.errorOut(ctx, em, "internal", "internal error", err)
	}
}

func (t *turn) errorOut(ctx context.Context, em api.Emitter, code, message string, cause error) error {
	t.finished = true
	usageJSON, _ := json.Marshal(t.usageDoc())
	if ferr := t.r.d.Store.FinishTurn(context.WithoutCancel(ctx), t.id, store.TurnError, usageJSON); ferr != nil && !errors.Is(ferr, store.ErrNotFound) {
		t.r.d.Logf("loop: turn %s: finish error: %v", t.id, ferr)
	}
	t.emit(em, api.EventTurnError, map[string]string{"code": code, "message": message})
	status := "error"
	if t.lostAs != "" {
		status = t.lostAs // the row's own terminal status (expired, superseded)
	}
	_ = t.rec.Finish(context.WithoutCancel(ctx), status, "", cause.Error())
	return fmt.Errorf("loop: turn %s: %s: %w", t.id, code, cause)
}

// emit sends one event (seq assigned here) and records it in the trace. A
// failed Emit marks the client gone; the turn keeps its state consistent
// but stops writing.
func (t *turn) emit(em api.Emitter, typ string, data any) {
	t.seq++
	raw, _ := json.Marshal(data)
	t.rec.Event(t.seq, typ, raw)
	if t.gone || em == nil {
		return
	}
	if err := em.Emit(api.Event{Type: typ, TurnID: t.id, Seq: t.seq, Data: data}); err != nil {
		t.gone = true
		t.r.d.Logf("loop: turn %s: client gone at seq %d: %v", t.id, t.seq, err)
	}
}

// persistAfter appends one turn only while the conversation's tail is still
// the last message this turn wrote or read (tailSeq); errLostTail otherwise.
// EVERY message a turn writes goes through here — its user text, each
// assistant turn, each results message, each is_error repair — so a batch
// is never answered twice however the reaper and the loop interleave, and a
// superseded turn's late write never lands on top of the newer turn's. The
// SDK's param types marshal themselves with HTML escaping on (`<context>`
// is stored as \u003ccontext\u003e); the JSON is equivalent and reads back
// identically, so it is left as the SDK writes it.
func (t *turn) persistAfter(ctx context.Context, m anthropic.BetaMessageParam) error {
	content, err := json.Marshal(m.Content)
	if err != nil {
		return err
	}
	seq, err := t.r.d.Store.AppendMessageAfter(ctx, t.conv.ID, string(m.Role), content, t.tailSeq)
	if errors.Is(err, store.ErrConflict) {
		// The store's own error rides along (errors.Is still matches it):
		// an ended conversation (store.ErrConversationEnded) is a lost tail
		// to the loop, and tells Start that a DELETE took the conversation.
		return fmt.Errorf("loop: turn %s: conversation %s moved past seq %d: %w: %w", t.id, t.conv.ID, t.tailSeq, errLostTail, err)
	}
	if err != nil {
		return err
	}
	t.tailSeq = seq
	return nil
}

// traceCall records the model call and its tool_use blocks.
func (t *turn) traceCall(resp *model.Response) {
	if t.rec == nil {
		return
	}
	blocks, _ := json.Marshal(resp.Content)
	t.rec.ModelCall(trace.ModelCall{
		Call: t.calls, Model: resp.Model, StopReason: resp.StopReason, RequestID: resp.RequestID,
		LatencyMS: resp.Latency.Milliseconds(),
		Usage: trace.Usage{InputTokens: resp.Usage.InputTokens, OutputTokens: resp.Usage.OutputTokens,
			CacheReadInputTokens: resp.Usage.CacheReadInputTokens, CacheCreationInputTokens: resp.Usage.CacheCreationInputTokens},
	}, blocks)
}

// recordUsage writes the usage row for one call (thermometer recordUsage).
func (t *turn) recordUsage(ctx context.Context, resp *model.Response, err error, latency time.Duration) {
	if t.r.d.Usage == nil {
		return
	}
	c := usage.Call{
		TurnID: t.id, Operation: t.operation, RequestModel: t.backend.Model(),
		SiteID: t.req.SiteID, StoreID: t.req.StoreID, UserEmail: t.req.UserEmail,
		Latency: latency, Err: err, OccurredAt: t.r.d.Now(),
	}
	if resp != nil {
		c.Latency = resp.Latency
		c.Resp = &usage.Response{ID: resp.ID, Model: resp.Model, StopReason: resp.StopReason,
			Usage: usage.Tokens{InputTokens: resp.Usage.InputTokens, OutputTokens: resp.Usage.OutputTokens,
				CacheCreationInputTokens: resp.Usage.CacheCreationInputTokens, CacheReadInputTokens: resp.Usage.CacheReadInputTokens}}
	}
	_ = t.r.d.Usage.Record(ctx, c)
}

// usageDoc is the turn.done usage object and the turns.usage column.
func (t *turn) usageDoc() map[string]any {
	return map[string]any{
		"input_tokens":                t.usage.InputTokens,
		"output_tokens":               t.usage.OutputTokens,
		"cache_read_input_tokens":     t.usage.CacheReadInputTokens,
		"cache_creation_input_tokens": t.usage.CacheCreationInputTokens,
		"calls":                       t.calls,
	}
}

func addUsage(a, b model.Usage) model.Usage {
	a.InputTokens += b.InputTokens
	a.OutputTokens += b.OutputTokens
	a.CacheReadInputTokens += b.CacheReadInputTokens
	a.CacheCreationInputTokens += b.CacheCreationInputTokens
	return a
}

func toolCallEvent(u model.Block) map[string]any {
	input := u.Input
	if len(input) == 0 {
		input = json.RawMessage(`{}`)
	}
	return map[string]any{"id": u.ID, "name": u.Name, "input": input}
}

func errJSON(code, message string) string {
	b, _ := json.Marshal(map[string]string{"code": code, "message": message})
	return string(b)
}
