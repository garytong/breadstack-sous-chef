package loop

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/breadstack/sous-chef/internal/api"
	"github.com/breadstack/sous-chef/internal/config"
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

// Escalation reasons (turn.escalated `reason`, README §4 "Hybrid").
const (
	// EscalateToolError: an op the model wrote was refused — the browser's
	// engine answered it is_error, or the service refused it before it was
	// shown (an unknown tool, a reset without keys). Nothing was applied
	// for that op.
	EscalateToolError = "tool_error"
	// EscalateUnsupported: the model called report_unsupported.
	EscalateUnsupported = "unsupported"
	// EscalateModelError: the model call failed or timed out.
	EscalateModelError = "model_error"
	// EscalateMaxTokens: the reply stopped at max_tokens.
	EscalateMaxTokens = "max_tokens"
	// EscalateHiddenWrite: an op the model wrote was applied, but a field it
	// set comes back `W_HIDDEN:<key>` with its gate closed — the merchant
	// will not see the change (plan Q9: the student's silent wrong change
	// `columnsPerRow`). The write stays; the fallback reads the result and
	// its `hidden` member and decides what to tell the merchant or change.
	EscalateHiddenWrite = "hidden_write"
)

// reasonRecheck answers a report_unsupported that escalated the turn: the
// fallback model reads it as "not accepted", so it looks for itself instead
// of agreeing with a verdict it did not reach.
const reasonRecheck = "not accepted: a second model is re-checking this request. Look it up again (the index, lookup_fields, describe_target) and make the edit if a field or list covers it; call report_unsupported again only if it really cannot be done by editing this site's configuration"

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

	// Hybrid (README §4): backend is the route the NEXT model call uses —
	// the route the turn started on until it escalates, its fallback after.
	escalated    *escalation // set once; a turn escalates at most once
	wantEscalate string      // reason of an escalation decided outside Run (resume), applied before the next call
	routes       []string    // every route called so far, first-use order
	// foreignBefore: the turn escalated from a messages-compatible route to
	// an anthropic one when the history held this many messages. Their
	// thinking blocks (the student's, whatever its route's `thinking` says —
	// a server may return one unasked — and any earlier turn's) are left out
	// of every request the fallback is sent: Claude rejects a thinking block
	// it did not sign, which would end the turn in turn.error exactly when
	// the fallback is meant to rescue it. The stored history is untouched.
	foreignBefore int

	// origin is the route the turn STARTED on (backend until it escalates);
	// fieldsContext whether its user message was built with the
	// candidate-fields option on. turn.done states both (`route`).
	origin        *model.Backend
	fieldsContext bool

	// models is, per route name, the model id the turn's calls on that route
	// were reported under (noteModel); turn.done states it.
	models map[string]string

	// primeNext: the turn ended turn.done; Run primes the conversation's next
	// turn once the slot is released (prime.go).
	primeNext bool
	// parkedAt: when the turn last parked on a batch (Runner.primeBusy).
	parkedAt time.Time

	// Fast finish (fastfinish.go, README §4). hadError: a results message of
	// this turn carried an is_error tool_result (pending doc `had_error`).
	// fastFinish: the clean batch's op count once resume armed a fast
	// finish (0 = the closing call runs); fastFinished once Run took it.
	hadError     bool
	fastFinish   int
	fastFinished bool
}

// sendable is the request the next model call sends: t.next, without the
// thinking blocks an anthropic fallback did not write (foreignBefore), and —
// on a messages-compatible route — with the history stabilised
// (prompt.StableHistory): earlier turns' user messages lose their <context>
// and <candidate_fields> blocks, only the turn being answered keeps its own.
// The student's prompt then grows by the merchant's words per turn instead
// of a page context (5–7k tokens each on a real site), and its KV prefix is
// the same from one turn to the next up to the previous turn's user text.
//
// An anthropic route is sent the history as stored (README §3 "History"):
// rewriting an earlier user message is a history edit, which Claude Sonnet
// 5.5 / Opus 5.5 check against every replayed thinking block's signature
// (preserved thinking — a 400 on accounts created on or after 2026-08-31),
// and it would also move the automatic cache breakpoint's prefix every
// turn, turning the history's cache reads (0.1× input) into fresh writes.
//
// On both kinds, a tool_result-only user message followed by another user
// message is sent as one (prompt.MergeUserRuns): results first, then the
// text. A fast-finished turn leaves the conversation on its results message
// and the next turn's text is stored as a row of its own, so its request
// would otherwise carry two user messages in a row. The merge runs last:
// foreignBefore counts stored messages.
func (t *turn) sendable() model.Request {
	req := t.next
	if t.backend.Kind() == config.RouteMessagesCompatible {
		req.Messages = prompt.StableHistory(req.Messages)
	}
	if t.foreignBefore > 0 {
		req.Messages = model.WithoutThinkingBefore(req.Messages, t.foreignBefore)
	}
	req.Messages = prompt.MergeUserRuns(req.Messages)
	return req
}

// routeDoc is turn.done's `route`: the route the turn started on, whether
// its user message was built with the candidate-fields option on (the block
// itself is absent when nothing matched), and the route it falls back to
// (null for none). The lab's collector and eval render their rows from it
// rather than from a setting of their own.
func (t *turn) routeDoc() map[string]any {
	origin := t.origin
	if origin == nil {
		origin = t.backend
	}
	doc := map[string]any{"name": origin.Name(), "fields_context": t.fieldsContext, "fallback": nil,
		"model": t.reportedModel(origin)}
	if fb := origin.Fallback(); fb != nil {
		doc["fallback"] = fb.Name()
	}
	return doc
}

// noteModel records the model id a call on the turn's current route was
// reported under (its usage row's and trace entry's `model`).
func (t *turn) noteModel(responseModel string) {
	if t.models == nil {
		t.models = map[string]string{}
	}
	t.models[t.backend.Name()] = t.backend.ReportedModel(responseModel)
}

// reportedModel is the model id b's calls of this turn were reported under:
// the last one noted, else (no call of b answered) what a call that names
// none reports — `report_as`, else the route's model id.
func (t *turn) reportedModel(b *model.Backend) string {
	if m := t.models[b.Name()]; m != "" {
		return m
	}
	return b.ReportedModel("")
}

// modelsDoc is turn.done's `models` on a hybrid turn: for each route the
// turn called (`routes`), the model id its calls were reported under — per
// route, so the student's and the fallback's are told apart.
func (t *turn) modelsDoc() map[string]string {
	out := make(map[string]string, len(t.routes))
	for _, name := range t.routes {
		if m := t.models[name]; m != "" {
			out[name] = m
			continue
		}
		// A route whose calls all failed: one of the turn's own backends.
		for _, b := range []*model.Backend{t.origin, t.backend, t.backend.Fallback()} {
			if b != nil && b.Name() == name {
				out[name] = b.ReportedModel("")
				break
			}
		}
	}
	return out
}

// canEscalate reports whether the turn may still move to a fallback route:
// its current route names one and it has not escalated yet.
func (t *turn) canEscalate() bool {
	return t.escalated == nil && t.backend.Fallback() != nil
}

// hybrid reports whether the turn runs (or started) on a route with a
// fallback; only such turns report `routes`.
func (t *turn) hybrid() bool {
	return t.escalated != nil || t.backend.Fallback() != nil
}

// escalate moves the turn to its route's fallback: every later model call
// of the turn — on this stream, after a tool-results continuation, after a
// rebuild on another instance (the pending doc carries it) — goes there,
// with the history as it stands. The event is informational.
func (t *turn) escalate(em api.Emitter, reason string) {
	from, to := t.backend, t.backend.Fallback()
	if to == nil || t.escalated != nil {
		return
	}
	t.escalated = &escalation{From: from.Name(), To: to.Name(), Reason: reason}
	t.backend = to
	if from.Kind() != config.RouteAnthropic && to.Kind() == config.RouteAnthropic {
		// Everything in the history so far was written before Claude took
		// the turn; none of its thinking blocks is Claude's to read back.
		t.foreignBefore = len(t.history)
	}
	t.rec.Escalated(trace.Escalation{From: from.Name(), To: to.Name(), Reason: reason, AfterCall: t.calls})
	t.r.d.Logf("loop: turn %s: escalated %s → %s after %d call(s): %s", t.id, from.Name(), to.Name(), t.calls, reason)
	t.emit(em, api.EventTurnEscalated, t.escalated)
}

// noteRoute records that the turn is calling route.
func (t *turn) noteRoute(route string) {
	for _, r := range t.routes {
		if r == route {
			return
		}
	}
	t.routes = append(t.routes, route)
}

// isOp reports whether a tool name is (or was meant as) an edit: a browser
// op, or a tool the service does not know. The lookups, report_unsupported
// and pick_asset are not.
func isOp(name string) bool {
	runs, known := toolgen.RunsWhere(name)
	return !known || runs == toolgen.RunsBrowser
}

func hasTool(uses []model.Block, name string) bool {
	for _, u := range uses {
		if u.Name == name {
			return true
		}
	}
	return false
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
		// After the slot is released, so the Primer's busy check does not
		// count this turn (prime.go). It holds Run — the turn's request —
		// until the prime's requests are on the wire (bounded): the
		// launcher has stopped reading at turn.done, and an instance with
		// request-based CPU would otherwise throttle the prime.
		if t.primeNext {
			t.primeNext = false
			t.primeAfterDone()
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
		// A clean batch on a fast_finish route (resume armed it): no closing
		// call, the service's own line and turn.done.
		if t.fastFinish > 0 {
			return t.finishFast(ctx, em)
		}
		// An escalation decided while the turn was not streaming (a refused
		// op in the tool-results just posted, or in an all-server batch):
		// this call is the fallback's.
		if t.wantEscalate != "" {
			t.escalate(em, t.wantEscalate)
			t.wantEscalate = ""
		}
		t.calls++
		last := t.calls >= t.r.d.MaxCalls
		// A call that may still be escalated away from (hybrid route, not
		// escalated yet, another call allowed) holds its text back until the
		// reply is known to stand: a reply that is discarded (error,
		// max_tokens) or overruled (report_unsupported) is never shown.
		mayEscalate := t.canEscalate() && !last
		var held []string
		var streamedText int
		onText := func(s string) {
			if s == "" {
				return
			}
			streamedText += len(s)
			if mayEscalate {
				held = append(held, s)
				return
			}
			t.emit(em, api.EventTextDelta, map[string]string{"text": s})
		}
		flush := func() {
			for _, s := range held {
				t.emit(em, api.EventTextDelta, map[string]string{"text": s})
			}
			held = nil
		}
		t.noteRoute(t.backend.Name())
		callStart := t.r.d.Now()
		resp, err := t.backend.Message(ctx, t.sendable(), onText)
		t.recordUsage(ctx, resp, err, t.r.d.Now().Sub(callStart))
		if err != nil {
			if pe, ok := model.AsParserError(err); ok {
				// A route with parser_retries "off": the model server could not
				// parse the model's own tool call, and the call was not redrawn.
				// The turn goes on below exactly as after any failed call.
				t.r.d.Logf("souschef: model parser error: turn %s: call %d on route %s (reported as %s), not retried, %d on this instance: %.200s",
					t.id, t.calls, t.backend.Name(), t.backend.ReportedModel(""), pe.Count, pe.Message)
			}
			if mayEscalate && ctx.Err() == nil {
				// The same request goes to the fallback: nothing of the
				// failed call is in the history.
				t.r.d.Logf("loop: turn %s: model call %d on route %s failed, escalating: %v", t.id, t.calls, t.backend.Name(), err)
				t.escalate(em, EscalateModelError)
				continue
			}
			return t.fail(ctx, em, err)
		}
		t.usage = addUsage(t.usage, resp.Usage)
		t.noteModel(resp.Model)
		t.traceCall(resp)
		t.observe(resp)
		if mayEscalate && resp.StopReason == string(anthropic.BetaStopReasonMaxTokens) {
			// A cut-off reply is discarded whole — not stored, not shown (its
			// usage and trace entry stay) — and the same request goes to the
			// fallback. Storing it would leave the fallback a history ending
			// in an assistant turn, which Claude refuses as prefill.
			t.escalate(em, EscalateMaxTokens)
			continue
		}

		// The assistant turn goes into the history before its results
		// (thermometer); thinking blocks ride along unchanged — the API wants
		// them back on the same model (preserved thinking) and their text is
		// empty under the default display. They are stored as returned; a
		// messages-compatible route with thinking disabled (the default) drops
		// them from the history at send time (model.buildParams), so what the
		// model sees matches its training rows. The append is a compare-and-set
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
			flush()
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

		// report_unsupported from a route with a fallback escalates the turn
		// instead of ending it: the call is answered is_error (reasonRecheck)
		// and not echoed to the page, the text that came with it is dropped,
		// and the fallback decides. The event goes out before dispatch, which
		// numbers the batch's tool.call events from t.seq.
		overruled := mayEscalate && hasTool(uses, toolgen.ReportUnsupported)
		if overruled {
			held = nil
			t.escalate(em, EscalateUnsupported)
		} else {
			flush()
		}
		results, pending, calls, refused := t.dispatch(uses, overruled)
		if pending == nil {
			// Every call was a server tool: tell the browser about the echoed
			// ones, answer and go round again.
			t.emitCalls(em, calls)
			if err := t.appendResults(ctx, results); err != nil {
				return t.fail(ctx, em, err)
			}
			if refused && t.canEscalate() {
				t.wantEscalate = EscalateToolError
			}
			continue
		}
		pending.Route = t.backend.Name()
		pending.Escalated = t.escalated
		pending.Routes = t.routes
		pending.Models = make(map[string]string, len(t.models))
		for k, v := range t.models {
			pending.Models[k] = v
		}
		pending.ForeignBefore = t.foreignBefore
		fc := t.fieldsContext
		pending.FieldsContext = &fc
		pending.HadError = t.hadError
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
//
// overruled is set when this batch's report_unsupported just escalated the
// turn: the call is answered is_error (reasonRecheck) and not echoed.
// refused reports that the service itself refused an op of the batch (an
// unknown tool, a reset without keys) — the hybrid trigger for a batch that
// never reaches the browser; a batch that does is judged again, whole, when
// its results arrive (resume).
func (t *turn) dispatch(uses []model.Block, overruled bool) (results []model.ToolResult, pending *pendingDoc, calls []model.Block, refused bool) {
	pd := &pendingDoc{}
	// Slugs of a create_page refused here: a later op of the same batch
	// aimed at one (the model passes the slug and targets it in the same
	// reply) would otherwise land on whatever page already holds it. The
	// browser applies the same rule after an engine refusal (client/turns.js
	// applyOps).
	notCreated := map[string]bool{}
	for _, u := range uses {
		runs, known := toolgen.RunsWhere(u.Name)
		call := pendingCall{ID: u.ID, Name: u.Name, Input: u.Input}
		switch {
		case !known:
			call.Server = true
			refused = true
			res := model.ToolResult{ID: u.ID, IsError: true, Content: errJSON("E_OP", fmt.Sprintf("unknown tool %q", u.Name))}
			results = append(results, res)
			pd.ServerResults = append(pd.ServerResults, serverResult{ID: u.ID, Content: res.Content, IsError: true})
		case (runs == toolgen.RunsBrowser || runs == toolgen.RunsSuspend) && notCreated[toolgen.PageOf(u.Input)]:
			call.Server = true
			refused = true
			t.rec.MarkServer(u.ID)
			msg := fmt.Sprintf("page %q was not created (create_page was refused earlier in this reply), so this op was not applied", toolgen.PageOf(u.Input))
			res := model.ToolResult{ID: u.ID, IsError: true, Content: errJSON("E_TARGET", msg)}
			results = append(results, res)
			pd.ServerResults = append(pd.ServerResults, serverResult{ID: u.ID, Content: res.Content, IsError: true})
		case runs == toolgen.RunsServer && overruled && u.Name == toolgen.ReportUnsupported:
			call.Server = true
			t.rec.MarkServer(u.ID)
			res := model.ToolResult{ID: u.ID, IsError: true, Content: errJSON("E_ESCALATED", reasonRecheck)}
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
			if out, bad := toolgen.ValidateBrowserOp(u.Name, u.Input); bad {
				// A reset / restore_default without keys would clear the
				// whole target in the engine, and a create_page without a
				// usable title cannot be applied: answered here, is_error,
				// and never emitted as a tool.call (README §2, §4 step 4).
				// `bad`, not `refused`: the named result must see it, or a
				// batch with no browser op left would not escalate.
				call.Server = true
				refused = true
				t.rec.MarkServer(u.ID)
				res := model.ToolResult{ID: u.ID, Content: out.Content, IsError: true}
				results = append(results, res)
				pd.ServerResults = append(pd.ServerResults, serverResult{ID: u.ID, Content: res.Content, IsError: true})
				if u.Name == toolgen.CreatePage {
					for _, slug := range toolgen.CreatePageSlugs(u.Input) {
						notCreated[slug] = true
					}
				}
				break
			}
			call.Seq = t.seq + len(calls) + 1
			calls = append(calls, u)
		}
		pd.Calls = append(pd.Calls, call)
	}
	if len(pd.browserIDs()) == 0 {
		return results, nil, calls, refused
	}
	return results, pd, calls, refused
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
	// refused: an op of this batch was refused — by the engine (the browser
	// answered ok:false; nothing was applied for it) or by the service
	// before it was shown. On a route with a fallback the next call
	// escalates (Run applies it: this is not a stream). A cancelled or
	// failed asset pick, and a result the browser never sent, are not the
	// model's doing and do not count.
	refused := false
	// hidden: an applied op set a field the merchant will not see
	// (hiddenWrites) — the other trigger judged here; a refusal wins when
	// both happen in one batch.
	var hidden []string
	for _, c := range t.pending.Calls {
		if c.Server {
			sr, ok := t.pending.serverResult(c.ID)
			if !ok {
				sr = serverResult{ID: c.ID, Content: errJSON("E_OP", reasonServerLost), IsError: true}
			} else if sr.IsError && isOp(c.Name) {
				refused = true
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
		if !res.OK && isOp(c.Name) {
			refused = true
		}
		if res.OK && t.canEscalate() {
			hidden = append(hidden, hiddenWrites(t.compiled.Manifest, t.viewedPage(), c.Name, c.Input, res)...)
		}
	}
	ops, clean := t.cleanBatch(answered)
	hadErrorBefore := t.hadError
	if err := t.appendResults(ctx, results); err != nil {
		return err // t.pending kept: failResume answers the batch from it
	}
	t.pending = nil
	switch {
	case !t.canEscalate():
	case refused:
		t.wantEscalate = EscalateToolError
	case len(hidden) > 0:
		t.r.d.Logf("loop: turn %s: hidden write(s) %v on route %s", t.id, hidden, t.backend.Name())
		t.wantEscalate = EscalateHiddenWrite
	}
	t.decideFastFinish(ops, clean, hadErrorBefore)
	return nil
}

// viewedPage is the page of the turn's context ("" when the turn was rebuilt
// from storage, which keeps no context, or the context names none).
func (t *turn) viewedPage() string {
	tc, err := prompt.ParseContext(t.req.Context)
	if err != nil {
		return ""
	}
	return tc.Page
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
	usageJSON, _ := json.Marshal(t.storedUsage())
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
	for _, r := range results {
		if r.IsError {
			t.hadError = true
			break
		}
	}
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
	usageJSON, _ := json.Marshal(t.storedUsage())
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
	data := map[string]any{"usage": t.usageDoc(), "stop": stop, "conversationId": t.conv.ID, "route": t.routeDoc()}
	if t.rotated {
		data["conversationRotated"] = true
	}
	if t.hybrid() {
		// Which routes this turn called, first-use order: the route alone,
		// or the route and its fallback after an escalation.
		data["routes"] = append([]string{}, t.routes...)
		data["models"] = t.modelsDoc()
	}
	if t.fastFinished {
		// No closing model call: the last text.delta was the service's own
		// (`source:"service"`), and no assistant message closes the turn.
		data["fast_finish"] = true
	}
	t.emit(em, api.EventTurnDone, data)
	t.primeNext = true
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
	usageJSON, _ := json.Marshal(t.storedUsage())
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
		Call: t.calls, Route: t.backend.Name(), Model: t.backend.ReportedModel(resp.Model), StopReason: resp.StopReason, RequestID: resp.RequestID,
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
		TurnID: t.id, Operation: t.operation, RequestModel: t.backend.Model(), ReportAs: t.backend.ReportAs(),
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

// storedUsage is the turns.usage column: usageDoc, plus — on a hybrid turn
// only — the routes it called and its escalation, and `fast_finish:true` on
// a turn that ended without its closing model call.
func (t *turn) storedUsage() map[string]any {
	doc := t.usageDoc()
	if t.hybrid() {
		doc["routes"] = append([]string{}, t.routes...)
		doc["models"] = t.modelsDoc()
		if t.escalated != nil {
			doc["escalated"] = t.escalated
		}
	}
	if t.fastFinished {
		doc["fast_finish"] = true
	}
	return doc
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
