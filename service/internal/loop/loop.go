// Package loop is the agent loop (README §4): the turn state machine behind
// POST /v1/turns and POST /v1/turns/{id}/tool-results.
//
// The service decides, the browser applies. A turn calls the model (at most
// MaxCalls times), answers the server tools (lookup_fields, describe_target,
// report_unsupported) inline, and hands every data op to the browser as a
// tool.call event; the turn is then persisted as waiting_tools (or suspended
// for pick_asset) and the stream closes. The browser applies the ops, posts
// one tool-results body, and the same turn continues — in memory when the
// continuation lands on the same instance, rebuilt from Postgres otherwise.
//
// The loop shape is breadstack-thermometer server/internal/chat/chat.go
// respond(): maxIterations, the assistant turn appended before its results,
// all results of a batch in ONE user message, the last iteration running no
// tool, noConclusion when the model still wanted tools.
package loop

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/breadstack/sous-chef/internal/api"
	"github.com/breadstack/sous-chef/internal/fields"
	"github.com/breadstack/sous-chef/internal/model"
	"github.com/breadstack/sous-chef/internal/prompt"
	"github.com/breadstack/sous-chef/internal/registry"
	"github.com/breadstack/sous-chef/internal/store"
	"github.com/breadstack/sous-chef/internal/trace"
	"github.com/breadstack/sous-chef/internal/usage"
)

// Defaults (README §4, §7).
const (
	DefaultMaxCalls      = 8
	DefaultTurnTTL       = 10 * time.Minute
	DefaultMaxConcurrent = 8
	// DefaultRotateAt is the message count at which a site's conversation is
	// rotated (README §3: 40 messages, no compaction in v1).
	DefaultRotateAt = 40
	// historyLimit bounds a replay read; a conversation rotates well below it.
	historyLimit = 1000
)

// Store is the slice of *store.Store the loop needs (conversations,
// messages, turns, plus trace.Store for the recorder).
type Store interface {
	GetConversation(ctx context.Context, id string) (*store.Conversation, error)
	ListMessages(ctx context.Context, conversationID string, beforeSeq, limit int) ([]store.Message, error)
	// AppendMessageAfter appends only while the tail is still afterSeq
	// (store.ErrConflict otherwise). Every message a turn writes goes
	// through it: a turn owns the tail of its conversation for as long as it
	// is the site's live turn, so a reaper that closed its batch first, or a
	// newer turn that took the conversation over, is noticed rather than
	// written over.
	AppendMessageAfter(ctx context.Context, conversationID, role string, content json.RawMessage, afterSeq int) (int, error)
	// StartTurn is the Start's storage block, serialized per site: resolve
	// (and rotate) the conversation, insert the running turn row, supersede
	// every other live turn of the site — one transaction, so the most
	// recently committed Start is the site's only live turn, whatever the
	// instances' clocks say and however two Starts interleave.
	StartTurn(ctx context.Context, in store.StartTurn) (*store.StartedTurn, error)
	GetTurn(ctx context.Context, id string) (*store.Turn, error)
	// ListTurns returns a conversation's turns oldest first; a Start's tail
	// repair uses it to name the turn that left the tail dangling when no
	// superseded pending doc explains it (repairHistory).
	ListTurns(ctx context.Context, conversationID string) ([]store.Turn, error)
	SetTurnWaiting(ctx context.Context, id, status string, pending json.RawMessage) error
	ResumeTurn(ctx context.Context, id string) (json.RawMessage, error)
	FinishTurn(ctx context.Context, id, status string, usage json.RawMessage) error
	ExpireStaleTurnsWithPending(ctx context.Context, ttl time.Duration) ([]store.ExpiredTurn, error)
	trace.Store
}

// Loader resolves a manifest hash to its compiled prefix (registry.Registry).
type Loader interface {
	Load(ctx context.Context, hash string) (*registry.Compiled, error)
}

// Deps wires a Runner.
type Deps struct {
	Store     Store
	Manifests Loader
	Models    *model.Table
	// Usage records one row per model call; nil = off.
	Usage *usage.Recorder
	// Tracing off when false (traces are on by default; per-store opt-out is
	// the store's flag).
	NoTraces bool

	MaxCalls      int           // 0 → DefaultMaxCalls
	TurnTTL       time.Duration // 0 → DefaultTurnTTL
	MaxConcurrent int           // 0 → DefaultMaxConcurrent
	RotateAt      int           // 0 → DefaultRotateAt
	Logf          func(string, ...any)
	Now           func() time.Time
}

// Runner implements api.TurnRunner.
type Runner struct {
	d Deps

	mu     sync.Mutex
	live   int              // turns holding a slot (between Start/Continue and the end of Run)
	parked map[string]*turn // waiting/suspended turns on this instance
	noSite noSiteLog        // turns whose siteId-less body was logged already
}

// noSiteLog is the once-per-turn-per-instance set behind the "body carries
// no siteId" log line (tool-results on the Runner, feedback on Feedback).
type noSiteLog struct {
	mu   sync.Mutex
	seen map[string]bool
}

// noSiteLogCap bounds the once-per-turn log set; past it the set is reset
// (a repeat line per turn, not a leak).
const noSiteLogCap = 4096

// first reports whether turnID is seen for the first time, recording it.
func (l *noSiteLog) first(turnID string) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.seen[turnID] {
		return false
	}
	if l.seen == nil || len(l.seen) >= noSiteLogCap {
		l.seen = map[string]bool{}
	}
	l.seen[turnID] = true
	return true
}

// New validates deps and builds a Runner.
func New(d Deps) (*Runner, error) {
	if d.Store == nil || d.Manifests == nil || d.Models == nil {
		return nil, errors.New("loop: Store, Manifests and Models are required")
	}
	if d.MaxCalls <= 0 {
		d.MaxCalls = DefaultMaxCalls
	}
	if d.TurnTTL <= 0 {
		d.TurnTTL = DefaultTurnTTL
	}
	if d.MaxConcurrent <= 0 {
		d.MaxConcurrent = DefaultMaxConcurrent
	}
	if d.RotateAt <= 0 {
		d.RotateAt = DefaultRotateAt
	}
	if d.Logf == nil {
		d.Logf = log.Printf
	}
	if d.Now == nil {
		d.Now = func() time.Time { return time.Now().UTC() }
	}
	return &Runner{d: d, parked: map[string]*turn{}}, nil
}

// Start implements api.TurnRunner.Start: the pre-checks that need an HTTP
// status (409 unknown manifest, 400 bad route/context, 503 busy), then the
// conversation, the turn row, the user message, and the first request.
func (r *Runner) Start(ctx context.Context, req api.TurnRequest) (api.Turn, error) {
	compiled, err := r.d.Manifests.Load(ctx, req.ManifestHash)
	if err != nil {
		return nil, err
	}
	backend, err := r.d.Models.Route(req.Model)
	if err != nil {
		return nil, api.ErrBadRequest("model: " + err.Error())
	}
	if _, err := prompt.ContextBlock(req.Context, req.Text); err != nil {
		return nil, api.ErrBadRequest(err.Error())
	}
	release, ok := r.acquire()
	if !ok {
		return nil, api.ErrBusy()
	}
	t, err := r.start(ctx, req, compiled, backend)
	if err != nil {
		release()
		return nil, err
	}
	t.release = release
	return t, nil
}

// startAttempts bounds the history-read → user-text append cycle of start:
// each compare-and-set conflict re-reads the tail once more.
const startAttempts = 3

func (r *Runner) start(ctx context.Context, req api.TurnRequest, compiled *registry.Compiled, backend *model.Backend) (*turn, error) {
	if req.ConversationID != "" {
		// The 400s a named conversation can earn, answered before the row
		// exists; StartTurn re-checks under the lock and the mapping below
		// covers a conversation that changed in between.
		if err := r.checkNamedConversation(ctx, req); err != nil {
			return nil, err
		}
	}
	// ONE storage transaction, serialized per site (store.StartTurn): the
	// conversation (rotated when full), this turn's row, and the supersede of
	// every other live turn of the site — running, waiting_tools or
	// suspended, on any of the site's conversations (the open one, the one
	// the rotation just closed, one a DELETE ended). Two Starts for one site
	// therefore commit one after another and the later one supersedes the
	// earlier, whatever the order their requests arrived in and whatever
	// their instances' clocks say (README §4 "One live turn per site"). A
	// superseded turn's late continuation is refused (409 turn_not_waiting
	// status=superseded) and its loop stops at its next status-guarded write
	// or compare-and-set append. Anything that fails from here on closes the
	// row as error so it is not left running until the reaper.
	st, err := r.d.Store.StartTurn(ctx, store.StartTurn{
		SiteID: req.SiteID, StoreID: req.StoreID, ConversationID: req.ConversationID, RotateAt: r.d.RotateAt,
		ManifestHash: req.ManifestHash, BuildID: req.BuildID, Model: backend.ReportedModel(""), UserEmail: req.UserEmail,
	})
	switch {
	case errors.Is(err, store.ErrConversationEnded):
		return nil, api.ErrBadRequest("conversationId: conversation has ended")
	case errors.Is(err, store.ErrNotFound):
		return nil, api.ErrBadRequest("conversationId: no such conversation")
	case err != nil:
		return nil, err
	}
	if st.Rotated {
		r.d.Logf("loop: conversation for site %s rotated at %d messages → %s", req.SiteID, r.d.RotateAt, st.Conversation.ID)
	}
	t, err := r.begin(ctx, req, compiled, backend, st)
	if err != nil {
		if ferr := r.d.Store.FinishTurn(context.WithoutCancel(ctx), st.Turn.ID, store.TurnError, nil); ferr != nil && !errors.Is(ferr, store.ErrNotFound) {
			r.d.Logf("loop: turn %s: finish after failed start: %v", st.Turn.ID, ferr)
		}
		return nil, err
	}
	return t, nil
}

// checkNamedConversation is the 400 pre-check of POST /v1/turns
// conversationId: it must exist, belong to the site and be open.
func (r *Runner) checkNamedConversation(ctx context.Context, req api.TurnRequest) error {
	c, err := r.d.Store.GetConversation(ctx, req.ConversationID)
	switch {
	case errors.Is(err, store.ErrNotFound):
		return api.ErrBadRequest("conversationId: no such conversation")
	case err != nil:
		return err
	case c.SiteID != req.SiteID:
		return api.ErrBadRequest("conversationId belongs to another site")
	case c.EndedAt != nil:
		return api.ErrBadRequest("conversationId: conversation has ended")
	}
	return nil
}

// begin is start after the row exists: the superseded turns' bookkeeping,
// the history, the tail repair and the turn's own user text.
func (r *Runner) begin(ctx context.Context, req api.TurnRequest, compiled *registry.Compiled, backend *model.Backend, st *store.StartedTurn) (*turn, error) {
	conv, row := st.Conversation, st.Turn
	// The turns StartTurn superseded: the ones parked here are dropped and
	// their traces finished; the dangling batch one of them left on THIS
	// conversation is the tail repaired below, with the stashed server
	// results it holds.
	superseded := r.superseded(ctx, req.SiteID, conv.ID, row, st.Superseded)
	candidates := r.candidateFields(req, compiled, backend)
	for attempt := 1; ; attempt++ {
		history, tailSeq, err := r.history(ctx, conv.ID)
		if err != nil {
			return nil, err
		}
		// A new turn must not start after an unanswered tool_use anywhere in
		// the history (a rebuild, by contrast, is about to answer its tail
		// itself).
		history, repair := r.repairHistory(ctx, conv.ID, row.ID, history, superseded)
		next, err := prompt.BuildWithFields(compiled.Prefix, history, req.Context, req.Text, candidates)
		if err != nil {
			return nil, api.ErrBadRequest(err.Error())
		}
		t := &turn{
			r: r, id: row.ID, conv: conv, rotated: st.Rotated, compiled: compiled, backend: backend,
			req: req, history: next.Messages, next: next, operation: usage.OpChat, started: row.Started, tailSeq: tailSeq,
			origin: backend, fieldsContext: backend.FieldsContext(),
		}
		if repair != nil {
			// The stored tail dangles too: persist the repair before this
			// turn's user text, so the NEXT turn's replay is valid as well (a
			// request-side fix alone would leave the id unanswered in storage
			// for every later turn). Compare-and-set, with the same exits as
			// the user-text append below: a row that is no longer running (a
			// DELETE, or a newer Start, committed between this one's StartTurn
			// and this write and superseded it — the DELETE's case also
			// refuses the append as store.ErrConversationEnded, the
			// conversation being ended) stores nothing and ends at Run's first
			// row check, never as a failed Start; a still-running row whose
			// tail moved re-reads it (the superseded turn's own close of its
			// batch, or the reaper's, landed first — it is answered by now).
			switch err := t.persistAfter(ctx, *repair); {
			case err == nil:
			case errors.Is(err, errLostTail):
				status, serr := t.rowStatus(ctx)
				if serr != nil {
					return nil, r.storeUnavailableStart(t, "repair", serr)
				}
				if status != store.TurnRunning || errors.Is(err, store.ErrConversationEnded) {
					return r.lostStart(ctx, t, status), nil
				}
				if attempt < startAttempts {
					continue
				}
				return nil, fmt.Errorf("loop: turn %s: conversation %s tail moved %d times during start: %w", row.ID, conv.ID, attempt, err)
			default:
				return nil, err
			}
		}
		// The row must still be running right before the user text is
		// stored: a newer Start for the site may have committed since this
		// one's StartTurn (superseding this row) and stored ITS text already
		// — the history above would then be the newer turn's, and a text
		// appended behind it would break that turn's own compare-and-set. A
		// supersede that lands after this check is caught by the newer turn
		// instead: its history read follows its StartTurn, so it either sees
		// this text or its own append conflicts and re-reads.
		switch self, err := r.d.Store.GetTurn(ctx, row.ID); {
		case errors.Is(err, store.ErrNotFound):
			return nil, fmt.Errorf("loop: turn %s: row check before start: %w", row.ID, err)
		case err != nil:
			return nil, r.storeUnavailableStart(t, "user text", err)
		case self.Status != store.TurnRunning:
			return r.lostStart(ctx, t, self.Status), nil
		}
		// The user text is a compare-and-set on the tail read above, so the
		// request this turn is about to send is exactly the stored history:
		// a message that landed in between (a superseded turn's last write,
		// another Start's user text) is seen and replayed, never built over.
		switch err := t.persistAfter(ctx, next.Messages[len(next.Messages)-1]); {
		case err == nil:
			return r.traced(ctx, t), nil
		case errors.Is(err, errLostTail):
			status, serr := t.rowStatus(ctx)
			if serr != nil {
				return nil, r.storeUnavailableStart(t, "user text", serr)
			}
			if status != store.TurnRunning || errors.Is(err, store.ErrConversationEnded) {
				return r.lostStart(ctx, t, status), nil
			}
			if attempt < startAttempts {
				continue
			}
			return nil, fmt.Errorf("loop: turn %s: conversation %s tail moved %d times during start: %w", row.ID, conv.ID, attempt, err)
		default:
			return nil, err
		}
	}
}

// candidateFields is the candidate-fields block of this turn's user message
// (README §3), "" when the route does not carry one (fields_context off —
// every anthropic route unless it says otherwise) or nothing in the manifest
// matches the text. It is computed from the turn's own context and text
// only, once per turn, and becomes part of the stored user message: later
// turns replay it as it was sent and get a block of their own. A context
// the retrieval cannot read costs the block, never the turn.
func (r *Runner) candidateFields(req api.TurnRequest, compiled *registry.Compiled, backend *model.Backend) string {
	if !backend.FieldsContext() || compiled.Manifest == nil {
		return ""
	}
	fc, err := fields.ParseContext(req.Context)
	if err != nil {
		r.d.Logf("loop: candidate fields skipped for site %s: %v", req.SiteID, err)
		return ""
	}
	return fields.Candidates(compiled.Manifest, fc, strings.TrimSpace(req.Text), fields.Options{}).Block
}

// storeUnavailableStart is the exit of a Start whose row status could not
// be read when it needed it (the check before its user text, or the re-read
// after a compare-and-set append lost the tail): 503 store_unavailable, so
// the launcher retries, instead of guessing that the row was lost and
// handing out a turn that would run on a history nobody stored (README §4
// "Storage failures"). start finishes the row error when storage takes it;
// nothing of the turn is in the conversation.
func (r *Runner) storeUnavailableStart(t *turn, at string, cause error) error {
	r.d.Logf("loop: turn %s: row status unreadable before its %s was stored; the Start is refused 503 store_unavailable: %v", t.id, at, cause)
	return fmt.Errorf("loop: turn %s: %w: %v", t.id, api.ErrStoreUnavailable(), cause)
}

// traced attaches the trace recorder to a turn that is about to run.
func (r *Runner) traced(ctx context.Context, t *turn) *turn {
	if r.d.NoTraces {
		return t
	}
	t.rec = trace.New(ctx, r.d.Store, trace.Begin{
		TurnID: t.id, SiteID: t.req.SiteID, StoreID: t.req.StoreID, ManifestHash: t.req.ManifestHash,
		BuildID: t.req.BuildID, Route: t.backend.Name(), UserEmail: t.req.UserEmail, Text: t.req.Text,
		Context: t.req.Context, StartedAt: t.started,
	})
	return t
}

// lostStart returns a turn that lost the race for its site before its user
// text was stored: a newer Start superseded it between CreateTurn and the
// append. Nothing of it is in the conversation; Run ends it turn.error
// turn_expired at its first row check, without a model call.
func (r *Runner) lostStart(ctx context.Context, t *turn, status string) *turn {
	r.d.Logf("loop: turn %s: row is %s before its user text was stored; the turn ends without a model call", t.id, status)
	return r.traced(ctx, t)
}

// superseded does this instance's share of the supersede StartTurn ran
// (store side, once per row across instances): it drops the superseded
// turns parked here and finishes their traces, and returns what the tail
// repair of conversationID (the new turn's) needs — the pending docs of the
// superseded turns that were waiting on it, and whether one of them was
// still running there (its just-persisted assistant message may be the
// tail). A running turn's loop is not touched here: it ends itself at its
// next status-guarded write or compare-and-set append.
func (r *Runner) superseded(ctx context.Context, siteID, conversationID string, self *store.Turn, rows []store.SupersededTurn) *supersededTurns {
	out := &supersededTurns{}
	for _, e := range rows {
		if e.ConversationID == conversationID {
			var pd pendingDoc
			switch {
			case len(e.PendingToolCalls) > 0 && json.Unmarshal(e.PendingToolCalls, &pd) == nil:
				out.docs = append(out.docs, &pd)
			case e.WasStatus == store.TurnRunning:
				out.running = true
			}
		}
		if t := r.unpark(e.ID); t != nil {
			_ = t.rec.Finish(context.WithoutCancel(ctx), store.TurnSuperseded, "", "a newer turn started for the site before this one finished")
		}
		r.d.Logf("loop: site %s: turn %s (%s, conversation %s) superseded by turn %s on conversation %s", siteID, e.ID, e.WasStatus, e.ConversationID, self.ID, conversationID)
	}
	return out
}

// history replays a conversation's stored messages as request params and
// returns the seq of the last stored message (0 when empty): the tail every
// later compare-and-set append of the turn builds on.
func (r *Runner) history(ctx context.Context, conversationID string) ([]anthropic.BetaMessageParam, int, error) {
	rows, err := r.d.Store.ListMessages(ctx, conversationID, 0, historyLimit)
	if err != nil {
		return nil, 0, err
	}
	if len(rows) == 0 {
		return nil, 0, nil
	}
	type wire struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	msgs := make([]wire, len(rows))
	for i, m := range rows {
		msgs[i] = wire{Role: m.Role, Content: m.Content}
	}
	raw, err := json.Marshal(msgs)
	if err != nil {
		return nil, 0, err
	}
	out, err := model.UnmarshalMessages(raw)
	if err != nil {
		return nil, 0, fmt.Errorf("loop: replay conversation %s: %w", conversationID, err)
	}
	return out, rows[len(rows)-1].Seq, nil
}

// Continue implements api.TurnRunner.Continue: validate the results against
// the pending calls, flip the turn back to running, append ONE user message
// of tool_result blocks in call order, and hand back the turn to Run.
func (r *Runner) Continue(ctx context.Context, turnID string, results api.ToolResultsRequest) (api.Turn, error) {
	// The row is the ground truth for what this turn is waiting on. A parked
	// entry is used only while its pending doc is the row's: the same turn
	// may have been continued on another instance and parked again there with
	// a new batch, leaving this entry stale (it would refuse the new ids).
	row, err := r.d.Store.GetTurn(ctx, turnID)
	if errors.Is(err, store.ErrNotFound) {
		if t := r.unpark(turnID); t != nil {
			_ = t.rec.Finish(context.WithoutCancel(ctx), "error", "", "turn row gone while parked")
		}
		return nil, api.ErrTurnNotFound(turnID)
	}
	if err != nil {
		return nil, err // a parked entry stays parked
	}
	// Tenant check before anything that would reveal the turn's state: a
	// caller naming another site's turn gets the same 404 as for no turn.
	if err := r.verifySite(ctx, row, results.SiteID, results.StoreID, "tool-results"); err != nil {
		return nil, err
	}
	if row.Status != store.TurnWaitingTools && row.Status != store.TurnSuspended {
		if t := r.unpark(turnID); t != nil {
			_ = t.rec.Finish(context.WithoutCancel(ctx), row.Status, "", "turn finished on another instance; this trace holds the part run here")
		}
		return nil, api.ErrTurnNotWaiting(turnID, row.Status)
	}
	t := r.unpark(turnID)
	if t != nil && !t.ownsRow(row) {
		r.d.Logf("loop: turn %s: parked entry superseded by a continuation on another instance; rebuilding from storage", turnID)
		_ = t.rec.Finish(context.WithoutCancel(ctx), "superseded", "", "turn continued on another instance; this trace holds the part run here")
		t = nil
	}
	if t == nil {
		t, err = r.rebuild(ctx, row)
		if err != nil {
			return nil, err
		}
	}
	if t.pending == nil {
		r.park(t)
		return nil, api.ErrTurnNotWaiting(turnID, store.TurnRunning)
	}
	// Validate before touching the row so a 400 leaves the turn waiting.
	for i, res := range results.Results {
		if _, ok := t.pending.browser(res.ToolCallID); !ok {
			r.park(t)
			return nil, api.ErrBadRequest(fmt.Sprintf("results[%d].toolCallId %q is not a pending call of this turn", i, res.ToolCallID))
		}
	}
	release, ok := r.acquire()
	if !ok {
		r.park(t)
		return nil, api.ErrBusy()
	}
	if _, err := r.d.Store.ResumeTurn(ctx, turnID); err != nil {
		release()
		if errors.Is(err, store.ErrNotFound) {
			row, gerr := r.d.Store.GetTurn(ctx, turnID)
			if gerr != nil {
				return nil, api.ErrTurnNotFound(turnID)
			}
			return nil, api.ErrTurnNotWaiting(turnID, row.Status)
		}
		return nil, err
	}
	if err := t.resume(ctx, results); err != nil {
		release()
		// The row is running and its pending doc is gone: nothing can
		// continue this turn. Close it as an error and answer the batch
		// is_error so the conversation's tail stays replayable; the reaper
		// is the fallback when this write fails too.
		t.failResume(context.WithoutCancel(ctx), err)
		if errors.Is(err, errLostTail) {
			// Someone else (the reaper) answered the batch between ResumeTurn
			// and the write: the turn is over, and the browser is told so.
			status := store.TurnExpired
			if row, gerr := r.d.Store.GetTurn(ctx, turnID); gerr == nil {
				status = row.Status
			}
			return nil, api.ErrTurnNotWaiting(turnID, status)
		}
		return nil, err
	}
	t.release = release
	return t, nil
}

// verifySite checks a body's siteId/storeId against the turn's conversation
// (README §1, §11: the launcher proxy sends them; the service verifies).
// Each id is verified whenever present; a mismatch is api.ErrTurnNotFound —
// never a hint that the turn exists. A body without siteId is logged once
// per turn per instance (the proxy should send it); without either id it
// is accepted unverified (older proxies).
func (r *Runner) verifySite(ctx context.Context, row *store.Turn, siteID, storeID, what string) error {
	return verifyTenant(ctx, r.d.Store, &r.noSite, r.d.Logf, row, siteID, storeID, what)
}

// convReader is the slice of the store verifyTenant reads.
type convReader interface {
	GetConversation(ctx context.Context, id string) (*store.Conversation, error)
}

// verifyTenant is verifySite for a loaded turn row, shared by tool-results
// (Runner) and feedback (Feedback).
func verifyTenant(ctx context.Context, st convReader, once *noSiteLog, logf func(string, ...any), row *store.Turn, siteID, storeID, what string) error {
	if siteID == "" && once.first(row.ID) {
		if storeID == "" {
			logf("loop: turn %s: %s body carries no siteId; accepted unverified (the proxy should send it)", row.ID, what)
		} else {
			logf("loop: turn %s: %s body carries no siteId; only its storeId is verified (the proxy should send both)", row.ID, what)
		}
	}
	if siteID == "" && storeID == "" {
		return nil
	}
	conv, err := st.GetConversation(ctx, row.ConversationID)
	if errors.Is(err, store.ErrNotFound) {
		return api.ErrTurnNotFound(row.ID)
	}
	if err != nil {
		return err
	}
	if (siteID != "" && conv.SiteID != siteID) || (storeID != "" && conv.StoreID != storeID) {
		logf("loop: turn %s: %s body names site %q store %q, turn belongs to site %q store %q; refused as not found", row.ID, what, siteID, storeID, conv.SiteID, conv.StoreID)
		return api.ErrTurnNotFound(row.ID)
	}
	return nil
}

// rebuild reconstructs a waiting turn from Postgres (a continuation that
// landed on another instance, or after a restart). The trace recorder
// cannot be rebuilt; the tail of such a turn is not traced.
func (r *Runner) rebuild(ctx context.Context, row *store.Turn) (*turn, error) {
	var pd pendingDoc
	if len(row.PendingToolCalls) == 0 || json.Unmarshal(row.PendingToolCalls, &pd) != nil {
		return nil, api.ErrTurnNotWaiting(row.ID, row.Status)
	}
	compiled, err := r.d.Manifests.Load(ctx, row.ManifestHash)
	if err != nil {
		return nil, err
	}
	backend, origin, err := r.pendingBackend(&pd)
	if err != nil {
		return nil, fmt.Errorf("loop: rebuild turn %s: %w", row.ID, err)
	}
	fieldsContext := origin.FieldsContext()
	if pd.FieldsContext != nil {
		fieldsContext = *pd.FieldsContext
	}
	conv, err := r.d.Store.GetConversation(ctx, row.ConversationID)
	if err != nil {
		return nil, err
	}
	history, tailSeq, err := r.history(ctx, conv.ID)
	if err != nil {
		return nil, err
	}
	r.d.Logf("loop: turn %s rebuilt from storage (%d messages); its trace tail is not recorded", row.ID, len(history))
	return &turn{
		r: r, id: row.ID, conv: conv, rotated: pd.Rotated, compiled: compiled, backend: backend,
		req:     api.TurnRequest{SiteID: conv.SiteID, StoreID: conv.StoreID, ManifestHash: row.ManifestHash, BuildID: row.BuildID, UserEmail: row.UserEmail},
		history: history, calls: pd.CallCount, usage: pd.Usage, seq: pd.Seq, started: row.Started, pending: &pd, tailSeq: tailSeq,
		escalated: pd.Escalated, routes: pd.Routes, models: pd.Models, foreignBefore: pd.ForeignBefore,
		origin: origin, fieldsContext: fieldsContext,
	}, nil
}

// pendingBackend is the route a waiting turn continues on. A turn that has
// not escalated names an allowlisted route, checked as at Start. One that
// has (pd.Escalated) continues on its fallback, which no request may name
// and the allowlist need not hold: it is reached through the route the turn
// started on, and only while that route still falls back to it.
//
// origin is the route the turn started on: the same backend, or the
// escalated turn's `from`.
func (r *Runner) pendingBackend(pd *pendingDoc) (backend, origin *model.Backend, err error) {
	if pd.Escalated == nil {
		b, err := r.d.Models.Route(pd.Route)
		return b, b, err
	}
	origin, err = r.d.Models.Route(pd.Escalated.From)
	if err != nil {
		return nil, nil, err
	}
	fb := origin.Fallback()
	if fb == nil || fb.Name() != pd.Route {
		return nil, nil, fmt.Errorf("route %q no longer falls back to %q, the route this turn escalated to", pd.Escalated.From, pd.Route)
	}
	return fb, origin, nil
}

// ownsRow reports whether this parked turn's pending doc is the one the row
// holds — i.e. no other instance has continued the turn since it was parked
// here. Compared by shape (seq, call count, the batch's ids in order), not
// bytes: jsonb does not preserve key order.
func (t *turn) ownsRow(row *store.Turn) bool {
	if t.pending == nil || len(row.PendingToolCalls) == 0 {
		return false
	}
	var pd pendingDoc
	if json.Unmarshal(row.PendingToolCalls, &pd) != nil {
		return false
	}
	return samePending(t.pending, &pd)
}

// --- concurrency and the parked map ---

func (r *Runner) acquire() (func(), bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.live >= r.d.MaxConcurrent {
		return nil, false
	}
	r.live++
	var once sync.Once
	return func() {
		once.Do(func() {
			r.mu.Lock()
			r.live--
			r.mu.Unlock()
		})
	}, true
}

// Live is the number of turns holding a slot right now.
func (r *Runner) Live() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.live
}

func (r *Runner) park(t *turn) {
	r.mu.Lock()
	r.parked[t.id] = t
	r.mu.Unlock()
}

func (r *Runner) unpark(id string) *turn {
	r.mu.Lock()
	defer r.mu.Unlock()
	t := r.parked[id]
	delete(r.parked, id)
	return t
}

// Parked is the number of waiting/suspended turns held in memory.
func (r *Runner) Parked() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.parked)
}

// --- TTL reaper ---

// Reap expires every live turn older than TurnTTL (README §4 step 8),
// answers each expired turn's own unanswered tool_use blocks is_error (from
// the stored pending doc, on whichever instance reaped it; never a younger
// turn's tail on the same conversation), drops the
// in-memory continuations, and evicts parked turns this instance no longer
// owns: rows that completed elsewhere (no longer waiting/suspended), rows
// waiting on a batch parked elsewhere (continued on another instance), rows
// that vanished, and entries past TurnTTL even when the database cannot be
// asked. Returns the number of turns expired or evicted.
func (r *Runner) Reap(ctx context.Context) (int, error) {
	expired, err := r.d.Store.ExpireStaleTurnsWithPending(ctx, r.d.TurnTTL)
	if err != nil {
		// The TTL age-out of parked entries does not need the database.
		n := r.evictParked(ctx, false)
		return n, err
	}
	handled := map[string]bool{}
	closed := 0
	for _, e := range expired {
		handled[e.ID] = true
		var pd *pendingDoc
		if len(e.PendingToolCalls) > 0 {
			var doc pendingDoc
			if json.Unmarshal(e.PendingToolCalls, &doc) == nil {
				pd = &doc
			}
		}
		// The tail is this turn's only when nothing newer owns it: with a
		// pending doc, closeDangling answers just the doc's ids; without
		// one (the turn was mid-call) the tail is attributable only when
		// this is the conversation's newest turn.
		if pd == nil && e.HasNewer {
			r.d.Logf("loop: reaper: turn %s expired while running; a newer turn owns conversation %s's tail, nothing closed", e.ID, e.ConversationID)
		} else {
			n, cerr := r.closeDangling(ctx, e.ConversationID, pd, reasonExpired)
			if cerr != nil {
				r.d.Logf("loop: reaper: turn %s: %v", e.ID, cerr)
			}
			closed += n
		}
		if t := r.unpark(e.ID); t != nil {
			_ = t.rec.Finish(context.WithoutCancel(ctx), "expired", "", "turn expired after "+r.d.TurnTTL.String())
		}
	}
	evicted := r.evictParked(ctx, true)
	if len(expired) > 0 || evicted > 0 {
		r.d.Logf("loop: reaper expired %d turn(s), answered %d dangling tool_use id(s), evicted %d parked turn(s)", len(expired), closed, evicted)
	}
	return len(expired) + evicted, nil
}

// evictParked drops parked turns this instance should no longer hold. With
// askDB, each entry's row is read: a row that is not waiting/suspended any
// more (continued and finished on another instance, or expired by another
// reaper) or that is gone is evicted and its partial trace closed with the
// row's status. Regardless of the database, an entry older than TurnTTL is
// evicted as expired. Returns the number evicted.
func (r *Runner) evictParked(ctx context.Context, askDB bool) int {
	r.mu.Lock()
	snapshot := make(map[string]*turn, len(r.parked))
	for id, t := range r.parked {
		snapshot[id] = t
	}
	r.mu.Unlock()
	cutoff := r.d.Now().Add(-r.d.TurnTTL)
	bg := context.WithoutCancel(ctx)
	n := 0
	for id, t := range snapshot {
		if t.started.Before(cutoff) {
			if r.unpark(id) != nil {
				_ = t.rec.Finish(bg, "expired", "", "parked turn aged out after "+r.d.TurnTTL.String())
				n++
			}
			continue
		}
		if !askDB {
			continue
		}
		row, err := r.d.Store.GetTurn(ctx, id)
		switch {
		case errors.Is(err, store.ErrNotFound):
			if r.unpark(id) != nil {
				_ = t.rec.Finish(bg, "error", "", "turn row gone while parked")
				n++
			}
		case err != nil:
			r.d.Logf("loop: reaper: parked turn %s: %v", id, err)
		case row.Status == store.TurnDone, row.Status == store.TurnError, row.Status == store.TurnExpired, row.Status == store.TurnSuperseded:
			if r.unpark(id) != nil {
				_ = t.rec.Finish(bg, row.Status, "", "turn finished on another instance; this trace holds the part run here")
				n++
			}
		case row.Status == store.TurnWaitingTools, row.Status == store.TurnSuspended:
			// Waiting on a batch this entry does not know: continued and
			// parked again on another instance. The entry would refuse the
			// new ids, so it goes; a continuation here rebuilds from the row.
			if !t.ownsRow(row) && r.unpark(id) != nil {
				_ = t.rec.Finish(bg, "superseded", "", "turn continued on another instance; this trace holds the part run here")
				n++
			}
		default:
			// running: another instance is mid-continuation; its finish
			// shows up on the next pass (or the TTL age-out above takes it).
		}
	}
	return n
}

// RunReaper calls Reap every interval until ctx ends (main's goroutine).
func (r *Runner) RunReaper(ctx context.Context, interval time.Duration) {
	if interval <= 0 {
		interval = time.Minute
	}
	tick := time.NewTicker(interval)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
			if _, err := r.Reap(ctx); err != nil && ctx.Err() == nil {
				r.d.Logf("loop: reaper: %v", err)
			}
		}
	}
}
