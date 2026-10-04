package loop

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/breadstack/sous-chef/internal/model"
	"github.com/breadstack/sous-chef/internal/store"
)

// A conversation whose last message is an assistant turn with tool_use
// blocks that no tool_result answers cannot be replayed: the Messages API
// rejects the next request (400, "tool_use ids were found without
// tool_result blocks") until the conversation rotates. Every path that
// persists an assistant tool_use turn and then stops — max_calls, a stop
// other than tool_use that still carries tool_use blocks, the reaper beating
// SetTurnWaiting, a continuation that fails after ResumeTurn, TTL expiry —
// appends ONE user message of is_error tool_result blocks, one per
// unanswered id, so the stored history stays valid (append-only; nothing is
// rewritten). history() keeps a read-side guard for anything that still
// slips through (a crash between the two writes).

// errLostTail is returned by a compare-and-set append when the conversation
// no longer ends where this writer last left it: someone else (the reaper,
// another instance) appended in between, and whatever this writer meant to
// answer has in all likelihood been answered already.
var errLostTail = errors.New("loop: conversation tail moved")

// errSuperseded is errLostTail's shape when the row check found the turn
// superseded (a newer turn started for the site, or its conversation was
// ended): the same turn_expired exit, with a message that says which.
var errSuperseded = fmt.Errorf("loop: turn superseded: %w", errLostTail)

// Reasons given to the model for a call it never got an answer to.
const (
	reasonMaxCalls      = "not executed: the turn reached its call limit"
	reasonNoToolStop    = "not executed: the response ended without requesting tool use"
	reasonExpired       = "not executed: the turn expired before the browser answered"
	reasonContinueFail  = "not executed: the continuation failed before the results were recorded"
	reasonExtraPick     = "only one pick_asset per batch: call pick_asset alone, in its own response, after the other edits are confirmed"
	reasonServerLost    = "server result lost"
	reasonHistoryRepair = "not executed: the turn ended before this call was answered"
	reasonSuperseded    = "unanswered: the user sent a new message before the browser reported this call's result; the page's current state is in this message's context block"
)

// toolUseRef is the id and name of one tool_use block.
type toolUseRef struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

// danglingToolUses returns the tool_use blocks of the conversation's last
// stored message when that message is an assistant turn — exactly the ids
// the next request would leave unanswered. Nil when the tail is a user
// message (every call answered) or the conversation is empty.
func danglingToolUses(msgs []store.Message) []toolUseRef {
	if len(msgs) == 0 {
		return nil
	}
	tail := msgs[len(msgs)-1]
	if tail.Role != "assistant" {
		return nil
	}
	return toolUsesIn(tail.Content)
}

// toolUsesIn lists the tool_use blocks of a stored content array.
func toolUsesIn(content json.RawMessage) []toolUseRef {
	var blocks []struct {
		Type string `json:"type"`
		ID   string `json:"id"`
		Name string `json:"name"`
	}
	if json.Unmarshal(content, &blocks) != nil {
		return nil
	}
	var out []toolUseRef
	for _, b := range blocks {
		if b.Type == "tool_use" && b.ID != "" {
			out = append(out, toolUseRef{ID: b.ID, Name: b.Name})
		}
	}
	return out
}

// paramToolUses lists the tool_use blocks of a request-side message.
func paramToolUses(m anthropic.BetaMessageParam) []toolUseRef {
	if m.Role != anthropic.BetaMessageParamRoleAssistant {
		return nil
	}
	var out []toolUseRef
	for _, c := range m.Content {
		if c.OfToolUse != nil && c.OfToolUse.ID != "" {
			out = append(out, toolUseRef{ID: c.OfToolUse.ID, Name: c.OfToolUse.Name})
		}
	}
	return out
}

// abandonResults builds the tool_result blocks that close the given calls:
// a server tool whose answer was stashed in pd gets that answer (the work
// was done), everything else is_error with reason.
func abandonResults(uses []toolUseRef, pd *pendingDoc, reason string) []model.ToolResult {
	out := make([]model.ToolResult, 0, len(uses))
	for _, u := range uses {
		if pd != nil {
			if sr, ok := pd.serverResult(u.ID); ok {
				out = append(out, model.ToolResult{ID: u.ID, Content: sr.Content, IsError: sr.IsError})
				continue
			}
		}
		out = append(out, model.ToolResult{ID: u.ID, IsError: true, Content: errJSON("E_OP", reason)})
	}
	return out
}

// closeDangling answers whatever tool_use blocks the conversation's tail
// leaves unanswered (ground truth = the stored messages, so a reaper and a
// loop racing on the same turn append at most once between them and a tail
// that is already closed is left alone). The append is a compare-and-set on
// the tail it read; when the tail moves under it the tail is re-read, and a
// tail that is closed by then is left alone. pd, when known, names the
// batch being closed: only tail ids in pd.Calls are answered (a tail that
// shares none of them belongs to a younger turn of the conversation and is
// left alone) and its stashed server results are used. With pd nil the
// whole tail is answered; callers pass nil only when the tail is known to
// be theirs. Returns the number of ids answered.
func (r *Runner) closeDangling(ctx context.Context, conversationID string, pd *pendingDoc, reason string) (int, error) {
	for attempt := 0; ; attempt++ {
		msgs, err := r.d.Store.ListMessages(ctx, conversationID, 0, 1)
		if err != nil {
			return 0, err
		}
		uses := danglingToolUses(msgs)
		if pd != nil {
			uses = ownUses(uses, pd)
		}
		if len(uses) == 0 {
			return 0, nil
		}
		m := model.ToolResultsTurn(abandonResults(uses, pd, reason))
		content, err := json.Marshal(m.Content)
		if err != nil {
			return 0, err
		}
		_, err = r.d.Store.AppendMessageAfter(ctx, conversationID, string(m.Role), content, msgs[len(msgs)-1].Seq)
		if err == nil {
			return len(uses), nil
		}
		if errors.Is(err, store.ErrConflict) && attempt < 2 {
			continue // someone appended meanwhile; look at the new tail
		}
		return 0, fmt.Errorf("loop: close %d dangling tool_use in conversation %s: %w", len(uses), conversationID, err)
	}
}

// ownUses keeps the tool_use refs that are calls of pd — the ids this
// writer is entitled to answer.
func ownUses(uses []toolUseRef, pd *pendingDoc) []toolUseRef {
	var out []toolUseRef
	for _, u := range uses {
		for _, c := range pd.Calls {
			if c.ID == u.ID {
				out = append(out, u)
				break
			}
		}
	}
	return out
}

// abandon closes this turn's own last assistant message: one user message
// of is_error tool_result blocks for every tool_use in uses, appended to
// the in-memory history and the store (compare-and-set on that assistant
// message: errLostTail when the reaper answered it first). The turn makes no
// further model call after this; the message exists so the NEXT turn's
// replay is valid.
func (t *turn) abandon(ctx context.Context, uses []model.Block, reason string) error {
	if len(uses) == 0 {
		return nil
	}
	refs := make([]toolUseRef, 0, len(uses))
	for _, u := range uses {
		refs = append(refs, toolUseRef{ID: u.ID, Name: u.Name})
	}
	m := model.ToolResultsTurn(abandonResults(refs, t.pending, reason))
	if err := t.persistAfter(ctx, m); err != nil {
		return fmt.Errorf("loop: turn %s: abandon %d tool_use: %w", t.id, len(uses), err)
	}
	t.history = append(t.history, m)
	return nil
}

// supersededTurns is what Start's supersede hands its tail repair: the
// pending docs of the turns it closed that were waiting on the new turn's
// conversation, and whether one it closed was still running there (between
// its model call and SetTurnWaiting, or about to finish — its assistant
// message may be the tail, with no doc to match).
type supersededTurns struct {
	docs    []*pendingDoc
	running bool
}

// owner names the superseded turn a dangling tail belongs to: the doc whose
// calls cover every missing id (matched by tool_use id — pd.Seq is a
// per-turn event counter and says nothing across turns), else nil with ok
// when a running turn was superseded (its tail, no stashed results); not ok
// when nothing superseded explains the tail.
func (s *supersededTurns) owner(missing []toolUseRef) (pd *pendingDoc, ok bool) {
	if s == nil {
		return nil, false
	}
	for _, d := range s.docs {
		if len(ownUses(missing, d)) == len(missing) {
			return d, true
		}
	}
	return nil, s.running
}

// repairHistory is the read-side guard for a replayed conversation: every
// assistant tool_use block that the following message does not answer gets
// a synthetic is_error user message right after it, so the request is valid
// whatever the stored history looks like (the Messages API merges adjacent
// user messages). The whole history is checked, not just the tail: a tail
// that dangled when an earlier turn started is mid-history now. The second
// result is the repair of the LAST message, when it needed one — the only
// repair storage can still take (Start persists it before the new user
// text); nil otherwise. superseded, when not nil, describes the turns Start
// just superseded: a tail made of one's calls is the expected outcome of
// the user moving on (answered with its stashed server results and
// reasonSuperseded, no warning). A dangling tail no superseded doc explains
// is looked up among the conversation's turns (tailOwner): when the turn
// that left it was closed by another actor — expired by a reaper whose
// closing append has not landed yet, or superseded by a Start that
// committed between this one's transaction and its history read (its doc
// went to that Start) — the repair is the expected outcome of that race,
// answered with that actor's reason and logged as information. Any other
// repair (a tail of a turn that ended done/error without closing it, a
// tail nothing explains, a dangle mid-history) means a write path failed
// to close its own tail and is logged as a WARNING worth a look. selfID is
// the starting turn's row, left out of the lookup.
func (r *Runner) repairHistory(ctx context.Context, conversationID, selfID string, msgs []anthropic.BetaMessageParam, superseded *supersededTurns) ([]anthropic.BetaMessageParam, *anthropic.BetaMessageParam) {
	var out []anthropic.BetaMessageParam
	var tail *anthropic.BetaMessageParam
	repaired := 0
	for i, m := range msgs {
		out = append(out, m)
		uses := paramToolUses(m)
		if len(uses) == 0 {
			continue
		}
		answered := map[string]bool{}
		if i+1 < len(msgs) && msgs[i+1].Role == anthropic.BetaMessageParamRoleUser {
			for _, c := range msgs[i+1].Content {
				if c.OfToolResult != nil {
					answered[c.OfToolResult.ToolUseID] = true
				}
			}
		}
		var missing []toolUseRef
		for _, u := range uses {
			if !answered[u.ID] {
				missing = append(missing, u)
			}
		}
		if len(missing) == 0 {
			continue
		}
		var fix anthropic.BetaMessageParam
		last := i == len(msgs)-1
		switch pd, ok := superseded.owner(missing); {
		case last && ok:
			fix = model.ToolResultsTurn(abandonResults(missing, pd, reasonSuperseded))
		case last:
			owner := r.tailOwner(ctx, conversationID, selfID)
			if owner == nil || !closedByAnother(owner.Status) {
				fix = model.ToolResultsTurn(abandonResults(missing, nil, reasonHistoryRepair))
				repaired += len(missing)
				break
			}
			fix = model.ToolResultsTurn(abandonResults(missing, nil, reasonForStatus(owner.Status)))
			r.d.Logf("loop: conversation %s: turn %s (%s) left %d tool_use id(s) unanswered on the tail; answering them is_error before this turn's text (the close of whoever %s it may still be landing on another instance)", conversationID, owner.ID, owner.Status, len(missing), owner.Status)
		default:
			fix = model.ToolResultsTurn(abandonResults(missing, nil, reasonHistoryRepair))
			repaired += len(missing)
		}
		out = append(out, fix)
		if last {
			tail = &fix
		}
	}
	if tail == nil && repaired == 0 {
		return msgs, nil
	}
	if repaired > 0 {
		r.d.Logf("loop: WARNING conversation %s replayed with %d unanswered tool_use id(s); answering them is_error for this request", conversationID, repaired)
	}
	return out, tail
}

// tailOwner names the turn of conversationID (other than selfID) that left
// the dangling tail: the conversation's newest other turn by `started` (the
// tail is the last message; its writer is the latest turn before this
// one). Nil when the conversation has no other turn or the turns cannot be
// read.
//
// This is a heuristic, not a lookup: no stored pending doc can identify the
// owner here, because every terminal transition NULLs pending_tool_calls
// (FinishTurn, expire, supersede) and StartTurn has already superseded
// every live turn of the site before this runs, so by now no turn of the
// conversation holds a doc. `turns.started` is the writing instance's
// clock, so the heuristic is clock-skew-sensitive: when two turns' rows
// were written by instances whose clocks differ by more than the gap
// between the turns, the "newest" row may be the older turn, and a tail
// left by a turn whose own write path failed (done/error → WARNING) can be
// attributed to the other one (expired/superseded → INFO). The repair
// itself is the same either way; only the log level and the tool_result
// text differ. A skew-proof owner needs a monotonic column on turns (none
// exists: uuid ids, instance-clock `started`); not added for a log-level
// precision issue (README §4 "No dangling").
func (r *Runner) tailOwner(ctx context.Context, conversationID, selfID string) *store.Turn {
	turns, err := r.d.Store.ListTurns(ctx, conversationID)
	if err != nil {
		r.d.Logf("loop: conversation %s: turns of a dangling tail: %v", conversationID, err)
		return nil
	}
	var newest *store.Turn
	for i := range turns {
		t := &turns[i]
		if t.ID == selfID {
			continue
		}
		if newest == nil || !t.Started.Before(newest.Started) {
			newest = t // ListTurns is oldest first; a tie goes to the later row
		}
	}
	return newest
}

// closedByAnother reports whether a turn row was closed by someone other
// than its own loop — the reaper (expired) or a newer Start / a DELETE
// (superseded) — whose closing of its tail is their business, not a write
// path of the turn's that failed. done and error are the turn's own exits.
func closedByAnother(status string) bool {
	return status == store.TurnExpired || status == store.TurnSuperseded
}

// reasonForStatus is the tool_result text for a call its turn never
// answered, by the status of whoever closed the turn.
func reasonForStatus(status string) string {
	if status == store.TurnExpired {
		return reasonExpired
	}
	return reasonSuperseded
}
