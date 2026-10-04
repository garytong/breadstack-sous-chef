package store

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

// Turn statuses (README §6; the loop's state machine in §4).
const (
	TurnRunning      = "running"
	TurnWaitingTools = "waiting_tools"
	TurnSuspended    = "suspended"
	TurnDone         = "done"
	TurnError        = "error"
	TurnExpired      = "expired"
	// TurnSuperseded: a live turn (running/waiting_tools/suspended) closed
	// by a newer POST /v1/turns for the same site, or by DELETE
	// /v1/conversations/{siteId}, before it finished (README §4). Terminal;
	// its late tool-results are 409 with this status, which the launcher
	// treats as "drop the pending results".
	TurnSuperseded = "superseded"
)

// Turn is one row of turns.
type Turn struct {
	ID               string
	ConversationID   string
	Status           string
	ManifestHash     string
	BuildID          string
	Model            string
	UserEmail        string
	Started          time.Time
	Ended            *time.Time
	Usage            json.RawMessage // nil until the turn ends
	PendingToolCalls json.RawMessage // nil unless waiting_tools/suspended
}

// NewTurn is what CreateTurn takes.
type NewTurn struct {
	ConversationID string
	ManifestHash   string
	BuildID        string
	Model          string
	UserEmail      string
}

// CreateTurn inserts a running turn and returns it.
func (s *Store) CreateTurn(ctx context.Context, in NewTurn) (*Turn, error) {
	row := s.pool.QueryRow(ctx, `
		INSERT INTO turns (conversation_id, status, manifest_hash, build_id, model, user_email, started)
		VALUES ($1, 'running', $2, $3, $4, $5, $6)
		RETURNING `+turnCols, in.ConversationID, in.ManifestHash, in.BuildID, in.Model, in.UserEmail, s.now())
	t, err := scanTurn(row)
	if err != nil {
		return nil, fmt.Errorf("store: create turn: %w", err)
	}
	return t, nil
}

// GetTurn by id.
func (s *Store) GetTurn(ctx context.Context, id string) (*Turn, error) {
	row := s.pool.QueryRow(ctx, `SELECT `+turnCols+` FROM turns WHERE id = $1`, id)
	t, err := scanTurn(row)
	if err != nil {
		return nil, notFound(err)
	}
	return t, nil
}

// SetTurnWaiting moves a running turn to waiting_tools or suspended with
// the tool_use blocks the browser must answer. ErrNotFound when the turn is
// not running (a stale continuation, the reaper got there first, or a newer
// turn superseded it mid-call).
func (s *Store) SetTurnWaiting(ctx context.Context, id, status string, pending json.RawMessage) error {
	if status != TurnWaitingTools && status != TurnSuspended {
		return fmt.Errorf("store: SetTurnWaiting: status %q", status)
	}
	tag, err := s.pool.Exec(ctx, `
		UPDATE turns SET status = $2, pending_tool_calls = $3
		WHERE id = $1 AND status = 'running'`, id, status, nullableJSON(pending))
	if err != nil {
		return fmt.Errorf("store: set turn waiting: %w", err)
	}
	if tag.RowsAffected() == 0 {
		return ErrNotFound
	}
	return nil
}

// ResumeTurn moves a waiting_tools/suspended turn back to running and
// clears pending_tool_calls, returning the pending blocks the caller must
// match results against. ErrNotFound when the turn is not waiting — the
// handler's 409 turn_not_waiting.
func (s *Store) ResumeTurn(ctx context.Context, id string) (pending json.RawMessage, err error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback(ctx) //nolint:errcheck
	err = tx.QueryRow(ctx, `
		SELECT pending_tool_calls FROM turns
		WHERE id = $1 AND status IN ('waiting_tools', 'suspended') FOR UPDATE`, id).Scan(&pending)
	if err != nil {
		return nil, notFound(err)
	}
	if _, err := tx.Exec(ctx, `UPDATE turns SET status = 'running', pending_tool_calls = NULL WHERE id = $1`, id); err != nil {
		return nil, fmt.Errorf("store: resume turn: %w", err)
	}
	if err := tx.Commit(ctx); err != nil {
		return nil, err
	}
	return pending, nil
}

// FinishTurn ends a turn as done or error with its summed usage.
func (s *Store) FinishTurn(ctx context.Context, id, status string, usage json.RawMessage) error {
	if status != TurnDone && status != TurnError {
		return fmt.Errorf("store: FinishTurn: status %q", status)
	}
	tag, err := s.pool.Exec(ctx, `
		UPDATE turns SET status = $2, ended = $3, usage = $4, pending_tool_calls = NULL
		WHERE id = $1 AND status IN ('running', 'waiting_tools', 'suspended')`, id, status, s.now(), nullableJSON(usage))
	if err != nil {
		return fmt.Errorf("store: finish turn: %w", err)
	}
	if tag.RowsAffected() == 0 {
		return ErrNotFound
	}
	return nil
}

// ExpiredTurn is one turn the TTL reaper expired, with what the reaping
// instance needs to close the conversation's dangling tool_use blocks: the
// conversation and the pending_tool_calls doc as it stood (nil for a turn
// that was still running).
type ExpiredTurn struct {
	ID               string
	ConversationID   string
	WasStatus        string          // running | waiting_tools | suspended
	PendingToolCalls json.RawMessage // the doc before it was cleared; nil when none
	// HasNewer is true when the conversation has a turn started after this
	// one (by (started, id)): the conversation's tail then belongs to that
	// turn, not to this one, and a reaper with no pending doc to match
	// against must leave it alone (loop.Runner.Reap).
	HasNewer bool
}

// ExpireStaleTurnsWithPending is the TTL reaper: every live turn started
// before now-ttl becomes expired. Each row is returned once, to exactly one
// caller (the UPDATE re-checks the live status under the row lock, so two
// instances reaping at once never both claim the same turn), with its
// pre-expiry pending doc so that caller can append the is_error tool_result
// message that keeps the conversation replayable (loop.Runner.Reap).
func (s *Store) ExpireStaleTurnsWithPending(ctx context.Context, ttl time.Duration) ([]ExpiredTurn, error) {
	cutoff := s.now().Add(-ttl)
	rows, err := s.pool.Query(ctx, `
		UPDATE turns t SET status = 'expired', ended = $2, pending_tool_calls = NULL
		FROM (
			SELECT id, conversation_id, status, pending_tool_calls, started FROM turns
			WHERE status IN ('running', 'waiting_tools', 'suspended') AND started < $1
		) old
		WHERE t.id = old.id AND t.status IN ('running', 'waiting_tools', 'suspended')
		RETURNING t.id, t.conversation_id, old.status, old.pending_tool_calls,
			EXISTS (SELECT 1 FROM turns n WHERE n.conversation_id = t.conversation_id
				AND (n.started, n.id) > (old.started, old.id))`, cutoff, s.now())
	if err != nil {
		return nil, fmt.Errorf("store: expire turns: %w", err)
	}
	defer rows.Close()
	var out []ExpiredTurn
	for rows.Next() {
		var e ExpiredTurn
		if err := rows.Scan(&e.ID, &e.ConversationID, &e.WasStatus, &e.PendingToolCalls, &e.HasNewer); err != nil {
			return nil, err
		}
		out = append(out, e)
	}
	return out, rows.Err()
}

// ExpireStaleTurns is ExpireStaleTurnsWithPending returning only the ids.
func (s *Store) ExpireStaleTurns(ctx context.Context, ttl time.Duration) ([]string, error) {
	expired, err := s.ExpireStaleTurnsWithPending(ctx, ttl)
	if err != nil {
		return nil, err
	}
	var ids []string
	for _, e := range expired {
		ids = append(ids, e.ID)
	}
	return ids, nil
}

// SupersededTurn is one turn SupersedeTurns/SupersedeTurnsForSite/StartTurn/
// EndSiteConversation closed,
// with the pending doc it was waiting on (nil for a turn that was still
// running: it was mid-model-call on some instance and has no batch yet).
type SupersededTurn struct {
	ID               string
	ConversationID   string
	WasStatus        string // running | waiting_tools | suspended
	PendingToolCalls json.RawMessage
}

// liveStatuses is the SQL list of the statuses a turn can still move from.
const liveStatuses = `('running', 'waiting_tools', 'suspended')`

// SupersedeTurns closes every live (running/waiting_tools/suspended) turn
// of one conversation as superseded. SupersedeTurnsForSite is what POST
// /v1/turns and DELETE /v1/conversations use (README §4); this is the
// per-conversation form. Each row is returned once, to exactly one caller
// (the UPDATE re-checks the status under the row lock). A running turn is
// closed too: its loop notices at its next row check, SetTurnWaiting or
// FinishTurn (all status-guarded) and ends turn.error turn_expired, after
// answering its own tail if it had just persisted one.
func (s *Store) SupersedeTurns(ctx context.Context, conversationID string) ([]SupersededTurn, error) {
	rows, err := s.pool.Query(ctx, `
		UPDATE turns t SET status = 'superseded', ended = $2, pending_tool_calls = NULL
		FROM (
			SELECT id, conversation_id, status, pending_tool_calls FROM turns
			WHERE conversation_id = $1 AND status IN `+liveStatuses+`
		) old
		WHERE t.id = old.id AND t.status IN `+liveStatuses+`
		RETURNING t.id, old.conversation_id, old.status, old.pending_tool_calls`, conversationID, s.now())
	if err != nil {
		return nil, fmt.Errorf("store: supersede turns: %w", err)
	}
	return scanSuperseded(rows)
}

// SupersedeTurnsForSite closes every live turn of EVERY conversation of a
// site (open, rotated or ended) as superseded, outside the per-site lock.
// StartTurn and EndSiteConversation do the same inside it; this is the
// stand-alone form (tests, operators). Same once-per-row guarantee as
// SupersedeTurns.
func (s *Store) SupersedeTurnsForSite(ctx context.Context, siteID string) ([]SupersededTurn, error) {
	return supersedeSite(ctx, s.pool, siteID, "", s.now())
}

// StartTurn is what POST /v1/turns asks of storage, as ONE transaction
// serialized per site by a transaction-scoped advisory lock on the site id
// (README §4 "One live turn per site"): the conversation is resolved (the
// named one, else the site's open one, created when none), rotated when it
// holds RotateAt or more messages, the turn row is inserted, and every
// other live turn of the site (running/waiting_tools/suspended, on any of
// its conversations) is superseded and returned with its pending doc.
// Because every Start for a site runs this block one after another, the
// most recently committed Start is always the only live turn of the site:
// two Starts racing on two instances cannot both survive, and clock skew
// between instances (turns.started is the instance's clock) is irrelevant.
// The rotation and the supersede sharing the lock also means two Starts at
// the rotation boundary never collide on conversations_open_site.
type StartTurn struct {
	SiteID  string
	StoreID string
	// ConversationID names the conversation (POST /v1/turns conversationId);
	// "" = the site's open conversation. It must be open and the site's:
	// ErrNotFound when it is gone or another site's, ErrConversationEnded
	// when it has ended.
	ConversationID string
	// RotateAt > 0 rotates the conversation when it already holds that many
	// messages: the open one is ended and a fresh one opened (README §3).
	RotateAt int

	ManifestHash string
	BuildID      string
	Model        string
	UserEmail    string
}

// StartedTurn is StartTurn's result.
type StartedTurn struct {
	Turn         *Turn
	Conversation *Conversation // the turn's conversation (the fresh one when Rotated)
	Rotated      bool
	// Superseded lists the site's live turns this Start closed, each with
	// its pending doc (nil for a turn that was still running).
	Superseded []SupersededTurn
}

// StartTurn runs the per-site Start block described on the StartTurn type.
func (s *Store) StartTurn(ctx context.Context, in StartTurn) (*StartedTurn, error) {
	if in.SiteID == "" {
		return nil, errors.New("store: siteID required")
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback(ctx) //nolint:errcheck
	if err := lockSite(ctx, tx, in.SiteID); err != nil {
		return nil, err
	}
	now := s.now()
	out := &StartedTurn{}
	if in.ConversationID != "" {
		c, err := scanConversation(tx.QueryRow(ctx, `SELECT `+conversationCols+` FROM conversations WHERE id = $1`, in.ConversationID))
		switch {
		case err != nil:
			return nil, fmt.Errorf("store: start turn: conversation: %w", notFound(err))
		case c.SiteID != in.SiteID:
			return nil, fmt.Errorf("store: start turn: conversation %s is not site %s's: %w", c.ID, in.SiteID, ErrNotFound)
		case c.EndedAt != nil:
			return nil, fmt.Errorf("store: start turn: conversation %s: %w", c.ID, ErrConversationEnded)
		}
		out.Conversation = c
	} else {
		c, err := scanConversation(tx.QueryRow(ctx, `SELECT `+conversationCols+` FROM conversations WHERE site_id = $1 AND ended_at IS NULL`, in.SiteID))
		switch {
		case errors.Is(err, pgx.ErrNoRows):
			c, err = insertConversation(ctx, tx, in.SiteID, in.StoreID, now)
			if err != nil {
				return nil, fmt.Errorf("store: start turn: open conversation: %w", err)
			}
		case err != nil:
			return nil, fmt.Errorf("store: start turn: open conversation: %w", err)
		}
		out.Conversation = c
	}
	if in.RotateAt > 0 {
		var n int
		if err := tx.QueryRow(ctx, `SELECT COUNT(*) FROM messages WHERE conversation_id = $1`, out.Conversation.ID).Scan(&n); err != nil {
			return nil, fmt.Errorf("store: start turn: count messages: %w", err)
		}
		if n >= in.RotateAt {
			if _, err := tx.Exec(ctx, `UPDATE conversations SET ended_at = $2, updated = $2 WHERE site_id = $1 AND ended_at IS NULL`, in.SiteID, now); err != nil {
				return nil, fmt.Errorf("store: start turn: rotate: end: %w", err)
			}
			c, err := insertConversation(ctx, tx, in.SiteID, in.StoreID, now)
			if err != nil {
				return nil, fmt.Errorf("store: start turn: rotate: create: %w", err)
			}
			out.Conversation, out.Rotated = c, true
		}
	}
	t, err := scanTurn(tx.QueryRow(ctx, `
		INSERT INTO turns (conversation_id, status, manifest_hash, build_id, model, user_email, started)
		VALUES ($1, 'running', $2, $3, $4, $5, $6)
		RETURNING `+turnCols, out.Conversation.ID, in.ManifestHash, in.BuildID, in.Model, in.UserEmail, now))
	if err != nil {
		return nil, fmt.Errorf("store: start turn: create turn: %w", err)
	}
	out.Turn = t
	superseded, err := supersedeSite(ctx, tx, in.SiteID, t.ID, now)
	if err != nil {
		return nil, err
	}
	out.Superseded = superseded
	if err := tx.Commit(ctx); err != nil {
		return nil, err
	}
	return out, nil
}

// querier is the slice of *pgxpool.Pool and pgx.Tx the shared SQL needs.
type querier interface {
	Exec(ctx context.Context, sql string, args ...any) (pgconn.CommandTag, error)
	Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error)
	QueryRow(ctx context.Context, sql string, args ...any) pgx.Row
}

// lockSite takes the transaction-scoped advisory lock that serializes the
// per-site operations (StartTurn, EndSiteConversation). Class 1 of the
// two-key form; released at commit or rollback.
func lockSite(ctx context.Context, tx querier, siteID string) error {
	if _, err := tx.Exec(ctx, `SELECT pg_advisory_xact_lock(1, hashtext($1))`, siteID); err != nil {
		return fmt.Errorf("store: lock site %s: %w", siteID, err)
	}
	return nil
}

// supersedeSite is the shared UPDATE: every live turn of the site becomes
// superseded, except the turn exceptID when not "" (the Start's own row).
// Each row is returned once, to exactly one caller (the UPDATE re-checks the
// live status under the row lock).
func supersedeSite(ctx context.Context, q querier, siteID, exceptID string, now time.Time) ([]SupersededTurn, error) {
	if siteID == "" {
		return nil, errors.New("store: siteID required")
	}
	var except *string
	if exceptID != "" {
		except = &exceptID
	}
	rows, err := q.Query(ctx, `
		UPDATE turns t SET status = 'superseded', ended = $2, pending_tool_calls = NULL
		FROM (
			SELECT tt.id, tt.conversation_id, tt.status, tt.pending_tool_calls
			FROM turns tt JOIN conversations c ON c.id = tt.conversation_id
			WHERE c.site_id = $1 AND tt.status IN `+liveStatuses+`
			  AND ($3::uuid IS NULL OR tt.id <> $3::uuid)
		) old
		WHERE t.id = old.id AND t.status IN `+liveStatuses+`
		RETURNING t.id, old.conversation_id, old.status, old.pending_tool_calls`, siteID, now, except)
	if err != nil {
		return nil, fmt.Errorf("store: supersede turns for site: %w", err)
	}
	return scanSuperseded(rows)
}

func scanSuperseded(rows pgx.Rows) ([]SupersededTurn, error) {
	defer rows.Close()
	var out []SupersededTurn
	for rows.Next() {
		var e SupersededTurn
		if err := rows.Scan(&e.ID, &e.ConversationID, &e.WasStatus, &e.PendingToolCalls); err != nil {
			return nil, err
		}
		out = append(out, e)
	}
	return out, rows.Err()
}

// ListTurns returns a conversation's turns oldest first.
func (s *Store) ListTurns(ctx context.Context, conversationID string) ([]Turn, error) {
	rows, err := s.pool.Query(ctx, `SELECT `+turnCols+` FROM turns WHERE conversation_id = $1 ORDER BY started`, conversationID)
	if err != nil {
		return nil, fmt.Errorf("store: list turns: %w", err)
	}
	defer rows.Close()
	out := []Turn{}
	for rows.Next() {
		t, err := scanTurn(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, *t)
	}
	return out, rows.Err()
}

// CountLiveTurns counts running/waiting/suspended turns across all
// conversations (an observability number; MAX_CONCURRENT_TURNS is enforced
// in memory per instance).
func (s *Store) CountLiveTurns(ctx context.Context) (int, error) {
	var n int
	err := s.pool.QueryRow(ctx, `SELECT COUNT(*) FROM turns WHERE status IN ('running', 'waiting_tools', 'suspended')`).Scan(&n)
	return n, err
}

const turnCols = `id, conversation_id, status, manifest_hash, build_id, model, user_email, started, ended, usage, pending_tool_calls`

func scanTurn(row pgx.Row) (*Turn, error) {
	var t Turn
	if err := row.Scan(&t.ID, &t.ConversationID, &t.Status, &t.ManifestHash, &t.BuildID, &t.Model, &t.UserEmail,
		&t.Started, &t.Ended, &t.Usage, &t.PendingToolCalls); err != nil {
		return nil, err
	}
	return &t, nil
}
