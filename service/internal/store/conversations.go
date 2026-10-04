package store

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
)

// Conversation is one row of conversations: a site's running Messages API
// history. At most one per site is open (ended_at IS NULL).
type Conversation struct {
	ID      string
	SiteID  string
	StoreID string
	Created time.Time
	Updated time.Time
	EndedAt *time.Time
}

// Message is one row of messages: a Messages API turn, content = the block
// array exactly as the request carried or the response returned it (minus
// thinking, which the trace package strips before storage).
type Message struct {
	ConversationID string
	Seq            int
	Role           string // "user" | "assistant"
	Content        json.RawMessage
	Created        time.Time
}

// OpenConversation returns the site's open conversation, creating one when
// none exists. Two concurrent first turns race on the partial unique index;
// the loser re-reads the winner's row.
func (s *Store) OpenConversation(ctx context.Context, siteID, storeID string) (*Conversation, error) {
	if siteID == "" {
		return nil, errors.New("store: siteID required")
	}
	if c, err := s.openConversation(ctx, siteID); err == nil {
		return c, nil
	} else if !errors.Is(err, ErrNotFound) {
		return nil, err
	}
	c, err := s.CreateConversation(ctx, siteID, storeID)
	if err == nil {
		return c, nil
	}
	if isUniqueViolation(err) {
		return s.openConversation(ctx, siteID)
	}
	return nil, err
}

// CreateConversation inserts a new open conversation. It fails with a
// unique violation while the site already has one open; callers that want
// "rotate" call EndConversation first (RotateConversation does both).
func (s *Store) CreateConversation(ctx context.Context, siteID, storeID string) (*Conversation, error) {
	c, err := insertConversation(ctx, s.pool, siteID, storeID, s.now())
	if err != nil {
		return nil, fmt.Errorf("store: create conversation: %w", err)
	}
	return c, nil
}

// conversationCols is the SELECT list scanConversation reads.
const conversationCols = `id, site_id, store_id, created, updated, ended_at`

// insertConversation is the INSERT behind CreateConversation, Rotate-
// Conversation and StartTurn; it fails with a unique violation while the
// site has an open conversation.
func insertConversation(ctx context.Context, q querier, siteID, storeID string, now time.Time) (*Conversation, error) {
	return scanConversation(q.QueryRow(ctx, `
		INSERT INTO conversations (site_id, store_id, created, updated)
		VALUES ($1, $2, $3, $3)
		RETURNING `+conversationCols, siteID, storeID, now))
}

// RotateConversation ends the site's open conversation (if any) and opens a
// fresh one in a single transaction — the 40-message rotation of README §3.
// POST /v1/turns rotates inside StartTurn (under the per-site lock, so two
// Starts at the boundary never collide on conversations_open_site); this is
// the stand-alone form.
func (s *Store) RotateConversation(ctx context.Context, siteID, storeID string) (*Conversation, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback(ctx) //nolint:errcheck
	now := s.now()
	if _, err := tx.Exec(ctx, `UPDATE conversations SET ended_at = $2, updated = $2 WHERE site_id = $1 AND ended_at IS NULL`, siteID, now); err != nil {
		return nil, fmt.Errorf("store: rotate: end: %w", err)
	}
	c, err := insertConversation(ctx, tx, siteID, storeID, now)
	if err != nil {
		return nil, fmt.Errorf("store: rotate: create: %w", err)
	}
	if err := tx.Commit(ctx); err != nil {
		return nil, err
	}
	return c, nil
}

// GetConversation by id.
func (s *Store) GetConversation(ctx context.Context, id string) (*Conversation, error) {
	row := s.pool.QueryRow(ctx, `SELECT `+conversationCols+` FROM conversations WHERE id = $1`, id)
	c, err := scanConversation(row)
	if err != nil {
		return nil, notFound(err)
	}
	return c, nil
}

// openConversation is the site's open conversation or ErrNotFound.
func (s *Store) openConversation(ctx context.Context, siteID string) (*Conversation, error) {
	row := s.pool.QueryRow(ctx, `SELECT `+conversationCols+` FROM conversations WHERE site_id = $1 AND ended_at IS NULL`, siteID)
	c, err := scanConversation(row)
	if err != nil {
		return nil, notFound(err)
	}
	return c, nil
}

// FindOpenConversation returns the site's open conversation without
// creating one (GET /v1/conversations/{siteId}); ErrNotFound when none.
func (s *Store) FindOpenConversation(ctx context.Context, siteID string) (*Conversation, error) {
	return s.openConversation(ctx, siteID)
}

// EndedConversation is EndSiteConversation's result.
type EndedConversation struct {
	Ended      bool             // the site had an open conversation
	Superseded []SupersededTurn // the site's live turns closed first
}

// EndSiteConversation is DELETE /v1/conversations/{siteId}: under the same
// per-site advisory lock as StartTurn, every live turn of the site is
// superseded (a turn still waiting on, or running in, the conversation must
// not continue late and write into it — README §4) and the open conversation
// is ended. Serialized with the site's Starts, so a Start either sees the
// ended conversation and opens a fresh one, or is superseded here.
func (s *Store) EndSiteConversation(ctx context.Context, siteID string) (*EndedConversation, error) {
	if siteID == "" {
		return nil, errors.New("store: siteID required")
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback(ctx) //nolint:errcheck
	if err := lockSite(ctx, tx, siteID); err != nil {
		return nil, err
	}
	now := s.now()
	out := &EndedConversation{}
	if out.Superseded, err = supersedeSite(ctx, tx, siteID, "", now); err != nil {
		return nil, err
	}
	tag, err := tx.Exec(ctx, `
		UPDATE conversations SET ended_at = $2, updated = $2
		WHERE site_id = $1 AND ended_at IS NULL`, siteID, now)
	if err != nil {
		return nil, fmt.Errorf("store: end conversation: %w", err)
	}
	out.Ended = tag.RowsAffected() > 0
	if err := tx.Commit(ctx); err != nil {
		return nil, err
	}
	return out, nil
}

// EndConversation sets ended_at on the site's open conversation and
// reports whether there was one, without superseding its turns or taking
// the per-site lock (EndSiteConversation does both; this is the bare form).
func (s *Store) EndConversation(ctx context.Context, siteID string) (bool, error) {
	tag, err := s.pool.Exec(ctx, `
		UPDATE conversations SET ended_at = $2, updated = $2
		WHERE site_id = $1 AND ended_at IS NULL`, siteID, s.now())
	if err != nil {
		return false, fmt.Errorf("store: end conversation: %w", err)
	}
	return tag.RowsAffected() > 0, nil
}

// AppendMessage appends one Messages API turn and returns its seq. seq is
// dense per conversation: the insert computes max(seq)+1 under a row lock
// on the conversation so two concurrent appends never collide.
func (s *Store) AppendMessage(ctx context.Context, conversationID, role string, content json.RawMessage) (int, error) {
	if role != "user" && role != "assistant" {
		return 0, fmt.Errorf("store: message role %q", role)
	}
	if len(content) == 0 {
		return 0, errors.New("store: message content required")
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return 0, err
	}
	defer tx.Rollback(ctx) //nolint:errcheck
	seq, err := appendMessageTx(ctx, tx, conversationID, role, content, s.now())
	if err != nil {
		return 0, err
	}
	if err := tx.Commit(ctx); err != nil {
		return 0, err
	}
	return seq, nil
}

// AppendMessageAfter appends one turn only while the conversation's last
// message is still afterSeq (0 = the conversation is empty) and returns the
// new seq; otherwise ErrConflict and nothing is written. The check and the
// insert run under the conversation's row lock, so it is a true
// compare-and-set: the loop answers an assistant tool_use message with this
// so a reaper (or another instance) that already closed the tail is
// detected instead of the same id being answered twice.
func (s *Store) AppendMessageAfter(ctx context.Context, conversationID, role string, content json.RawMessage, afterSeq int) (int, error) {
	if role != "user" && role != "assistant" {
		return 0, fmt.Errorf("store: message role %q", role)
	}
	if len(content) == 0 {
		return 0, errors.New("store: message content required")
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return 0, err
	}
	defer tx.Rollback(ctx) //nolint:errcheck
	if err := lockOpenConversation(ctx, tx, conversationID); err != nil {
		return 0, err
	}
	var tail int
	if err := tx.QueryRow(ctx, `SELECT COALESCE(MAX(seq), 0) FROM messages WHERE conversation_id = $1`, conversationID).Scan(&tail); err != nil {
		return 0, fmt.Errorf("store: append message: tail: %w", err)
	}
	if tail != afterSeq {
		return 0, ErrConflict
	}
	seq, err := appendMessageTx(ctx, tx, conversationID, role, content, s.now())
	if err != nil {
		return 0, err
	}
	if err := tx.Commit(ctx); err != nil {
		return 0, err
	}
	return seq, nil
}

// AppendMessages appends several turns atomically (an assistant turn and
// the user tool_result turn that answers it) and returns the first seq.
func (s *Store) AppendMessages(ctx context.Context, conversationID string, msgs []Message) (firstSeq int, err error) {
	if len(msgs) == 0 {
		return 0, nil
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return 0, err
	}
	defer tx.Rollback(ctx) //nolint:errcheck
	now := s.now()
	for i, m := range msgs {
		seq, err := appendMessageTx(ctx, tx, conversationID, m.Role, m.Content, now)
		if err != nil {
			return 0, err
		}
		if i == 0 {
			firstSeq = seq
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return 0, err
	}
	return firstSeq, nil
}

// lockOpenConversation takes the conversation's row lock (so concurrent
// appends serialise) and refuses an ended conversation: ErrConversationEnded
// (which is an ErrConflict). A conversation the 40-message rotation closed,
// or a DELETE ended, is never replayed again, so nothing may be written
// into it — in particular not the late reply of a turn superseded by the
// Start that rotated it away, whose compare-and-set would otherwise pass
// against its own text still on that tail (README §4 "Late writes").
func lockOpenConversation(ctx context.Context, tx pgx.Tx, conversationID string) error {
	var ended *time.Time
	if err := tx.QueryRow(ctx, `SELECT ended_at FROM conversations WHERE id = $1 FOR UPDATE`, conversationID).Scan(&ended); err != nil {
		return fmt.Errorf("store: append message: lock conversation: %w", notFound(err))
	}
	if ended != nil {
		return fmt.Errorf("store: append message: conversation %s: %w", conversationID, ErrConversationEnded)
	}
	return nil
}

func appendMessageTx(ctx context.Context, tx pgx.Tx, conversationID, role string, content json.RawMessage, now time.Time) (int, error) {
	if err := lockOpenConversation(ctx, tx, conversationID); err != nil {
		return 0, err
	}
	var seq int
	err := tx.QueryRow(ctx, `
		INSERT INTO messages (conversation_id, seq, role, content, created)
		SELECT $1, COALESCE(MAX(seq), 0) + 1, $2, $3, $4 FROM messages WHERE conversation_id = $1
		RETURNING seq`, conversationID, role, content, now).Scan(&seq)
	if err != nil {
		return 0, fmt.Errorf("store: append message: %w", err)
	}
	if _, err := tx.Exec(ctx, `UPDATE conversations SET updated = $2 WHERE id = $1`, conversationID, now); err != nil {
		return 0, fmt.Errorf("store: append message: touch: %w", err)
	}
	return seq, nil
}

// ListMessages returns up to limit messages in seq order. beforeSeq > 0
// returns only messages with seq < beforeSeq (the ?before= page); the slice
// is the newest `limit` of those, still ascending — what the replay and the
// GET /v1/conversations/{siteId} view both want.
func (s *Store) ListMessages(ctx context.Context, conversationID string, beforeSeq, limit int) ([]Message, error) {
	if limit <= 0 {
		limit = 200
	}
	rows, err := s.pool.Query(ctx, `
		SELECT conversation_id, seq, role, content, created FROM (
			SELECT conversation_id, seq, role, content, created
			FROM messages
			WHERE conversation_id = $1 AND ($2 <= 0 OR seq < $2)
			ORDER BY seq DESC
			LIMIT $3
		) newest ORDER BY seq ASC`, conversationID, beforeSeq, limit)
	if err != nil {
		return nil, fmt.Errorf("store: list messages: %w", err)
	}
	defer rows.Close()
	out := []Message{}
	for rows.Next() {
		var m Message
		if err := rows.Scan(&m.ConversationID, &m.Seq, &m.Role, &m.Content, &m.Created); err != nil {
			return nil, err
		}
		out = append(out, m)
	}
	return out, rows.Err()
}

// CountMessages is what the 40-message rotation check reads.
func (s *Store) CountMessages(ctx context.Context, conversationID string) (int, error) {
	var n int
	if err := s.pool.QueryRow(ctx, `SELECT COUNT(*) FROM messages WHERE conversation_id = $1`, conversationID).Scan(&n); err != nil {
		return 0, fmt.Errorf("store: count messages: %w", err)
	}
	return n, nil
}

func scanConversation(row pgx.Row) (*Conversation, error) {
	var c Conversation
	if err := row.Scan(&c.ID, &c.SiteID, &c.StoreID, &c.Created, &c.Updated, &c.EndedAt); err != nil {
		return nil, err
	}
	return &c, nil
}
