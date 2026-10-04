package store

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

// Trace is one row of traces: a production turn kept for 90 days (user
// decision) unless the store opted out. Ops are the model's raw tool calls;
// Events what the stream emitted and the browser reported back — stored
// apart so training never learns a browser correction as model output.
type Trace struct {
	TurnID   string
	SiteID   string
	StoreID  string
	Context  json.RawMessage
	Events   json.RawMessage
	Ops      json.RawMessage
	Feedback json.RawMessage // nil until POST /v1/feedback
	Created  time.Time
}

// InsertTrace stores t. A second insert for the same turn replaces events,
// ops and context (the loop writes once at turn end; a retry after a
// transient failure must not fail on the primary key) and keeps feedback.
func (s *Store) InsertTrace(ctx context.Context, t Trace) error {
	if t.TurnID == "" || t.SiteID == "" {
		return errors.New("store: trace needs turn_id and site_id")
	}
	if len(t.Events) == 0 {
		t.Events = json.RawMessage(`[]`)
	}
	if len(t.Ops) == 0 {
		t.Ops = json.RawMessage(`[]`)
	}
	created := t.Created
	if created.IsZero() {
		created = s.now()
	}
	_, err := s.pool.Exec(ctx, `
		INSERT INTO traces (turn_id, site_id, store_id, context, events, ops, feedback, created)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
		ON CONFLICT (turn_id) DO UPDATE SET
			context = EXCLUDED.context,
			events  = EXCLUDED.events,
			ops     = EXCLUDED.ops`,
		t.TurnID, t.SiteID, t.StoreID, nullableJSON(t.Context), t.Events, t.Ops, nullableJSON(t.Feedback), created)
	if err != nil {
		return fmt.Errorf("store: insert trace: %w", err)
	}
	return nil
}

// GetTrace by turn id.
func (s *Store) GetTrace(ctx context.Context, turnID string) (*Trace, error) {
	row := s.pool.QueryRow(ctx, `
		SELECT turn_id, site_id, store_id, context, events, ops, feedback, created
		FROM traces WHERE turn_id = $1`, turnID)
	var t Trace
	if err := row.Scan(&t.TurnID, &t.SiteID, &t.StoreID, &t.Context, &t.Events, &t.Ops, &t.Feedback, &t.Created); err != nil {
		return nil, notFound(err)
	}
	return &t, nil
}

// MaxFeedbackBytes caps the MERGED traces.feedback document per turn (its
// compact JSON text). POST /v1/feedback is append-by-merge, so without a cap
// a client could grow one turn's feedback without bound.
const MaxFeedbackBytes = 16 << 10

// ErrFeedbackTooLarge: merging this feedback would push the turn's feedback
// document past MaxFeedbackBytes. Nothing was written.
var ErrFeedbackTooLarge = errors.New("store: merged feedback exceeds 16 KiB")

// MergeFeedback shallow-merges feedback (a JSON object) into the trace's
// feedback column: `feedback = coalesce(feedback,'{}') || $2`, so a later
// {saved:true} keeps an earlier {kept_paths:[…]}. ErrNotFound when the
// turn has no trace (opted-out store, or already pruned) — the feedback
// handler still answers 204 in that case. ErrFeedbackTooLarge when the
// merged document would exceed MaxFeedbackBytes (the row is untouched).
func (s *Store) MergeFeedback(ctx context.Context, turnID string, feedback json.RawMessage) error {
	if len(feedback) == 0 {
		return errors.New("store: feedback required")
	}
	if len(feedback) > MaxFeedbackBytes {
		return ErrFeedbackTooLarge
	}
	tag, err := s.pool.Exec(ctx, `
		UPDATE traces SET feedback = COALESCE(feedback, '{}'::jsonb) || $2::jsonb
		WHERE turn_id = $1
		  AND octet_length((COALESCE(feedback, '{}'::jsonb) || $2::jsonb)::text) <= $3`,
		turnID, feedback, MaxFeedbackBytes)
	if err != nil {
		return fmt.Errorf("store: merge feedback: %w", err)
	}
	if tag.RowsAffected() == 0 {
		// Distinguish "no trace" from "would exceed the cap".
		var exists bool
		if err := s.pool.QueryRow(ctx, `SELECT EXISTS (SELECT 1 FROM traces WHERE turn_id = $1)`, turnID).Scan(&exists); err != nil {
			return fmt.Errorf("store: merge feedback: %w", err)
		}
		if exists {
			return ErrFeedbackTooLarge
		}
		return ErrNotFound
	}
	return nil
}

// ListTraces returns traces created in [from, to) oldest first, up to
// limit — the lab export's read (Phase 5).
func (s *Store) ListTraces(ctx context.Context, from, to time.Time, limit int) ([]Trace, error) {
	if limit <= 0 {
		limit = 500
	}
	rows, err := s.pool.Query(ctx, `
		SELECT turn_id, site_id, store_id, context, events, ops, feedback, created
		FROM traces WHERE created >= $1 AND created < $2
		ORDER BY created LIMIT $3`, from, to, limit)
	if err != nil {
		return nil, fmt.Errorf("store: list traces: %w", err)
	}
	defer rows.Close()
	out := []Trace{}
	for rows.Next() {
		var t Trace
		if err := rows.Scan(&t.TurnID, &t.SiteID, &t.StoreID, &t.Context, &t.Events, &t.Ops, &t.Feedback, &t.Created); err != nil {
			return nil, err
		}
		out = append(out, t)
	}
	return out, rows.Err()
}

// --- store_optout ---

// IsOptedOut reports whether storeID's turns must not be traced. An empty
// store id is never opted out (there is nothing to look up).
func (s *Store) IsOptedOut(ctx context.Context, storeID string) (bool, error) {
	if storeID == "" {
		return false, nil
	}
	var one int
	err := s.pool.QueryRow(ctx, `SELECT 1 FROM store_optout WHERE store_id = $1`, storeID).Scan(&one)
	if err != nil {
		if notFound(err) == ErrNotFound {
			return false, nil
		}
		return false, fmt.Errorf("store: optout lookup: %w", err)
	}
	return true, nil
}

// SetOptOut records the opt-out (PUT /v1/stores/{storeId}/trace-optout);
// idempotent, the first by_email wins.
func (s *Store) SetOptOut(ctx context.Context, storeID, byEmail string) error {
	if storeID == "" {
		return errors.New("store: storeID required")
	}
	_, err := s.pool.Exec(ctx, `
		INSERT INTO store_optout (store_id, created, by_email) VALUES ($1, $2, $3)
		ON CONFLICT (store_id) DO NOTHING`, storeID, s.now(), byEmail)
	if err != nil {
		return fmt.Errorf("store: set optout: %w", err)
	}
	return nil
}

// ClearOptOut removes the flag (DELETE …/trace-optout); reports whether it
// was set.
func (s *Store) ClearOptOut(ctx context.Context, storeID string) (bool, error) {
	tag, err := s.pool.Exec(ctx, `DELETE FROM store_optout WHERE store_id = $1`, storeID)
	if err != nil {
		return false, fmt.Errorf("store: clear optout: %w", err)
	}
	return tag.RowsAffected() > 0, nil
}
