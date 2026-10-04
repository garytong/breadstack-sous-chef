package store

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// UsageRow is one model call (README §6 usage_rows). Field names follow the
// thermometer's llm_usage row (server/internal/therm, migration 0009) so the
// shipper maps 1:1 onto /api/ingest/usage.
type UsageRow struct {
	ID               int64
	TurnID           string // "" when the call belonged to no turn
	Operation        string // "chat" | "tool_turn"
	Model            string
	SiteID           string
	SiteHost         string
	StoreID          string
	UserEmail        string
	InputTokens      int64
	OutputTokens     int64
	CacheWriteTokens int64
	CacheReadTokens  int64
	LatencyMS        *int64
	Status           string // "ok" | "error" | "refusal"
	StopReason       string
	RequestID        string
	OccurredAt       time.Time
	Shipped          bool
}

// InsertUsage stores one row and returns its id. Callers pass
// context.WithoutCancel(ctx) (thermometer recordUsage) so a client hanging
// up never loses the row.
func (s *Store) InsertUsage(ctx context.Context, r UsageRow) (int64, error) {
	if r.Model == "" {
		return 0, errors.New("store: usage row needs a model")
	}
	if r.Operation == "" {
		r.Operation = "chat"
	}
	if r.OccurredAt.IsZero() {
		r.OccurredAt = s.now()
	}
	var turnID any
	if r.TurnID != "" {
		turnID = r.TurnID
	}
	var id int64
	err := s.pool.QueryRow(ctx, `
		INSERT INTO usage_rows (turn_id, operation, model, site_id, site_host, store_id, user_email,
			input_tokens, output_tokens, cache_write_tokens, cache_read_tokens,
			latency_ms, status, stop_reason, request_id, occurred_at, shipped)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, false)
		RETURNING id`,
		turnID, r.Operation, r.Model, r.SiteID, r.SiteHost, r.StoreID, r.UserEmail,
		r.InputTokens, r.OutputTokens, r.CacheWriteTokens, r.CacheReadTokens,
		r.LatencyMS, r.Status, r.StopReason, r.RequestID, r.OccurredAt).Scan(&id)
	if err != nil {
		return 0, fmt.Errorf("store: insert usage: %w", err)
	}
	return id, nil
}

// ListUnshippedUsage returns up to limit unshipped rows, oldest first — the
// shipper's batch (the thermometer accepts 1..500 per POST).
func (s *Store) ListUnshippedUsage(ctx context.Context, limit int) ([]UsageRow, error) {
	if limit <= 0 || limit > 500 {
		limit = 500
	}
	rows, err := s.pool.Query(ctx, `
		SELECT `+usageCols+` FROM usage_rows WHERE NOT shipped ORDER BY id LIMIT $1`, limit)
	if err != nil {
		return nil, fmt.Errorf("store: list unshipped usage: %w", err)
	}
	defer rows.Close()
	out := []UsageRow{}
	for rows.Next() {
		var r UsageRow
		var turnID *string
		if err := rows.Scan(&r.ID, &turnID, &r.Operation, &r.Model, &r.SiteID, &r.SiteHost, &r.StoreID, &r.UserEmail,
			&r.InputTokens, &r.OutputTokens, &r.CacheWriteTokens, &r.CacheReadTokens,
			&r.LatencyMS, &r.Status, &r.StopReason, &r.RequestID, &r.OccurredAt, &r.Shipped); err != nil {
			return nil, err
		}
		if turnID != nil {
			r.TurnID = *turnID
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

// ListUsageByTurn returns a turn's rows oldest first (tests, debugging).
func (s *Store) ListUsageByTurn(ctx context.Context, turnID string) ([]UsageRow, error) {
	rows, err := s.pool.Query(ctx, `SELECT `+usageCols+` FROM usage_rows WHERE turn_id = $1 ORDER BY id`, turnID)
	if err != nil {
		return nil, fmt.Errorf("store: list usage by turn: %w", err)
	}
	defer rows.Close()
	out := []UsageRow{}
	for rows.Next() {
		var r UsageRow
		var tid *string
		if err := rows.Scan(&r.ID, &tid, &r.Operation, &r.Model, &r.SiteID, &r.SiteHost, &r.StoreID, &r.UserEmail,
			&r.InputTokens, &r.OutputTokens, &r.CacheWriteTokens, &r.CacheReadTokens,
			&r.LatencyMS, &r.Status, &r.StopReason, &r.RequestID, &r.OccurredAt, &r.Shipped); err != nil {
			return nil, err
		}
		if tid != nil {
			r.TurnID = *tid
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

// MarkUsageShipped flips shipped on ids after the thermometer answered 2xx.
func (s *Store) MarkUsageShipped(ctx context.Context, ids []int64) (int64, error) {
	if len(ids) == 0 {
		return 0, nil
	}
	tag, err := s.pool.Exec(ctx, `UPDATE usage_rows SET shipped = true WHERE id = ANY($1)`, ids)
	if err != nil {
		return 0, fmt.Errorf("store: mark usage shipped: %w", err)
	}
	return tag.RowsAffected(), nil
}

const usageCols = `id, turn_id, operation, model, site_id, site_host, store_id, user_email,
	input_tokens, output_tokens, cache_write_tokens, cache_read_tokens,
	latency_ms, status, stop_reason, request_id, occurred_at, shipped`
