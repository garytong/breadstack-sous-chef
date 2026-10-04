// Package usage records one row per model call and ships the rows to the
// thermometer's /api/ingest/usage (README §6).
//
// Two patterns are copied:
//   - breadstack-thermometer server/internal/chat/chat.go recordUsage: one
//     row per Messages API call, written with context.WithoutCancel so a
//     caller hanging up never loses it; the response's own model field wins
//     over the one asked for (a fallback may have served the call).
//   - cookie-cutter branch sous-chef launcher/server/thermometer.go
//     (reportSousChefUsage/postUsage): config-gated on THERMOMETER_API_URL +
//     THERMOMETER_API_KEY, fire-and-forget in its own goroutine with a 5 s
//     timeout, recover()ed, failures only log; wire shape = the thermometer's
//     ingestUsage (server/internal/api/usage.go:21-39), 16 keys.
//
// What is new here: the row is stored locally first (usage_rows) and only
// flagged shipped on a 2xx, so a thermometer outage loses nothing — the
// Sweep later re-ships whatever is still unshipped.
package usage

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/breadstack/sous-chef/internal/store"
)

// Source is the thermometer's source tag for this service.
const Source = "sous-chef"

// Operations.
const (
	OpChat     = "chat"      // a call that started from the merchant's text
	OpToolTurn = "tool_turn" // a call that started from tool results
)

// ShipTimeout bounds one POST to the thermometer.
const ShipTimeout = 5 * time.Second

// MaxBatch is the thermometer's per-POST limit (usage.go maxBatch = 500).
const MaxBatch = 500

// Store is the slice of *store.Store this package needs.
type Store interface {
	InsertUsage(ctx context.Context, r store.UsageRow) (int64, error)
	ListUnshippedUsage(ctx context.Context, limit int) ([]store.UsageRow, error)
	MarkUsageShipped(ctx context.Context, ids []int64) (int64, error)
}

// Call is what the loop knows about one model call. Resp is nil when the
// call itself failed (network, 5xx, deadline).
type Call struct {
	TurnID       string
	Operation    string // OpChat | OpToolTurn
	RequestModel string // the route's model id, used when Resp names none
	SiteID       string
	SiteHost     string
	StoreID      string
	UserEmail    string
	Latency      time.Duration
	Resp         *Response
	Err          error
	OccurredAt   time.Time // zero = now
}

// Response is the slice of a Messages API response the row reads.
type Response struct {
	ID         string
	Model      string
	StopReason string
	Usage      Tokens
}

// Tokens mirrors the usage block.
type Tokens struct {
	InputTokens              int64
	OutputTokens             int64
	CacheCreationInputTokens int64
	CacheReadInputTokens     int64
}

// RowFor builds the usage_rows row for c (pure; tested on its own).
func RowFor(c Call, now time.Time) store.UsageRow {
	lat := c.Latency.Milliseconds()
	row := store.UsageRow{
		TurnID:     c.TurnID,
		Operation:  c.Operation,
		Model:      c.RequestModel,
		SiteID:     c.SiteID,
		SiteHost:   c.SiteHost,
		StoreID:    c.StoreID,
		UserEmail:  c.UserEmail,
		LatencyMS:  &lat,
		OccurredAt: c.OccurredAt,
	}
	if row.Operation == "" {
		row.Operation = OpChat
	}
	if row.OccurredAt.IsZero() {
		row.OccurredAt = now
	}
	switch {
	case c.Err != nil:
		row.Status = "error"
		row.StopReason = ""
		if c.Resp != nil && c.Resp.Model != "" {
			row.Model = c.Resp.Model
		}
	case c.Resp == nil:
		row.Status = "error"
	default:
		if c.Resp.Model != "" {
			row.Model = c.Resp.Model
		}
		row.RequestID = c.Resp.ID
		row.StopReason = c.Resp.StopReason
		row.InputTokens = c.Resp.Usage.InputTokens
		row.OutputTokens = c.Resp.Usage.OutputTokens
		row.CacheWriteTokens = c.Resp.Usage.CacheCreationInputTokens
		row.CacheReadTokens = c.Resp.Usage.CacheReadInputTokens
		row.Status = "ok"
		if row.StopReason == "refusal" {
			row.Status = "refusal"
		}
	}
	return row
}

// Recorder stores rows and (when a Shipper is configured) ships them.
type Recorder struct {
	st      Store
	shipper *Shipper // nil = shipping off
	now     func() time.Time
	wg      sync.WaitGroup
	logf    func(string, ...any)
}

// NewRecorder wires the store and an optional shipper (nil when the
// thermometer env is unset: rows are still stored).
func NewRecorder(st Store, shipper *Shipper) *Recorder {
	return &Recorder{st: st, shipper: shipper, now: func() time.Time { return time.Now().UTC() }, logf: log.Printf}
}

// Enabled reports whether rows ship anywhere.
func (r *Recorder) ShippingEnabled() bool { return r != nil && r.shipper != nil }

// Record stores the row for c (with context.WithoutCancel so the caller
// hanging up never loses it) and, when shipping is on, posts it to the
// thermometer in its own goroutine — fire-and-forget: Record returns as
// soon as the insert has. The insert error is returned for the log; it
// never fails a turn.
func (r *Recorder) Record(ctx context.Context, c Call) error {
	if r == nil || r.st == nil {
		return nil
	}
	row := RowFor(c, r.now())
	id, err := r.st.InsertUsage(context.WithoutCancel(ctx), row)
	if err != nil {
		r.logf("usage: record: %v", err)
		return err
	}
	row.ID = id
	if r.shipper == nil {
		return nil
	}
	r.wg.Add(1)
	go func() {
		defer r.wg.Done()
		defer func() {
			if rec := recover(); rec != nil {
				r.logf("usage: shipper panicked: %v", rec)
			}
		}()
		r.shipRows(context.WithoutCancel(ctx), []store.UsageRow{row})
	}()
	return nil
}

// Sweep re-ships up to MaxBatch unshipped rows (a boot-time catch-up and a
// periodic retry). Returns how many were marked shipped. No-op when
// shipping is off.
func (r *Recorder) Sweep(ctx context.Context) (int64, error) {
	if r == nil || r.shipper == nil {
		return 0, nil
	}
	rows, err := r.st.ListUnshippedUsage(ctx, MaxBatch)
	if err != nil {
		return 0, err
	}
	if len(rows) == 0 {
		return 0, nil
	}
	return r.shipRows(ctx, rows), nil
}

// Wait blocks until in-flight fire-and-forget ships finish (shutdown and
// tests).
func (r *Recorder) Wait() {
	if r != nil {
		r.wg.Wait()
	}
}

func (r *Recorder) shipRows(ctx context.Context, rows []store.UsageRow) int64 {
	if err := r.shipper.Ship(ctx, rows); err != nil {
		r.logf("usage: ship %d row(s) failed (kept unshipped): %v", len(rows), err)
		return 0
	}
	ids := make([]int64, len(rows))
	for i, row := range rows {
		ids[i] = row.ID
	}
	n, err := r.st.MarkUsageShipped(context.WithoutCancel(ctx), ids)
	if err != nil {
		r.logf("usage: mark shipped: %v", err)
	}
	return n
}

// --- shipper ---

// ingestRow is the thermometer's /api/ingest/usage wire shape, field for
// field (breadstack-thermometer server/internal/api/usage.go:21-39). Every
// key is always present: the thermometer's decoder treats a missing
// latency_ms as "unknown", so it is a pointer there and here.
type ingestRow struct {
	Source           string `json:"source"`
	Operation        string `json:"operation"`
	Model            string `json:"model"`
	SiteID           string `json:"site_id"`
	SiteHost         string `json:"site_host"`
	StoreID          string `json:"store_id"`
	UserEmail        string `json:"user_email"`
	InputTokens      int64  `json:"input_tokens"`
	OutputTokens     int64  `json:"output_tokens"`
	CacheWriteTokens int64  `json:"cache_creation_input_tokens"`
	CacheReadTokens  int64  `json:"cache_read_input_tokens"`
	LatencyMS        *int64 `json:"latency_ms"`
	Status           string `json:"status"`
	StopReason       string `json:"stop_reason"`
	RequestID        string `json:"request_id"`
	OccurredAt       string `json:"occurred_at"`
}

// IngestRow maps a stored row onto the wire shape (pure; tested).
func IngestRow(r store.UsageRow) ingestRow {
	return ingestRow{
		Source:           Source,
		Operation:        r.Operation,
		Model:            r.Model,
		SiteID:           r.SiteID,
		SiteHost:         r.SiteHost,
		StoreID:          r.StoreID,
		UserEmail:        r.UserEmail,
		InputTokens:      r.InputTokens,
		OutputTokens:     r.OutputTokens,
		CacheWriteTokens: r.CacheWriteTokens,
		CacheReadTokens:  r.CacheReadTokens,
		LatencyMS:        r.LatencyMS,
		Status:           r.Status,
		StopReason:       r.StopReason,
		RequestID:        r.RequestID,
		OccurredAt:       r.OccurredAt.UTC().Format(time.RFC3339),
	}
}

// Shipper posts batches to the thermometer.
type Shipper struct {
	url     string // base URL, no trailing slash
	key     string
	client  *http.Client
	timeout time.Duration
}

// NewShipperFromEnv returns a Shipper when both THERMOMETER_API_URL and
// THERMOMETER_API_KEY are set, else nil (shipping off, rows still stored).
func NewShipperFromEnv(getenv func(string) string) *Shipper {
	url := strings.TrimRight(strings.TrimSpace(getenv("THERMOMETER_API_URL")), "/")
	key := strings.TrimSpace(getenv("THERMOMETER_API_KEY"))
	if url == "" || key == "" {
		return nil
	}
	return NewShipper(url, key)
}

// NewShipper builds a shipper for url with key.
func NewShipper(url, key string) *Shipper {
	return &Shipper{url: strings.TrimRight(url, "/"), key: key, client: http.DefaultClient, timeout: ShipTimeout}
}

// SetHTTPClient and SetTimeout are for tests.
func (s *Shipper) SetHTTPClient(c *http.Client) { s.client = c }
func (s *Shipper) SetTimeout(d time.Duration)   { s.timeout = d }
func (s *Shipper) Endpoint() string             { return s.url + "/api/ingest/usage" }

// Ship POSTs {"usage":[...]} with X-API-Key under ShipTimeout. Any 2xx is
// success (the thermometer answers 202). It never retries: the rows stay
// unshipped and Sweep tries again.
func (s *Shipper) Ship(ctx context.Context, rows []store.UsageRow) error {
	if s == nil {
		return errors.New("usage: shipping is off")
	}
	if len(rows) == 0 {
		return nil
	}
	if len(rows) > MaxBatch {
		return fmt.Errorf("usage: batch of %d exceeds %d", len(rows), MaxBatch)
	}
	wire := make([]ingestRow, len(rows))
	for i, r := range rows {
		wire[i] = IngestRow(r)
	}
	body, err := json.Marshal(map[string]any{"usage": wire})
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.Endpoint(), bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-API-Key", s.key)
	res, err := s.client.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		snippet, _ := io.ReadAll(io.LimitReader(res.Body, 512))
		return fmt.Errorf("usage: thermometer answered %s: %s", res.Status, strings.TrimSpace(string(snippet)))
	}
	return nil
}
