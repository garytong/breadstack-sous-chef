package usage

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/breadstack/sous-chef/internal/store"
)

var fixed = time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)

// memStore is an in-memory Store.
type memStore struct {
	mu     sync.Mutex
	rows   []store.UsageRow
	insErr error
}

func (m *memStore) InsertUsage(_ context.Context, r store.UsageRow) (int64, error) {
	if m.insErr != nil {
		return 0, m.insErr
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	r.ID = int64(len(m.rows) + 1)
	m.rows = append(m.rows, r)
	return r.ID, nil
}

func (m *memStore) ListUnshippedUsage(_ context.Context, limit int) ([]store.UsageRow, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var out []store.UsageRow
	for _, r := range m.rows {
		if !r.Shipped && len(out) < limit {
			out = append(out, r)
		}
	}
	return out, nil
}

func (m *memStore) MarkUsageShipped(_ context.Context, ids []int64) (int64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var n int64
	for _, id := range ids {
		for i := range m.rows {
			if m.rows[i].ID == id && !m.rows[i].Shipped {
				m.rows[i].Shipped = true
				n++
			}
		}
	}
	return n, nil
}

func (m *memStore) shippedCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	n := 0
	for _, r := range m.rows {
		if r.Shipped {
			n++
		}
	}
	return n
}

func TestRowFor(t *testing.T) {
	ok := Call{TurnID: "t", Operation: OpToolTurn, RequestModel: "default-model", SiteID: "s", StoreID: "st", UserEmail: "u",
		Latency: 1500 * time.Millisecond,
		Resp: &Response{ID: "msg_1", Model: "claude-sonnet-5-5", StopReason: "tool_use",
			Usage: Tokens{InputTokens: 11, OutputTokens: 22, CacheCreationInputTokens: 33, CacheReadInputTokens: 44}}}
	row := RowFor(ok, fixed)
	if row.Model != "claude-sonnet-5-5" || row.Status != "ok" || row.StopReason != "tool_use" || row.RequestID != "msg_1" ||
		row.InputTokens != 11 || row.OutputTokens != 22 || row.CacheWriteTokens != 33 || row.CacheReadTokens != 44 ||
		row.LatencyMS == nil || *row.LatencyMS != 1500 || !row.OccurredAt.Equal(fixed) || row.Operation != OpToolTurn || row.TurnID != "t" {
		t.Errorf("ok row = %+v", row)
	}
	// Response without a model keeps the requested one; refusal status.
	ref := RowFor(Call{RequestModel: "m", Resp: &Response{StopReason: "refusal"}}, fixed)
	if ref.Model != "m" || ref.Status != "refusal" || ref.Operation != OpChat {
		t.Errorf("refusal row = %+v", ref)
	}
	// Failed call: error status, no tokens, no stop reason.
	bad := RowFor(Call{RequestModel: "m", Err: errors.New("timeout"), Resp: &Response{Model: "served", StopReason: "end_turn", Usage: Tokens{InputTokens: 5}}}, fixed)
	if bad.Status != "error" || bad.Model != "served" || bad.InputTokens != 0 || bad.StopReason != "" {
		t.Errorf("error row = %+v", bad)
	}
	nilResp := RowFor(Call{RequestModel: "m"}, fixed)
	if nilResp.Status != "error" {
		t.Errorf("nil response must be error: %+v", nilResp)
	}
	// A route with report_as: every row carries it — the echoed tag is
	// ignored on an answered call, a failed one and one with no response.
	echo := &Response{ID: "msg_2", Model: "sous-chef", StopReason: "end_turn", Usage: Tokens{InputTokens: 7, OutputTokens: 3}}
	for name, c := range map[string]Call{
		"ok":       {RequestModel: "sous-chef", ReportAs: "blueberry-muffin", Resp: echo},
		"error":    {RequestModel: "sous-chef", ReportAs: "blueberry-muffin", Resp: echo, Err: errors.New("cut")},
		"no reply": {RequestModel: "sous-chef", ReportAs: "blueberry-muffin", Err: errors.New("timeout")},
	} {
		if r := RowFor(c, fixed); r.Model != "blueberry-muffin" {
			t.Errorf("report_as, %s: model = %q", name, r.Model)
		}
	}
	if r := RowFor(Call{RequestModel: "sous-chef", ReportAs: "blueberry-muffin", Resp: echo}, fixed); r.InputTokens != 7 || r.OutputTokens != 3 || r.Status != "ok" || r.RequestID != "msg_2" {
		t.Errorf("report_as changes only the model: %+v", r)
	}
	if w := IngestRow(RowFor(Call{RequestModel: "sous-chef", ReportAs: "blueberry-muffin", Resp: echo}, fixed)); w.Model != "blueberry-muffin" {
		t.Errorf("wire model = %q", w.Model)
	}
	// Without it the same call reports the echoed tag, as before.
	if r := RowFor(Call{RequestModel: "sous-chef", Resp: echo}, fixed); r.Model != "sous-chef" {
		t.Errorf("no report_as: model = %q", r.Model)
	}
	// Explicit OccurredAt wins.
	at := fixed.Add(-time.Hour)
	if r := RowFor(Call{RequestModel: "m", OccurredAt: at}, fixed); !r.OccurredAt.Equal(at) {
		t.Errorf("OccurredAt not kept: %v", r.OccurredAt)
	}
}

// TestIngestRowWireShape pins the exact 16-key thermometer ingest shape
// (breadstack-thermometer server/internal/api/usage.go:21-39).
func TestIngestRowWireShape(t *testing.T) {
	lat := int64(321)
	row := store.UsageRow{Operation: "chat", Model: "claude-sonnet-5-5", SiteID: "s", SiteHost: "h", StoreID: "st", UserEmail: "u",
		InputTokens: 1, OutputTokens: 2, CacheWriteTokens: 3, CacheReadTokens: 4, LatencyMS: &lat, Status: "ok",
		StopReason: "end_turn", RequestID: "msg", OccurredAt: fixed}
	raw, err := json.Marshal(IngestRow(row))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	want := []string{"cache_creation_input_tokens", "cache_read_input_tokens", "input_tokens", "latency_ms", "model", "occurred_at",
		"operation", "output_tokens", "request_id", "site_host", "site_id", "source", "status", "stop_reason", "store_id", "user_email"}
	if !reflect.DeepEqual(keys, want) {
		t.Fatalf("keys = %v\nwant %v", keys, want)
	}
	if m["source"] != Source || m["cache_creation_input_tokens"] != float64(3) || m["cache_read_input_tokens"] != float64(4) ||
		m["latency_ms"] != float64(321) || m["occurred_at"] != "2026-10-01T12:00:00Z" {
		t.Errorf("values = %v", m)
	}
	// latency_ms is null, not absent, when unknown.
	row.LatencyMS = nil
	raw, _ = json.Marshal(IngestRow(row))
	if !strings.Contains(string(raw), `"latency_ms":null`) {
		t.Errorf("latency_ms must be null when unknown: %s", raw)
	}
}

func TestNewShipperFromEnvGate(t *testing.T) {
	get := func(m map[string]string) func(string) string { return func(k string) string { return m[k] } }
	if NewShipperFromEnv(get(nil)) != nil {
		t.Error("unset env must give nil shipper")
	}
	if NewShipperFromEnv(get(map[string]string{"THERMOMETER_API_URL": "https://t.example"})) != nil {
		t.Error("URL without key must give nil")
	}
	if NewShipperFromEnv(get(map[string]string{"THERMOMETER_API_KEY": "k"})) != nil {
		t.Error("key without URL must give nil")
	}
	s := NewShipperFromEnv(get(map[string]string{"THERMOMETER_API_URL": " https://t.example/ ", "THERMOMETER_API_KEY": "k"}))
	if s == nil || s.Endpoint() != "https://t.example/api/ingest/usage" {
		t.Errorf("shipper = %+v", s)
	}
	rec := NewRecorder(&memStore{}, nil)
	if rec.ShippingEnabled() {
		t.Error("nil shipper must report shipping off")
	}
}

// fakeThermometer records what it receives.
type fakeThermometer struct {
	mu      sync.Mutex
	batches [][]map[string]any
	keys    []string
	status  int
	delay   time.Duration
}

func (f *fakeThermometer) handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if f.delay > 0 {
			select {
			case <-time.After(f.delay):
			case <-r.Context().Done():
				return
			}
		}
		body, _ := io.ReadAll(r.Body)
		var in struct {
			Usage []map[string]any `json:"usage"`
		}
		_ = json.Unmarshal(body, &in)
		f.mu.Lock()
		f.batches = append(f.batches, in.Usage)
		f.keys = append(f.keys, r.Header.Get("X-API-Key"))
		f.mu.Unlock()
		if r.URL.Path != "/api/ingest/usage" || r.Header.Get("Content-Type") != "application/json" {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		status := f.status
		if status == 0 {
			status = http.StatusAccepted
		}
		w.WriteHeader(status)
	})
}

func (f *fakeThermometer) count() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.batches)
}

func TestRecordStoresAndShips(t *testing.T) {
	therm := &fakeThermometer{}
	srv := httptest.NewServer(therm.handler())
	defer srv.Close()
	ms := &memStore{}
	rec := NewRecorder(ms, NewShipper(srv.URL, "secret"))
	rec.now = func() time.Time { return fixed }
	if !rec.ShippingEnabled() {
		t.Fatal("shipping should be on")
	}

	// A cancelled context must not stop the insert (WithoutCancel).
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := rec.Record(ctx, Call{TurnID: "t1", RequestModel: "m", SiteID: "s", Latency: 50 * time.Millisecond,
		Resp: &Response{ID: "msg_1", Model: "claude-sonnet-5-5", StopReason: "end_turn", Usage: Tokens{InputTokens: 7, OutputTokens: 3}}})
	if err != nil {
		t.Fatal(err)
	}
	rec.Wait()
	if len(ms.rows) != 1 || ms.rows[0].Model != "claude-sonnet-5-5" || ms.rows[0].Status != "ok" {
		t.Fatalf("stored rows = %+v", ms.rows)
	}
	if therm.count() != 1 || therm.keys[0] != "secret" {
		t.Fatalf("thermometer got %d batches, key %v", therm.count(), therm.keys)
	}
	got := therm.batches[0][0]
	if got["source"] != "sous-chef" || got["model"] != "claude-sonnet-5-5" || got["input_tokens"] != float64(7) ||
		got["operation"] != "chat" || got["latency_ms"] != float64(50) || got["occurred_at"] != "2026-10-01T12:00:00Z" {
		t.Errorf("wire row = %v", got)
	}
	if ms.shippedCount() != 1 {
		t.Error("row not marked shipped after 202")
	}
}

func TestRecordWithoutShipperStoresOnly(t *testing.T) {
	ms := &memStore{}
	rec := NewRecorder(ms, nil)
	if err := rec.Record(context.Background(), Call{RequestModel: "m", Err: errors.New("down")}); err != nil {
		t.Fatal(err)
	}
	rec.Wait()
	if len(ms.rows) != 1 || ms.rows[0].Status != "error" || ms.rows[0].Shipped {
		t.Errorf("rows = %+v", ms.rows)
	}
	if n, err := rec.Sweep(context.Background()); n != 0 || err != nil {
		t.Errorf("Sweep with shipping off = %d, %v", n, err)
	}
	// Insert failure is returned, not panicked.
	bad := NewRecorder(&memStore{insErr: errors.New("pg")}, nil)
	bad.logf = func(string, ...any) {}
	if err := bad.Record(context.Background(), Call{RequestModel: "m"}); err == nil {
		t.Error("insert error swallowed")
	}
	// nil recorder is safe.
	var none *Recorder
	if err := none.Record(context.Background(), Call{}); err != nil {
		t.Error(err)
	}
	none.Wait()
}

func TestShipFailureKeepsRowUnshippedAndSweepRetries(t *testing.T) {
	therm := &fakeThermometer{status: http.StatusInternalServerError}
	srv := httptest.NewServer(therm.handler())
	defer srv.Close()
	ms := &memStore{}
	var logs []string
	var lmu sync.Mutex
	rec := NewRecorder(ms, NewShipper(srv.URL, "k"))
	rec.logf = func(f string, a ...any) { lmu.Lock(); logs = append(logs, f); lmu.Unlock() }
	for i := 0; i < 3; i++ {
		if err := rec.Record(context.Background(), Call{RequestModel: "m", Resp: &Response{StopReason: "end_turn"}}); err != nil {
			t.Fatal(err)
		}
	}
	rec.Wait()
	if ms.shippedCount() != 0 || therm.count() != 3 {
		t.Fatalf("after 500s: shipped=%d posts=%d", ms.shippedCount(), therm.count())
	}
	lmu.Lock()
	if len(logs) != 3 || !strings.Contains(logs[0], "ship") {
		t.Errorf("failures not logged: %v", logs)
	}
	lmu.Unlock()
	// Thermometer recovers; one sweep ships all three in one batch.
	therm.mu.Lock()
	therm.status = http.StatusAccepted
	therm.mu.Unlock()
	n, err := rec.Sweep(context.Background())
	if err != nil || n != 3 {
		t.Fatalf("Sweep = %d, %v", n, err)
	}
	if therm.count() != 4 || len(therm.batches[3]) != 3 || ms.shippedCount() != 3 {
		t.Errorf("sweep batch: posts=%d size=%d shipped=%d", therm.count(), len(therm.batches[3]), ms.shippedCount())
	}
	// Nothing left.
	if n, _ := rec.Sweep(context.Background()); n != 0 {
		t.Errorf("second sweep shipped %d", n)
	}
}

func TestShipTimeoutIsBoundedAndNonBlocking(t *testing.T) {
	therm := &fakeThermometer{delay: 2 * time.Second}
	srv := httptest.NewServer(therm.handler())
	defer srv.Close()
	ms := &memStore{}
	sh := NewShipper(srv.URL, "k")
	sh.SetTimeout(100 * time.Millisecond)
	rec := NewRecorder(ms, sh)
	rec.logf = func(string, ...any) {}
	start := time.Now()
	if err := rec.Record(context.Background(), Call{RequestModel: "m", Resp: &Response{}}); err != nil {
		t.Fatal(err)
	}
	if took := time.Since(start); took > 500*time.Millisecond {
		t.Fatalf("Record blocked on the ship for %s; it must be fire-and-forget", took)
	}
	rec.Wait()
	if took := time.Since(start); took > 1500*time.Millisecond {
		t.Fatalf("ship not bounded by its timeout: %s", took)
	}
	if ms.shippedCount() != 0 {
		t.Error("timed-out ship must leave the row unshipped")
	}
	if sh.timeout != 100*time.Millisecond || NewShipper("u", "k").timeout != ShipTimeout {
		t.Error("timeout defaults wrong")
	}
}

func TestShipValidation(t *testing.T) {
	var nilShipper *Shipper
	if err := nilShipper.Ship(context.Background(), []store.UsageRow{{Model: "m"}}); err == nil {
		t.Error("nil shipper Ship must error")
	}
	sh := NewShipper("http://127.0.0.1:1", "k")
	if err := sh.Ship(context.Background(), nil); err != nil {
		t.Errorf("empty batch must be a no-op: %v", err)
	}
	big := make([]store.UsageRow, MaxBatch+1)
	if err := sh.Ship(context.Background(), big); err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Errorf("oversize batch: %v", err)
	}
}

func TestRecoverFromPanickingShipper(t *testing.T) {
	ms := &memStore{}
	sh := NewShipper("http://127.0.0.1:1", "k")
	sh.SetHTTPClient(&http.Client{Transport: panicTransport{}})
	rec := NewRecorder(ms, sh)
	var logged bool
	rec.logf = func(f string, a ...any) {
		if strings.Contains(f, "panicked") {
			logged = true
		}
	}
	if err := rec.Record(context.Background(), Call{RequestModel: "m"}); err != nil {
		t.Fatal(err)
	}
	rec.Wait()
	if !logged {
		t.Error("panic in the shipper must be recovered and logged")
	}
}

type panicTransport struct{}

func (panicTransport) RoundTrip(*http.Request) (*http.Response, error) { panic("boom") }
