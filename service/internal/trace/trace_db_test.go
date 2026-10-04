package trace_test

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/breadstack/sous-chef/internal/store"
	"github.com/breadstack/sous-chef/internal/store/storetest"
	"github.com/breadstack/sous-chef/internal/trace"
)

func TestMain(m *testing.M) { os.Exit(storetest.Main(m)) }

// Against Postgres: a traced turn lands in traces; an opted-out store's
// does not; the retention job removes an aged trace.
func TestRecorderAgainstPostgres(t *testing.T) {
	s := storetest.Open(t)
	ctx := context.Background()
	start := time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)

	b := trace.Begin{TurnID: "aaaaaaaa-0000-0000-0000-000000000001", SiteID: "site-1", StoreID: "store-1",
		ManifestHash: "h", BuildID: "b", Route: "default", Text: "hi", Context: json.RawMessage(`{"page":"home"}`), StartedAt: start}
	r := trace.New(ctx, s, b)
	if r == nil {
		t.Fatal("recorder nil for a store that did not opt out")
	}
	r.ModelCall(trace.ModelCall{Model: "m", StopReason: "end_turn"}, json.RawMessage(`[{"type":"thinking","thinking":"X"},{"type":"text","text":"ok"}]`))
	r.Event(1, "turn.done", json.RawMessage(`{}`))
	if err := r.Finish(ctx, "done", "end_turn", ""); err != nil {
		t.Fatal(err)
	}
	got, err := s.GetTrace(ctx, b.TurnID)
	if err != nil {
		t.Fatal(err)
	}
	var ops struct {
		Calls []struct {
			Text string `json:"text"`
		} `json:"calls"`
	}
	if err := json.Unmarshal(got.Ops, &ops); err != nil {
		t.Fatal(err)
	}
	if !got.Created.Equal(start) || strings.Contains(string(got.Ops), `thinking`) || len(ops.Calls) != 1 || ops.Calls[0].Text != "ok" {
		t.Errorf("stored trace ops = %s (created %v)", got.Ops, got.Created)
	}
	// Feedback merges onto it.
	if err := s.MergeFeedback(ctx, b.TurnID, json.RawMessage(`{"saved":true}`)); err != nil {
		t.Fatal(err)
	}

	// Opted-out store: no recorder, no row.
	if err := s.SetOptOut(ctx, "store-2", "admin@x.test"); err != nil {
		t.Fatal(err)
	}
	b2 := b
	b2.TurnID, b2.StoreID = "aaaaaaaa-0000-0000-0000-000000000002", "store-2"
	r2 := trace.New(ctx, s, b2)
	if r2 != nil {
		t.Fatal("opted-out store got a recorder")
	}
	_ = r2.Finish(ctx, "done", "end_turn", "")
	if _, err := s.GetTrace(ctx, b2.TurnID); err != store.ErrNotFound {
		t.Errorf("opted-out turn has a trace: %v", err)
	}

	// Retention: 91 days later the trace is gone.
	s.SetClock(func() time.Time { return start.AddDate(0, 0, 91) })
	res, err := trace.Run(ctx, s, store.DefaultRetention, func(string, ...any) {})
	if err != nil || res.Traces != 1 {
		t.Fatalf("retention = %+v, %v", res, err)
	}
	if _, err := s.GetTrace(ctx, b.TurnID); err != store.ErrNotFound {
		t.Errorf("trace survived retention: %v", err)
	}
}

func TestCommandAgainstPostgres(t *testing.T) {
	dsn := storetest.OpenDSN(t)
	var out strings.Builder
	err := trace.Command(context.Background(), []string{"--days", "45"},
		func(k string) string {
			if k == "DATABASE_URL" {
				return dsn
			}
			return ""
		}, &out)
	if err != nil {
		t.Fatalf("Command: %v (%s)", err, out.String())
	}
	if !strings.Contains(out.String(), "retention: ok") || !strings.Contains(out.String(), "traces=45d") {
		t.Errorf("output = %q", out.String())
	}
}
