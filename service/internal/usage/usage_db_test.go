package usage_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"sync/atomic"
	"testing"
	"time"

	"github.com/breadstack/sous-chef/internal/store"
	"github.com/breadstack/sous-chef/internal/store/storetest"
	"github.com/breadstack/sous-chef/internal/usage"
)

func TestMain(m *testing.M) { os.Exit(storetest.Main(m)) }

// Against Postgres: Record inserts a usage_rows row, the thermometer's 202
// flips shipped, a 503 leaves it unshipped for Sweep.
func TestRecorderAgainstPostgres(t *testing.T) {
	s := storetest.Open(t)
	ctx := context.Background()
	c, _ := s.OpenConversation(ctx, "site-1", "store-1")
	turn, _ := s.CreateTurn(ctx, store.NewTurn{ConversationID: c.ID, ManifestHash: "h"})

	var status atomic.Int32
	status.Store(http.StatusServiceUnavailable)
	var posts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		posts.Add(1)
		w.WriteHeader(int(status.Load()))
	}))
	defer srv.Close()

	rec := usage.NewRecorder(s, usage.NewShipperFromEnv(func(k string) string {
		switch k {
		case "THERMOMETER_API_URL":
			return srv.URL
		case "THERMOMETER_API_KEY":
			return "k"
		}
		return ""
	}))
	err := rec.Record(ctx, usage.Call{TurnID: turn.ID, RequestModel: "claude-sonnet-5-5", SiteID: "site-1", StoreID: "store-1",
		UserEmail: "m@x.test", Latency: 900 * time.Millisecond,
		Resp: &usage.Response{ID: "msg_1", Model: "claude-sonnet-5-5", StopReason: "tool_use",
			Usage: usage.Tokens{InputTokens: 12, OutputTokens: 34, CacheCreationInputTokens: 56, CacheReadInputTokens: 78}}})
	if err != nil {
		t.Fatal(err)
	}
	rec.Wait()
	rows, err := s.ListUsageByTurn(ctx, turn.ID)
	if err != nil || len(rows) != 1 {
		t.Fatalf("rows = %+v, %v", rows, err)
	}
	r := rows[0]
	if r.Shipped || r.CacheWriteTokens != 56 || r.CacheReadTokens != 78 || r.LatencyMS == nil || *r.LatencyMS != 900 || r.Status != "ok" || r.RequestID != "msg_1" {
		t.Errorf("row = %+v", r)
	}
	if posts.Load() != 1 {
		t.Errorf("posts = %d", posts.Load())
	}
	// Thermometer back: Sweep ships it.
	status.Store(http.StatusAccepted)
	n, err := rec.Sweep(ctx)
	if err != nil || n != 1 {
		t.Fatalf("Sweep = %d, %v", n, err)
	}
	rows, _ = s.ListUsageByTurn(ctx, turn.ID)
	if !rows[0].Shipped {
		t.Error("row not marked shipped")
	}
	unshipped, _ := s.ListUnshippedUsage(ctx, 10)
	if len(unshipped) != 0 {
		t.Errorf("unshipped left: %d", len(unshipped))
	}
}
