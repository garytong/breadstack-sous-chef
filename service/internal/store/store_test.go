package store_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/breadstack/sous-chef/internal/store"
	"github.com/breadstack/sous-chef/internal/store/storetest"
)

// Integration tests against a real Postgres (storetest: TEST_DATABASE_URL,
// else a docker container, else skip).

func TestMain(m *testing.M) { os.Exit(storetest.Main(m)) }

const hashA = "3e18656a000000000000000000000000000000000000000000000000000000a1"
const hashB = "3e18656a000000000000000000000000000000000000000000000000000000b2"
const hashC = "3e18656a000000000000000000000000000000000000000000000000000000c3"

// jsonEq compares two JSON documents semantically: jsonb re-spaces what it
// stores, so byte equality is the wrong test.
func jsonEq(a, b json.RawMessage) bool {
	var x, y any
	if json.Unmarshal(a, &x) != nil || json.Unmarshal(b, &y) != nil {
		return false
	}
	return reflect.DeepEqual(x, y)
}

func ctxT(t *testing.T) context.Context {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)
	return ctx
}

func TestMigrateIdempotentAndDown(t *testing.T) {
	dsn := storetest.OpenDSN(t) // already migrated once
	v, err := store.Migrate(dsn)
	if err != nil {
		t.Fatalf("second Migrate: %v", err)
	}
	if v != 6 {
		t.Fatalf("schema version = %d, want 6", v)
	}
	if err := store.MigrateDown(dsn); err != nil {
		t.Fatalf("MigrateDown: %v", err)
	}
	v, dirty, err := store.MigrationVersion(dsn)
	if err != nil || dirty || v != 0 {
		t.Fatalf("after down: version=%d dirty=%v err=%v", v, dirty, err)
	}
	if _, err := store.Migrate(dsn); err != nil {
		t.Fatalf("re-up: %v", err)
	}
}

func TestManifests(t *testing.T) {
	s := storetest.Open(t)
	ctx := ctxT(t)

	s.SetClock(func() time.Time { return time.Date(2026, 10, 1, 0, 0, 0, 0, time.UTC) })
	body := json.RawMessage(`{"schema":1,"sections":{"hero":{}}}`)
	m := store.Manifest{Hash: hashA, Body: body, Index: json.RawMessage(`{"lines":["s:hero"]}`),
		GeneratedFrom: json.RawMessage(`{"build":"abc123","at":"2026-10-01T00:00:00Z"}`),
		BuildLine:     "int", Counts: json.RawMessage(`{"fields":2094,"lists":69}`)}
	created, err := s.UpsertManifest(ctx, m)
	if err != nil || !created {
		t.Fatalf("first upsert: created=%v err=%v", created, err)
	}
	created, err = s.UpsertManifest(ctx, m)
	if err != nil || created {
		t.Fatalf("second upsert: created=%v err=%v (want false, nil)", created, err)
	}
	ok, err := s.ManifestExists(ctx, hashA)
	if err != nil || !ok {
		t.Fatalf("exists: %v %v", ok, err)
	}
	ok, err = s.ManifestExists(ctx, hashB)
	if err != nil || ok {
		t.Fatalf("exists(unknown): %v %v", ok, err)
	}
	got, err := s.GetManifest(ctx, hashA, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Body) != 0 {
		t.Errorf("body=false must leave Body empty, got %s", got.Body)
	}
	var counts map[string]int
	if err := json.Unmarshal(got.Counts, &counts); err != nil || counts["fields"] != 2094 {
		t.Errorf("counts = %s (%v)", got.Counts, err)
	}
	got, err = s.GetManifest(ctx, hashA, true)
	if err != nil || !jsonEq(got.Body, body) {
		t.Errorf("GetManifest(body) = %s, %v", got.Body, err)
	}
	if _, err := s.GetManifest(ctx, hashB, false); !errors.Is(err, store.ErrNotFound) {
		t.Errorf("unknown hash err = %v, want ErrNotFound", err)
	}

	// Bad hash rejected before SQL; bad build line rejected by the CHECK.
	if _, err := s.UpsertManifest(ctx, store.Manifest{Hash: "short", Body: body}); err == nil {
		t.Error("short hash accepted")
	}
	if _, err := s.UpsertManifest(ctx, store.Manifest{Hash: hashB, Body: body, BuildLine: "staging"}); err == nil {
		t.Error("build_line 'staging' accepted")
	}

	// PreviousManifest: same line, newest other hash.
	s.SetClock(func() time.Time { return time.Date(2026, 10, 2, 0, 0, 0, 0, time.UTC) })
	if _, err := s.UpsertManifest(ctx, store.Manifest{Hash: hashB, Body: body, BuildLine: "int"}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.UpsertManifest(ctx, store.Manifest{Hash: hashC, Body: body, BuildLine: "prod"}); err != nil {
		t.Fatal(err)
	}
	prev, err := s.PreviousManifest(ctx, "int", hashB)
	if err != nil || prev.Hash != hashA {
		t.Errorf("PreviousManifest(int, excl B) = %v, %v; want A", prev, err)
	}
	prev, err = s.PreviousManifest(ctx, "int", hashA)
	if err != nil || prev.Hash != hashB {
		t.Errorf("PreviousManifest(int, excl A) = %v, %v; want B", prev, err)
	}
	if _, err := s.PreviousManifest(ctx, "prod", hashC); !errors.Is(err, store.ErrNotFound) {
		t.Errorf("prod line has no other manifest; err = %v", err)
	}
	all, err := s.ListManifests(ctx)
	if err != nil || len(all) != 3 {
		t.Errorf("ListManifests = %d, %v", len(all), err)
	}
	if all[0].Hash == hashA {
		t.Errorf("newest first expected; got A first")
	}
}

func TestConversationsAndMessages(t *testing.T) {
	s := storetest.Open(t)
	ctx := ctxT(t)

	c1, err := s.OpenConversation(ctx, "site-1", "store-1")
	if err != nil {
		t.Fatal(err)
	}
	c1again, err := s.OpenConversation(ctx, "site-1", "store-1")
	if err != nil || c1again.ID != c1.ID {
		t.Fatalf("OpenConversation must return the open one: %v %v", c1again, err)
	}
	if _, err := s.CreateConversation(ctx, "site-1", "store-1"); err == nil {
		t.Fatal("second open conversation for a site must violate the partial unique index")
	}

	seq, err := s.AppendMessage(ctx, c1.ID, "user", json.RawMessage(`[{"type":"text","text":"hi"}]`))
	if err != nil || seq != 1 {
		t.Fatalf("first seq = %d, %v", seq, err)
	}
	first, err := s.AppendMessages(ctx, c1.ID, []store.Message{
		{Role: "assistant", Content: json.RawMessage(`[{"type":"tool_use","id":"tu_1","name":"set_fields","input":{}}]`)},
		{Role: "user", Content: json.RawMessage(`[{"type":"tool_result","tool_use_id":"tu_1","content":"ok"}]`)},
	})
	if err != nil || first != 2 {
		t.Fatalf("AppendMessages first seq = %d, %v", first, err)
	}
	if _, err := s.AppendMessage(ctx, c1.ID, "system", json.RawMessage(`[]`)); err == nil {
		t.Error("role system accepted")
	}
	if _, err := s.AppendMessage(ctx, c1.ID, "user", nil); err == nil {
		t.Error("empty content accepted")
	}
	n, err := s.CountMessages(ctx, c1.ID)
	if err != nil || n != 3 {
		t.Fatalf("count = %d, %v", n, err)
	}
	msgs, err := s.ListMessages(ctx, c1.ID, 0, 200)
	if err != nil || len(msgs) != 3 || msgs[0].Seq != 1 || msgs[2].Seq != 3 || msgs[1].Role != "assistant" {
		t.Fatalf("ListMessages = %+v, %v", msgs, err)
	}
	// Paging: before=3 limit=1 → just seq 2.
	msgs, err = s.ListMessages(ctx, c1.ID, 3, 1)
	if err != nil || len(msgs) != 1 || msgs[0].Seq != 2 {
		t.Fatalf("ListMessages(before=3,limit=1) = %+v, %v", msgs, err)
	}
	// Newest `limit`, ascending.
	msgs, err = s.ListMessages(ctx, c1.ID, 0, 2)
	if err != nil || len(msgs) != 2 || msgs[0].Seq != 2 || msgs[1].Seq != 3 {
		t.Fatalf("ListMessages(limit=2) = %+v, %v", msgs, err)
	}

	// Concurrent appends get dense, distinct seqs.
	var wg sync.WaitGroup
	seqs := make(chan int, 20)
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			seq, err := s.AppendMessage(ctx, c1.ID, "user", json.RawMessage(`[{"type":"text","text":"x"}]`))
			if err != nil {
				t.Error(err)
				return
			}
			seqs <- seq
		}()
	}
	wg.Wait()
	close(seqs)
	seen := map[int]bool{}
	for seq := range seqs {
		if seen[seq] {
			t.Errorf("duplicate seq %d", seq)
		}
		seen[seq] = true
	}
	if len(seen) != 20 {
		t.Fatalf("got %d seqs, want 20", len(seen))
	}
	for i := 4; i <= 23; i++ {
		if !seen[i] {
			t.Errorf("seq %d missing: not dense", i)
		}
	}

	// AppendMessageAfter is a compare-and-set on the tail seq (N2: the loop
	// answering a tool_use batch must notice the reaper closed it first).
	seq, err = s.AppendMessageAfter(ctx, c1.ID, "assistant", json.RawMessage(`[{"type":"text","text":"tail"}]`), 23)
	if err != nil || seq != 24 {
		t.Fatalf("AppendMessageAfter(23) = %d, %v", seq, err)
	}
	if _, err := s.AppendMessageAfter(ctx, c1.ID, "user", json.RawMessage(`[{"type":"text","text":"late"}]`), 23); !errors.Is(err, store.ErrConflict) {
		t.Fatalf("AppendMessageAfter on a moved tail err = %v, want ErrConflict", err)
	}
	if n, _ := s.CountMessages(ctx, c1.ID); n != 24 {
		t.Fatalf("a conflicting append wrote: count = %d, want 24", n)
	}
	if _, err := s.AppendMessageAfter(ctx, c1.ID, "system", json.RawMessage(`[]`), 24); err == nil {
		t.Error("AppendMessageAfter accepted role system")
	}
	if _, err := s.AppendMessageAfter(ctx, "00000000-0000-0000-0000-000000000000", "user", json.RawMessage(`[]`), 0); !errors.Is(err, store.ErrNotFound) {
		t.Errorf("AppendMessageAfter on unknown conversation err = %v, want ErrNotFound", err)
	}
	// Twenty racers on the same tail: exactly one wins.
	var wins atomic.Int32
	var wg2 sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg2.Add(1)
		go func() {
			defer wg2.Done()
			_, err := s.AppendMessageAfter(ctx, c1.ID, "user", json.RawMessage(`[{"type":"text","text":"race"}]`), 24)
			switch {
			case err == nil:
				wins.Add(1)
			case errors.Is(err, store.ErrConflict):
			default:
				t.Error(err)
			}
		}()
	}
	wg2.Wait()
	if wins.Load() != 1 {
		t.Fatalf("AppendMessageAfter racers: %d wins, want exactly 1", wins.Load())
	}
	if n, _ := s.CountMessages(ctx, c1.ID); n != 25 {
		t.Fatalf("after the race count = %d, want 25", n)
	}

	// End + reopen gives a new conversation; rotation does both at once.
	ended, err := s.EndConversation(ctx, "site-1")
	if err != nil || !ended {
		t.Fatalf("EndConversation = %v, %v", ended, err)
	}
	ended, err = s.EndConversation(ctx, "site-1")
	if err != nil || ended {
		t.Fatalf("second EndConversation = %v, %v (want false)", ended, err)
	}
	old, _ := s.GetConversation(ctx, c1.ID)
	if old.EndedAt == nil {
		t.Error("ended_at not set")
	}
	c2, err := s.OpenConversation(ctx, "site-1", "store-1")
	if err != nil || c2.ID == c1.ID {
		t.Fatalf("after end, OpenConversation must create: %v %v", c2, err)
	}
	c3, err := s.RotateConversation(ctx, "site-1", "store-1")
	if err != nil || c3.ID == c2.ID {
		t.Fatalf("rotate: %v %v", c3, err)
	}
	c2after, _ := s.GetConversation(ctx, c2.ID)
	if c2after.EndedAt == nil {
		t.Error("rotate must end the previous conversation")
	}
	if _, err := s.GetConversation(ctx, "00000000-0000-0000-0000-000000000000"); !errors.Is(err, store.ErrNotFound) {
		t.Errorf("unknown conversation err = %v", err)
	}
	if _, err := s.AppendMessage(ctx, "00000000-0000-0000-0000-000000000000", "user", json.RawMessage(`[]`)); !errors.Is(err, store.ErrNotFound) {
		t.Errorf("append to unknown conversation err = %v, want ErrNotFound", err)
	}
}

func TestTurnsStateMachine(t *testing.T) {
	s := storetest.Open(t)
	ctx := ctxT(t)
	c, err := s.OpenConversation(ctx, "site-2", "store-2")
	if err != nil {
		t.Fatal(err)
	}
	turn, err := s.CreateTurn(ctx, store.NewTurn{ConversationID: c.ID, ManifestHash: hashA, BuildID: "abc", Model: "default", UserEmail: "m@x.test"})
	if err != nil {
		t.Fatal(err)
	}
	if turn.Status != store.TurnRunning || turn.Ended != nil || len(turn.PendingToolCalls) != 0 {
		t.Fatalf("new turn = %+v", turn)
	}
	// running → waiting_tools with pending calls.
	pending := json.RawMessage(`[{"id":"tu_1","name":"set_fields","input":{"values":{"title":"x"}}}]`)
	if err := s.SetTurnWaiting(ctx, turn.ID, store.TurnWaitingTools, pending); err != nil {
		t.Fatal(err)
	}
	if err := s.SetTurnWaiting(ctx, turn.ID, store.TurnWaitingTools, pending); !errors.Is(err, store.ErrNotFound) {
		t.Errorf("waiting twice must fail (not running): %v", err)
	}
	if err := s.SetTurnWaiting(ctx, turn.ID, store.TurnDone, nil); err == nil {
		t.Error("SetTurnWaiting(done) accepted")
	}
	got, _ := s.GetTurn(ctx, turn.ID)
	if got.Status != store.TurnWaitingTools || !jsonEq(got.PendingToolCalls, pending) {
		t.Fatalf("after waiting: %+v", got)
	}
	// Resume returns the pending blocks and clears them.
	back, err := s.ResumeTurn(ctx, turn.ID)
	if err != nil || !jsonEq(back, pending) {
		t.Fatalf("ResumeTurn = %s, %v", back, err)
	}
	got, _ = s.GetTurn(ctx, turn.ID)
	if got.Status != store.TurnRunning || len(got.PendingToolCalls) != 0 {
		t.Fatalf("after resume: %+v", got)
	}
	if _, err := s.ResumeTurn(ctx, turn.ID); !errors.Is(err, store.ErrNotFound) {
		t.Errorf("resume while running must be ErrNotFound (409 turn_not_waiting): %v", err)
	}
	// suspended (pick_asset) → resume → done.
	if err := s.SetTurnWaiting(ctx, turn.ID, store.TurnSuspended, json.RawMessage(`[{"id":"tu_2","name":"pick_asset"}]`)); err != nil {
		t.Fatal(err)
	}
	if _, err := s.ResumeTurn(ctx, turn.ID); err != nil {
		t.Fatal(err)
	}
	usage := json.RawMessage(`{"input_tokens":10,"output_tokens":5,"calls":2}`)
	if err := s.FinishTurn(ctx, turn.ID, store.TurnDone, usage); err != nil {
		t.Fatal(err)
	}
	got, _ = s.GetTurn(ctx, turn.ID)
	if got.Status != store.TurnDone || got.Ended == nil || !jsonEq(got.Usage, usage) {
		t.Fatalf("after finish: %+v", got)
	}
	if err := s.FinishTurn(ctx, turn.ID, store.TurnError, nil); !errors.Is(err, store.ErrNotFound) {
		t.Errorf("finishing a finished turn: %v", err)
	}
	if err := s.FinishTurn(ctx, turn.ID, "expired", nil); err == nil {
		t.Error("FinishTurn(expired) accepted; the reaper owns that status")
	}
	if _, err := s.GetTurn(ctx, "00000000-0000-0000-0000-000000000000"); !errors.Is(err, store.ErrNotFound) {
		t.Errorf("unknown turn: %v", err)
	}

	// TTL reaper: an old waiting turn expires, a young one and the done one don't.
	base := time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)
	s.SetClock(func() time.Time { return base.Add(-30 * time.Minute) })
	stale, _ := s.CreateTurn(ctx, store.NewTurn{ConversationID: c.ID, ManifestHash: hashA})
	_ = s.SetTurnWaiting(ctx, stale.ID, store.TurnWaitingTools, pending)
	s.SetClock(func() time.Time { return base.Add(-1 * time.Minute) })
	young, _ := s.CreateTurn(ctx, store.NewTurn{ConversationID: c.ID, ManifestHash: hashA})
	s.SetClock(func() time.Time { return base })
	ids, err := s.ExpireStaleTurns(ctx, 10*time.Minute)
	if err != nil || len(ids) != 1 || ids[0] != stale.ID {
		t.Fatalf("ExpireStaleTurns = %v, %v; want [%s]", ids, err, stale.ID)
	}
	st, _ := s.GetTurn(ctx, stale.ID)
	if st.Status != store.TurnExpired || st.Ended == nil || len(st.PendingToolCalls) != 0 {
		t.Errorf("stale turn after reap: %+v", st)
	}
	yt, _ := s.GetTurn(ctx, young.ID)
	if yt.Status != store.TurnRunning {
		t.Errorf("young turn reaped: %+v", yt)
	}
	live, _ := s.CountLiveTurns(ctx)
	if live != 1 {
		t.Errorf("live turns = %d, want 1", live)
	}
	turns, err := s.ListTurns(ctx, c.ID)
	if err != nil || len(turns) != 3 || turns[0].ID != stale.ID /* earliest started */ || turns[2].ID != turn.ID {
		t.Errorf("ListTurns = %d, %v (order by started)", len(turns), err)
	}
}

// TestTurnsSupersedeAndHasNewer: SupersedeTurns closes only the
// conversation's waiting/suspended turns (not running, not another
// conversation's) and returns each once with its pending doc; the reaper's
// ExpiredTurn.HasNewer is set exactly when a later-started turn exists in
// the same conversation (loop.Reap uses it to leave a younger turn's tail
// alone — refuter N4).
func TestTurnsSupersedeAndHasNewer(t *testing.T) {
	s := storetest.Open(t)
	ctx := ctxT(t)
	base := time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)
	c, err := s.OpenConversation(ctx, "site-sup", "store-sup")
	if err != nil {
		t.Fatal(err)
	}
	other, err := s.OpenConversation(ctx, "site-sup-other", "store-sup")
	if err != nil {
		t.Fatal(err)
	}
	pending := json.RawMessage(`{"calls":[{"id":"pa1","name":"pick_asset"}],"seq":1}`)
	s.SetClock(func() time.Time { return base.Add(-20 * time.Minute) })
	waiting, _ := s.CreateTurn(ctx, store.NewTurn{ConversationID: c.ID, ManifestHash: hashA})
	_ = s.SetTurnWaiting(ctx, waiting.ID, store.TurnWaitingTools, pending)
	s.SetClock(func() time.Time { return base.Add(-15 * time.Minute) })
	suspended, _ := s.CreateTurn(ctx, store.NewTurn{ConversationID: c.ID, ManifestHash: hashA})
	_ = s.SetTurnWaiting(ctx, suspended.ID, store.TurnSuspended, pending)
	s.SetClock(func() time.Time { return base.Add(-12 * time.Minute) })
	running, _ := s.CreateTurn(ctx, store.NewTurn{ConversationID: c.ID, ManifestHash: hashA})
	done, _ := s.CreateTurn(ctx, store.NewTurn{ConversationID: c.ID, ManifestHash: hashA})
	_ = s.FinishTurn(ctx, done.ID, store.TurnDone, nil)
	foreign, _ := s.CreateTurn(ctx, store.NewTurn{ConversationID: other.ID, ManifestHash: hashA})
	_ = s.SetTurnWaiting(ctx, foreign.ID, store.TurnWaitingTools, pending)
	s.SetClock(func() time.Time { return base })

	got, err := s.SupersedeTurns(ctx, c.ID)
	if err != nil || len(got) != 3 {
		t.Fatalf("SupersedeTurns = %+v, %v; want 3 (waiting, suspended, running)", got, err)
	}
	byID := map[string]store.SupersededTurn{}
	for _, e := range got {
		byID[e.ID] = e
	}
	if e := byID[waiting.ID]; e.WasStatus != store.TurnWaitingTools || e.ConversationID != c.ID || !jsonEq(e.PendingToolCalls, pending) {
		t.Errorf("superseded waiting = %+v", e)
	}
	if e := byID[suspended.ID]; e.WasStatus != store.TurnSuspended || e.ConversationID != c.ID || !jsonEq(e.PendingToolCalls, pending) {
		t.Errorf("superseded suspended = %+v", e)
	}
	if e := byID[running.ID]; e.WasStatus != store.TurnRunning || e.ConversationID != c.ID || e.PendingToolCalls != nil {
		t.Errorf("superseded running = %+v (no pending doc expected)", e)
	}
	for _, id := range []string{waiting.ID, suspended.ID, running.ID} {
		row, _ := s.GetTurn(ctx, id)
		if row.Status != store.TurnSuperseded || row.Ended == nil || len(row.PendingToolCalls) != 0 {
			t.Errorf("row %s after supersede = %+v", id, row)
		}
		if _, err := s.ResumeTurn(ctx, id); !errors.Is(err, store.ErrNotFound) {
			t.Errorf("resume of a superseded turn = %v, want ErrNotFound (409)", err)
		}
		if err := s.FinishTurn(ctx, id, store.TurnError, nil); !errors.Is(err, store.ErrNotFound) {
			t.Errorf("finish of a superseded turn = %v, want ErrNotFound (terminal)", err)
		}
		if err := s.SetTurnWaiting(ctx, id, store.TurnWaitingTools, pending); !errors.Is(err, store.ErrNotFound) {
			t.Errorf("SetTurnWaiting on a superseded turn = %v, want ErrNotFound (the running loop's guard)", err)
		}
	}
	if row, _ := s.GetTurn(ctx, done.ID); row.Status != store.TurnDone {
		t.Errorf("done turn touched: %+v", row)
	}
	if row, _ := s.GetTurn(ctx, foreign.ID); row.Status != store.TurnWaitingTools {
		t.Errorf("other conversation's turn touched: %+v", row)
	}
	if again, err := s.SupersedeTurns(ctx, c.ID); err != nil || len(again) != 0 {
		t.Errorf("second SupersedeTurns = %+v, %v; want none (each row once)", again, err)
	}
	live, _ := s.CountLiveTurns(ctx)
	if live != 1 { // foreign
		t.Errorf("live turns = %d, want 1", live)
	}

	// HasNewer: an old running turn (started -12m) is older than a fresh turn
	// in its conversation → true; the other conversation's turn has nothing
	// after it → false.
	s.SetClock(func() time.Time { return base.Add(-12 * time.Minute) })
	oldRunning, _ := s.CreateTurn(ctx, store.NewTurn{ConversationID: c.ID, ManifestHash: hashA})
	s.SetClock(func() time.Time { return base.Add(-1 * time.Minute) })
	young, _ := s.CreateTurn(ctx, store.NewTurn{ConversationID: c.ID, ManifestHash: hashA})
	s.SetClock(func() time.Time { return base })
	expired, err := s.ExpireStaleTurnsWithPending(ctx, 10*time.Minute)
	if err != nil || len(expired) != 2 {
		t.Fatalf("ExpireStaleTurnsWithPending = %+v, %v; want 2 (running + foreign)", expired, err)
	}
	for _, e := range expired {
		switch e.ID {
		case oldRunning.ID:
			if !e.HasNewer || e.WasStatus != store.TurnRunning || e.PendingToolCalls != nil {
				t.Errorf("expired running turn = %+v; want HasNewer", e)
			}
		case foreign.ID:
			if e.HasNewer || e.WasStatus != store.TurnWaitingTools || !jsonEq(e.PendingToolCalls, pending) {
				t.Errorf("expired foreign turn = %+v; want !HasNewer", e)
			}
		default:
			t.Errorf("unexpected expired turn %+v", e)
		}
	}
	if row, _ := s.GetTurn(ctx, young.ID); row.Status != store.TurnRunning {
		t.Errorf("young turn reaped: %+v", row)
	}
}

// SupersedeTurnsForSite closes the live turns of EVERY conversation of the
// site — the open one, a rotated one and an ended one — running included,
// each row once, and leaves other sites alone (README §4: a Start or a
// DELETE for the site).
func TestTurnsSupersedeForSite(t *testing.T) {
	s := storetest.Open(t)
	ctx := ctxT(t)
	pending := json.RawMessage(`{"calls":[{"id":"op1","name":"set_fields"}],"seq":1}`)
	if _, err := s.SupersedeTurnsForSite(ctx, ""); err == nil {
		t.Error("empty siteID accepted")
	}
	// Conversation 1: ended by DELETE, with a suspended turn.
	c1, _ := s.OpenConversation(ctx, "site-wide", "store-wide")
	onEnded, _ := s.CreateTurn(ctx, store.NewTurn{ConversationID: c1.ID, ManifestHash: hashA})
	_ = s.SetTurnWaiting(ctx, onEnded.ID, store.TurnSuspended, pending)
	if ok, err := s.EndConversation(ctx, "site-wide"); err != nil || !ok {
		t.Fatalf("end = %v %v", ok, err)
	}
	// Conversation 2: rotated away, with a waiting turn; conversation 3 open
	// with a running turn and a finished one.
	c2, _ := s.OpenConversation(ctx, "site-wide", "store-wide")
	onRotated, _ := s.CreateTurn(ctx, store.NewTurn{ConversationID: c2.ID, ManifestHash: hashA})
	_ = s.SetTurnWaiting(ctx, onRotated.ID, store.TurnWaitingTools, pending)
	c3, err := s.RotateConversation(ctx, "site-wide", "store-wide")
	if err != nil || c3.ID == c2.ID {
		t.Fatalf("rotate = %+v %v", c3, err)
	}
	running, _ := s.CreateTurn(ctx, store.NewTurn{ConversationID: c3.ID, ManifestHash: hashA})
	finished, _ := s.CreateTurn(ctx, store.NewTurn{ConversationID: c3.ID, ManifestHash: hashA})
	_ = s.FinishTurn(ctx, finished.ID, store.TurnError, nil)
	// Another site's live turn.
	co, _ := s.OpenConversation(ctx, "site-wide-other", "store-wide")
	foreign, _ := s.CreateTurn(ctx, store.NewTurn{ConversationID: co.ID, ManifestHash: hashA})
	_ = s.SetTurnWaiting(ctx, foreign.ID, store.TurnWaitingTools, pending)

	got, err := s.SupersedeTurnsForSite(ctx, "site-wide")
	if err != nil || len(got) != 3 {
		t.Fatalf("SupersedeTurnsForSite = %+v, %v; want 3", got, err)
	}
	want := map[string]struct {
		conv, status string
		doc          bool
	}{
		onEnded.ID:   {c1.ID, store.TurnSuspended, true},
		onRotated.ID: {c2.ID, store.TurnWaitingTools, true},
		running.ID:   {c3.ID, store.TurnRunning, false},
	}
	for _, e := range got {
		w, ok := want[e.ID]
		if !ok || e.ConversationID != w.conv || e.WasStatus != w.status || (e.PendingToolCalls != nil) != w.doc || (w.doc && !jsonEq(e.PendingToolCalls, pending)) {
			t.Errorf("superseded %+v, want %+v", e, w)
		}
		row, _ := s.GetTurn(ctx, e.ID)
		if row.Status != store.TurnSuperseded || row.Ended == nil || row.PendingToolCalls != nil {
			t.Errorf("row after supersede = %+v", row)
		}
	}
	if row, _ := s.GetTurn(ctx, finished.ID); row.Status != store.TurnError {
		t.Errorf("finished turn touched: %+v", row)
	}
	if row, _ := s.GetTurn(ctx, foreign.ID); row.Status != store.TurnWaitingTools {
		t.Errorf("other site's turn touched: %+v", row)
	}
	if again, err := s.SupersedeTurnsForSite(ctx, "site-wide"); err != nil || len(again) != 0 {
		t.Errorf("second SupersedeTurnsForSite = %+v, %v; want none", again, err)
	}
	if live, _ := s.CountLiveTurns(ctx); live != 1 {
		t.Errorf("live turns = %d, want 1 (the other site's)", live)
	}
}

// StartTurn: the conversation (named, open, created, rotated), the row and
// the supersede of every other live turn of the site, in one transaction.
func TestTurnsStartTurn(t *testing.T) {
	s := storetest.Open(t)
	ctx := ctxT(t)
	base := time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)
	s.SetClock(func() time.Time { return base })
	pending := json.RawMessage(`{"calls":[{"id":"op1","name":"set_fields"}],"seq":1}`)
	in := store.StartTurn{SiteID: "site-st", StoreID: "store-st", RotateAt: 4, ManifestHash: hashA, BuildID: "b1", Model: "m", UserEmail: "o@x"}

	if _, err := s.StartTurn(ctx, store.StartTurn{ManifestHash: hashA}); err == nil {
		t.Error("empty siteID accepted")
	}
	// First Start: creates the conversation, nothing to supersede.
	first, err := s.StartTurn(ctx, in)
	if err != nil {
		t.Fatal(err)
	}
	if first.Conversation == nil || first.Conversation.SiteID != "site-st" || first.Conversation.StoreID != "store-st" || first.Rotated || len(first.Superseded) != 0 {
		t.Fatalf("first = %+v", first)
	}
	if row := first.Turn; row.Status != store.TurnRunning || row.ConversationID != first.Conversation.ID || row.ManifestHash != hashA || row.BuildID != "b1" || row.Model != "m" || row.UserEmail != "o@x" || !row.Started.Equal(base) {
		t.Fatalf("first row = %+v", row)
	}
	if open, _ := s.FindOpenConversation(ctx, "site-st"); open.ID != first.Conversation.ID {
		t.Fatalf("open = %s, want %s", open.ID, first.Conversation.ID)
	}
	_ = s.SetTurnWaiting(ctx, first.Turn.ID, store.TurnWaitingTools, pending)
	// A live turn of another site is never touched.
	co, _ := s.OpenConversation(ctx, "site-st-other", "store-st")
	foreign, _ := s.CreateTurn(ctx, store.NewTurn{ConversationID: co.ID, ManifestHash: hashA})

	// Second Start: same conversation, supersedes the waiting first (with
	// its pending doc), not self, not the other site.
	second, err := s.StartTurn(ctx, in)
	if err != nil {
		t.Fatal(err)
	}
	if second.Conversation.ID != first.Conversation.ID || second.Rotated || len(second.Superseded) != 1 {
		t.Fatalf("second = %+v", second)
	}
	if e := second.Superseded[0]; e.ID != first.Turn.ID || e.WasStatus != store.TurnWaitingTools || e.ConversationID != first.Conversation.ID || !jsonEq(e.PendingToolCalls, pending) {
		t.Errorf("superseded = %+v", e)
	}
	if row, _ := s.GetTurn(ctx, first.Turn.ID); row.Status != store.TurnSuperseded || row.Ended == nil || row.PendingToolCalls != nil {
		t.Errorf("first row after second Start = %+v", row)
	}
	for name, id := range map[string]string{"self": second.Turn.ID, "other site": foreign.ID} {
		if row, _ := s.GetTurn(ctx, id); row.Status != store.TurnRunning {
			t.Errorf("%s turn touched: %+v", name, row)
		}
	}

	// Rotation: at RotateAt messages the open conversation is ended and a
	// fresh one opened, in the same transaction; the running second turn
	// (on the old conversation) is superseded too.
	for i := 0; i < 4; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		if _, err := s.AppendMessage(ctx, second.Conversation.ID, role, json.RawMessage(`[{"type":"text","text":"m"}]`)); err != nil {
			t.Fatal(err)
		}
	}
	third, err := s.StartTurn(ctx, in)
	if err != nil {
		t.Fatal(err)
	}
	if !third.Rotated || third.Conversation.ID == second.Conversation.ID || third.Turn.ConversationID != third.Conversation.ID || len(third.Superseded) != 1 || third.Superseded[0].ID != second.Turn.ID || third.Superseded[0].WasStatus != store.TurnRunning || third.Superseded[0].PendingToolCalls != nil {
		t.Fatalf("third = %+v superseded=%+v", third, third.Superseded)
	}
	if old, _ := s.GetConversation(ctx, second.Conversation.ID); old.EndedAt == nil {
		t.Error("rotated conversation not ended")
	}
	if open, _ := s.FindOpenConversation(ctx, "site-st"); open.ID != third.Conversation.ID {
		t.Errorf("open = %s, want the fresh %s", open.ID, third.Conversation.ID)
	}
	// RotateAt 0 = never rotate.
	noRotate := in
	noRotate.RotateAt = 0
	for i := 0; i < 6; i++ {
		_, _ = s.AppendMessage(ctx, third.Conversation.ID, []string{"user", "assistant"}[i%2], json.RawMessage(`[{"type":"text","text":"m"}]`))
	}
	fourth, err := s.StartTurn(ctx, noRotate)
	if err != nil || fourth.Rotated || fourth.Conversation.ID != third.Conversation.ID {
		t.Fatalf("fourth = %+v %v", fourth, err)
	}

	// Named conversation: the open one is fine; an ended one is
	// ErrConversationEnded; a missing one or another site's is ErrNotFound.
	named := noRotate
	named.ConversationID = fourth.Conversation.ID
	fifth, err := s.StartTurn(ctx, named)
	if err != nil || fifth.Conversation.ID != fourth.Conversation.ID || len(fifth.Superseded) != 1 || fifth.Superseded[0].ID != fourth.Turn.ID {
		t.Fatalf("named open = %+v %v", fifth, err)
	}
	named.ConversationID = second.Conversation.ID
	if _, err := s.StartTurn(ctx, named); !errors.Is(err, store.ErrConversationEnded) || !errors.Is(err, store.ErrConflict) {
		t.Errorf("named ended = %v", err)
	}
	named.ConversationID = co.ID
	if _, err := s.StartTurn(ctx, named); !errors.Is(err, store.ErrNotFound) {
		t.Errorf("named other site = %v", err)
	}
	named.ConversationID = "00000000-0000-0000-0000-000000000000"
	if _, err := s.StartTurn(ctx, named); !errors.Is(err, store.ErrNotFound) {
		t.Errorf("named missing = %v", err)
	}
	// Failed Starts left nothing behind: exactly one live turn for the site
	// (fifth) plus the other site's.
	if live, _ := s.CountLiveTurns(ctx); live != 2 {
		t.Errorf("live turns = %d, want 2", live)
	}
	if row, _ := s.GetTurn(ctx, fifth.Turn.ID); row.Status != store.TurnRunning {
		t.Errorf("fifth = %s", row.Status)
	}
	// The unconditional form still closes everything of the site.
	rest, err := s.SupersedeTurnsForSite(ctx, "site-st")
	if err != nil || len(rest) != 1 || rest[0].ID != fifth.Turn.ID {
		t.Errorf("SupersedeTurnsForSite = %+v, %v", rest, err)
	}
}

// Appends refuse an ended conversation (rotated away or DELETE'd) with
// ErrConversationEnded, which is an ErrConflict: the compare-and-set of a
// turn superseded by the Start that rotated its conversation fails even
// though the tail is still its own text.
func TestMessagesRefuseEndedConversation(t *testing.T) {
	s := storetest.Open(t)
	ctx := ctxT(t)
	c, _ := s.OpenConversation(ctx, "site-ended", "store-ended")
	text := json.RawMessage(`[{"type":"text","text":"hi"}]`)
	seq, err := s.AppendMessageAfter(ctx, c.ID, "user", text, 0)
	if err != nil || seq != 1 {
		t.Fatal(seq, err)
	}
	if ok, _ := s.EndConversation(ctx, "site-ended"); !ok {
		t.Fatal("not ended")
	}
	if _, err := s.AppendMessageAfter(ctx, c.ID, "assistant", text, 1); !errors.Is(err, store.ErrConversationEnded) || !errors.Is(err, store.ErrConflict) {
		t.Errorf("AppendMessageAfter on ended = %v", err)
	}
	if _, err := s.AppendMessage(ctx, c.ID, "assistant", text); !errors.Is(err, store.ErrConversationEnded) {
		t.Errorf("AppendMessage on ended = %v", err)
	}
	if _, err := s.AppendMessages(ctx, c.ID, []store.Message{{Role: "assistant", Content: text}}); !errors.Is(err, store.ErrConversationEnded) {
		t.Errorf("AppendMessages on ended = %v", err)
	}
	if n, _ := s.CountMessages(ctx, c.ID); n != 1 {
		t.Errorf("messages = %d, want 1", n)
	}
}

// EndSiteConversation (DELETE): supersedes the site's live turns and ends
// the open conversation in one transaction; a no-op on a site with none.
func TestEndSiteConversation(t *testing.T) {
	s := storetest.Open(t)
	ctx := ctxT(t)
	if out, err := s.EndSiteConversation(ctx, "site-none"); err != nil || out.Ended || len(out.Superseded) != 0 {
		t.Fatalf("nothing to end = %+v %v", out, err)
	}
	if _, err := s.EndSiteConversation(ctx, ""); err == nil {
		t.Error("empty siteID accepted")
	}
	st, err := s.StartTurn(ctx, store.StartTurn{SiteID: "site-end", StoreID: "s", ManifestHash: hashA})
	if err != nil {
		t.Fatal(err)
	}
	_ = s.SetTurnWaiting(ctx, st.Turn.ID, store.TurnSuspended, json.RawMessage(`{"calls":[{"id":"pa1","name":"pick_asset"}],"seq":1,"suspended_id":"pa1"}`))
	out, err := s.EndSiteConversation(ctx, "site-end")
	if err != nil || !out.Ended || len(out.Superseded) != 1 || out.Superseded[0].ID != st.Turn.ID || out.Superseded[0].WasStatus != store.TurnSuspended || out.Superseded[0].PendingToolCalls == nil {
		t.Fatalf("end = %+v %v", out, err)
	}
	if c, _ := s.GetConversation(ctx, st.Conversation.ID); c.EndedAt == nil {
		t.Error("conversation not ended")
	}
	if row, _ := s.GetTurn(ctx, st.Turn.ID); row.Status != store.TurnSuperseded {
		t.Errorf("row = %s", row.Status)
	}
	if out, _ := s.EndSiteConversation(ctx, "site-end"); out.Ended || len(out.Superseded) != 0 {
		t.Errorf("second end = %+v", out)
	}
}

// twoStores opens two *store.Store on ONE database: two service instances.
// skewB shifts the second's clock (turns.started is the instance's clock).
func twoStores(t *testing.T, skewB time.Duration) (*store.Store, *store.Store) {
	t.Helper()
	dsn := storetest.OpenDSN(t)
	ctx := ctxT(t)
	a, err := store.Open(ctx, dsn)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(a.Close)
	b, err := store.Open(ctx, dsn)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(b.Close)
	if skewB != 0 {
		b.SetClock(func() time.Time { return time.Now().UTC().Add(skewB) })
	}
	return a, b
}

// raceRounds is the number of rounds the racing-Start tests run; raised
// with STORE_RACE_ROUNDS for a longer soak.
func raceRounds(t *testing.T, def int) int {
	t.Helper()
	if v := os.Getenv("STORE_RACE_ROUNDS"); v != "" {
		var n int
		if _, err := fmt.Sscan(v, &n); err == nil && n > 0 {
			return n
		}
	}
	return def
}

// E6 shape: two instances Start for one site at the same moment, round after
// round. After both commit exactly one row is live — the one committed last
// (the one that superseded the other) — never both, never none.
func TestStartTurn_TwoInstancesRacingNeverBothLive(t *testing.T) {
	a, b := twoStores(t, 0)
	ctx := ctxT(t)
	rounds := raceRounds(t, 60)
	in := store.StartTurn{SiteID: "site-race", StoreID: "s", ManifestHash: hashA}
	bothLive, zeroLive := 0, 0
	for iter := 0; iter < rounds; iter++ {
		var wg sync.WaitGroup
		res := make([]*store.StartedTurn, 2)
		errs := make([]error, 2)
		for i, st := range []*store.Store{a, b} {
			wg.Add(1)
			go func(i int, st *store.Store) {
				defer wg.Done()
				res[i], errs[i] = st.StartTurn(ctx, in)
			}(i, st)
		}
		wg.Wait()
		if errs[0] != nil || errs[1] != nil {
			t.Fatalf("iter %d: %v", iter, errs)
		}
		live, err := a.CountLiveTurns(ctx)
		if err != nil {
			t.Fatal(err)
		}
		r0, _ := a.GetTurn(ctx, res[0].Turn.ID)
		r1, _ := a.GetTurn(ctx, res[1].Turn.ID)
		switch live {
		case 2:
			bothLive++
			t.Errorf("iter %d: BOTH LIVE (row0 %s started=%v, row1 %s started=%v)", iter, r0.Status, r0.Started, r1.Status, r1.Started)
		case 0:
			zeroLive++
			t.Errorf("iter %d: no live turn", iter)
		}
		// The survivor is the one whose StartTurn reported superseding the
		// other; the loser reported nothing of this round.
		var surv, loser int
		if r0.Status == store.TurnRunning {
			surv, loser = 0, 1
		} else {
			surv, loser = 1, 0
		}
		if live == 1 {
			if !supersededIDs(res[surv].Superseded)[res[loser].Turn.ID] {
				t.Errorf("iter %d: the survivor's StartTurn did not report superseding the loser (%v)", iter, res[surv].Superseded)
			}
			if supersededIDs(res[loser].Superseded)[res[surv].Turn.ID] {
				t.Errorf("iter %d: the loser reported superseding the survivor", iter)
			}
		}
		if _, err := a.SupersedeTurnsForSite(ctx, "site-race"); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("rounds=%d bothLive=%d zeroLive=%d", rounds, bothLive, zeroLive)
}

func supersededIDs(rows []store.SupersededTurn) map[string]bool {
	out := map[string]bool{}
	for _, r := range rows {
		out[r.ID] = true
	}
	return out
}

// E3 shape: instance B's clock is 2s behind. A Start on A goes waiting; a
// Start on B afterwards gets an EARLIER started, and must still supersede
// A's turn — the supersede is by commit order, not by started.
func TestStartTurn_ClockSkewDoesNotProtectOlderTurn(t *testing.T) {
	a, b := twoStores(t, -2*time.Second)
	ctx := ctxT(t)
	in := store.StartTurn{SiteID: "site-skew", StoreID: "s", ManifestHash: hashA}
	first, err := a.StartTurn(ctx, in)
	if err != nil {
		t.Fatal(err)
	}
	_ = a.SetTurnWaiting(ctx, first.Turn.ID, store.TurnWaitingTools, json.RawMessage(`{"calls":[{"id":"op1","name":"set_fields"}],"seq":1}`))
	second, err := b.StartTurn(ctx, in)
	if err != nil {
		t.Fatal(err)
	}
	if !second.Turn.Started.Before(first.Turn.Started) {
		t.Fatalf("skew not applied: first.started=%v second.started=%v", first.Turn.Started, second.Turn.Started)
	}
	if len(second.Superseded) != 1 || second.Superseded[0].ID != first.Turn.ID || second.Superseded[0].WasStatus != store.TurnWaitingTools {
		t.Fatalf("second superseded = %+v, want the waiting first turn", second.Superseded)
	}
	if row, _ := a.GetTurn(ctx, first.Turn.ID); row.Status != store.TurnSuperseded {
		t.Errorf("first = %s", row.Status)
	}
	if live, _ := a.CountLiveTurns(ctx); live != 1 {
		t.Errorf("live = %d", live)
	}
	// And the other way round: a Start on the fast instance after one on the
	// slow instance supersedes it too (started order = commit order here).
	third, err := a.StartTurn(ctx, in)
	if err != nil || len(third.Superseded) != 1 || third.Superseded[0].ID != second.Turn.ID {
		t.Fatalf("third = %+v %v", third.Superseded, err)
	}
}

// E7 shape: two instances Start at the rotation boundary (both see RotateAt
// reached). The rotation runs inside the serialized block, so the second
// Start sees the fresh conversation instead of colliding on
// conversations_open_site; exactly one Start rotates per round.
func TestStartTurn_ConcurrentRotation(t *testing.T) {
	a, b := twoStores(t, 0)
	ctx := ctxT(t)
	rounds := raceRounds(t, 15)
	in := store.StartTurn{SiteID: "site-rot", StoreID: "s", RotateAt: 4, ManifestHash: hashA}
	for iter := 0; iter < rounds; iter++ {
		open, err := a.OpenConversation(ctx, "site-rot", "s")
		if err != nil {
			t.Fatal(err)
		}
		n, _ := a.CountMessages(ctx, open.ID)
		for i := n; i < 4; i++ {
			if _, err := a.AppendMessage(ctx, open.ID, []string{"user", "assistant"}[i%2], json.RawMessage(`[{"type":"text","text":"m"}]`)); err != nil {
				t.Fatal(err)
			}
		}
		var wg sync.WaitGroup
		res := make([]*store.StartedTurn, 2)
		errs := make([]error, 2)
		for i, st := range []*store.Store{a, b} {
			wg.Add(1)
			go func(i int, st *store.Store) {
				defer wg.Done()
				res[i], errs[i] = st.StartTurn(ctx, in)
			}(i, st)
		}
		wg.Wait()
		if errs[0] != nil || errs[1] != nil {
			t.Fatalf("iter %d: Start failed at the rotation boundary: %v", iter, errs)
		}
		rotated := 0
		for _, r := range res {
			if r.Rotated {
				rotated++
			}
			if r.Conversation.ID == open.ID {
				t.Errorf("iter %d: a Start stayed on the full conversation %s", iter, open.ID)
			}
		}
		if rotated != 1 {
			t.Errorf("iter %d: %d Starts rotated, want exactly 1", iter, rotated)
		}
		if res[0].Conversation.ID != res[1].Conversation.ID {
			t.Errorf("iter %d: the two Starts are on different conversations %s / %s", iter, res[0].Conversation.ID, res[1].Conversation.ID)
		}
		if live, _ := a.CountLiveTurns(ctx); live != 1 {
			t.Errorf("iter %d: live = %d", iter, live)
		}
		if c, _ := a.GetConversation(ctx, open.ID); c.EndedAt == nil {
			t.Errorf("iter %d: the full conversation was not ended", iter)
		}
		if _, err := a.SupersedeTurnsForSite(ctx, "site-rot"); err != nil {
			t.Fatal(err)
		}
	}
}

// A Start racing a DELETE (EndSiteConversation) on another instance: both
// serialize on the site; afterwards either the Start's turn is superseded
// and its conversation ended (End committed second) or the Start opened a
// fresh conversation after the End (Start committed second). Never a live
// turn on an ended conversation.
func TestStartTurn_RacesEndSiteConversation(t *testing.T) {
	a, b := twoStores(t, 0)
	ctx := ctxT(t)
	rounds := raceRounds(t, 20)
	in := store.StartTurn{SiteID: "site-del", StoreID: "s", ManifestHash: hashA}
	startSecond, endSecond := 0, 0
	for iter := 0; iter < rounds; iter++ {
		if _, err := a.OpenConversation(ctx, "site-del", "s"); err != nil {
			t.Fatal(err)
		}
		var wg sync.WaitGroup
		var st *store.StartedTurn
		var ended *store.EndedConversation
		var errS, errE error
		wg.Add(2)
		go func() { defer wg.Done(); st, errS = a.StartTurn(ctx, in) }()
		go func() { defer wg.Done(); ended, errE = b.EndSiteConversation(ctx, "site-del") }()
		wg.Wait()
		if errS != nil || errE != nil {
			t.Fatalf("iter %d: start=%v end=%v", iter, errS, errE)
		}
		row, _ := a.GetTurn(ctx, st.Turn.ID)
		conv, _ := a.GetConversation(ctx, st.Conversation.ID)
		switch {
		case row.Status == store.TurnSuperseded && conv.EndedAt != nil && len(ended.Superseded) == 1 && ended.Superseded[0].ID == st.Turn.ID:
			endSecond++
		case row.Status == store.TurnRunning && conv.EndedAt == nil && len(ended.Superseded) == 0 && ended.Ended:
			startSecond++
		default:
			t.Errorf("iter %d: row=%s conv.ended=%v end=%+v", iter, row.Status, conv.EndedAt != nil, ended)
		}
		if _, err := a.SupersedeTurnsForSite(ctx, "site-del"); err != nil {
			t.Fatal(err)
		}
		if _, err := a.EndConversation(ctx, "site-del"); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("rounds=%d start-committed-second=%d end-committed-second=%d", rounds, startSecond, endSecond)
}

func TestTracesFeedbackAndOptOut(t *testing.T) {
	s := storetest.Open(t)
	ctx := ctxT(t)
	turnID := "11111111-1111-1111-1111-111111111111"
	tr := store.Trace{TurnID: turnID, SiteID: "site-3", StoreID: "store-3",
		Context: json.RawMessage(`{"page":"home"}`),
		Ops:     json.RawMessage(`[{"name":"set_fields"}]`),
		Events:  json.RawMessage(`[{"event":"tool.call"}]`)}
	if err := s.InsertTrace(ctx, tr); err != nil {
		t.Fatal(err)
	}
	if err := s.InsertTrace(ctx, store.Trace{SiteID: "x"}); err == nil {
		t.Error("trace without turn_id accepted")
	}
	got, err := s.GetTrace(ctx, turnID)
	if err != nil || !jsonEq(got.Ops, tr.Ops) || len(got.Feedback) != 0 {
		t.Fatalf("GetTrace = %+v, %v", got, err)
	}
	// Feedback merges shallowly and survives a re-insert.
	if err := s.MergeFeedback(ctx, turnID, json.RawMessage(`{"kept_paths":["a.b"]}`)); err != nil {
		t.Fatal(err)
	}
	if err := s.MergeFeedback(ctx, turnID, json.RawMessage(`{"saved":true}`)); err != nil {
		t.Fatal(err)
	}
	tr.Ops = json.RawMessage(`[{"name":"set_fields"},{"name":"edit_list"}]`)
	if err := s.InsertTrace(ctx, tr); err != nil {
		t.Fatal(err)
	}
	got, _ = s.GetTrace(ctx, turnID)
	var fb map[string]any
	if err := json.Unmarshal(got.Feedback, &fb); err != nil || fb["saved"] != true || fb["kept_paths"] == nil {
		t.Errorf("feedback = %s", got.Feedback)
	}
	if !jsonEq(got.Ops, tr.Ops) {
		t.Errorf("re-insert must replace ops: %s", got.Ops)
	}
	if err := s.MergeFeedback(ctx, "22222222-2222-2222-2222-222222222222", json.RawMessage(`{}`)); !errors.Is(err, store.ErrNotFound) {
		t.Errorf("feedback on missing trace: %v", err)
	}
	if err := s.MergeFeedback(ctx, turnID, nil); err == nil {
		t.Error("empty feedback accepted")
	}
	// Merged size is capped at MaxFeedbackBytes: a single oversize doc is
	// refused before the DB; a doc that fits alone but not merged is
	// refused by the DB and leaves the row untouched; the existing trace
	// is distinguished from a missing one.
	big := func(key string, n int) json.RawMessage {
		return json.RawMessage(`{"` + key + `":"` + strings.Repeat("x", n) + `"}`)
	}
	if err := s.MergeFeedback(ctx, turnID, big("note", store.MaxFeedbackBytes)); !errors.Is(err, store.ErrFeedbackTooLarge) {
		t.Errorf("oversize feedback: %v", err)
	}
	before, _ := s.GetTrace(ctx, turnID)
	if err := s.MergeFeedback(ctx, turnID, big("a", 9000)); err != nil {
		t.Fatalf("9 KB feedback: %v", err)
	}
	if err := s.MergeFeedback(ctx, turnID, big("b", 9000)); !errors.Is(err, store.ErrFeedbackTooLarge) {
		t.Errorf("second 9 KB (merged ~18 KB) accepted: %v", err)
	}
	after, _ := s.GetTrace(ctx, turnID)
	var fb2 map[string]any
	_ = json.Unmarshal(after.Feedback, &fb2)
	if _, has := fb2["b"]; has || fb2["a"] == nil || fb2["saved"] != true || len(after.Feedback) > store.MaxFeedbackBytes || len(after.Feedback) <= len(before.Feedback) {
		t.Errorf("feedback after capped merge = %d bytes, keys %v", len(after.Feedback), keys(fb2))
	}
	if err := s.MergeFeedback(ctx, "22222222-2222-2222-2222-222222222222", big("a", 9000)); !errors.Is(err, store.ErrNotFound) {
		t.Errorf("feedback on missing trace with a fitting doc: %v", err)
	}
	list, err := s.ListTraces(ctx, time.Now().Add(-time.Hour), time.Now().Add(time.Hour), 10)
	if err != nil || len(list) != 1 {
		t.Errorf("ListTraces = %d, %v", len(list), err)
	}

	// Opt-out.
	out, err := s.IsOptedOut(ctx, "store-3")
	if err != nil || out {
		t.Fatalf("IsOptedOut before = %v, %v", out, err)
	}
	if out, _ := s.IsOptedOut(ctx, ""); out {
		t.Error("empty store id must never be opted out")
	}
	if err := s.SetOptOut(ctx, "store-3", "admin@x.test"); err != nil {
		t.Fatal(err)
	}
	if err := s.SetOptOut(ctx, "store-3", "other@x.test"); err != nil {
		t.Fatalf("SetOptOut must be idempotent: %v", err)
	}
	if err := s.SetOptOut(ctx, "", "x"); err == nil {
		t.Error("SetOptOut(\"\") accepted")
	}
	out, _ = s.IsOptedOut(ctx, "store-3")
	if !out {
		t.Error("not opted out after SetOptOut")
	}
	cleared, err := s.ClearOptOut(ctx, "store-3")
	if err != nil || !cleared {
		t.Fatalf("ClearOptOut = %v, %v", cleared, err)
	}
	cleared, _ = s.ClearOptOut(ctx, "store-3")
	if cleared {
		t.Error("second clear must report false")
	}
}

func TestUsageRows(t *testing.T) {
	s := storetest.Open(t)
	ctx := ctxT(t)
	c, _ := s.OpenConversation(ctx, "site-4", "store-4")
	turn, _ := s.CreateTurn(ctx, store.NewTurn{ConversationID: c.ID, ManifestHash: hashA})
	lat := int64(1234)
	id1, err := s.InsertUsage(ctx, store.UsageRow{TurnID: turn.ID, Model: "claude-sonnet-5-5", SiteID: "site-4", StoreID: "store-4",
		UserEmail: "m@x.test", InputTokens: 100, OutputTokens: 20, CacheWriteTokens: 3000, CacheReadTokens: 0,
		LatencyMS: &lat, Status: "ok", StopReason: "tool_use", RequestID: "msg_1"})
	if err != nil || id1 == 0 {
		t.Fatalf("InsertUsage = %d, %v", id1, err)
	}
	id2, err := s.InsertUsage(ctx, store.UsageRow{Model: "claude-sonnet-5-5", Operation: "tool_turn", Status: "error"})
	if err != nil || id2 <= id1 {
		t.Fatalf("second InsertUsage = %d, %v", id2, err)
	}
	if _, err := s.InsertUsage(ctx, store.UsageRow{}); err == nil {
		t.Error("usage row without model accepted")
	}
	if _, err := s.InsertUsage(ctx, store.UsageRow{Model: "m", InputTokens: -1}); err == nil {
		t.Error("negative tokens accepted")
	}
	rows, err := s.ListUnshippedUsage(ctx, 10)
	if err != nil || len(rows) != 2 || rows[0].ID != id1 || rows[0].Operation != "chat" || rows[0].TurnID != turn.ID ||
		rows[0].LatencyMS == nil || *rows[0].LatencyMS != 1234 || rows[1].TurnID != "" || rows[1].LatencyMS != nil {
		t.Fatalf("ListUnshippedUsage = %+v, %v", rows, err)
	}
	byTurn, err := s.ListUsageByTurn(ctx, turn.ID)
	if err != nil || len(byTurn) != 1 || byTurn[0].ID != id1 {
		t.Fatalf("ListUsageByTurn = %+v, %v", byTurn, err)
	}
	n, err := s.MarkUsageShipped(ctx, []int64{id1})
	if err != nil || n != 1 {
		t.Fatalf("MarkUsageShipped = %d, %v", n, err)
	}
	if n, _ := s.MarkUsageShipped(ctx, nil); n != 0 {
		t.Error("MarkUsageShipped(nil) must be a no-op")
	}
	rows, _ = s.ListUnshippedUsage(ctx, 10)
	if len(rows) != 1 || rows[0].ID != id2 {
		t.Fatalf("unshipped after mark = %+v", rows)
	}
}

func TestRetention(t *testing.T) {
	s := storetest.Open(t)
	ctx := ctxT(t)
	now := time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)
	old := now.AddDate(0, 0, -91)   // past both 90-day cutoffs
	young := now.AddDate(0, 0, -89) // inside
	usageOld := now.AddDate(0, 0, -31)

	// Old ended conversation with a message and a turn (cascade); young
	// ended conversation; old but still-open conversation (never pruned).
	s.SetClock(func() time.Time { return old })
	cOld, _ := s.CreateConversation(ctx, "site-old", "")
	_, _ = s.AppendMessage(ctx, cOld.ID, "user", json.RawMessage(`[{"type":"text","text":"old"}]`))
	tOld, _ := s.CreateTurn(ctx, store.NewTurn{ConversationID: cOld.ID, ManifestHash: hashA})
	_ = s.FinishTurn(ctx, tOld.ID, store.TurnDone, nil)
	_, _ = s.EndConversation(ctx, "site-old")
	cOpen, _ := s.CreateConversation(ctx, "site-open", "")
	// Still-open conversation whose last activity (updated) is 91 days old:
	// inactivity ends it and prunes it in the same run, messages included.
	cIdle, _ := s.CreateConversation(ctx, "site-idle", "")
	_, _ = s.AppendMessage(ctx, cIdle.ID, "user", json.RawMessage(`[{"type":"text","text":"merchant text that must not live forever"}]`))
	// cOpen was created old but is touched recently (updated moves): kept.
	s.SetClock(func() time.Time { return young })
	_, _ = s.AppendMessage(ctx, cOpen.ID, "user", json.RawMessage(`[{"type":"text","text":"recent"}]`))
	cYoung, _ := s.CreateConversation(ctx, "site-young", "")
	_, _ = s.EndConversation(ctx, "site-young")

	// Traces: one old, one young.
	if err := s.InsertTrace(ctx, store.Trace{TurnID: tOld.ID, SiteID: "site-old", Created: old}); err != nil {
		t.Fatal(err)
	}
	if err := s.InsertTrace(ctx, store.Trace{TurnID: "33333333-3333-3333-3333-333333333333", SiteID: "site-young", Created: young}); err != nil {
		t.Fatal(err)
	}
	// Usage: old shipped (pruned), old unshipped (kept), young shipped (kept).
	idOldShipped, _ := s.InsertUsage(ctx, store.UsageRow{Model: "m", OccurredAt: usageOld})
	_, _ = s.MarkUsageShipped(ctx, []int64{idOldShipped})
	_, _ = s.InsertUsage(ctx, store.UsageRow{Model: "m", OccurredAt: usageOld})
	idYoungShipped, _ := s.InsertUsage(ctx, store.UsageRow{Model: "m", OccurredAt: now.AddDate(0, 0, -1)})
	_, _ = s.MarkUsageShipped(ctx, []int64{idYoungShipped})

	s.SetClock(func() time.Time { return now })
	res, err := s.Retention(ctx, store.DefaultRetention)
	if err != nil {
		t.Fatal(err)
	}
	if res.Traces != 1 || res.Ended != 1 || res.Conversations != 2 || res.UsageRows != 1 {
		t.Fatalf("Retention = %+v; want traces=1 ended=1 conversations=2 usage=1", res)
	}
	if _, err := s.GetConversation(ctx, cIdle.ID); !errors.Is(err, store.ErrNotFound) {
		t.Errorf("idle open conversation survived: %v", err)
	}
	if n, _ := s.CountMessages(ctx, cIdle.ID); n != 0 {
		t.Errorf("messages of idle conversation survived: %d", n)
	}
	if _, err := s.FindOpenConversation(ctx, "site-idle"); !errors.Is(err, store.ErrNotFound) {
		t.Errorf("site-idle still has an open conversation: %v", err)
	}
	if _, err := s.GetConversation(ctx, cOld.ID); !errors.Is(err, store.ErrNotFound) {
		t.Errorf("old ended conversation survived: %v", err)
	}
	if _, err := s.GetTurn(ctx, tOld.ID); !errors.Is(err, store.ErrNotFound) {
		t.Errorf("turn of pruned conversation survived (cascade): %v", err)
	}
	if n, _ := s.CountMessages(ctx, cOld.ID); n != 0 {
		t.Errorf("messages of pruned conversation survived: %d", n)
	}
	if _, err := s.GetConversation(ctx, cOpen.ID); err != nil {
		t.Errorf("open conversation pruned: %v", err)
	}
	if _, err := s.GetConversation(ctx, cYoung.ID); err != nil {
		t.Errorf("young ended conversation pruned: %v", err)
	}
	if _, err := s.GetTrace(ctx, tOld.ID); !errors.Is(err, store.ErrNotFound) {
		t.Errorf("old trace survived: %v", err)
	}
	if _, err := s.GetTrace(ctx, "33333333-3333-3333-3333-333333333333"); err != nil {
		t.Errorf("young trace pruned: %v", err)
	}
	unshipped, _ := s.ListUnshippedUsage(ctx, 10)
	if len(unshipped) != 1 {
		t.Errorf("unshipped old usage row must survive; have %d", len(unshipped))
	}
	// A second run removes nothing.
	res, err = s.Retention(ctx, store.DefaultRetention)
	if err != nil || res.Traces+res.Ended+res.Conversations+res.UsageRows != 0 {
		t.Errorf("second run = %+v, %v", res, err)
	}
	// The boundary is strict: a conversation idle exactly InactivityDays is
	// kept; one second more and it goes.
	s.SetClock(func() time.Time { return now.AddDate(0, 0, -90) })
	cEdge, _ := s.CreateConversation(ctx, "site-edge", "")
	s.SetClock(func() time.Time { return now })
	if res, err = s.Retention(ctx, store.DefaultRetention); err != nil || res.Ended != 0 || res.Conversations != 0 {
		t.Errorf("exactly-90d conversation pruned: %+v, %v", res, err)
	}
	s.SetClock(func() time.Time { return now.Add(time.Second) })
	if res, err = s.Retention(ctx, store.DefaultRetention); err != nil || res.Ended != 1 || res.Conversations != 1 {
		t.Errorf("90d+1s conversation kept: %+v, %v", res, err)
	}
	if _, err := s.GetConversation(ctx, cEdge.ID); !errors.Is(err, store.ErrNotFound) {
		t.Errorf("edge conversation survived: %v", err)
	}
	if _, err := s.Retention(ctx, store.RetentionPolicy{}); err == nil {
		t.Error("zero policy accepted")
	}
}

func keys(m map[string]any) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
