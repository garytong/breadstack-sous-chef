package e2e

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/breadstack/sous-chef/internal/api"
	"github.com/breadstack/sous-chef/internal/config"
	"github.com/breadstack/sous-chef/internal/loop"
	"github.com/breadstack/sous-chef/internal/manifest"
	"github.com/breadstack/sous-chef/internal/model"
	"github.com/breadstack/sous-chef/internal/model/modeltest"
	"github.com/breadstack/sous-chef/internal/registry"
	"github.com/breadstack/sous-chef/internal/store"
	"github.com/breadstack/sous-chef/internal/store/storetest"
)

// Round-7 regressions (refuter E3, E4, E6, E7, E8): two service instances —
// two *store.Store (own clocks) and two loop.Runner — on ONE Postgres. The
// Start's storage block is serialized per site by the database
// (store.StartTurn), so racing Starts on two instances never leave two live
// turns, clock skew between the instances does not shield an older turn, and
// two Starts at the rotation boundary never collide on the open-conversation
// index. Skipped without a database (storetest).

// instance is one service instance: its own *store.Store and Runner.
type instance struct {
	st     *store.Store
	runner *loop.Runner
	logMu  sync.Mutex
	logs   []string
}

func (in *instance) logged(s string) bool {
	in.logMu.Lock()
	defer in.logMu.Unlock()
	for _, l := range in.logs {
		if strings.Contains(l, s) {
			return true
		}
	}
	return false
}

type fleet struct {
	t    *testing.T
	fake *modeltest.Server
	hash string
	a, b *instance
	in   []*instance // every instance; a and b are in[0] and in[1]
}

// newFleet builds two instances on a fresh database; skewB shifts instance
// B's clock (turns.started is the instance's clock).
func newFleet(t *testing.T, skewB time.Duration) *fleet {
	t.Helper()
	return newFleetN(t, 2, 0, skewB)
}

// newFleetN builds n instances on a fresh database; skews[i], when given,
// shifts instance i's clock.
func newFleetN(t *testing.T, n int, skews ...time.Duration) *fleet {
	t.Helper()
	dsn := storetest.OpenDSN(t)
	ctx := context.Background()
	body, err := os.ReadFile("testdata/manifest.snapshot.json")
	if err != nil {
		t.Fatal(err)
	}
	hash, _ := manifest.HashBytes(body)
	fake := modeltest.New()
	t.Cleanup(fake.Close)
	envv := map[string]string{
		"DATABASE_URL": "unused", "SERVICE_KEYS": serviceKey, "ANTHROPIC_API_KEY": "sk-e2e",
		"ANTHROPIC_BASE_URL": fake.URL, "MODEL_CALL_TIMEOUT": "5s", "MAX_CONCURRENT_TURNS": "8",
	}
	getenv := func(k string) string { return envv[k] }
	cfg, err := config.Load(getenv)
	if err != nil {
		t.Fatal(err)
	}
	opts := model.FromConfig(cfg, getenv)
	retries := 0
	opts.MaxRetries = &retries
	models, err := model.NewTable(opts)
	if err != nil {
		t.Fatal(err)
	}
	mk := func(skew time.Duration, first bool) *instance {
		st, err := store.Open(ctx, dsn)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(st.Close)
		if skew != 0 {
			st.SetClock(func() time.Time { return time.Now().UTC().Add(skew) })
		}
		reg := registry.New(st)
		if first {
			if _, _, err := reg.Register(ctx, hash, body, "v3.0.1"); err != nil {
				t.Fatal(err)
			}
		}
		in := &instance{st: st}
		r, err := loop.New(loop.Deps{Store: st, Manifests: reg, Models: models, TurnTTL: cfg.TurnTTL, MaxConcurrent: 8, RotateAt: 1000, Logf: func(f string, a ...any) {
			in.logMu.Lock()
			in.logs = append(in.logs, fmt.Sprintf(f, a...))
			in.logMu.Unlock()
			t.Logf(f, a...)
		}})
		if err != nil {
			t.Fatal(err)
		}
		in.runner = r
		return in
	}
	f := &fleet{t: t, fake: fake, hash: hash}
	for i := 0; i < n; i++ {
		var skew time.Duration
		if i < len(skews) {
			skew = skews[i]
		}
		f.in = append(f.in, mk(skew, i == 0))
	}
	f.a, f.b = f.in[0], f.in[1]
	return f
}

// warned reports whether any instance logged a history-repair WARNING.
func (f *fleet) warned() bool {
	for _, in := range f.in {
		if in.logged("WARNING") {
			return true
		}
	}
	return false
}

// clearLogs drops every instance's log lines (per-round WARNING checks).
func (f *fleet) clearLogs() {
	for _, in := range f.in {
		in.logMu.Lock()
		in.logs = nil
		in.logMu.Unlock()
	}
}

func evData(ev api.CollectedEvent) map[string]any {
	var m map[string]any
	_ = json.Unmarshal(ev.Data, &m)
	return m
}

func (f *fleet) req(text string) api.TurnRequest {
	return api.TurnRequest{SiteID: "site-fleet", StoreID: "store-fleet", ManifestHash: f.hash, BuildID: "abc123",
		Context: []byte(`{"page":"home","sections":[{"id":"hero-1","type":"hero"}]}`), Text: text, UserEmail: "owner@shop.com"}
}

func (f *fleet) run(tr api.Turn) []api.CollectedEvent {
	col := &api.Collector{}
	_ = tr.Run(context.Background(), col)
	return col.Events()
}

// racingStarts starts one turn on each instance concurrently.
func (f *fleet) racingStarts(ctx context.Context, iter int) ([]api.Turn, []error) {
	var wg sync.WaitGroup
	turns := make([]api.Turn, 2)
	errs := make([]error, 2)
	inst := []*instance{f.a, f.b}
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			turns[i], errs[i] = inst[i].runner.Start(ctx, f.req(fmt.Sprintf("iter %d text %d", iter, i)))
		}(i)
	}
	wg.Wait()
	return turns, errs
}

func ekinds(evs []api.CollectedEvent) string {
	var p []string
	for _, e := range evs {
		p = append(p, e.Event)
	}
	return strings.Join(p, " ")
}

func rolesOf(t *testing.T, st *store.Store, conv string) string {
	t.Helper()
	msgs, err := st.ListMessages(context.Background(), conv, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	var rl []string
	for _, m := range msgs {
		rl = append(rl, m.Role)
	}
	return strings.Join(rl, " ")
}

func opStep(id, headline string) modeltest.Step {
	return modeltest.Reply(modeltest.Message{Content: []modeltest.Block{
		modeltest.ToolUse(id, "set_fields", map[string]any{"target": heroTarget, "values": map[string]any{"headline": headline}}),
	}})
}

func textStep(s string) modeltest.Step {
	return modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text(s)}})
}

// e2eRounds is the number of rounds the racing tests run; raised with
// E2E_RACE_ROUNDS for a longer soak.
func e2eRounds(def int) int {
	if v := os.Getenv("E2E_RACE_ROUNDS"); v != "" {
		var n int
		if _, err := fmt.Sscan(v, &n); err == nil && n > 0 {
			return n
		}
	}
	return def
}

func fleetResults(id string) api.ToolResultsRequest {
	return api.ToolResultsRequest{SiteID: "site-fleet", StoreID: "store-fleet", Results: []api.ToolResult{{ToolCallID: id, OK: true}}}
}

// E3: instance B's clock is 2s behind. A Start on A goes waiting_tools; a
// Start on B 100ms later gets an EARLIER started and must still supersede
// the waiting turn: the supersede is by commit order, not by started.
func TestFleet_SkewedInstanceSupersedesOlderTurn(t *testing.T) {
	f := newFleet(t, -2*time.Second)
	ctx := context.Background()
	f.fake.Append(opStep("opA", "a"), textStep("B reply."), textStep("C reply."))
	tA, err := f.a.runner.Start(ctx, f.req("A text"))
	if err != nil {
		t.Fatal(err)
	}
	if k := ekinds(f.run(tA)); k != "tool.call turn.waiting" {
		t.Fatalf("A = %s", k)
	}
	rowA, _ := f.a.st.GetTurn(ctx, tA.ID())
	time.Sleep(100 * time.Millisecond)
	tB, err := f.b.runner.Start(ctx, f.req("B text"))
	if err != nil {
		t.Fatal(err)
	}
	rowA2, _ := f.a.st.GetTurn(ctx, tA.ID())
	rowB, _ := f.a.st.GetTurn(ctx, tB.ID())
	if !rowB.Started.Before(rowA.Started) {
		t.Fatalf("skew not applied: A.started=%v B.started=%v", rowA.Started, rowB.Started)
	}
	if rowA2.Status != store.TurnSuperseded {
		t.Errorf("the waiting turn on instance A was not superseded by the Start on the slow-clock instance B (rowA=%s)", rowA2.Status)
	}
	if f.b.logged("WARNING") {
		t.Error("B's Start repaired A's live tail as an unexplained dangle (WARNING)")
	}
	if k := ekinds(f.run(tB)); !strings.HasSuffix(k, "turn.done") {
		t.Errorf("B = %s", k)
	}
	_, err = f.a.runner.Continue(ctx, tA.ID(), fleetResults("opA"))
	var apiErr *api.Error
	if !errors.As(err, &apiErr) || apiErr.Status != 409 || apiErr.Extra["status"] != store.TurnSuperseded {
		t.Errorf("A's late results → %v (launcher drops only 409 status=superseded)", err)
	}
	if live, _ := f.a.st.CountLiveTurns(ctx); live != 0 {
		t.Errorf("live turns = %d", live)
	}
	if got := rolesOf(t, f.a.st, rowA.ConversationID); got != "user assistant user user assistant" {
		t.Errorf("roles = %q, want A's batch answered by B's Start then B's exchange", got)
	}
	reqs := f.fake.Requests()
	assertReplayable(t, reqs[len(reqs)-1].Body["messages"].([]any))
}

// E4: two instances racing Starts for one site, repeatedly, with the
// survivor driven to completion each round: exactly one live turn after
// both return; the loser ends turn.error (superseded) without a model call;
// the survivor waits, is continued on the OTHER instance (rebuilt from
// storage) and finishes; no WARNING; nothing leaks.
func TestFleet_RacingStartsSurvivorCompletes(t *testing.T) {
	f := newFleet(t, 0)
	ctx := context.Background()
	rounds := e2eRounds(12)
	inst := []*instance{f.a, f.b}
	for iter := 0; iter < rounds; iter++ {
		op := fmt.Sprintf("op%d", iter)
		f.fake.Append(opStep(op, "one"), textStep("after "+op+"."))
		before := len(f.fake.Requests())
		turns, errs := f.racingStarts(ctx, iter)
		if errs[0] != nil || errs[1] != nil {
			t.Fatalf("iter %d: %v", iter, errs)
		}
		r0, _ := f.a.st.GetTurn(ctx, turns[0].ID())
		r1, _ := f.a.st.GetTurn(ctx, turns[1].ID())
		if live, _ := f.a.st.CountLiveTurns(ctx); live != 1 {
			t.Fatalf("iter %d: live=%d (row0=%s started=%v, row1=%s started=%v)", iter, live, r0.Status, r0.Started, r1.Status, r1.Started)
		}
		var surv, loser int
		if r0.Status == store.TurnRunning {
			surv, loser = 0, 1
		} else {
			surv, loser = 1, 0
		}
		if rl, _ := f.a.st.GetTurn(ctx, turns[loser].ID()); rl.Status != store.TurnSuperseded {
			t.Errorf("iter %d: loser=%s", iter, rl.Status)
		}
		if k := ekinds(f.run(turns[loser])); k != "turn.error" {
			t.Errorf("iter %d: loser events = %s", iter, k)
		}
		if n := len(f.fake.Requests()) - before; n != 0 {
			t.Errorf("iter %d: loser made %d model call(s)", iter, n)
		}
		if k := ekinds(f.run(turns[surv])); k != "tool.call turn.waiting" {
			t.Errorf("iter %d: survivor events = %s", iter, k)
		}
		// Continue on the OTHER instance (rebuild from storage).
		tc, err := inst[1-surv].runner.Continue(ctx, turns[surv].ID(), fleetResults(op))
		if err != nil {
			t.Fatalf("iter %d: survivor continue on the other instance = %v", iter, err)
		}
		if k := ekinds(f.run(tc)); !strings.HasSuffix(k, "turn.done") {
			t.Errorf("iter %d: survivor resume = %s", iter, k)
		}
		if f.a.logged("WARNING") || f.b.logged("WARNING") {
			t.Fatalf("iter %d: WARNING logged", iter)
		}
		if live, _ := f.a.st.CountLiveTurns(ctx); live != 0 {
			t.Fatalf("iter %d: live=%d at end", iter, live)
		}
		// Reap on both so the survivor's stale parked entry on its first
		// instance is evicted.
		_, _ = f.a.runner.Reap(ctx)
		_, _ = f.b.runner.Reap(ctx)
		if f.a.runner.Parked()+f.b.runner.Parked() != 0 || f.a.runner.Live()+f.b.runner.Live() != 0 {
			t.Fatalf("iter %d: parked=%d/%d live=%d/%d", iter, f.a.runner.Parked(), f.b.runner.Parked(), f.a.runner.Live(), f.b.runner.Live())
		}
	}
	reqs := f.fake.Requests()
	assertReplayable(t, reqs[len(reqs)-1].Body["messages"].([]any))
}

// E6: the both-live rate. Two instances Start for one site at the same
// moment, N rounds, no model script (a turn that reaches the model fails
// loudly): never two live rows, never none. Before round 7 this was 8/60
// (started taken Go-side before the INSERT; the supersede UPDATE saw only
// committed rows).
func TestFleet_RacingStartsNeverBothLive(t *testing.T) {
	f := newFleet(t, 0)
	ctx := context.Background()
	rounds := e2eRounds(60)
	bothLive, zeroLive := 0, 0
	for iter := 0; iter < rounds; iter++ {
		turns, errs := f.racingStarts(ctx, iter)
		if errs[0] != nil || errs[1] != nil {
			t.Fatalf("iter %d: %v", iter, errs)
		}
		live, _ := f.a.st.CountLiveTurns(ctx)
		r0, _ := f.a.st.GetTurn(ctx, turns[0].ID())
		r1, _ := f.a.st.GetTurn(ctx, turns[1].ID())
		switch live {
		case 2:
			bothLive++
			t.Errorf("iter %d: BOTH LIVE: row0 started=%v id=%s; row1 started=%v id=%s (gap %v)", iter, r0.Started.Format("15:04:05.000000"), r0.ID[:8], r1.Started.Format("15:04:05.000000"), r1.ID[:8], r1.Started.Sub(r0.Started))
		case 0:
			zeroLive++
			t.Errorf("iter %d: no live turn (row0=%s row1=%s)", iter, r0.Status, r1.Status)
		}
		if _, err := f.a.st.SupersedeTurnsForSite(ctx, "site-fleet"); err != nil {
			t.Fatal(err)
		}
		before := len(f.fake.Requests())
		for _, tr := range turns {
			f.run(tr)
		}
		if n := len(f.fake.Requests()) - before; n != 0 {
			t.Fatalf("iter %d: %d model call(s) by superseded turns", iter, n)
		}
	}
	t.Logf("rounds=%d bothLive=%d zeroLive=%d", rounds, bothLive, zeroLive)
}

// E7: two Starts racing at the rotation boundary (both see RotateAt
// reached): the rotation runs inside the serialized Start block, so neither
// fails on conversations_open_site, exactly one rotates, and both land on
// the fresh conversation with exactly one live turn.
func TestFleet_ConcurrentRotation(t *testing.T) {
	f := newFleet(t, 0)
	ctx := context.Background()
	rounds := e2eRounds(15)
	const rotateAt = 1000 // the fleet runners' RotateAt
	for iter := 0; iter < rounds; iter++ {
		open, err := f.a.st.OpenConversation(ctx, "site-fleet", "store-fleet")
		if err != nil {
			t.Fatal(err)
		}
		n, _ := f.a.st.CountMessages(ctx, open.ID)
		for i := n; i < rotateAt; i++ {
			role := "user"
			if i%2 == 1 {
				role = "assistant"
			}
			if _, err := f.a.st.AppendMessage(ctx, open.ID, role, []byte(`[{"type":"text","text":"m"}]`)); err != nil {
				t.Fatal(err)
			}
		}
		turns, errs := f.racingStarts(ctx, iter)
		if errs[0] != nil || errs[1] != nil {
			t.Fatalf("iter %d: a Start failed at the rotation boundary: %v", iter, errs)
		}
		r0, _ := f.a.st.GetTurn(ctx, turns[0].ID())
		r1, _ := f.a.st.GetTurn(ctx, turns[1].ID())
		if r0.ConversationID == open.ID || r1.ConversationID == open.ID || r0.ConversationID != r1.ConversationID {
			t.Errorf("iter %d: conversations %s / %s (full one %s)", iter, r0.ConversationID, r1.ConversationID, open.ID)
		}
		if c, _ := f.a.st.GetConversation(ctx, open.ID); c.EndedAt == nil {
			t.Errorf("iter %d: the full conversation was not ended", iter)
		}
		if live, _ := f.a.st.CountLiveTurns(ctx); live != 1 {
			t.Errorf("iter %d: live=%d", iter, live)
		}
		if _, err := f.a.st.SupersedeTurnsForSite(ctx, "site-fleet"); err != nil {
			t.Fatal(err)
		}
		for _, tr := range turns {
			f.run(tr)
		}
	}
}

// E8 (I7 over HTTP + Postgres): A is in the model on conversation C1 when
// B's Start rotates to C2 (RotateAt=1) and supersedes A. A's late batch's
// compare-and-set runs against C1, whose tail is still A's own text; the
// ended C1 refuses the write: nothing is written into C1, no tool.call is
// streamed for the superseded batch, A ends turn.error turn_expired
// (superseded), its results are 409 superseded, C replays with no WARNING.
func TestEndToEnd_LateBatchOnRotatedConversationNotStored(t *testing.T) {
	e := newEnvWith(t, func(d *loop.Deps) { d.RotateAt = 1 })
	ctx := context.Background()
	if _, _, err := e.manifes.Register(ctx, e.hash, e.body, "v3.0.1"); err != nil {
		t.Fatal(err)
	}
	e.fake.Append(
		modeltest.Step{Delay: 1500 * time.Millisecond, Message: &modeltest.Message{Content: []modeltest.Block{
			modeltest.ToolUse("opA", "set_fields", map[string]any{"target": heroTarget, "values": map[string]any{"headline": "a"}}),
		}}},
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("B reply.")}}),
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("C reply.")}}),
	)
	seen := len(e.fake.Requests())
	type reply struct {
		status int
		body   []byte
	}
	aCh := make(chan reply, 1)
	go func() {
		s, b := e.post("/v1/turns", e.turnBody("A text", ""))
		aCh <- reply{s, b}
	}()
	deadline := time.Now().Add(5 * time.Second)
	for len(e.fake.Requests()) < seen+1 {
		if time.Now().After(deadline) {
			t.Fatal("A never reached the model")
		}
		time.Sleep(10 * time.Millisecond)
	}
	status, body, _ := e.do("POST", "/v1/turns", e.turnBody("B text", ""))
	evsB := parseSSE(t, body)
	if status != 200 || !strings.HasSuffix(kinds(evsB), "turn.done") || evsB[len(evsB)-1].Data["conversationRotated"] != true {
		t.Fatalf("B → %d %s %v", status, kinds(evsB), evsB[len(evsB)-1].Data)
	}
	turnB := evsB[0].Data["turnId"].(string)
	a := <-aCh
	if a.status != 200 {
		t.Fatalf("A → %d %s", a.status, a.body)
	}
	evsA := parseSSE(t, a.body)
	turnA := evsA[0].Data["turnId"].(string)
	rowA, _ := e.st.GetTurn(ctx, turnA)
	rowB, _ := e.st.GetTurn(ctx, turnB)
	c1, _ := e.st.GetConversation(ctx, rowA.ConversationID)
	if rowA.ConversationID == rowB.ConversationID || c1.EndedAt == nil {
		t.Fatal("no rotation")
	}
	lastA := evsA[len(evsA)-1]
	if kinds(evsA) != "turn.error" || lastA.Data["code"] != "turn_expired" || !strings.Contains(lastA.Data["message"].(string), "superseded") {
		t.Errorf("A → %s %v; want turn.error turn_expired (superseded) and no tool.call for a turn the user moved on from", kinds(evsA), lastA.Data)
	}
	if got := rolesOf(t, e.st, rowA.ConversationID); got != "user" {
		t.Errorf("the superseded turn wrote into the ended conversation C1: %q", got)
	}
	if got := rolesOf(t, e.st, rowB.ConversationID); got != "user assistant" {
		t.Errorf("C2 roles = %q", got)
	}
	if rowA.Status != store.TurnSuperseded || rowB.Status != store.TurnDone {
		t.Errorf("rowA=%s rowB=%s", rowA.Status, rowB.Status)
	}
	status, body, _ = e.do("POST", "/v1/turns/"+turnA+"/tool-results", []byte(`{"siteId":"site-e2e","storeId":"store-e2e","results":[{"toolCallId":"opA","ok":true}]}`))
	if status != 409 || !strings.Contains(string(body), `"status":"superseded"`) {
		t.Errorf("A's late results → %d %s", status, body)
	}
	mark := e.logCount()
	status, body, _ = e.do("POST", "/v1/turns", e.turnBody("C text", ""))
	if status != 200 || !strings.HasSuffix(kinds(parseSSE(t, body)), "turn.done") || e.logged(mark, "WARNING") {
		t.Errorf("C → %d %s warned=%v", status, kinds(parseSSE(t, body)), e.logged(mark, "WARNING"))
	}
	reqs := e.fake.Requests()
	assertReplayable(t, reqs[len(reqs)-1].Body["messages"].([]any))
	if e.runner.Live() != 0 || e.runner.Parked() != 0 {
		t.Fatalf("leaked state: live=%d parked=%d", e.runner.Live(), e.runner.Parked())
	}
}

// Round-8 regression (refuter F3 / defect D1): a Start racing a DELETE
// (loop.Conversations.End on the other instance) with a WAITING turn — a
// dangling a(opW) tail — on the conversation, every round. The Start either
// lands on a fresh conversation (the DELETE committed first) and runs to
// turn.done, or is superseded by the DELETE (its StartTurn committed first,
// so the dangling tail was its to repair — into a conversation the DELETE
// has ended by then) and ends turn.error turn_expired naming the supersede
// with no model call. It never fails: a failed Start is a 500 to the
// launcher. E2E_RACE_ROUNDS raises the rounds for a soak.
func TestFleet_StartRacesDeleteWithWaitingTurn(t *testing.T) {
	f := newFleetN(t, 2)
	ctx := context.Background()
	convs := loop.Conversations{St: f.b.st, Logf: func(format string, a ...any) { t.Logf("[del] "+format, a...) }}
	rounds := e2eRounds(40)
	fresh, superseded, startErrors := 0, 0, 0
	for iter := 0; iter < rounds; iter++ {
		f.fake.Append(opStep("opW", "w"), textStep("S reply."))
		tW, err := f.a.runner.Start(ctx, f.req("W text"))
		if err != nil {
			t.Fatalf("iter %d: W: %v", iter, err)
		}
		if k := ekinds(f.run(tW)); k != "tool.call turn.waiting" {
			t.Fatalf("iter %d: W = %s", iter, k)
		}
		rowW, _ := f.a.st.GetTurn(ctx, tW.ID())
		var wg sync.WaitGroup
		var tS api.Turn
		var errS, errE error
		wg.Add(2)
		go func() { defer wg.Done(); tS, errS = f.a.runner.Start(ctx, f.req("S text")) }()
		go func() { defer wg.Done(); errE = convs.End(ctx, "site-fleet") }()
		wg.Wait()
		if errE != nil {
			t.Fatalf("iter %d: End: %v", iter, errE)
		}
		if errS != nil {
			startErrors++
			t.Errorf("iter %d: Start racing the DELETE failed (a 500 to the launcher): %v", iter, errS)
			// The fake still holds "S reply."; consume it and reset the site.
			if tX, xerr := f.a.runner.Start(ctx, f.req("X text")); xerr == nil {
				_ = f.run(tX)
			}
			_, _ = f.a.st.SupersedeTurnsForSite(ctx, "site-fleet")
			_, _ = f.a.st.EndConversation(ctx, "site-fleet")
			_, _ = f.a.runner.Reap(ctx)
			continue
		}
		rowS, _ := f.a.st.GetTurn(ctx, tS.ID())
		before := len(f.fake.Requests())
		evs := f.run(tS)
		if rowS.ConversationID != rowW.ConversationID {
			fresh++
			if !strings.HasSuffix(ekinds(evs), "turn.done") {
				t.Errorf("iter %d: S on a fresh conversation = %s", iter, ekinds(evs))
			}
		} else {
			superseded++
			if ekinds(evs) != "turn.error" || len(f.fake.Requests())-before != 0 {
				t.Errorf("iter %d: superseded S = %s, model calls = %d", iter, ekinds(evs), len(f.fake.Requests())-before)
			} else if d := evData(evs[0]); d["code"] != "turn_expired" || !strings.Contains(fmt.Sprint(d["message"]), "superseded") {
				t.Errorf("iter %d: superseded S error = %v", iter, d)
			}
			// The ended conversation holds W's text and batch, plus whatever
			// S stored before the DELETE landed (its repair of W's batch, its
			// own text — a DELETE after the row check is caught by Run's
			// first row check instead); never a reply of S's.
			if r := rolesOf(t, f.a.st, rowW.ConversationID); !strings.HasPrefix(r, "user assistant") || strings.Count(r, "assistant") != 1 {
				t.Errorf("iter %d: the ended conversation holds %q", iter, r)
			}
			// The fake still holds "S reply."; a plain Start consumes it.
			tX, err := f.a.runner.Start(ctx, f.req("X text"))
			if err != nil {
				t.Fatalf("iter %d: X: %v", iter, err)
			}
			if k := ekinds(f.run(tX)); !strings.HasSuffix(k, "turn.done") {
				t.Errorf("iter %d: X = %s", iter, k)
			}
		}
		if _, err := f.a.st.SupersedeTurnsForSite(ctx, "site-fleet"); err != nil {
			t.Fatal(err)
		}
		_, _ = f.a.st.EndConversation(ctx, "site-fleet")
		_, _ = f.a.runner.Reap(ctx)
	}
	t.Logf("rounds=%d freshConversation=%d supersededByDelete=%d startErrors=%d warned=%v", rounds, fresh, superseded, startErrors, f.warned())
	if f.warned() {
		t.Error("WARNING logged")
	}
	for i, in := range f.in {
		if in.runner.Live() != 0 || in.runner.Parked() != 0 {
			t.Errorf("instance %d: live=%d parked=%d", i, in.runner.Live(), in.runner.Parked())
		}
	}
}

// Round-8 regression (refuter F4 / defect D2): a Start racing the TTL
// reaper on another instance. A waiting turn W, 11 minutes old (started by
// an instance whose clock is behind), is claimed by the Start's supersede or
// by the other instance's Reap; either way the Start's turn completes, the
// history replays with no dangling tool_use, and NO WARNING is logged: when
// the reaper claims W first but its closing append has not landed when the
// Start reads the history, the dangling tail is the reaper's (W's row is
// expired), repaired with the reaper's reason and logged as information.
func TestFleet_StartRacesReaperNoWarning(t *testing.T) {
	f := newFleetN(t, 3, 0, 0, -11*time.Minute)
	ctx := context.Background()
	rounds := e2eRounds(40)
	bySupersede, byReaper, warnRounds := 0, 0, 0
	for iter := 0; iter < rounds; iter++ {
		f.fake.Append(opStep("opW", "w"), textStep("S reply."))
		tW, err := f.in[2].runner.Start(ctx, f.req("W text"))
		if err != nil {
			t.Fatalf("iter %d: W: %v", iter, err)
		}
		if k := ekinds(f.run(tW)); k != "tool.call turn.waiting" {
			t.Fatalf("iter %d: W = %s", iter, k)
		}
		f.clearLogs()
		var wg sync.WaitGroup
		var tS api.Turn
		var errS, errR error
		wg.Add(2)
		go func() { defer wg.Done(); tS, errS = f.in[0].runner.Start(ctx, f.req("S text")) }()
		go func() { defer wg.Done(); _, errR = f.in[1].runner.Reap(ctx) }()
		wg.Wait()
		if errS != nil || errR != nil {
			t.Fatalf("iter %d: start=%v reap=%v", iter, errS, errR)
		}
		evs := f.run(tS)
		if !strings.HasSuffix(ekinds(evs), "turn.done") {
			t.Errorf("iter %d: S = %s", iter, ekinds(evs))
		}
		rowW, _ := f.in[0].st.GetTurn(ctx, tW.ID())
		rowS, _ := f.in[0].st.GetTurn(ctx, tS.ID())
		switch rowW.Status {
		case store.TurnSuperseded:
			bySupersede++
		case store.TurnExpired:
			byReaper++
		default:
			t.Errorf("iter %d: W = %s", iter, rowW.Status)
		}
		if f.warned() {
			warnRounds++
			t.Errorf("iter %d: WARNING logged (W=%s)", iter, rowW.Status)
		}
		if r := rolesOf(t, f.in[0].st, rowS.ConversationID); r != "user assistant user user assistant" {
			t.Errorf("iter %d: roles = %q", iter, r)
		}
		_, _ = f.in[0].st.SupersedeTurnsForSite(ctx, "site-fleet")
		_, _ = f.in[0].st.EndConversation(ctx, "site-fleet")
		for _, in := range f.in {
			_, _ = in.runner.Reap(ctx)
		}
	}
	t.Logf("rounds=%d claimedBySupersede=%d claimedByReaper=%d roundsWithWARNING=%d", rounds, bySupersede, byReaper, warnRounds)
	for _, r := range f.fake.Requests() {
		assertReplayable(t, r.Body["messages"].([]any))
	}
}
