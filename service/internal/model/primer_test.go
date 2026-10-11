package model

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/breadstack/sous-chef/internal/config"
	"github.com/breadstack/sous-chef/internal/model/modeltest"
)

// primeReply is the model server's answer to a prime: one token, the
// re-read and restored counts as llama-server reports them.
func primeReply(reread, cached int64) modeltest.Step {
	return modeltest.Step{Message: &modeltest.Message{
		Model: "sous-chef", StopReason: "max_tokens", Content: []modeltest.Block{modeltest.Text("x")},
		Usage: modeltest.Usage{InputTokens: reread, OutputTokens: 1, CacheReadInputTokens: cached},
	}}
}

func primeTable(t *testing.T, fake *modeltest.Server, mutate func(*config.Route)) *Backend {
	t.Helper()
	temp := 0.3
	r := config.Route{Kind: config.RouteMessagesCompatible, URL: fake.URL, Auth: config.AuthNone, Model: "sous-chef", MaxTokens: 1024, Temperature: &temp, Prime: config.PrimeOn}
	if mutate != nil {
		mutate(&r)
	}
	tbl, err := NewTable(Options{Routes: map[string]config.Route{"gpu": r}, Default: "gpu", Allowlist: []string{"gpu"}, CallTimeout: 5 * time.Second})
	if err != nil {
		t.Fatalf("NewTable: %v", err)
	}
	b, _ := tbl.Route("")
	return b
}

// primeHarness is a Primer on a fake model server with a settable clock
// and busy flag, and a log it can be asserted on.
type primeHarness struct {
	t      *testing.T
	fake   *modeltest.Server
	b      *Backend
	p      *Primer
	mu     sync.Mutex
	now    time.Time
	busy   bool
	logs   []string
	builds int
}

func newPrimeHarness(t *testing.T, steps ...modeltest.Step) *primeHarness {
	t.Helper()
	h := &primeHarness{t: t, now: time.Date(2026, 10, 10, 20, 0, 0, 0, time.UTC)}
	h.fake = modeltest.New(steps...)
	t.Cleanup(h.fake.Close)
	h.b = primeTable(t, h.fake, nil)
	h.p = NewPrimer(PrimerOptions{
		MaxDefer: 300 * time.Millisecond, PollInterval: 5 * time.Millisecond,
		Busy: func() bool { h.mu.Lock(); defer h.mu.Unlock(); return h.busy },
		Now:  func() time.Time { h.mu.Lock(); defer h.mu.Unlock(); return h.now },
		Logf: func(f string, a ...any) {
			h.mu.Lock()
			h.logs = append(h.logs, fmt.Sprintf(f, a...))
			h.mu.Unlock()
			t.Logf(f, a...)
		},
	})
	return h
}

func (h *primeHarness) advance(d time.Duration) { h.mu.Lock(); h.now = h.now.Add(d); h.mu.Unlock() }
func (h *primeHarness) setBusy(b bool)          { h.mu.Lock(); h.busy = b; h.mu.Unlock() }

func (h *primeHarness) logged(sub string) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, l := range h.logs {
		if strings.Contains(l, sub) {
			return true
		}
	}
	return false
}

// job primes prefix; with a tail key, the tail request too. The requests'
// single user message names which one it is.
func (h *primeHarness) job(prefix, tailKey string) PrimeJob {
	return PrimeJob{Backend: h.b, Prefix: prefix, Reason: "test", Site: "site-1",
		Build: func(context.Context) (PrimeRequests, error) {
			h.mu.Lock()
			h.builds++
			h.mu.Unlock()
			req := sampleRequest()
			req.Messages = []anthropic.BetaMessageParam{UserTextTurn("FLOOR " + prefix)}
			out := PrimeRequests{Floor: req}
			if tailKey != "" {
				tail := sampleRequest()
				tail.Messages = []anthropic.BetaMessageParam{UserTextTurn("TAIL " + tailKey)}
				out.Tail, out.TailKey = &tail, tailKey
			}
			return out, nil
		}}
}

func (h *primeHarness) run(job PrimeJob) {
	h.t.Helper()
	h.p.Schedule(job)
	h.p.Wait()
}

// sent lists what the fake was sent: each request's first user text.
func (h *primeHarness) sent() []string {
	var out []string
	for _, c := range h.fake.Requests() {
		msgs, _ := c.Body["messages"].([]any)
		m, _ := msgs[len(msgs)-1].(map[string]any)
		blocks, _ := m["content"].([]any)
		b, _ := blocks[0].(map[string]any)
		out = append(out, fmt.Sprint(b["text"]))
	}
	return out
}

func (h *primeHarness) wantSent(want ...string) {
	h.t.Helper()
	if got := h.sent(); strings.Join(got, "|") != strings.Join(want, "|") {
		h.t.Fatalf("sent %q, want %q", got, want)
	}
}

func TestPrime_RequestIsTheRoutesOwnWithMaxTokensOne(t *testing.T) {
	fake := modeltest.New(primeReply(1024, 12983))
	defer fake.Close()
	b := primeTable(t, fake, nil)
	req := sampleRequest()
	req.Messages = append(historyWithThinking(t), UserTextTurn("x x x"))
	resp, err := b.Prime(context.Background(), req, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Usage.InputTokens != 1024 || resp.Usage.CacheReadInputTokens != 12983 {
		t.Errorf("usage = %+v", resp.Usage)
	}
	got := fake.Requests()
	if len(got) != 1 {
		t.Fatalf("requests = %d", len(got))
	}
	body := got[0].Body
	if body["max_tokens"] != float64(1) || got[0].Stream || body["temperature"] != 0.3 || body["model"] != "sous-chef" {
		t.Errorf("max_tokens=%v stream=%v temperature=%v model=%v", body["max_tokens"], got[0].Stream, body["temperature"], body["model"])
	}
	// The same plain request a real call on the route sends (buildParams):
	// thinking stated disabled, history thinking dropped, no cache_control.
	fake2 := modeltest.New(primeReply(10, 0))
	defer fake2.Close()
	b2 := primeTable(t, fake2, nil)
	if _, err := b2.Message(context.Background(), req, nil); err != nil {
		t.Fatal(err)
	}
	callBody := fake2.Requests()[0].Body
	for _, k := range []string{"system", "tools", "messages", "thinking", "tool_choice", "temperature"} {
		if fmt.Sprint(body[k]) != fmt.Sprint(callBody[k]) {
			t.Errorf("%s differs between the prime and a real call:\nprime %v\ncall  %v", k, body[k], callBody[k])
		}
	}
	if strings.Contains(string(got[0].Raw), "cache_control") || strings.Contains(string(got[0].Raw), `"type":"thinking"`) {
		t.Errorf("prime body carries Claude-only fields or thinking blocks: %s", got[0].Raw)
	}
}

func TestPrime_NoRetriesAndAnthropicUnsupported(t *testing.T) {
	fake := modeltest.New(modeltest.Step{Status: 500, Body: `{"type":"error","error":{"type":"api_error","message":"boom"}}`}, primeReply(1, 1))
	defer fake.Close()
	b := primeTable(t, fake, nil)
	if _, err := b.Prime(context.Background(), sampleRequest(), time.Second); err == nil {
		t.Fatal("want the 500")
	}
	if n := len(fake.Requests()); n != 1 {
		t.Errorf("requests = %d, want 1 (no retry)", n)
	}
	a := anthropicTable(t, fake, nil)
	ab, _ := a.Route("")
	if _, err := ab.Prime(context.Background(), sampleRequest(), time.Second); !errors.Is(err, ErrPrimeUnsupported) {
		t.Errorf("anthropic Prime err = %v", err)
	}
	p := NewPrimer(PrimerOptions{Logf: t.Logf})
	if p.Schedule(PrimeJob{Backend: ab, Prefix: "h", Build: func(context.Context) (PrimeRequests, error) { return PrimeRequests{}, nil }}) != nil {
		t.Error("an anthropic route was scheduled")
	}
}

func TestPrimer_OffRouteIsNeverPrimed(t *testing.T) {
	fake := modeltest.New()
	defer fake.Close()
	b := primeTable(t, fake, func(r *config.Route) { r.Prime = config.PrimeOff })
	p := NewPrimer(PrimerOptions{Logf: t.Logf})
	if p.Schedule(PrimeJob{Backend: b, Prefix: "h", Build: func(context.Context) (PrimeRequests, error) { return PrimeRequests{}, nil }}) != nil {
		t.Error("prime:off route was scheduled")
	}
	p.Wait()
	if len(fake.Requests()) != 0 {
		t.Error("requests sent")
	}
}

func TestPrimer_FloorThenTail_ThenDedupedUntilStale(t *testing.T) {
	h := newPrimeHarness(t,
		primeReply(14007, 0), primeReply(1307, 12983), // floor (cold), tail
		primeReply(1024, 12983), primeReply(1307, 12983), // after FloorFresh: floor, tail again
	)
	h.run(h.job("aaa", "conv-1:4"))
	h.wantSent("FLOOR aaa", "TAIL conv-1:4")
	if !h.logged("kind=floor") || !h.logged("cold=true") || !h.logged("reread=1307 cached=12983") {
		t.Errorf("log lines missing: %v", h.logs)
	}
	// The same state again: both checkpoints are in place.
	h.run(h.job("aaa", "conv-1:4"))
	h.wantSent("FLOOR aaa", "TAIL conv-1:4")
	if !h.logged("checkpoint already in place") {
		t.Error("no skip line")
	}
	// A new tail state: only the tail.
	h.fake.Append(primeReply(1400, 13266))
	h.run(h.job("aaa", "conv-1:8"))
	h.wantSent("FLOOR aaa", "TAIL conv-1:4", "TAIL conv-1:8")
	// Past FloorFresh the floor is re-sent, and the tail after it (the floor
	// prime erases the tail checkpoint).
	h.advance(DefaultFloorFresh + time.Second)
	h.fake.Append(primeReply(1024, 12983), primeReply(1400, 13266))
	h.run(h.job("aaa", "conv-1:8"))
	h.wantSent("FLOOR aaa", "TAIL conv-1:4", "TAIL conv-1:8", "FLOOR aaa", "TAIL conv-1:8")
}

// A job that finds a turn in progress (holding a slot, or parked on its
// batch) waits for the instance to go idle — releasing its trigger at once
// — and is dropped past MaxDefer.
func TestPrimer_DefersWhileATurnIsInProgress_ThenDrops(t *testing.T) {
	h := newPrimeHarness(t)
	h.setBusy(true)
	d := h.p.Schedule(h.job("aaa", "conv-1:4"))
	select {
	case <-d:
	case <-time.After(time.Second):
		t.Fatal("a deferred job held its trigger")
	}
	h.p.Wait()
	h.wantSent()
	if h.builds != 0 || !h.logged("deferred: a turn is in progress") || !h.logged("dropped: still busy") {
		t.Errorf("builds=%d logs=%v", h.builds, h.logs)
	}
}

// A deferred job runs once the turn is over (Kick), and a newer job queued
// while it waited replaces it.
func TestPrimer_DeferredJobRunsWhenIdle_NewestWins(t *testing.T) {
	h := newPrimeHarness(t, primeReply(14007, 0), primeReply(1307, 12983))
	h.p.o.MaxDefer = 5 * time.Second
	h.p.o.PollInterval = time.Hour // only Kick wakes it
	h.setBusy(true)
	h.p.Schedule(h.job("aaa", "conv-1:4"))
	time.Sleep(30 * time.Millisecond)
	h.p.Schedule(h.job("aaa", "conv-2:6"))
	time.Sleep(30 * time.Millisecond)
	h.setBusy(false)
	h.p.Kick()
	h.p.Wait()
	h.wantSent("FLOOR aaa", "TAIL conv-2:6")
	if !h.logged("replaced by test") {
		t.Errorf("no replace line: %v", h.logs)
	}
}

// A turn that starts while the job builds or between its floor and its
// tail sends it back to waiting; the retry sends only what is missing.
func TestPrimer_TurnStartingMidJobDefersTheRest(t *testing.T) {
	h := newPrimeHarness(t, primeReply(14007, 0), primeReply(1307, 12983))
	h.p.o.MaxDefer = 5 * time.Second
	job := h.job("aaa", "conv-1:4")
	build := job.Build
	first := true
	job.Build = func(ctx context.Context) (PrimeRequests, error) {
		if first {
			first = false
			h.setBusy(true) // a turn starts while storage is read
			go func() { time.Sleep(50 * time.Millisecond); h.setBusy(false); h.p.Kick() }()
		}
		return build(ctx)
	}
	h.run(job)
	h.wantSent("FLOOR aaa", "TAIL conv-1:4")
	if h.builds != 2 {
		t.Errorf("builds = %d, want 2 (rebuilt after the wait)", h.builds)
	}
}

// The trigger's Dispatch is released once the job's last request is on the
// wire — before the model server answers it.
func TestPrimer_DispatchReleasesWhenTheLastRequestIsWritten(t *testing.T) {
	slowTail := primeReply(1307, 12983)
	slowTail.Delay = 800 * time.Millisecond
	h := newPrimeHarness(t, primeReply(14007, 0), slowTail)
	started := time.Now()
	d := h.p.Schedule(h.job("aaa", "conv-1:4"))
	d.Wait(5 * time.Second)
	if el := time.Since(started); el > 600*time.Millisecond {
		t.Errorf("Dispatch released after %s: it waited for the tail's answer", el)
	}
	if n := len(h.fake.Requests()); n < 1 {
		t.Errorf("released before the floor was sent (%d requests)", n)
	}
	h.p.Wait()
	h.wantSent("FLOOR aaa", "TAIL conv-1:4")
	// A skipped job releases too; a nil Dispatch never blocks.
	h.p.Schedule(h.job("aaa", "conv-1:4")).Wait(time.Second)
	Dispatch(nil).Wait(time.Hour)
}

// SiteInterval: a site's tail is not re-primed within the interval (two
// panels re-reading their conversations), while the floor is still kept.
func TestPrimer_SiteIntervalThrottlesTails(t *testing.T) {
	h := newPrimeHarness(t, primeReply(14007, 0), primeReply(1307, 12983))
	job := func(key string) PrimeJob {
		j := h.job("aaa", key)
		j.SiteInterval = time.Minute
		return j
	}
	h.run(job("conv-1:4"))
	h.run(job("conv-1:6")) // another state of site-1's conversation, 0 s later
	h.wantSent("FLOOR aaa", "TAIL conv-1:4")
	if !h.logged("primed less than 1m0s ago") {
		t.Errorf("no throttle line: %v", h.logs)
	}
	h.advance(time.Minute + time.Second)
	h.fake.Append(primeReply(1400, 12983))
	h.run(job("conv-1:6"))
	h.wantSent("FLOOR aaa", "TAIL conv-1:4", "TAIL conv-1:6")
}

func TestPrimer_ObserveMissAndPrefixChangeRePrimeTheFloor(t *testing.T) {
	h := newPrimeHarness(t, primeReply(14007, 0))
	h.run(h.job("aaa", ""))
	h.wantSent("FLOOR aaa")
	// A real call that hit the cache: the floor stays trusted.
	h.p.Observe(h.b, "aaa", Usage{InputTokens: 2300, CacheReadInputTokens: 12983})
	h.run(h.job("aaa", ""))
	h.wantSent("FLOOR aaa")
	// A real call that re-read everything: the server lost it.
	h.p.Observe(h.b, "aaa", Usage{InputTokens: 15283})
	h.fake.Append(primeReply(14007, 0))
	h.run(h.job("aaa", ""))
	h.wantSent("FLOOR aaa", "FLOOR aaa")
	if !h.logged("missed the cache") {
		t.Error("no miss line")
	}
	// Another prefix took the slot (a real call on manifest bbb): aaa's
	// floor is re-primed on the next trigger.
	h.p.Observe(h.b, "bbb", Usage{InputTokens: 2300, CacheReadInputTokens: 9328})
	h.fake.Append(primeReply(5000, 9328))
	h.run(h.job("aaa", ""))
	h.wantSent("FLOOR aaa", "FLOOR aaa", "FLOOR aaa")
	// A small real call without cache (a short prompt) is not a miss.
	h.p.Observe(h.b, "aaa", Usage{InputTokens: 120})
	h.run(h.job("aaa", ""))
	h.wantSent("FLOOR aaa", "FLOOR aaa", "FLOOR aaa")
}

func TestPrimer_TailThatRestoresNothingForgetsTheFloor(t *testing.T) {
	h := newPrimeHarness(t, primeReply(14007, 0), primeReply(1307, 12983))
	h.run(h.job("aaa", "conv-1:4"))
	// Another instance's traffic erased the floor behind this one's back: the
	// next tail prime restores nothing. It re-read everything and left the
	// next turn's checkpoint, so nothing is re-sent now; the floor is
	// forgotten and the next trigger lays it (and the tail after it).
	h.fake.Append(primeReply(15300, 0))
	h.run(h.job("aaa", "conv-1:8"))
	h.wantSent("FLOOR aaa", "TAIL conv-1:4", "TAIL conv-1:8")
	if !h.logged("floor for prefix aaa is forgotten") {
		t.Errorf("logs = %v", h.logs)
	}
	h.fake.Append(primeReply(14007, 0), primeReply(1400, 12983))
	h.run(h.job("aaa", "conv-1:8"))
	h.wantSent("FLOOR aaa", "TAIL conv-1:4", "TAIL conv-1:8", "FLOOR aaa", "TAIL conv-1:8")
}

func TestPrimer_OneAtATimeNewestQueuedWins(t *testing.T) {
	slow := primeReply(14007, 0)
	slow.Delay = 300 * time.Millisecond
	h := newPrimeHarness(t, slow, primeReply(14007, 0))
	h.p.Schedule(h.job("aaa", ""))
	time.Sleep(50 * time.Millisecond) // the first prime is in flight
	h.p.Schedule(h.job("bbb", ""))    // queued…
	h.p.Schedule(h.job("ccc", ""))    // …and replaced
	h.p.Wait()
	h.wantSent("FLOOR aaa", "FLOOR ccc")
	if !h.logged("replaced queued") {
		t.Error("no replace line")
	}
}

func TestPrimer_FailureLeavesTheFloorStale(t *testing.T) {
	h := newPrimeHarness(t, modeltest.Step{Status: 503, Body: `{"type":"error","error":{"type":"overloaded_error","message":"cold"}}`}, primeReply(14007, 0))
	h.run(h.job("aaa", ""))
	if !h.logged("failed after") {
		t.Errorf("no failure line: %v", h.logs)
	}
	h.run(h.job("aaa", ""))
	h.wantSent("FLOOR aaa", "FLOOR aaa")
}

func TestPrimer_BuildErrorSendsNothing(t *testing.T) {
	h := newPrimeHarness(t)
	h.run(PrimeJob{Backend: h.b, Prefix: "aaa", Reason: "test", Build: func(context.Context) (PrimeRequests, error) {
		return PrimeRequests{}, errors.New("unknown manifest")
	}})
	h.wantSent()
	if !h.logged("build failed: unknown manifest") {
		t.Errorf("logs = %v", h.logs)
	}
}
