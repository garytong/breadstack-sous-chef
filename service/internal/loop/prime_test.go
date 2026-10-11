package loop

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/breadstack/sous-chef/internal/config"
	"github.com/breadstack/sous-chef/internal/model"
	"github.com/breadstack/sous-chef/internal/model/modeltest"
	"github.com/breadstack/sous-chef/internal/prompt"
)

// primed is a harness whose DEFAULT route is the in-house one ("muffin",
// messages-compatible, primed) with a Claude route beside it, and a Primer
// wired as main wires it.
type primed struct {
	*harness
	gpu, claude *modeltest.Server
	primer      *model.Primer
	mu          sync.Mutex
	logs        []string
}

func newPrimed(t *testing.T, gpuSteps []modeltest.Step, routeMutate func(*config.Route)) *primed {
	t.Helper()
	pr := &primed{gpu: modeltest.New(answering("sous-chef", gpuSteps)...), claude: modeltest.New()}
	t.Cleanup(pr.gpu.Close)
	t.Cleanup(pr.claude.Close)
	pr.harness = newHarness(t, func(d *Deps) {
		muffin := config.Route{Kind: config.RouteMessagesCompatible, URL: pr.gpu.URL, Auth: config.AuthNone, Model: "sous-chef", MaxTokens: 512, Prime: config.PrimeOn}
		if routeMutate != nil {
			routeMutate(&muffin)
		}
		tbl, err := model.NewTable(model.Options{
			Routes: map[string]config.Route{
				"muffin": muffin,
				"claude": {Kind: config.RouteAnthropic, Auth: config.AuthAPIKey, Model: "claude-sonnet-5-5", MaxTokens: 1024, Effort: "medium"},
			},
			Default: "muffin", Allowlist: []string{"muffin", "claude"},
			APIKey: "sk-test", BaseURL: pr.claude.URL, CallTimeout: 5 * time.Second, MaxRetries: intp(0),
		})
		if err != nil {
			t.Fatal(err)
		}
		d.Models = tbl
		pr.primer = model.NewPrimer(model.PrimerOptions{MaxDefer: 300 * time.Millisecond, PollInterval: 5 * time.Millisecond, Logf: func(f string, a ...any) {
			pr.mu.Lock()
			pr.logs = append(pr.logs, fmt.Sprintf(f, a...))
			pr.mu.Unlock()
			t.Logf(f, a...)
		}})
		d.Primer = pr.primer
	})
	return pr
}

func (pr *primed) logged(sub string) bool {
	pr.mu.Lock()
	defer pr.mu.Unlock()
	for _, l := range pr.logs {
		if strings.Contains(l, sub) {
			return true
		}
	}
	return false
}

func primeStep(reread, cached int64) modeltest.Step {
	return modeltest.Step{Message: &modeltest.Message{Model: "sous-chef", StopReason: "max_tokens", Content: []modeltest.Block{modeltest.Text("x")},
		Usage: modeltest.Usage{InputTokens: reread, OutputTokens: 1, CacheReadInputTokens: cached}}}
}

// msgsJSON is each message of a captured body, as JSON.
func msgsJSON(t *testing.T, c modeltest.Captured) []string {
	t.Helper()
	msgs, _ := c.Body["messages"].([]any)
	out := make([]string, len(msgs))
	for i, m := range msgs {
		b, _ := json.Marshal(m)
		out[i] = string(b)
	}
	return out
}

func lastUserText(t *testing.T, c modeltest.Captured) string {
	t.Helper()
	u := userTexts(t, c)
	if len(u) == 0 {
		return ""
	}
	return u[len(u)-1]
}

// After turn.done the conversation's next turn is primed: the floor (the
// empty history) and the tail (the history as the next Start will read it),
// each a filler user message with max_tokens 1. The next real turn's first
// request then shares every message before its own user message with the
// tail prime, byte for byte, and the same system and tools.
func TestPrime_AfterTurnDone_SharesTheNextTurnsPrefix(t *testing.T) {
	pr := newPrimed(t, []modeltest.Step{
		reply(setFields("s1", "headline")), reply(modeltest.Text("Done.")),
		primeStep(14007, 0), primeStep(1307, 12983), // after turn 1: floor (cold), tail
		reply(modeltest.Text("Sure.")),
		primeStep(1200, 13266), // after turn 2: the tail only
	}, nil)
	turn1, _ := pr.start(pr.request("set the hero headline to Fresh cookies"))
	if n := len(pr.gpu.Requests()); n != 1 {
		t.Fatalf("a waiting turn primed: %d requests", n)
	}
	evs, err := pr.cont(turn1.ID(), writtenResult("s1", "headline"))
	if err != nil || last(evs).Event != "turn.done" {
		t.Fatalf("turn 1 = %s (%v)", kinds(evs), err)
	}
	pr.primer.Wait()
	reqs := pr.gpu.Requests()
	if len(reqs) != 4 {
		t.Fatalf("after turn 1: %d requests, want 2 calls + floor + tail", len(reqs))
	}
	floor, tail := reqs[2], reqs[3]
	for name, c := range map[string]modeltest.Captured{"floor": floor, "tail": tail} {
		if c.Body["max_tokens"] != float64(1) || c.Stream || lastUserText(t, c) != prompt.PrimeFiller(config.DefaultPrimeFillerTokens) {
			t.Errorf("%s: max_tokens=%v stream=%v last user text %.30q", name, c.Body["max_tokens"], c.Stream, lastUserText(t, c))
		}
	}
	if got := msgRoles(floor); got != "user" {
		t.Errorf("floor roles = %s", got)
	}
	if got := msgRoles(tail); got != "user assistant user assistant user" {
		t.Errorf("tail roles = %s", got)
	}

	pr.start(pr.request("now make the buttons rounded"))
	pr.primer.Wait()
	reqs = pr.gpu.Requests()
	if len(reqs) != 6 {
		t.Fatalf("after turn 2: %d requests", len(reqs))
	}
	next := reqs[4]
	pm, nm := msgsJSON(t, tail), msgsJSON(t, next)
	if len(pm) != len(nm) {
		t.Fatalf("tail prime has %d messages, turn 2's request %d", len(pm), len(nm))
	}
	for i := 0; i < len(pm)-1; i++ {
		if pm[i] != nm[i] {
			t.Errorf("message %d differs:\nprime %s\nturn  %s", i, pm[i], nm[i])
		}
	}
	if !strings.HasPrefix(lastUserText(t, next), "<context>\n") {
		t.Errorf("turn 2's own message = %.40q", lastUserText(t, next))
	}
	for _, k := range []string{"system", "tools", "thinking", "tool_choice"} {
		if fmt.Sprint(tail.Body[k]) != fmt.Sprint(next.Body[k]) || fmt.Sprint(floor.Body[k]) != fmt.Sprint(next.Body[k]) {
			t.Errorf("%s differs between the primes and the real request", k)
		}
	}
	// Turn 2's prime: the floor is still trusted, so only the tail.
	if got := msgRoles(reqs[5]); got != "user assistant user assistant user assistant user" {
		t.Errorf("turn 2's tail prime roles = %s", got)
	}
	if !pr.logged("kind=tail reason=turn_done site=site-1") || !pr.logged("cold=true") {
		t.Error("prime log lines missing")
	}
}

// The triggers outside a turn (the api.Primer methods): a manifest upload
// and a conversation end prime the floor once; the panel opening on a
// conversation primes its next turn; a conversation waiting on a batch gets
// the floor only; a route with prime:"off" and an anthropic route never
// prime.
func TestPrime_APITriggers(t *testing.T) {
	pr := newPrimed(t, []modeltest.Step{
		primeStep(14007, 0),                // ManifestRegistered: floor
		reply(modeltest.Text("Hello.")),    // a turn
		primeStep(1307, 12983),             // its tail (floor fresh)
		reply(setFields("s1", "headline")), // a turn left waiting
	}, nil)
	pr.runner.ManifestRegistered(pr.hash)
	pr.primer.Wait()
	pr.runner.ConversationEnded("site-1", pr.hash) // floor fresh: nothing
	pr.runner.ManifestRegistered("deadbeef")       // unknown manifest: build fails, nothing sent
	pr.primer.Wait()
	if n := len(pr.gpu.Requests()); n != 1 {
		t.Fatalf("requests = %d, want the one floor", n)
	}
	if !pr.logged("build failed") {
		t.Error("unknown manifest not logged")
	}
	t1, _ := pr.start(pr.request("hello"))
	pr.primer.Wait()
	conv := pr.st.turn(t1.ID()).ConversationID
	pr.runner.ConversationOpened("site-1", conv, pr.hash) // already primed: nothing
	pr.primer.Wait()
	if n := len(pr.gpu.Requests()); n != 3 {
		t.Fatalf("requests = %d, want floor + turn + tail", n)
	}
	if !pr.logged("reason=panel_open site=site-1 prefix=") || !pr.logged("checkpoint already in place") {
		t.Error("panel_open skip not logged")
	}
	// A turn waiting on its batch: no turn.done, no prime; opening the
	// panel then primes nothing new (the floor is fresh, the tail is a
	// continuation's business).
	pr.start(pr.request("set the headline"))
	pr.runner.ConversationOpened("site-1", conv, pr.hash)
	pr.primer.Wait()
	if n := len(pr.gpu.Requests()); n != 4 {
		t.Fatalf("requests = %d after a waiting turn", n)
	}

	// An anthropic turn never primes.
	pr.claude.Append(reply(modeltest.Text("Hi.")))
	r := pr.request("hello claude")
	r.SiteID = "site-2"
	r.Model = "claude"
	pr.start(r)
	pr.primer.Wait()
	if n := len(pr.claude.Requests()); n != 1 {
		t.Errorf("claude requests = %d", n)
	}

	// prime:"off" on the route: nothing.
	off := newPrimed(t, []modeltest.Step{reply(modeltest.Text("Hello."))}, func(r *config.Route) { r.Prime = config.PrimeOff })
	off.runner.ManifestRegistered(off.hash)
	off.start(off.request("hello"))
	off.primer.Wait()
	if n := len(off.gpu.Requests()); n != 1 {
		t.Errorf("prime off: %d requests", n)
	}
}

// A conversation about to rotate primes the floor only: its next turn
// starts a fresh conversation.
func TestPrime_RotatingConversationPrimesTheFloorOnly(t *testing.T) {
	pr := newPrimed(t, []modeltest.Step{reply(modeltest.Text("Hello.")), primeStep(14007, 0)}, nil)
	pr.runner.d.RotateAt = 2
	pr.start(pr.request("hello"))
	pr.primer.Wait()
	reqs := pr.gpu.Requests()
	if len(reqs) != 2 || msgRoles(reqs[1]) != "user" {
		t.Fatalf("requests = %d, last roles %s", len(reqs), msgRoles(reqs[len(reqs)-1]))
	}
}

// A turn parked on its batch keeps primes waiting for PrimeParkedBusy: its
// continuation resumes from the slot's state at its last call, which a
// prime would overwrite. Past the window the batch is likely abandoned and
// primes run again.
func TestPrime_ParkedTurnDefersPrimes(t *testing.T) {
	pr := newPrimed(t, []modeltest.Step{
		reply(setFields("s1", "headline")), // site-1's turn parks on its batch
		primeStep(14007, 0),                // the floor, once the window passed
	}, nil)
	pr.start(pr.request("set the headline"))
	if pr.runner.Parked() != 1 || pr.runner.Live() != 0 {
		t.Fatalf("parked=%d live=%d", pr.runner.Parked(), pr.runner.Live())
	}
	model.Dispatch(pr.runner.ConversationEnded("site-2", pr.hash)).Wait(time.Second) // released at once: deferred
	pr.primer.Wait()
	if n := len(pr.gpu.Requests()); n != 1 {
		t.Fatalf("requests = %d: a prime ran while a turn was parked", n)
	}
	if !pr.logged("deferred: a turn is in progress") || !pr.logged("dropped: still busy") {
		t.Error("deferral not logged")
	}
	pr.clock = pr.clock.Add(PrimeParkedBusy + time.Second) // no goroutine reads it now
	pr.runner.ConversationEnded("site-2", pr.hash)
	pr.primer.Wait()
	reqs := pr.gpu.Requests()
	if len(reqs) != 2 || reqs[1].Body["max_tokens"] != float64(1) {
		t.Fatalf("requests = %d, want the floor after the window", len(reqs))
	}
}
