package model

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/http/httptrace"
	"sync"
	"time"

	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/breadstack/sous-chef/internal/config"
)

// Prompt-cache priming (README §4 "Prompt-cache priming").
//
// The in-house model server (Ollama → llama-server, one slot, a hybrid-
// attention model) can resume a prompt only from the slot's current state
// or from a saved context checkpoint at or before the common prefix with
// the previous prompt. It saves checkpoints only at prompt end −n_ubatch
// (−1024) and −4 — never where the next turn's request starts to differ
// (the new user message's `<context>`), so every first call of a turn
// re-read the whole 13–15k-token prefix (6.7–11 s on the L4).
//
// A prime is one extra request, off the critical path: the next turn's
// exact prefix — system, tools, the stable history — plus a user message of
// PrimeFiller "x" tokens and max_tokens 1. Its −1024 checkpoint lands
// exactly at that message's `<|im_start|>user\n`, so the next real request
// restores it and re-reads only its own page context (measured on int
// 2026-10-10: 15,283 tokens / 7.3 s → 2,295–2,312 tokens / 1.4–1.6 s).
//
// Two kinds of prime per model server and fixed prefix (manifest):
//   - floor: the empty history. Its checkpoint at the first user message
//     serves every conversation on the manifest and survives every request
//     that shares the prefix; it is lost on a model-server restart (cold
//     start, instance churn) or when another prefix takes the slot.
//   - tail: a conversation's next turn (its history up to the last reply).
//     It restores the floor (or the previous tail checkpoint) and re-reads
//     only the last exchange plus the filler, ~1 s warm.

// Prime defaults.
const (
	// DefaultPrimeTimeout bounds one prime request. It is long on purpose: a
	// prime is never on a merchant's path, and cutting it short throws the
	// server's work away (Ollama cancels the prefill when the client goes),
	// while the case it exists for — a model server that just cold-started
	// (~1 min 50 s) and runs its first prefill at ~320 tok/s (~45 s for 14k
	// tokens) — needs the time.
	DefaultPrimeTimeout = 3 * time.Minute
	// DefaultFloorFresh is how long a floor checkpoint this instance made (or
	// saw a real call hit) is trusted without being re-sent. Past it, a
	// trigger re-primes the floor: ~1 s when the checkpoint is still there,
	// and the wake-up of a scaled-to-zero model server when it is not.
	DefaultFloorFresh = 5 * time.Minute
	// DefaultMinMissTokens: a real call that read nothing from the server's
	// cache while re-reading at least this many tokens means the server lost
	// its checkpoints for the prefix (a restart, or another prefix took the
	// slot). Continuations re-read ~120 tokens; a turn's first call with the
	// floor in place re-reads its page context, 1.7–3.1k.
	DefaultMinMissTokens = 4096
	// DefaultMaxDefer is how long a job that found a turn in progress on
	// this instance (Busy) waits for the instance to go idle before it is
	// dropped. A newer job for the same server replaces it meanwhile.
	DefaultMaxDefer = 2 * time.Minute
	// DefaultPollInterval is how often a deferred job re-checks Busy between
	// Kicks (a parked turn stops counting as busy by time alone).
	DefaultPollInterval = 500 * time.Millisecond
	// DefaultHold bounds how long a trigger (the request that scheduled a
	// job) waits for the job's requests to reach the model server
	// (Dispatch.Wait). Cloud Run with request-based CPU (`cpu_idle`)
	// throttles a goroutine once no request is in flight, so a prime started
	// after its trigger's response would stall until the next request —
	// likely the merchant's own next turn. Holding the trigger open until
	// the prime's request is written keeps the CPU; the model server then
	// does the work on its own. A warm floor + tail takes ~1 s; a cold floor
	// outlasts the bound (its request is already on the wire).
	DefaultHold       = 3 * time.Second
	primeBuildTimeout = 30 * time.Second
)

// ErrPrimeUnsupported is returned by Backend.Prime on a route that is not
// messages-compatible.
var ErrPrimeUnsupported = errors.New("model: prime needs a messages-compatible route")

// Prime sends req as a prime: the route's own request (buildParams — the
// same system, tools, thinking and sampling fields as a real call, so the
// server renders the same prefix) with max_tokens 1, never streamed, no
// retries, under timeout (≤0 = DefaultPrimeTimeout). The response's usage
// says what the server re-read (input_tokens) and restored
// (cache_read_input_tokens).
func (b *Backend) Prime(ctx context.Context, req Request, timeout time.Duration) (*Response, error) {
	if b.route.Kind != config.RouteMessagesCompatible {
		return nil, ErrPrimeUnsupported
	}
	if timeout <= 0 {
		timeout = DefaultPrimeTimeout
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	params := b.buildParams(req)
	params.MaxTokens = 1
	started := b.now()
	msg, err := b.client.Beta.Messages.New(ctx, params, option.WithRequestTimeout(timeout), option.WithMaxRetries(0))
	if err != nil {
		return nil, err
	}
	resp := fromMessage(msg)
	resp.Latency = b.now().Sub(started)
	return resp, nil
}

// PrimeJob is one request to prime a backend's model server.
type PrimeJob struct {
	Backend *Backend
	// Prefix identifies the fixed prefix (system + tools): the manifest hash.
	Prefix string
	// Reason names the trigger, for the log: turn_done, manifest,
	// panel_open, conversation_end.
	Reason string
	// Site is the site the prime is for ("" for a floor-only trigger).
	Site string
	// SiteInterval, when > 0, drops the job's tail if this server primed a
	// tail for Site less than SiteInterval ago (the panel_open trigger: two
	// open panels that re-read their conversations must not re-prime each
	// other's tails on every read). The floor is still checked.
	SiteInterval time.Duration
	// Build makes the requests, in the primer's goroutine (it may read
	// storage).
	Build func(ctx context.Context) (PrimeRequests, error)

	sig *dispatch
}

// PrimeRequests is what a PrimeJob primes.
type PrimeRequests struct {
	// Floor is the empty-history prime of the job's prefix.
	Floor Request
	// Tail is a conversation's next turn; nil for none (no conversation, one
	// about to rotate, one waiting on a batch). TailKey identifies the
	// history state it covers (conversation id and tail seq), so the same
	// state is not primed twice in a row.
	Tail    *Request
	TailKey string
}

// Dispatch is closed once a scheduled job no longer needs this instance's
// CPU before the model server has its work: its last request is written to
// the wire, or the job ended, was deferred (a turn is in progress), was
// replaced or dropped. nil for a job that was not scheduled.
type Dispatch <-chan struct{}

// Wait blocks until d is closed or max (≤0 = DefaultHold) has passed. A
// nil Dispatch returns at once.
func (d Dispatch) Wait(max time.Duration) {
	if d == nil {
		return
	}
	if max <= 0 {
		max = DefaultHold
	}
	t := time.NewTimer(max)
	defer t.Stop()
	select {
	case <-d:
	case <-t.C:
	}
}

type dispatch struct {
	once sync.Once
	ch   chan struct{}
}

func newDispatch() *dispatch { return &dispatch{ch: make(chan struct{})} }

func (d *dispatch) signal() {
	if d != nil {
		d.once.Do(func() { close(d.ch) })
	}
}

// PrimerOptions configures a Primer. Zero values take the defaults.
type PrimerOptions struct {
	Timeout       time.Duration // one prime request; DefaultPrimeTimeout
	FloorFresh    time.Duration // DefaultFloorFresh
	MinMissTokens int64         // DefaultMinMissTokens
	MaxDefer      time.Duration // DefaultMaxDefer
	PollInterval  time.Duration // DefaultPollInterval
	// Busy reports that a merchant's turn is in progress on this instance:
	// holding a slot, or parked on its batch recently enough that its next
	// call is coming (loop.New wires the Runner's). A prime is not sent then
	// — the server has one slot, the turn's next call must not queue behind
	// it, and a prime would overwrite the slot the turn's continuation
	// resumes from. The job waits (Kick, PollInterval) up to MaxDefer.
	Busy func() bool
	Logf func(string, ...any)
	Now  func() time.Time
}

// Primer keeps a checkpoint at the fixed-prefix boundary (and at the next
// turn's) on each in-house model server. Schedule never blocks: per server
// one prime runs (or waits for the instance to go idle) at a time, and a
// job that arrives meanwhile replaces any job already queued (the slot can
// hold one conversation's tail at a time, so the newest wins). State is per
// instance.
type Primer struct {
	o    PrimerOptions
	mu   sync.Mutex
	st   map[string]*primeServer
	idle chan struct{} // closed and replaced by Kick
	wg   sync.WaitGroup
}

// primeServer is one model server's state as this instance knows it.
type primeServer struct {
	running bool
	pending *PrimeJob
	// lastPrefix is the fixed prefix of the last request this instance sent
	// the server (a real call or a prime); lastSeen when.
	lastPrefix string
	lastSeen   time.Time
	// floors: prefix → when its floor checkpoint was made or last confirmed.
	// Cleared when another prefix takes the slot or a real call misses.
	floors map[string]time.Time
	// tail / tailPrefix: the tail state the last prime checkpointed; reset by
	// any real call (the slot moved on).
	tail, tailPrefix string
	// siteTails: site → when this server last primed a tail for it
	// (PrimeJob.SiteInterval). Pruned past an hour.
	siteTails map[string]time.Time
	// counters, for the log lines
	primes, failures, skipped, cold, misses, deferred, dropped int
}

// NewPrimer builds a Primer.
func NewPrimer(o PrimerOptions) *Primer {
	if o.Timeout <= 0 {
		o.Timeout = DefaultPrimeTimeout
	}
	if o.FloorFresh <= 0 {
		o.FloorFresh = DefaultFloorFresh
	}
	if o.MinMissTokens <= 0 {
		o.MinMissTokens = DefaultMinMissTokens
	}
	if o.MaxDefer <= 0 {
		o.MaxDefer = DefaultMaxDefer
	}
	if o.PollInterval <= 0 {
		o.PollInterval = DefaultPollInterval
	}
	if o.Logf == nil {
		o.Logf = log.Printf
	}
	if o.Now == nil {
		o.Now = time.Now
	}
	return &Primer{o: o, st: map[string]*primeServer{}, idle: make(chan struct{})}
}

// SetBusy installs the busy check when none was given (loop.New: the
// Runner's turns in progress).
func (p *Primer) SetBusy(busy func() bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.o.Busy == nil {
		p.o.Busy = busy
	}
}

func (p *Primer) busy() bool {
	p.mu.Lock()
	b := p.o.Busy
	p.mu.Unlock()
	return b != nil && b()
}

// Kick wakes deferred jobs to re-check Busy now (the Runner calls it when a
// turn releases its slot).
func (p *Primer) Kick() {
	if p == nil {
		return
	}
	p.mu.Lock()
	close(p.idle)
	p.idle = make(chan struct{})
	p.mu.Unlock()
}

func (p *Primer) server(key string) *primeServer {
	s := p.st[key]
	if s == nil {
		s = &primeServer{floors: map[string]time.Time{}, siteTails: map[string]time.Time{}}
		p.st[key] = s
	}
	return s
}

// Schedule queues job and returns at once. The Dispatch is nil when the
// job's route is not primed (anthropic, or `prime` not "on") or the job is
// incomplete; otherwise the trigger may Wait on it (bounded) before it
// lets its request end.
func (p *Primer) Schedule(job PrimeJob) Dispatch {
	if p == nil || job.Backend == nil || !job.Backend.PrimeEnabled() || job.Build == nil || job.Prefix == "" {
		return nil
	}
	job.sig = newDispatch()
	key := job.Backend.Server()
	p.mu.Lock()
	s := p.server(key)
	if s.running {
		if s.pending != nil {
			s.skipped++
			p.o.Logf("prime: server=%s replaced queued %s prime (site %s) with %s (site %s)", key, s.pending.Reason, orDash(s.pending.Site), job.Reason, orDash(job.Site))
			s.pending.sig.signal()
		}
		j := job
		s.pending = &j
		p.mu.Unlock()
		return job.sig.ch
	}
	s.running = true
	p.wg.Add(1)
	p.mu.Unlock()
	j := job
	go p.run(key, s, &j)
	return job.sig.ch
}

// Wait blocks until no prime is running or queued (tests, shutdown).
func (p *Primer) Wait() { p.wg.Wait() }

// Observe records a real model call on b with the fixed prefix prefix: the
// slot now holds that request (any tail checkpoint a prime left is
// consumed), another prefix's floor is gone, and a call that read nothing
// from the cache while re-reading MinMissTokens or more means the server
// lost its checkpoints (cold start, restart, another prefix) — the next
// trigger re-primes the floor.
func (p *Primer) Observe(b *Backend, prefix string, u Usage) {
	if p == nil || b == nil || !b.PrimeEnabled() {
		return
	}
	key := b.Server()
	p.mu.Lock()
	defer p.mu.Unlock()
	s := p.server(key)
	if s.lastPrefix != prefix {
		s.floors = map[string]time.Time{}
	}
	s.lastPrefix = prefix
	s.lastSeen = p.o.Now()
	s.tail, s.tailPrefix = "", ""
	if u.CacheReadInputTokens == 0 && u.InputTokens >= p.o.MinMissTokens {
		s.misses++
		s.floors = map[string]time.Time{}
		p.o.Logf("prime: server=%s real call missed the cache (reread=%d cached=0, prefix %s): floor will be re-primed (misses=%d)", key, u.InputTokens, short(prefix), s.misses)
	} else if u.CacheReadInputTokens > 0 {
		if _, ok := s.floors[prefix]; ok {
			s.floors[prefix] = s.lastSeen // still there: confirmed
		}
	}
}

func (p *Primer) run(key string, s *primeServer, job *PrimeJob) {
	defer p.wg.Done()
	for job != nil {
		p.process(key, s, job)
		p.mu.Lock()
		job, s.pending = s.pending, nil
		if job == nil {
			s.running = false
		}
		p.mu.Unlock()
	}
}

// errBusy: runOne found a turn in progress before its first (or a further)
// request; the job goes back to waiting.
var errBusy = errors.New("busy")

// process runs job once the instance is idle: while Busy it waits (Kick,
// PollInterval) up to MaxDefer, switching to any newer job queued for the
// server meanwhile, then drops it.
func (p *Primer) process(key string, s *primeServer, job *PrimeJob) {
	route := job.Backend.Name()
	deadline := time.NewTimer(p.o.MaxDefer)
	defer deadline.Stop()
	waited := false
	for {
		p.mu.Lock()
		idle := p.idle
		p.mu.Unlock()
		if !p.busy() {
			if p.runOne(key, s, job) != errBusy {
				return
			}
		}
		// A turn is in progress: release the trigger and wait.
		job.sig.signal()
		if !waited {
			waited = true
			p.mu.Lock()
			s.deferred++
			n := s.deferred
			p.mu.Unlock()
			p.o.Logf("prime: server=%s route=%s reason=%s site=%s prefix=%s deferred: a turn is in progress on this instance (deferred=%d)", key, route, job.Reason, orDash(job.Site), short(job.Prefix), n)
		}
		poll := time.NewTimer(p.o.PollInterval)
		select {
		case <-idle:
		case <-poll.C:
		case <-deadline.C:
			poll.Stop()
			p.mu.Lock()
			s.dropped++
			n := s.dropped
			p.mu.Unlock()
			p.o.Logf("prime: server=%s route=%s reason=%s site=%s prefix=%s dropped: still busy after %s (dropped=%d)", key, route, job.Reason, orDash(job.Site), short(job.Prefix), p.o.MaxDefer, n)
			return
		}
		poll.Stop()
		p.mu.Lock()
		if s.pending != nil {
			p.o.Logf("prime: server=%s deferred %s prime (site %s) replaced by %s (site %s)", key, job.Reason, orDash(job.Site), s.pending.Reason, orDash(s.pending.Site))
			job, s.pending = s.pending, nil
			route = job.Backend.Name()
			waited = false
		}
		p.mu.Unlock()
	}
}

// floorFresh reports whether s holds a trusted floor checkpoint for prefix.
// Called with p.mu held.
func (p *Primer) floorFresh(s *primeServer, prefix string) bool {
	at, ok := s.floors[prefix]
	return ok && s.lastPrefix == prefix && p.o.Now().Sub(at) < p.o.FloorFresh
}

// runOne builds and sends job's requests: the floor when this instance does
// not trust one for the prefix, then the tail when there is one not already
// checkpointed. errBusy when a turn started before a request went out.
func (p *Primer) runOne(key string, s *primeServer, job *PrimeJob) error {
	defer job.sig.signal()
	route := job.Backend.Name()
	skip := func(why string) {
		p.mu.Lock()
		s.skipped++
		n := s.skipped
		p.mu.Unlock()
		p.o.Logf("prime: server=%s route=%s reason=%s site=%s prefix=%s skipped: %s (skipped=%d)", key, route, job.Reason, orDash(job.Site), short(job.Prefix), why, n)
	}
	bctx, cancel := context.WithTimeout(context.Background(), primeBuildTimeout)
	reqs, err := job.Build(bctx)
	cancel()
	if err != nil {
		p.mu.Lock()
		s.failures++
		p.mu.Unlock()
		p.o.Logf("prime: server=%s route=%s reason=%s site=%s prefix=%s build failed: %v", key, route, job.Reason, orDash(job.Site), short(job.Prefix), err)
		return nil
	}
	floor, tail := reqs.Floor, reqs.Tail
	now := p.o.Now()
	p.mu.Lock()
	needFloor := !p.floorFresh(s, job.Prefix)
	tailDone := tail != nil && reqs.TailKey != "" && s.tail == reqs.TailKey && s.tailPrefix == job.Prefix
	recent := false
	if tail != nil && job.SiteInterval > 0 && job.Site != "" {
		at, ok := s.siteTails[job.Site]
		recent = ok && now.Sub(at) < job.SiteInterval
	}
	p.mu.Unlock()
	if tailDone && !needFloor {
		tail = nil // a floor prime would erase it, so it is re-sent after one
	}
	if recent && tail != nil {
		tail = nil
		if !needFloor {
			skip(fmt.Sprintf("this site's tail was primed less than %s ago", job.SiteInterval))
			return nil
		}
	}
	if !needFloor && tail == nil {
		skip("checkpoint already in place")
		return nil
	}
	// The build read storage: a turn may have started meanwhile.
	if p.busy() {
		return errBusy
	}
	if needFloor {
		if _, ok := p.send(key, s, job, "floor", floor, tail == nil); !ok {
			return nil
		}
		if tail == nil {
			return nil
		}
		if p.busy() {
			return errBusy // the floor is fresh now: the retry sends the tail only
		}
	}
	resp, ok := p.send(key, s, job, "tail", *tail, true)
	if !ok {
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	s.tail, s.tailPrefix = reqs.TailKey, job.Prefix
	if job.Site != "" {
		for site, at := range s.siteTails {
			if now.Sub(at) > time.Hour {
				delete(s.siteTails, site)
			}
		}
		s.siteTails[job.Site] = now
	}
	if resp.Usage.CacheReadInputTokens == 0 && !needFloor {
		// The floor this instance trusted is gone (another instance's prefix,
		// a model-server restart it did not see). This prime re-read
		// everything and left the checkpoint the next turn needs; the floor
		// is forgotten, so the next trigger lays it again.
		delete(s.floors, job.Prefix)
		p.o.Logf("prime: server=%s tail prime restored nothing: the floor for prefix %s is forgotten", key, short(job.Prefix))
	}
	return nil
}

// send makes one prime request and records its outcome; last: this is the
// job's last request, so writing it releases the trigger (Dispatch).
func (p *Primer) send(key string, s *primeServer, job *PrimeJob, kind string, req Request, last bool) (*Response, bool) {
	started := p.o.Now()
	ctx := context.Background()
	if last {
		ctx = httptrace.WithClientTrace(ctx, &httptrace.ClientTrace{
			WroteRequest: func(httptrace.WroteRequestInfo) { job.sig.signal() },
		})
	}
	resp, err := job.Backend.Prime(ctx, req, p.o.Timeout)
	ms := p.o.Now().Sub(started).Milliseconds()
	p.mu.Lock()
	defer p.mu.Unlock()
	if err != nil {
		s.failures++
		p.o.Logf("prime: server=%s route=%s kind=%s reason=%s site=%s prefix=%s failed after %dms: %v (failures=%d)", key, job.Backend.Name(), kind, job.Reason, orDash(job.Site), short(job.Prefix), ms, err, s.failures)
		return nil, false
	}
	s.primes++
	now := p.o.Now()
	if s.lastPrefix != job.Prefix {
		s.floors = map[string]time.Time{}
	}
	s.lastPrefix, s.lastSeen = job.Prefix, now
	s.tail, s.tailPrefix = "", ""
	if kind == "floor" {
		s.floors[job.Prefix] = now
	}
	u := resp.Usage
	note := ""
	if u.CacheReadInputTokens == 0 {
		// Nothing to restore: the server was cold (just started, restarted,
		// or another prefix had the slot). This prime rebuilt it.
		s.cold++
		note = fmt.Sprintf(" cold=true colds=%d", s.cold)
		if ms > 0 && u.InputTokens*1000/ms < 1000 {
			note += " slow_prefill=true" // the first prefill after a model load runs at ~320 tok/s
		}
	}
	p.o.Logf("prime: server=%s route=%s kind=%s reason=%s site=%s prefix=%s reread=%d cached=%d ms=%d primes=%d%s",
		key, job.Backend.Name(), kind, job.Reason, orDash(job.Site), short(job.Prefix), u.InputTokens, u.CacheReadInputTokens, ms, s.primes, note)
	return resp, true
}

func short(s string) string {
	if len(s) > 12 {
		return s[:12]
	}
	return s
}

func orDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}
