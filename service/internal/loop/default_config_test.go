package loop

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/breadstack/sous-chef/internal/api"
	"github.com/breadstack/sous-chef/internal/config"
	"github.com/breadstack/sous-chef/internal/model"
	"github.com/breadstack/sous-chef/internal/model/modeltest"
	"github.com/breadstack/sous-chef/internal/registry"
	"github.com/breadstack/sous-chef/internal/store"
	"github.com/breadstack/sous-chef/internal/usage"
)

// TestDefaultConfig_UnchangedFromToday proves that deploying this build with
// int's configuration as it is today changes nothing: the environment the
// live sous-chef revision runs with (MODEL_ROUTES = blueberry-muffin as the
// default route plus Claude's two, DEFAULT_ROUTE, the allowlist, the call
// timeout; no parser_retries, no MODEL_ROUTE_ALIASES) goes through
// config.Load → model.FromConfig → the loop with the Primer on, and three
// turns run against fake model servers:
//
//  1. no model named (int's traffic): the in-house model writes an op, the
//     batch comes back clean, fast finish ends the turn; the Primer primes
//     the next turn;
//  2. no model named: the in-house server answers every attempt with Ollama's
//     MinistralParser 500 ("tool 'set_fieldz' not found"): the SDK retries
//     it twice, as today, and the turn escalates to Claude (model_error);
//  3. "model":"opus".
//
// Every request each server saw (method, path, query, every header but the
// three that name the test machine, the body's sha256 — byte for byte — and
// a readable digest of it), every turn's events (ids left out: they are random),
// the usage rows and the log lines are a transcript compared byte for byte
// with testdata/default_config.golden.json — written by this same test on
// the code before this change (sous-chef main 2742dc7 + the Mistral arm,
// SOUS_CHEF_WRITE_GOLDEN=1). Retry timing is the SDK's and is not recorded;
// the attempt count is.
func TestDefaultConfig_UnchangedFromToday(t *testing.T) {
	if testing.Short() {
		t.Skip("about 4 s of SDK retry backoff")
	}
	c := snapshotCompiled(t)
	inhouse := newScriptedOllama(t)
	claude := modeltest.New(
		modeltest.Reply(modeltest.Message{ID: "msg_c1", Model: "claude-sonnet-5-5", Content: []modeltest.Block{modeltest.Text("I made the headline say Fresh cookies.")},
			Usage: modeltest.Usage{InputTokens: 900, OutputTokens: 12}}),
		modeltest.Reply(modeltest.Message{ID: "msg_c2", Model: "claude-opus-5-5", Content: []modeltest.Block{modeltest.Text("Opus here.")},
			Usage: modeltest.Usage{InputTokens: 800, OutputTokens: 3}}),
	)
	t.Cleanup(claude.Close)

	// int's sous-chef environment on 2026-10-19 (gcloud run services describe
	// sous-chef), the model service's URL replaced by the fake's.
	envMap := map[string]string{
		"DATABASE_URL":          "postgres://unused",
		"SERVICE_KEYS":          "k",
		"ANTHROPIC_API_KEY":     "sk-test",
		"ANTHROPIC_BASE_URL":    claude.URL,
		"DEFAULT_ROUTE":         "blueberry-muffin",
		"MODEL_ROUTE_ALLOWLIST": "default,opus,blueberry-muffin",
		"MODEL_CALL_TIMEOUT":    "60s",
		"MAX_CONCURRENT_TURNS":  "8",
		"TURN_TTL":              "10m",
		"MODEL_ROUTES": `{"blueberry-muffin":{"auth":"id-token","fallback":"default","fast_finish":"on","kind":"messages-compatible","max_tokens":1024,"model":"sous-chef","prime":"on","report_as":"blueberry-muffin","stream":false,"temperature":0.3,"url":"` + inhouse.URL + `"},` +
			`"default":{"auth":"api-key","effort":"medium","kind":"anthropic","max_tokens":8192,"model":"claude-sonnet-5-5"},` +
			`"opus":{"auth":"api-key","effort":"medium","kind":"anthropic","max_tokens":8192,"model":"claude-opus-5-5"}}`,
	}
	cfg, err := config.Load(func(k string) string { return envMap[k] })
	if err != nil {
		t.Fatalf("config.Load: %v", err)
	}
	opts := model.FromConfig(cfg, func(k string) string { return envMap[k] })
	opts.Tokens = constToken("id-token-for-the-model-service")
	tbl, err := model.NewTable(opts)
	if err != nil {
		t.Fatalf("NewTable: %v", err)
	}
	var logMu sync.Mutex
	var logs []string
	logf := func(f string, a ...any) {
		logMu.Lock()
		logs = append(logs, fmt.Sprintf(f, a...))
		logMu.Unlock()
	}
	st := newMemStore()
	clock := time.Date(2026, 10, 19, 12, 0, 0, 0, time.UTC)
	st.now = func() time.Time { return clock }
	runner, err := New(Deps{
		Store: st, Manifests: memLoader{byHash: map[string]*registry.Compiled{c.Hash: c}}, Models: tbl,
		Usage: usage.NewRecorder(st, nil), Logf: logf, Now: func() time.Time { return clock },
		Primer: model.NewPrimer(model.PrimerOptions{Logf: logf}),
	})
	if err != nil {
		t.Fatal(err)
	}
	req := func(text, route string) api.TurnRequest {
		return api.TurnRequest{SiteID: "site-1", StoreID: "store-1", ManifestHash: c.Hash, BuildID: "abc123", Model: route,
			Context: json.RawMessage(`{"page":"home","sections":[{"id":"hero-1","type":"hero"}]}`), Text: text, UserEmail: "owner@shop.com"}
	}
	run := func(tr api.Turn) []api.CollectedEvent {
		col := &api.Collector{}
		if err := tr.Run(context.Background(), col); err != nil {
			logf("test: Run: %v", err)
		}
		return col.Events()
	}
	var turns []map[string]any
	record := func(name string, evs []api.CollectedEvent, from transcriptMarks) {
		inhouse.settle(t)
		turns = append(turns, map[string]any{
			"name":    name,
			"events":  sanitizeEvents(t, evs),
			"inhouse": inhouse.since(from.inhouse, false),
			"claude":  capturedSince(claude.Requests(), from.claude),
		})
	}
	marks := func() transcriptMarks {
		return transcriptMarks{inhouse: inhouse.count(), claude: len(claude.Requests())}
	}

	// 1. int's default route: an op, a clean batch, fast finish.
	inhouse.script(okStep(`{"id":"msg_s1","type":"message","role":"assistant","model":"sous-chef","content":[{"type":"text","text":"Updating the headline."},{"type":"tool_use","id":"call_s1","name":"set_fields","input":{"target":{"kind":"section","page":"home","sectionId":"hero-1"},"values":{"headline":"Fresh cookies"}}}],"stop_reason":"tool_use","usage":{"input_tokens":15100,"output_tokens":41,"cache_read_input_tokens":0}}`))
	m := marks()
	tr, err := runner.Start(context.Background(), req("make the hero headline say Fresh cookies", ""))
	if err != nil {
		t.Fatalf("Start 1: %v", err)
	}
	evs := run(tr)
	cont, err := runner.Continue(context.Background(), tr.ID(), api.ToolResultsRequest{SiteID: "site-1", StoreID: "store-1",
		Results: []api.ToolResult{{ToolCallID: "call_s1", OK: true, Written: []string{"sections[0].headline"}}}})
	if err != nil {
		t.Fatalf("Continue 1: %v", err)
	}
	evs = append(evs, run(cont)...)
	record("default route: op, clean batch, fast finish", evs, m)

	// 2. int's default route: Ollama's parser 500 on every attempt → the SDK's
	// two retries, then Claude (model_error).
	parser500 := `{"type":"error","error":{"type":"api_error","message":"tool 'set_fieldz' not found"},"request_id":"req_ollama"}`
	inhouse.script(errStep(500, parser500), errStep(500, parser500), errStep(500, parser500))
	m = marks()
	tr, err = runner.Start(context.Background(), req("now make it bigger", ""))
	if err != nil {
		t.Fatalf("Start 2: %v", err)
	}
	record("default route: parser 500 on every attempt, retried, escalated", run(tr), m)

	// 3. a turn naming opus.
	m = marks()
	tr, err = runner.Start(context.Background(), req("what can you change on this page?", "opus"))
	if err != nil {
		t.Fatalf("Start 3: %v", err)
	}
	record("opus", run(tr), m)

	st.mu.Lock()
	usageRows := append([]store.UsageRow(nil), st.usage...)
	st.mu.Unlock()
	var rows []map[string]any
	for _, r := range usageRows {
		rows = append(rows, map[string]any{"operation": r.Operation, "model": r.Model, "status": r.Status, "stop_reason": r.StopReason,
			"input_tokens": r.InputTokens, "output_tokens": r.OutputTokens, "cache_read_tokens": r.CacheReadTokens})
	}
	logMu.Lock()
	var souschefLines []string
	for _, l := range logs {
		if strings.HasPrefix(l, "souschef:") {
			souschefLines = append(souschefLines, l)
		}
	}
	logMu.Unlock()
	inhouse.settle(t)
	transcript := map[string]any{
		"about": "TestDefaultConfig_UnchangedFromToday (service/internal/loop/default_config_test.go): int's configuration today, three turns; every model request, event, usage row",
		"turns": turns,
		// The Primer's requests, all turns', sorted: when each lands is the
		// Primer's timing, what it sends is the transcript.
		"primes":          inhouse.since(0, true),
		"usage_rows":      rows,
		"souschef_lines":  souschefLines,
		"inhouse_unasked": inhouse.unused(),
	}
	got, err := json.MarshalIndent(transcript, "", " ")
	if err != nil {
		t.Fatal(err)
	}
	got = append(got, '\n')
	golden := filepath.Join("testdata", "default_config.golden.json")
	if os.Getenv("SOUS_CHEF_WRITE_GOLDEN") == "1" {
		if err := os.MkdirAll("testdata", 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(golden, got, 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %s (%d bytes)", golden, len(got))
		return
	}
	want, err := os.ReadFile(golden)
	if err != nil {
		t.Fatalf("golden: %v", err)
	}
	if !bytes.Equal(got, want) {
		out := filepath.Join(t.TempDir(), "default_config.got.json")
		_ = os.WriteFile(out, got, 0o644)
		t.Fatalf("the default configuration's transcript differs from the code before this change (%s): first difference at %s; got written to %s",
			golden, firstDiff(got, want), out)
	}
	// The transcript's own sanity, so a golden that recorded a broken run cannot pass.
	if n := len(turns[1]["inhouse"].([]map[string]any)); n != 1+model.DefaultMaxRetries {
		t.Errorf("parser 500 on the default route: %d attempts, want %d (the SDK's retries, as today)", n, 1+model.DefaultMaxRetries)
	}
	if len(souschefLines) != 0 {
		t.Errorf("souschef lines on the default configuration: %q", souschefLines)
	}
}

type transcriptMarks struct{ inhouse, claude int }

type constToken string

func (c constToken) IDToken(context.Context, string) (string, error) { return string(c), nil }

// scriptedOllama is the in-house model server: real calls take the next
// scripted step; primes (max_tokens 1, the Primer's) are answered with one
// token and recorded beside them, so their timing never shifts the script.
type scriptedOllama struct {
	*httptest.Server
	mu    sync.Mutex
	steps []ollamaStep
	next  int
	seen  []map[string]any
	last  time.Time
}

type ollamaStep struct {
	status int
	body   string
}

func okStep(body string) ollamaStep            { return ollamaStep{status: 200, body: body} }
func errStep(code int, body string) ollamaStep { return ollamaStep{status: code, body: body} }

func newScriptedOllama(t *testing.T) *scriptedOllama {
	s := &scriptedOllama{}
	s.Server = httptest.NewServer(http.HandlerFunc(s.handle))
	t.Cleanup(s.Close)
	return s
}

func (s *scriptedOllama) script(steps ...ollamaStep) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.steps = append(s.steps, steps...)
}

func (s *scriptedOllama) handle(w http.ResponseWriter, r *http.Request) {
	raw, _ := io.ReadAll(r.Body)
	var body struct {
		MaxTokens int `json:"max_tokens"`
	}
	_ = json.Unmarshal(raw, &body)
	prime := body.MaxTokens == 1
	s.mu.Lock()
	s.seen = append(s.seen, captured(r, raw, prime))
	s.last = time.Now()
	step := ollamaStep{status: 500, body: `{"type":"error","error":{"type":"api_error","message":"script exhausted"}}`}
	switch {
	case prime:
		step = okStep(`{"id":"msg_prime","type":"message","role":"assistant","model":"sous-chef","content":[{"type":"text","text":"x"}],"stop_reason":"max_tokens","usage":{"input_tokens":1030,"output_tokens":1,"cache_read_input_tokens":15000}}`)
	case s.next < len(s.steps):
		step = s.steps[s.next]
		s.next++
	}
	s.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(step.status)
	_, _ = io.WriteString(w, step.body)
}

func (s *scriptedOllama) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.seen)
}

func (s *scriptedOllama) unused() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.steps) - s.next
}

// settle waits until the server has seen no request for a while: the
// Primer's requests go out after turn.done, off the turn's path.
func (s *scriptedOllama) settle(t *testing.T) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		s.mu.Lock()
		quiet := time.Since(s.last) > time.Second
		s.mu.Unlock()
		if quiet {
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
}

// since lists the requests seen from index from on, real calls or primes;
// primes are sorted by body (two may race).
func (s *scriptedOllama) since(from int, primes bool) []map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := []map[string]any{}
	for _, c := range s.seen[from:] {
		if c["prime"].(bool) == primes {
			c = maps.Clone(c)
			delete(c, "prime")
			out = append(out, c)
		}
	}
	if primes {
		sort.Slice(out, func(i, j int) bool { return out[i]["body_sha256"].(string) < out[j]["body_sha256"].(string) })
	}
	return out
}

func captured(r *http.Request, raw []byte, prime bool) map[string]any {
	return map[string]any{"method": r.Method, "path": r.URL.Path, "query": r.URL.RawQuery, "headers": headersOf(r.Header),
		"body_bytes": len(raw), "body_sha256": fmt.Sprintf("%x", sha256.Sum256(raw)), "shape": bodyShape(raw), "prime": prime}
}

func capturedSince(reqs []modeltest.Captured, from int) []map[string]any {
	out := []map[string]any{}
	for _, c := range reqs[from:] {
		out = append(out, map[string]any{"path": c.Path, "headers": headersOf(c.Header),
			"body_bytes": len(c.Raw), "body_sha256": fmt.Sprintf("%x", sha256.Sum256(c.Raw)), "shape": bodyShape(c.Raw)})
	}
	return out
}

// headersOf is every request header but the three that name the machine the
// test runs on (the SDK's X-Stainless-Os/-Arch/-Runtime-Version).
func headersOf(h http.Header) map[string]string {
	hdr := map[string]string{}
	for k, v := range h {
		switch k = strings.ToLower(k); k {
		case "x-stainless-os", "x-stainless-arch", "x-stainless-runtime-version":
			continue
		}
		hdr[k] = strings.Join(v, ", ")
	}
	return hdr
}

// bodyShape is a readable digest of a Messages request body (the sha256
// beside it is the byte-exact check): its members, the sampling and
// thinking fields, the system blocks' sizes, the tools' names and the
// messages' roles and block types.
func bodyShape(raw []byte) map[string]any {
	var b map[string]any
	if json.Unmarshal(raw, &b) != nil {
		return map[string]any{"json": false}
	}
	keys := make([]string, 0, len(b))
	for k := range b {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	shape := map[string]any{"members": keys}
	for _, k := range []string{"model", "max_tokens", "temperature", "top_p", "stream", "thinking", "tool_choice", "output_config", "fallbacks", "cache_control"} {
		if v, ok := b[k]; ok {
			shape[k] = v
		}
	}
	if sys, ok := b["system"].([]any); ok {
		var sizes []int
		for _, blk := range sys {
			m, _ := blk.(map[string]any)
			txt, _ := m["text"].(string)
			sizes = append(sizes, len(txt))
		}
		shape["system_chars"] = sizes
	}
	if tools, ok := b["tools"].([]any); ok {
		var names []string
		for _, tl := range tools {
			m, _ := tl.(map[string]any)
			n, _ := m["name"].(string)
			names = append(names, n)
		}
		shape["tools"] = names
	}
	if msgs, ok := b["messages"].([]any); ok {
		var turns []string
		for _, mm := range msgs {
			m, _ := mm.(map[string]any)
			role, _ := m["role"].(string)
			var types []string
			if blocks, ok := m["content"].([]any); ok {
				for _, blk := range blocks {
					bm, _ := blk.(map[string]any)
					tp, _ := bm["type"].(string)
					types = append(types, tp)
				}
			}
			turns = append(turns, role+":"+strings.Join(types, "+"))
		}
		shape["messages"] = turns
	}
	return shape
}

// sanitizeEvents keeps every event's kind and data but the random ids.
func sanitizeEvents(t *testing.T, evs []api.CollectedEvent) []any {
	t.Helper()
	out := []any{}
	for _, e := range evs {
		var d any
		raw, err := json.Marshal(e.Data)
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(raw, &d); err != nil {
			t.Fatal(err)
		}
		out = append(out, map[string]any{"event": e.Event, "data": dropIDs(d)})
	}
	return out
}

func dropIDs(v any) any {
	switch x := v.(type) {
	case map[string]any:
		out := map[string]any{}
		for k, val := range x {
			if k == "turnId" || k == "conversationId" {
				continue
			}
			out[k] = dropIDs(val)
		}
		return out
	case []any:
		for i := range x {
			x[i] = dropIDs(x[i])
		}
		return x
	}
	return v
}

func firstDiff(a, b []byte) string {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			lo := i - 120
			if lo < 0 {
				lo = 0
			}
			hi := i + 120
			if hi > n {
				hi = n
			}
			return fmt.Sprintf("byte %d: got …%s… want …%s…", i, a[lo:min(hi, len(a))], b[lo:min(hi, len(b))])
		}
	}
	return fmt.Sprintf("length %d vs %d", len(a), len(b))
}
