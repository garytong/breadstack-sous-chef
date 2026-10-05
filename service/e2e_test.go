// End-to-end test (README §10): the real HTTP server, real Postgres (via
// storetest: TEST_DATABASE_URL, else a docker postgres:16-alpine, else
// skip), the manifest snapshot as the launcher's manifest, an httptest fake
// model speaking /v1/messages with scripted tool calls, and an httptest fake
// thermometer ingest. No build tag: `go test ./...` runs it and skips
// cleanly when no database can be had; `make e2e` runs it against the
// docker-compose Postgres.
package e2e

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/breadstack/sous-chef/internal/api"
	"github.com/breadstack/sous-chef/internal/auth"
	"github.com/breadstack/sous-chef/internal/config"
	"github.com/breadstack/sous-chef/internal/loop"
	"github.com/breadstack/sous-chef/internal/manifest"
	"github.com/breadstack/sous-chef/internal/model"
	"github.com/breadstack/sous-chef/internal/model/modeltest"
	"github.com/breadstack/sous-chef/internal/registry"
	"github.com/breadstack/sous-chef/internal/store"
	"github.com/breadstack/sous-chef/internal/store/storetest"
	"github.com/breadstack/sous-chef/internal/usage"
)

func TestMain(m *testing.M) { os.Exit(storetest.Main(m)) }

const serviceKey = "e2e-service-key"

// ingest is the fake thermometer: it records every /api/ingest/usage batch.
type ingest struct {
	mu   sync.Mutex
	rows []map[string]any
	keys []string
}

func (in *ingest) handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/ingest/usage" || r.Method != http.MethodPost {
			http.NotFound(w, r)
			return
		}
		var body struct {
			Usage []map[string]any `json:"usage"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		in.mu.Lock()
		in.rows = append(in.rows, body.Usage...)
		in.keys = append(in.keys, r.Header.Get("X-API-Key"))
		in.mu.Unlock()
		w.WriteHeader(http.StatusAccepted)
	})
}

func (in *ingest) count() int {
	in.mu.Lock()
	defer in.mu.Unlock()
	return len(in.rows)
}

type env struct {
	t       *testing.T
	st      *store.Store
	fake    *modeltest.Server
	ingest  *ingest
	usage   *usage.Recorder
	server  *httptest.Server
	runner  *loop.Runner
	body    []byte
	hash    string
	client  *http.Client
	manifes *registry.Registry

	logMu sync.Mutex
	logs  []string // every runner log line (WARNING checks)
}

func (e *env) logf(f string, a ...any) {
	e.logMu.Lock()
	e.logs = append(e.logs, fmt.Sprintf(f, a...))
	e.logMu.Unlock()
	e.t.Logf(f, a...)
}

// logged reports whether a runner log line since index from contains s;
// logCount is the index to pass later.
func (e *env) logged(from int, s string) bool {
	e.logMu.Lock()
	defer e.logMu.Unlock()
	for _, l := range e.logs[from:] {
		if strings.Contains(l, s) {
			return true
		}
	}
	return false
}

func (e *env) logCount() int {
	e.logMu.Lock()
	defer e.logMu.Unlock()
	return len(e.logs)
}

func newEnv(t *testing.T) *env { return newEnvWith(t, nil) }

// newEnvWith is newEnv with a hook on the runner's Deps (RotateAt, say).
func newEnvWith(t *testing.T, mutate func(*loop.Deps)) *env {
	t.Helper()
	st := storetest.Open(t) // skips when no database is available
	body, err := os.ReadFile("testdata/manifest.snapshot.json")
	if err != nil {
		t.Fatal(err)
	}
	hash, err := manifest.HashBytes(body)
	if err != nil {
		t.Fatal(err)
	}
	fake := modeltest.New()
	t.Cleanup(fake.Close)
	in := &ingest{}
	ingestSrv := httptest.NewServer(in.handler())
	t.Cleanup(ingestSrv.Close)

	// The same configuration path as main: env → config → model table.
	envv := map[string]string{
		"DATABASE_URL": "unused-here", "SERVICE_KEYS": serviceKey, "ANTHROPIC_API_KEY": "sk-e2e",
		"ANTHROPIC_BASE_URL": fake.URL, "MODEL_CALL_TIMEOUT": "5s", "MAX_CONCURRENT_TURNS": "4",
		"THERMOMETER_API_URL": ingestSrv.URL, "THERMOMETER_API_KEY": "therm-key",
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
	reg := registry.New(st)
	shipper := usage.NewShipperFromEnv(getenv)
	if shipper == nil {
		t.Fatal("shipper must be on")
	}
	rec := usage.NewRecorder(st, shipper)
	e := &env{t: t, st: st, fake: fake, ingest: in, usage: rec, body: body, hash: hash, manifes: reg}
	deps := loop.Deps{Store: st, Manifests: reg, Models: models, Usage: rec,
		TurnTTL: cfg.TurnTTL, MaxConcurrent: cfg.MaxConcurrentTurns, Logf: e.logf}
	if mutate != nil {
		mutate(&deps)
	}
	runner, err := loop.New(deps)
	if err != nil {
		t.Fatal(err)
	}
	_, defaultOK := cfg.ModelRoutes[cfg.DefaultRoute]
	srv := api.New(auth.NewKeys(cfg.ServiceKeys), api.Options{BuildID: "e2e", Routes: cfg.RouteNames(), RouteInfo: cfg.RouteInfo(), DefaultRoute: cfg.DefaultRoute, DefaultRouteConfigured: defaultOK, Heartbeat: -1, Logf: t.Logf},
		api.Deps{Manifests: reg, HashManifest: manifest.HashBytes, Turns: runner, Conversations: loop.Conversations{St: st, Logf: t.Logf}, Feedback: &loop.Feedback{St: st, Logf: t.Logf}, OptOut: loop.OptOut{St: st}, DB: st})
	hs := httptest.NewServer(srv.Handler())
	t.Cleanup(hs.Close)
	e.server, e.runner, e.client = hs, runner, hs.Client()
	return e
}

// do sends one authenticated request and returns status, body, headers.
func (e *env) do(method, path string, body []byte, hdr ...string) (int, []byte, http.Header) {
	e.t.Helper()
	req, err := http.NewRequest(method, e.server.URL+path, bytes.NewReader(body))
	if err != nil {
		e.t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer "+serviceKey)
	req.Header.Set("X-User-Email", "owner@shop.com")
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	for i := 0; i+1 < len(hdr); i += 2 {
		req.Header.Set(hdr[i], hdr[i+1])
	}
	res, err := e.client.Do(req)
	if err != nil {
		e.t.Fatal(err)
	}
	defer res.Body.Close()
	b, _ := io.ReadAll(res.Body)
	return res.StatusCode, b, res.Header
}

// post is do for goroutines (no t.Fatal): status and body, or 0 and the
// transport error's text.
func (e *env) post(path string, body []byte) (int, []byte) {
	req, err := http.NewRequest("POST", e.server.URL+path, bytes.NewReader(body))
	if err != nil {
		return 0, []byte(err.Error())
	}
	req.Header.Set("Authorization", "Bearer "+serviceKey)
	req.Header.Set("X-User-Email", "owner@shop.com")
	req.Header.Set("Content-Type", "application/json")
	res, err := e.client.Do(req)
	if err != nil {
		return 0, []byte(err.Error())
	}
	defer res.Body.Close()
	b, _ := io.ReadAll(res.Body)
	return res.StatusCode, b
}

type event struct {
	Type string
	Data map[string]any
}

// parseSSE reads `event:`/`data:` frames.
func parseSSE(t *testing.T, body []byte) []event {
	t.Helper()
	var out []event
	var cur event
	sc := bufio.NewScanner(bytes.NewReader(body))
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		switch {
		case strings.HasPrefix(line, "event: "):
			cur = event{Type: strings.TrimPrefix(line, "event: ")}
		case strings.HasPrefix(line, "data: "):
			if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &cur.Data); err != nil {
				t.Fatalf("bad data line %q: %v", line, err)
			}
		case line == "":
			if cur.Type != "" {
				out = append(out, cur)
				cur = event{}
			}
		}
	}
	return out
}

func kinds(evs []event) string {
	parts := make([]string, len(evs))
	for i, e := range evs {
		parts[i] = e.Type
	}
	return strings.Join(parts, " ")
}

func textOf(evs []event) string {
	var sb strings.Builder
	for _, e := range evs {
		if e.Type == "text.delta" {
			sb.WriteString(e.Data["text"].(string))
		}
	}
	return sb.String()
}

func (e *env) turnBody(text string, extra string) []byte {
	return []byte(fmt.Sprintf(`{"siteId":"site-e2e","storeId":"store-e2e","manifestHash":%q,"buildId":"abc123","context":{"page":"home","template":"cozy","sections":[{"id":"hero-1","type":"hero","title":"Hero","enabled":true}]},"text":%q%s}`, e.hash, text, extra))
}

var heroTarget = map[string]any{"kind": "section", "page": "home", "sectionId": "hero-1"}

// heroSrcset is the picker's {width: url} srcset object as the model echoes it
// into set_fields.assets.
var heroSrcset = map[string]any{"480": "https://cdn.example/hero-480.jpg", "960": "https://cdn.example/hero.jpg"}

func TestEndToEnd(t *testing.T) {
	e := newEnv(t)
	ctx := context.Background()

	// --- open routes ---
	status, body, _ := e.do("GET", "/healthz", nil)
	if status != 200 {
		t.Fatalf("healthz %d %s", status, body)
	}
	status, body, _ = e.do("GET", "/readyz", nil)
	if status != 200 || !strings.Contains(string(body), `"db":"ok"`) {
		t.Fatalf("readyz %d %s", status, body)
	}
	status, body, _ = e.do("GET", "/version", nil)
	if status != 200 || !strings.Contains(string(body), `"build_id":"e2e"`) || !strings.Contains(string(body), `"opus"`) ||
		!strings.Contains(string(body), `"route_info":{"default":{"kind":"anthropic","fields_context":false,"fallback":null,"report_as":null}`) || !strings.Contains(string(body), `"default_route":"default"`) {
		t.Fatalf("version %d %s", status, body)
	}
	// Auth gate.
	req, _ := http.NewRequest("GET", e.server.URL+"/v1/manifests/"+e.hash, nil)
	res, err := e.client.Do(req)
	if err != nil || res.StatusCode != 401 {
		t.Fatalf("no key → %v %v", res.StatusCode, err)
	}
	res.Body.Close()

	// --- unknown hash → 409 unknown_manifest carrying the hash ---
	status, body, _ = e.do("POST", "/v1/turns", e.turnBody("hello", ""))
	var errBody map[string]any
	_ = json.Unmarshal(body, &errBody)
	if status != 409 || errBody["error"] != "unknown_manifest" || errBody["hash"] != e.hash {
		t.Fatalf("unknown manifest → %d %s", status, body)
	}
	// --- storeId is required (the trace opt-out is keyed by it) ---
	status, body, _ = e.do("POST", "/v1/turns", []byte(fmt.Sprintf(`{"siteId":"site-e2e","manifestHash":%q,"buildId":"abc123","context":{"page":"home"},"text":"hello"}`, e.hash)))
	if status != 400 || !strings.Contains(string(body), "storeId is required") {
		t.Fatalf("missing storeId → %d %s", status, body)
	}
	// --- 256 KB body cap → 413 (300 KB is under the old 1 MiB cap) ---
	status, body, _ = e.do("POST", "/v1/turns", e.turnBody("hello", `,"pad":"`+strings.Repeat("x", 300*1024)+`"`))
	if status != 413 || !strings.Contains(string(body), "too large") {
		t.Fatalf("oversized turn → %d %s", status, body)
	}

	// --- register: 201 with counts, then 200; GET with index ---
	status, body, _ = e.do("PUT", "/v1/manifests/"+e.hash, e.body, "X-Launcher-Build", "abc123")
	var reg map[string]any
	_ = json.Unmarshal(body, &reg)
	if status != 201 || reg["fields"] != 2094.0 || reg["lists"] != 69.0 || reg["sections"] != 19.0 || reg["globalPanels"] != 12.0 || reg["pages"] != 17.0 || reg["details"] != 11.0 || reg["diff"] != nil {
		t.Fatalf("register → %d %s", status, body)
	}
	status, _, _ = e.do("PUT", "/v1/manifests/"+e.hash, e.body, "X-Launcher-Build", "abc123")
	if status != 200 {
		t.Fatalf("re-register → %d", status)
	}
	// POST /v1/manifests computes the same hash server-side (the launcher
	// cannot reproduce the Go canonicalisation from JS).
	status, body, _ = e.do("POST", "/v1/manifests", e.body, "X-Launcher-Build", "abc123")
	var posted map[string]any
	_ = json.Unmarshal(body, &posted)
	if status != 200 || posted["hash"] != e.hash || posted["fields"] != 2094.0 {
		t.Fatalf("POST /v1/manifests → %d %s", status, body)
	}
	wrong := strings.Repeat("0", 64)
	status, body, _ = e.do("PUT", "/v1/manifests/"+wrong, e.body)
	if status != 400 || !strings.Contains(string(body), "hash_mismatch") {
		t.Fatalf("mismatch → %d %s", status, body)
	}
	status, body, _ = e.do("GET", "/v1/manifests/"+e.hash, nil)
	var info map[string]any
	_ = json.Unmarshal(body, &info)
	if status != 200 || !strings.Contains(info["index"].(string), "s:heroBanner") || info["counts"].(map[string]any)["fields"] != 2094.0 || info["body"] != nil {
		t.Fatalf("get manifest → %d %.200s", status, body)
	}

	// --- turn 1: set_fields → tool.call + turn.waiting → tool-results → text → done ---
	e.fake.Append(
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{
			modeltest.Text("Making the headline bigger."),
			modeltest.ToolUse("op1", "set_fields", map[string]any{"target": heroTarget, "values": map[string]any{"headlineSize": "xl"}}),
		}, Usage: modeltest.Usage{InputTokens: 120, OutputTokens: 30, CacheCreationInputTokens: 9000}}),
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("Done — the headline is larger.")},
			Usage: modeltest.Usage{InputTokens: 10, OutputTokens: 12, CacheReadInputTokens: 9000}}),
	)
	status, body, hdr := e.do("POST", "/v1/turns", e.turnBody("make the hero title bigger", `,"model":"default"`))
	if status != 200 || !strings.HasPrefix(hdr.Get("Content-Type"), "text/event-stream") {
		t.Fatalf("turn 1 → %d %s %s", status, hdr.Get("Content-Type"), body)
	}
	evs := parseSSE(t, body)
	if kinds(evs) != "text.delta text.delta tool.call turn.waiting" {
		t.Fatalf("turn 1 events = %s", kinds(evs))
	}
	turnID := evs[0].Data["turnId"].(string)
	call := evs[2].Data
	if call["name"] != "set_fields" || call["id"] != "op1" || call["seq"] != 3.0 || call["input"].(map[string]any)["values"].(map[string]any)["headlineSize"] != "xl" {
		t.Fatalf("tool.call = %v", call)
	}
	if ids, _ := json.Marshal(evs[3].Data["toolCallIds"]); string(ids) != `["op1"]` {
		t.Fatalf("waiting = %v", evs[3].Data)
	}
	row, err := e.st.GetTurn(ctx, turnID)
	if err != nil || row.Status != store.TurnWaitingTools || row.UserEmail != "owner@shop.com" || row.Model != "claude-sonnet-5-5" {
		t.Fatalf("turn row = %+v %v", row, err)
	}

	status, body, _ = e.do("POST", "/v1/turns/"+turnID+"/tool-results", []byte(`{"results":[{"toolCallId":"op1","ok":true,"written":["headlineSize"],"sectionId":"hero-1"}]}`))
	if status != 200 {
		t.Fatalf("tool-results → %d %s", status, body)
	}
	evs = parseSSE(t, body)
	if !strings.HasSuffix(kinds(evs), "turn.done") || textOf(evs) != "Done — the headline is larger." {
		t.Fatalf("continuation events = %s text=%q", kinds(evs), textOf(evs))
	}
	done := evs[len(evs)-1].Data
	if done["stop"] != "end_turn" || done["seq"].(float64) <= 4 || done["turnId"] != turnID {
		t.Fatalf("done = %v", done)
	}
	u := done["usage"].(map[string]any)
	if u["calls"] != 2.0 || u["input_tokens"] != 130.0 || u["output_tokens"] != 42.0 || u["cache_read_input_tokens"] != 9000.0 || u["cache_creation_input_tokens"] != 9000.0 {
		t.Fatalf("usage = %v", u)
	}
	// Second continuation → 409.
	status, body, _ = e.do("POST", "/v1/turns/"+turnID+"/tool-results", []byte(`{"results":[{"toolCallId":"op1","ok":true}]}`))
	if status != 409 || !strings.Contains(string(body), "turn_not_waiting") {
		t.Fatalf("after done → %d %s", status, body)
	}

	// --- turn 2: pick_asset → suspended → resume with asset → set_fields (URL in values, alt/srcset in assets) → done ---
	e.fake.Append(
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{
			modeltest.ToolUse("pa1", "pick_asset", map[string]any{"purpose": "Hero background image", "asset_type": "image"}),
		}}),
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{
			modeltest.ToolUse("sf1", "set_fields", map[string]any{"target": heroTarget,
				"values": map[string]any{"bgImage": "https://cdn.example/hero.jpg"},
				"assets": map[string]any{"bgImage": map[string]any{"alt": "cookies on a tray", "srcset": heroSrcset}}}),
		}}),
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("New background applied.")}}),
	)
	status, body, _ = e.do("POST", "/v1/turns", e.turnBody("change the hero background", ""))
	evs = parseSSE(t, body)
	if status != 200 || kinds(evs) != "turn.suspended" {
		t.Fatalf("turn 2 → %d %s", status, kinds(evs))
	}
	turn2 := evs[0].Data["turnId"].(string)
	if evs[0].Data["reason"] != "pick_asset" || evs[0].Data["toolCallId"] != "pa1" || evs[0].Data["purpose"] != "Hero background image" || evs[0].Data["assetType"] != "image" {
		t.Fatalf("suspended = %v", evs[0].Data)
	}
	if row, _ := e.st.GetTurn(ctx, turn2); row.Status != store.TurnSuspended {
		t.Fatalf("status = %s", row.Status)
	}
	// The browser picker reports srcset as a {width: url} OBJECT (compactSrcsetObject).
	status, body, _ = e.do("POST", "/v1/turns/"+turn2+"/tool-results", []byte(`{"results":[{"toolCallId":"pa1","ok":true,"asset":{"public_url":"https://cdn.example/hero.jpg","alt":"cookies on a tray","srcset":{"480":"https://cdn.example/hero-480.jpg","960":"https://cdn.example/hero.jpg"}}}]}`))
	evs = parseSSE(t, body)
	if status != 200 || kinds(evs) != "tool.call turn.waiting" {
		t.Fatalf("resume → %d %s %s", status, kinds(evs), body)
	}
	in := evs[0].Data["input"].(map[string]any)
	bgAsset := in["assets"].(map[string]any)["bgImage"].(map[string]any)
	if in["values"].(map[string]any)["bgImage"] != "https://cdn.example/hero.jpg" || bgAsset["alt"] != "cookies on a tray" {
		t.Fatalf("set_fields after pick_asset = %v", in)
	}
	if ss, _ := bgAsset["srcset"].(map[string]any); ss["480"] != "https://cdn.example/hero-480.jpg" || ss["960"] != "https://cdn.example/hero.jpg" {
		t.Fatalf("set_fields assets.bgImage.srcset must be the object: %v", bgAsset["srcset"])
	}
	// And the model saw the object verbatim in pa1's tool_result.
	{
		paReqs := e.fake.Requests()
		msgs := paReqs[len(paReqs)-1].Body["messages"].([]any)
		last := msgs[len(msgs)-1].(map[string]any)
		tr := last["content"].([]any)[0].(map[string]any)
		// tool_result content is either a string or [{type:text,text}].
		content, _ := tr["content"].(string)
		if blocks, ok := tr["content"].([]any); ok && len(blocks) > 0 {
			content, _ = blocks[0].(map[string]any)["text"].(string)
		}
		var seen map[string]any
		if last["role"] != "user" || tr["type"] != "tool_result" || tr["tool_use_id"] != "pa1" || json.Unmarshal([]byte(content), &seen) != nil {
			t.Fatalf("pa1 tool_result = %v", last)
		}
		if ss, _ := seen["srcset"].(map[string]any); seen["public_url"] != "https://cdn.example/hero.jpg" || ss["480"] != "https://cdn.example/hero-480.jpg" || ss["960"] != "https://cdn.example/hero.jpg" {
			t.Fatalf("model-facing asset = %v", seen)
		}
	}
	// JSON fallback for the last leg.
	status, body, hdr = e.do("POST", "/v1/turns/"+turn2+"/tool-results", []byte(`{"results":[{"toolCallId":"sf1","ok":true,"written":["bgImage"]}]}`), "Accept", "application/json")
	if status != 200 || !strings.HasPrefix(hdr.Get("Content-Type"), "application/json") {
		t.Fatalf("json leg → %d %s", status, hdr.Get("Content-Type"))
	}
	var envlp struct {
		TurnID string `json:"turnId"`
		Events []struct {
			Event string         `json:"event"`
			Data  map[string]any `json:"data"`
		} `json:"events"`
	}
	if err := json.Unmarshal(body, &envlp); err != nil || envlp.TurnID != turn2 || len(envlp.Events) == 0 || envlp.Events[len(envlp.Events)-1].Event != "turn.done" {
		t.Fatalf("json envelope = %s (%v)", body, err)
	}
	if envlp.Events[len(envlp.Events)-1].Data["usage"].(map[string]any)["calls"] != 3.0 {
		t.Fatalf("turn 2 calls = %v", envlp.Events[len(envlp.Events)-1].Data["usage"])
	}

	// --- every model call carried identical tools + system bytes (the cached prefix) ---
	reqs := e.fake.Requests()
	if len(reqs) != 5 {
		t.Fatalf("model calls = %d, want 5", len(reqs))
	}
	first := prefixBytes(t, reqs[0])
	for i, r := range reqs[1:] {
		if prefixBytes(t, r) != first {
			t.Fatalf("request %d: tools+system differ from request 0", i+1)
		}
		if r.Header.Get("X-Api-Key") != "sk-e2e" || !r.Stream {
			t.Fatalf("request %d: key=%q stream=%v", i+1, r.Header.Get("X-Api-Key"), r.Stream)
		}
	}
	// Request rules: tool_choice auto, effort medium, no thinking param, 2 cached system blocks, strict tools.
	b0 := reqs[0].Body
	if b0["tool_choice"].(map[string]any)["type"] != "auto" || b0["thinking"] != nil || b0["output_config"].(map[string]any)["effort"] != "medium" || b0["model"] != "claude-sonnet-5-5" {
		t.Fatalf("request shape: %v %v %v", b0["tool_choice"], b0["thinking"], b0["output_config"])
	}
	sys := b0["system"].([]any)
	if len(sys) != 2 || sys[0].(map[string]any)["cache_control"] == nil || sys[1].(map[string]any)["cache_control"] == nil {
		t.Fatalf("system blocks = %v", sys)
	}
	// The tool_result for pick_asset carried the asset.
	if !strings.Contains(string(reqs[3].Raw), `public_url`) || !strings.Contains(string(reqs[3].Raw), `cookies on a tray`) {
		t.Fatalf("pick_asset result missing from request 3")
	}
	// Conversation history replayed across turns (turn 2's first request saw turn 1's 4 messages + its own).
	if n := len(reqs[2].Body["messages"].([]any)); n != 5 {
		t.Fatalf("turn 2 replayed %d messages, want 5", n)
	}

	// --- one usage row per model call, shipped to the fake thermometer ---
	rows1, _ := e.st.ListUsageByTurn(ctx, turnID)
	rows2, _ := e.st.ListUsageByTurn(ctx, turn2)
	if len(rows1) != 2 || len(rows2) != 3 {
		t.Fatalf("usage rows = %d + %d", len(rows1), len(rows2))
	}
	if rows1[0].Operation != "chat" || rows1[1].Operation != "tool_turn" || rows1[0].InputTokens != 120 || rows1[0].CacheWriteTokens != 9000 || rows1[0].Status != "ok" || rows1[0].Model != "fake-model" {
		t.Fatalf("usage row 0 = %+v", rows1[0])
	}
	e.usage.Wait()
	if e.ingest.count() != 5 {
		t.Fatalf("thermometer received %d rows, want 5", e.ingest.count())
	}
	e.ingest.mu.Lock()
	r0 := e.ingest.rows[0]
	key := e.ingest.keys[0]
	e.ingest.mu.Unlock()
	if r0["source"] != "sous-chef" || r0["user_email"] != "owner@shop.com" || r0["site_id"] != "site-e2e" || key != "therm-key" || len(r0) != 16 {
		t.Fatalf("ingest row = %v key=%s", r0, key)
	}
	unshipped, _ := e.st.ListUnshippedUsage(ctx, 10)
	if len(unshipped) != 0 {
		t.Fatalf("%d rows still unshipped", len(unshipped))
	}

	// --- traces: both turns recorded; feedback merges ---
	tr, err := e.st.GetTrace(ctx, turnID)
	if err != nil || tr.SiteID != "site-e2e" || !has(tr.Ops, `"name":"set_fields"`) || !has(tr.Events, `"event":"tool.result"`) || !has(tr.Context, `"page":"home"`) {
		t.Fatalf("trace 1 = %+v %v", tr, err)
	}
	status, _, _ = e.do("POST", "/v1/feedback", []byte(fmt.Sprintf(`{"turnId":%q,"kept_paths":["hero.headlineSize"],"saved":true}`, turnID)))
	if status != 204 {
		t.Fatalf("feedback → %d", status)
	}
	tr, _ = e.st.GetTrace(ctx, turnID)
	if !has(tr.Feedback, `"saved":true`) || !has(tr.Feedback, `hero.headlineSize`) {
		t.Fatalf("feedback = %s", tr.Feedback)
	}

	// --- conversation view and end ---
	status, body, _ = e.do("GET", "/v1/conversations/site-e2e", nil)
	var conv struct {
		ConversationID string           `json:"conversationId"`
		Messages       []map[string]any `json:"messages"`
		Turns          []map[string]any `json:"turns"`
	}
	_ = json.Unmarshal(body, &conv)
	if status != 200 || conv.ConversationID != done["conversationId"] || len(conv.Messages) != 10 || len(conv.Turns) != 2 || conv.Turns[0]["status"] != "done" {
		t.Fatalf("conversation → %d msgs=%d turns=%d", status, len(conv.Messages), len(conv.Turns))
	}
	status, _, _ = e.do("DELETE", "/v1/conversations/site-e2e", nil)
	if status != 204 {
		t.Fatalf("end → %d", status)
	}
	status, _, _ = e.do("GET", "/v1/conversations/site-e2e", nil)
	if status != 404 {
		t.Fatalf("after end → %d", status)
	}

	// --- opt-out: a traced store stops being traced; usage still flows ---
	status, _, _ = e.do("PUT", "/v1/stores/store-e2e/trace-optout", nil)
	if status != 204 {
		t.Fatalf("opt-out → %d", status)
	}
	e.fake.Append(modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("Hi.")}}))
	status, body, _ = e.do("POST", "/v1/turns", e.turnBody("hi", ""))
	evs = parseSSE(t, body)
	if status != 200 || !strings.HasSuffix(kinds(evs), "turn.done") {
		t.Fatalf("opted-out turn → %d %s", status, kinds(evs))
	}
	turn3 := evs[0].Data["turnId"].(string)
	if _, err := e.st.GetTrace(ctx, turn3); err == nil {
		t.Fatal("opted-out store was traced")
	}
	if rows, _ := e.st.ListUsageByTurn(ctx, turn3); len(rows) != 1 {
		t.Fatalf("opted-out turn usage rows = %d", len(rows))
	}
	// A fresh conversation after the DELETE.
	if evs[len(evs)-1].Data["conversationId"] == conv.ConversationID {
		t.Fatal("ended conversation reused")
	}
	status, _, _ = e.do("DELETE", "/v1/stores/store-e2e/trace-optout", nil)
	if status != 204 {
		t.Fatalf("clear opt-out → %d", status)
	}

	// --- server tool answered inline, model error surfaces as turn.error ---
	e.fake.Append(
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.ToolUse("lk1", "lookup_fields", map[string]any{"target": heroTarget, "query": "headline"})}}),
		modeltest.Error(529, "overloaded_error", "busy", nil),
	)
	status, body, _ = e.do("POST", "/v1/turns", e.turnBody("what is on the hero", ""))
	evs = parseSSE(t, body)
	if status != 200 || kinds(evs) != "turn.error" || evs[0].Data["code"] != "model_unavailable" {
		t.Fatalf("error turn → %d %s %v", status, kinds(evs), evs)
	}
	reqs = e.fake.Requests()
	if !strings.Contains(string(reqs[len(reqs)-1].Raw), `"tool_use_id":"lk1"`) {
		t.Fatal("lookup_fields result was not fed back inline")
	}
	turn4 := evs[0].Data["turnId"].(string)
	if row, _ := e.st.GetTurn(ctx, turn4); row.Status != store.TurnError {
		t.Fatalf("error turn status = %s", row.Status)
	}
	if rows, _ := e.st.ListUsageByTurn(ctx, turn4); len(rows) != 2 || rows[1].Status != "error" {
		t.Fatalf("error turn usage rows = %+v", rows)
	}

	// --- bad requests ---
	status, body, _ = e.do("POST", "/v1/turns", e.turnBody("x", `,"model":"haiku"`))
	if status != 400 || !strings.Contains(string(body), "model") {
		t.Fatalf("bad route → %d %s", status, body)
	}
	status, _, _ = e.do("POST", "/v1/turns/00000000-0000-0000-0000-000000000000/tool-results", []byte(`{"results":[{"toolCallId":"x","ok":true}]}`))
	if status != 404 {
		t.Fatalf("unknown turn → %d", status)
	}
	if e.runner.Live() != 0 || e.runner.Parked() != 0 {
		t.Fatalf("leaked state: live=%d parked=%d", e.runner.Live(), e.runner.Parked())
	}
}

// TestEndToEnd_TTLReaper expires a waiting turn through the real store.
func TestEndToEnd_TTLReaper(t *testing.T) {
	e := newEnv(t)
	ctx := context.Background()
	if _, _, err := e.manifes.Register(ctx, e.hash, e.body, "v3.0.1"); err != nil {
		t.Fatal(err)
	}
	e.fake.Append(modeltest.Reply(modeltest.Message{Content: []modeltest.Block{
		modeltest.ToolUse("op1", "set_fields", map[string]any{"target": heroTarget, "values": map[string]any{"headline": "x"}}),
	}}))
	status, body, _ := e.do("POST", "/v1/turns", e.turnBody("x", ""))
	evs := parseSSE(t, body)
	if status != 200 || kinds(evs) != "tool.call turn.waiting" {
		t.Fatalf("turn → %d %s", status, kinds(evs))
	}
	turnID := evs[0].Data["turnId"].(string)
	if n, _ := e.runner.Reap(ctx); n != 0 {
		t.Fatal("fresh turn reaped")
	}
	// Move the store clock 11 minutes ahead (TTL 10m).
	e.st.SetClock(func() time.Time { return time.Now().UTC().Add(11 * time.Minute) })
	n, err := e.runner.Reap(ctx)
	if err != nil || n != 1 {
		t.Fatalf("reap = %d %v", n, err)
	}
	row, _ := e.st.GetTurn(ctx, turnID)
	if row.Status != store.TurnExpired || row.PendingToolCalls != nil {
		t.Fatalf("row = %+v", row)
	}
	status, body, _ = e.do("POST", "/v1/turns/"+turnID+"/tool-results", []byte(`{"results":[{"toolCallId":"op1","ok":true}]}`))
	if status != 409 || !strings.Contains(string(body), `"status":"expired"`) {
		t.Fatalf("continue after expiry → %d %s", status, body)
	}
	tr, err := e.st.GetTrace(ctx, turnID)
	if err != nil || !has(tr.Ops, `"status":"expired"`) {
		t.Fatalf("trace = %v %s", err, tr.Ops)
	}
}

// TestEndToEnd_SiteCheckAndSupersede: the proxy's tenant assertions on
// tool-results and feedback (README §1, §11) and the one-live-turn-per-
// conversation rule (README §4) through the real HTTP server and Postgres.
func TestEndToEnd_SiteCheckAndSupersede(t *testing.T) {
	e := newEnv(t)
	ctx := context.Background()
	if _, _, err := e.manifes.Register(ctx, e.hash, e.body, "v3.0.1"); err != nil {
		t.Fatal(err)
	}

	// --- tool-results: wrong siteId → 404 (turn untouched); right siteId+storeId → ok ---
	e.fake.Append(
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{
			modeltest.ToolUse("op1", "set_fields", map[string]any{"target": heroTarget, "values": map[string]any{"headline": "x"}}),
		}}),
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("Done.")}}),
	)
	status, body, _ := e.do("POST", "/v1/turns", e.turnBody("set the headline", ""))
	evs := parseSSE(t, body)
	if status != 200 || kinds(evs) != "tool.call turn.waiting" {
		t.Fatalf("turn → %d %s", status, kinds(evs))
	}
	turnA := evs[0].Data["turnId"].(string)
	status, body, _ = e.do("POST", "/v1/turns/"+turnA+"/tool-results", []byte(`{"siteId":"site-other","results":[{"toolCallId":"op1","ok":true}]}`))
	if status != 404 || !strings.Contains(string(body), "turn_not_found") {
		t.Fatalf("wrong siteId → %d %s", status, body)
	}
	status, body, _ = e.do("POST", "/v1/turns/"+turnA+"/tool-results", []byte(`{"siteId":"site-e2e","storeId":"store-other","results":[{"toolCallId":"op1","ok":true}]}`))
	if status != 404 || !strings.Contains(string(body), "turn_not_found") {
		t.Fatalf("wrong storeId → %d %s", status, body)
	}
	if row, _ := e.st.GetTurn(ctx, turnA); row.Status != store.TurnWaitingTools {
		t.Fatalf("turn after refused continuations = %s", row.Status)
	}
	status, body, _ = e.do("POST", "/v1/turns/"+turnA+"/tool-results", []byte(`{"siteId":"site-e2e","storeId":"store-e2e","results":[{"toolCallId":"op1","ok":true,"written":["headline"]}]}`))
	evs = parseSSE(t, body)
	if status != 200 || !strings.HasSuffix(kinds(evs), "turn.done") {
		t.Fatalf("right siteId → %d %s %s", status, kinds(evs), body)
	}

	// --- feedback: wrong siteId → 404, nothing merged; right → 204 ---
	status, body, _ = e.do("POST", "/v1/feedback", []byte(fmt.Sprintf(`{"turnId":%q,"siteId":"site-other","saved":true}`, turnA)))
	if status != 404 || !strings.Contains(string(body), "turn_not_found") {
		t.Fatalf("feedback wrong siteId → %d %s", status, body)
	}
	if tr, err := e.st.GetTrace(ctx, turnA); err != nil || has(tr.Feedback, "saved") {
		t.Fatalf("feedback merged despite mismatch: %v %s", err, tr.Feedback)
	}
	status, _, _ = e.do("POST", "/v1/feedback", []byte(fmt.Sprintf(`{"turnId":%q,"siteId":"site-e2e","storeId":"store-e2e","saved":true}`, turnA)))
	if status != 204 {
		t.Fatalf("feedback right siteId → %d", status)
	}
	if tr, _ := e.st.GetTrace(ctx, turnA); !has(tr.Feedback, `"saved":true`) || has(tr.Feedback, "siteId") {
		t.Fatalf("feedback = %s", tr.Feedback)
	}

	// --- a Start while the previous turn is suspended supersedes it ---
	e.fake.Append(
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{
			modeltest.ToolUse("pa1", "pick_asset", map[string]any{"purpose": "Hero background image", "asset_type": "image"}),
		}}),
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("Changed the headline instead.")}}),
	)
	status, body, _ = e.do("POST", "/v1/turns", e.turnBody("change the hero background", ""))
	evs = parseSSE(t, body)
	if status != 200 || kinds(evs) != "turn.suspended" {
		t.Fatalf("suspended turn → %d %s", status, kinds(evs))
	}
	turnB := evs[0].Data["turnId"].(string)
	if e.runner.Parked() != 1 {
		t.Fatalf("parked = %d", e.runner.Parked())
	}
	status, body, _ = e.do("POST", "/v1/turns", e.turnBody("actually just change the headline", ""))
	evs = parseSSE(t, body)
	if status != 200 || !strings.HasSuffix(kinds(evs), "turn.done") {
		t.Fatalf("superseding turn → %d %s %s", status, kinds(evs), body)
	}
	turnC := evs[0].Data["turnId"].(string)
	rowB, _ := e.st.GetTurn(ctx, turnB)
	if rowB.Status != store.TurnSuperseded || rowB.Ended == nil || rowB.PendingToolCalls != nil || e.runner.Parked() != 0 {
		t.Fatalf("superseded row = %+v parked=%d", rowB, e.runner.Parked())
	}
	if tr, err := e.st.GetTrace(ctx, turnB); err != nil || !has(tr.Ops, `"status":"superseded"`) {
		t.Fatalf("superseded trace = %v %s", err, tr.Ops)
	}
	// The stale pick arrives: 409 turn_not_waiting status=superseded, nothing appended.
	msgsBefore, _ := e.st.ListMessages(ctx, rowB.ConversationID, 0, 0)
	status, body, _ = e.do("POST", "/v1/turns/"+turnB+"/tool-results", []byte(`{"siteId":"site-e2e","results":[{"toolCallId":"pa1","ok":true,"asset":{"public_url":"https://cdn.example/hero.jpg"}}]}`))
	if status != 409 || !strings.Contains(string(body), "turn_not_waiting") || !strings.Contains(string(body), `"status":"superseded"`) {
		t.Fatalf("stale Continue → %d %s", status, body)
	}
	msgsAfter, _ := e.st.ListMessages(ctx, rowB.ConversationID, 0, 0)
	if len(msgsAfter) != len(msgsBefore) {
		t.Fatalf("stale Continue appended: %d → %d messages", len(msgsBefore), len(msgsAfter))
	}
	// Stored: the superseded batch was answered before turn C's user text.
	tail := msgsAfter[len(msgsAfter)-3]
	if tail.Role != "user" || !has(tail.Content, `"tool_use_id":"pa1"`) || !has(tail.Content, "the user sent a new message") {
		t.Fatalf("superseded repair = %s %s", tail.Role, tail.Content)
	}
	// Turn C's request replayed a valid history; the conversation view lists the status.
	reqs := e.fake.Requests()
	lastReq := reqs[len(reqs)-1].Body["messages"].([]any)
	if n := len(lastReq); n != 8 { // u a u(res) a | u a(pa1) u(superseded) | u
		t.Fatalf("superseding turn replayed %d messages, want 8", n)
	}
	status, body, _ = e.do("GET", "/v1/conversations/site-e2e", nil)
	if status != 200 || !strings.Contains(string(body), `"status":"superseded"`) || !strings.Contains(string(body), turnC) {
		t.Fatalf("conversation view → %d %.300s", status, body)
	}
	if e.runner.Live() != 0 || e.runner.Parked() != 0 {
		t.Fatalf("leaked state: live=%d parked=%d", e.runner.Live(), e.runner.Parked())
	}
}

// TestEndToEnd_SupersedeAcrossConversations: the one-live-turn-per-SITE
// rule (README §4) where the earlier turn is not on the new turn's
// conversation or is still running — the 40-message rotation, DELETE
// /v1/conversations, and a Start while the previous stream is open — through
// the real HTTP server and Postgres. In every case the earlier turn ends
// superseded, its late tool-results are 409 status=superseded, and the next
// turn replays a clean history (no WARNING repair).
func TestEndToEnd_SupersedeAcrossConversations(t *testing.T) {
	e := newEnvWith(t, func(d *loop.Deps) { d.RotateAt = 2 })
	ctx := context.Background()
	if _, _, err := e.manifes.Register(ctx, e.hash, e.body, "v3.0.1"); err != nil {
		t.Fatal(err)
	}
	pick := []byte(`{"siteId":"site-e2e","storeId":"store-e2e","results":[{"toolCallId":"pa1","ok":true,"asset":{"public_url":"https://cdn.example/hero.jpg"}}]}`)
	expect409Superseded := func(turnID string, results []byte) {
		t.Helper()
		status, body, _ := e.do("POST", "/v1/turns/"+turnID+"/tool-results", results)
		if status != 409 || !strings.Contains(string(body), "turn_not_waiting") || !strings.Contains(string(body), `"status":"superseded"`) {
			t.Fatalf("late Continue → %d %s, want 409 turn_not_waiting status=superseded", status, body)
		}
	}

	// --- rotation: turn A suspended on C1 ([u a] = RotateAt); turn B rotates to C2 ---
	e.fake.Append(
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{
			modeltest.ToolUse("pa1", "pick_asset", map[string]any{"purpose": "Hero background image", "asset_type": "image"}),
		}}),
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("Fresh conversation.")}}),
	)
	status, body, _ := e.do("POST", "/v1/turns", e.turnBody("change the hero background", ""))
	evs := parseSSE(t, body)
	if status != 200 || kinds(evs) != "turn.suspended" {
		t.Fatalf("turn A → %d %s", status, kinds(evs))
	}
	turnA := evs[0].Data["turnId"].(string)
	rowA, _ := e.st.GetTurn(ctx, turnA)
	c1 := rowA.ConversationID
	status, body, _ = e.do("POST", "/v1/turns", e.turnBody("new topic", ""))
	evs = parseSSE(t, body)
	if status != 200 || !strings.HasSuffix(kinds(evs), "turn.done") || evs[len(evs)-1].Data["conversationRotated"] != true {
		t.Fatalf("turn B → %d %s %s", status, kinds(evs), body)
	}
	turnB := evs[0].Data["turnId"].(string)
	rowB, _ := e.st.GetTurn(ctx, turnB)
	if rowB.ConversationID == c1 {
		t.Fatal("turn B did not rotate")
	}
	rowA, _ = e.st.GetTurn(ctx, turnA)
	if rowA.Status != store.TurnSuperseded || rowA.Ended == nil || rowA.PendingToolCalls != nil || e.runner.Parked() != 0 {
		t.Fatalf("rotated-away turn A = %+v parked=%d; want superseded", rowA, e.runner.Parked())
	}
	if tr, err := e.st.GetTrace(ctx, turnA); err != nil || !has(tr.Ops, `"status":"superseded"`) {
		t.Fatalf("turn A trace = %v %s", err, tr.Ops)
	}
	c1Before, _ := e.st.ListMessages(ctx, c1, 0, 0)
	expect409Superseded(turnA, pick)
	if c1After, _ := e.st.ListMessages(ctx, c1, 0, 0); len(c1After) != len(c1Before) {
		t.Fatalf("late pick wrote into the rotated-away conversation: %d → %d messages", len(c1Before), len(c1After))
	}

	// --- DELETE: turn C waiting on C2; DELETE supersedes it before any new Start ---
	e.fake.Append(modeltest.Reply(modeltest.Message{Content: []modeltest.Block{
		modeltest.ToolUse("op1", "set_fields", map[string]any{"target": heroTarget, "values": map[string]any{"headline": "x"}}),
	}}))
	status, body, _ = e.do("POST", "/v1/turns", e.turnBody("set the headline", ""))
	evs = parseSSE(t, body)
	if status != 200 || kinds(evs) != "tool.call turn.waiting" {
		t.Fatalf("turn C → %d %s", status, kinds(evs))
	}
	turnC := evs[0].Data["turnId"].(string)
	if status, body, _ = e.do("DELETE", "/v1/conversations/site-e2e", nil); status != 204 {
		t.Fatalf("DELETE → %d %s", status, body)
	}
	rowC, _ := e.st.GetTurn(ctx, turnC)
	if rowC.Status != store.TurnSuperseded || rowC.PendingToolCalls != nil {
		t.Fatalf("turn C after DELETE = %+v; want superseded", rowC)
	}
	c2Before, _ := e.st.ListMessages(ctx, rowC.ConversationID, 0, 0)
	expect409Superseded(turnC, []byte(`{"siteId":"site-e2e","storeId":"store-e2e","results":[{"toolCallId":"op1","ok":true}]}`))
	if c2After, _ := e.st.ListMessages(ctx, rowC.ConversationID, 0, 0); len(c2After) != len(c2Before) || e.runner.Parked() != 0 {
		t.Fatalf("late results wrote into the ended conversation: %d → %d messages, parked=%d", len(c2Before), len(c2After), e.runner.Parked())
	}
	if status, _, _ = e.do("GET", "/v1/conversations/site-e2e", nil); status != 404 {
		t.Fatalf("GET after DELETE → %d", status)
	}

	// --- Start while running: turn D is in the model (delayed reply) when turn E starts ---
	e.fake.Append(
		modeltest.Step{Delay: 1500 * time.Millisecond, Message: &modeltest.Message{Content: []modeltest.Block{
			modeltest.ToolUse("opD", "set_fields", map[string]any{"target": heroTarget, "values": map[string]any{"headline": "d"}}),
		}}},
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("E reply.")}}),
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("F reply.")}}),
	)
	seen := len(e.fake.Requests())
	type reply struct {
		status int
		body   []byte
	}
	dCh := make(chan reply, 1)
	go func() {
		s, b := e.post("/v1/turns", e.turnBody("D text", ""))
		dCh <- reply{s, b}
	}()
	// Wait until D's model call is in flight, then start E.
	deadline := time.Now().Add(5 * time.Second)
	for len(e.fake.Requests()) < seen+1 {
		if time.Now().After(deadline) {
			t.Fatal("turn D never reached the model")
		}
		time.Sleep(10 * time.Millisecond)
	}
	status, body, _ = e.do("POST", "/v1/turns", e.turnBody("E text", ""))
	evs = parseSSE(t, body)
	if status != 200 || !strings.HasSuffix(kinds(evs), "turn.done") {
		t.Fatalf("turn E → %d %s %s", status, kinds(evs), body)
	}
	turnE := evs[0].Data["turnId"].(string)
	d := <-dCh
	if d.status != 200 {
		t.Fatalf("turn D → %d %s", d.status, d.body)
	}
	evsD := parseSSE(t, d.body)
	lastD := evsD[len(evsD)-1]
	// D's late batch is not stored (its compare-and-set append finds E's
	// text on the tail), so no tool.call is streamed for ops the page must
	// not apply; D ends turn_expired naming the supersede.
	if kinds(evsD) != "turn.error" || lastD.Data["code"] != "turn_expired" || !strings.Contains(lastD.Data["message"].(string), "superseded") {
		t.Fatalf("turn D (running when E started) → %s %v; want turn.error turn_expired (superseded), no tool.call", kinds(evsD), lastD.Data)
	}
	turnD := evsD[0].Data["turnId"].(string)
	rowD, _ := e.st.GetTurn(ctx, turnD)
	rowE, _ := e.st.GetTurn(ctx, turnE)
	if rowD.Status != store.TurnSuperseded || rowE.Status != store.TurnDone || rowD.ConversationID != rowE.ConversationID {
		t.Fatalf("rowD=%s rowE=%s (conv %s / %s)", rowD.Status, rowE.Status, rowD.ConversationID, rowE.ConversationID)
	}
	if tr, err := e.st.GetTrace(ctx, turnD); err != nil || !has(tr.Ops, `"status":"superseded"`) {
		t.Fatalf("turn D trace = %v %s", err, tr.Ops)
	}
	expect409Superseded(turnD, []byte(`{"siteId":"site-e2e","storeId":"store-e2e","results":[{"toolCallId":"opD","ok":true}]}`))
	// Stored: u(D) u(E) a(E) — D's batch never landed, nothing to answer.
	msgs, _ := e.st.ListMessages(ctx, rowD.ConversationID, 0, 0)
	var roles []string
	for _, m := range msgs {
		roles = append(roles, m.Role)
		if has(m.Content, "opD") {
			t.Fatalf("the superseded turn's batch was stored: %s", m.Content)
		}
	}
	if strings.Join(roles, " ") != "user user assistant" {
		t.Fatalf("stored roles = %v", roles)
	}
	// Turn F replays cleanly: no WARNING repair, every tool_use answered once.
	mark := e.logCount()
	status, body, _ = e.do("POST", "/v1/turns", e.turnBody("F text", ""))
	evs = parseSSE(t, body)
	if status != 200 || !strings.HasSuffix(kinds(evs), "turn.done") {
		t.Fatalf("turn F → %d %s %s", status, kinds(evs), body)
	}
	if e.logged(mark, "WARNING") {
		t.Fatal("turn F logged a history-repair WARNING: a mid-history dangle survived the running window")
	}
	reqs := e.fake.Requests()
	assertReplayable(t, reqs[len(reqs)-1].Body["messages"].([]any))
	if e.runner.Live() != 0 || e.runner.Parked() != 0 {
		t.Fatalf("leaked state: live=%d parked=%d", e.runner.Live(), e.runner.Parked())
	}
}

// Refuter H4b over HTTP + Postgres: turn A is in the model (delayed) when
// turn B starts, supersedes it and goes waiting_tools on its own batch; A's
// late batch arrives behind B's. A must write nothing (its compare-and-set
// append fails on B's text) and end turn_expired without a tool.call; the
// browser's results for B then land normally. Before round 6 a(opA) was
// stored on top of a(opB) and answered by A, so B's own results conflicted:
// B's tool-results 409 status=error, rowB error, a(opB) dangling mid-history
// with a WARNING on every later turn until rotation.
func TestEndToEnd_LateBatchDoesNotBuryWaitingNewerTurn(t *testing.T) {
	e := newEnv(t)
	ctx := context.Background()
	if _, _, err := e.manifes.Register(ctx, e.hash, e.body, "v3.0.1"); err != nil {
		t.Fatal(err)
	}
	e.fake.Append(
		modeltest.Step{Delay: 1500 * time.Millisecond, Message: &modeltest.Message{Content: []modeltest.Block{
			modeltest.ToolUse("opA", "set_fields", map[string]any{"target": heroTarget, "values": map[string]any{"headline": "a"}}),
		}}},
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{
			modeltest.ToolUse("opB", "set_fields", map[string]any{"target": heroTarget, "values": map[string]any{"headline": "b"}}),
		}}),
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("B done.")}}),
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
			t.Fatal("turn A never reached the model")
		}
		time.Sleep(10 * time.Millisecond)
	}
	status, body, _ := e.do("POST", "/v1/turns", e.turnBody("B text", ""))
	evsB := parseSSE(t, body)
	if status != 200 || kinds(evsB) != "tool.call turn.waiting" {
		t.Fatalf("turn B → %d %s", status, kinds(evsB))
	}
	turnB := evsB[0].Data["turnId"].(string)
	a := <-aCh
	if a.status != 200 {
		t.Fatalf("turn A → %d %s", a.status, a.body)
	}
	evsA := parseSSE(t, a.body)
	lastA := evsA[len(evsA)-1]
	if kinds(evsA) != "turn.error" || lastA.Data["code"] != "turn_expired" || !strings.Contains(lastA.Data["message"].(string), "superseded") {
		t.Fatalf("turn A → %s %v; want turn.error turn_expired (superseded), no tool.call", kinds(evsA), lastA.Data)
	}
	turnA := evsA[0].Data["turnId"].(string)
	rowA, _ := e.st.GetTurn(ctx, turnA)
	rowB, _ := e.st.GetTurn(ctx, turnB)
	if rowA.Status != store.TurnSuperseded || rowB.Status != store.TurnWaitingTools || rowA.ConversationID != rowB.ConversationID {
		t.Fatalf("rowA=%s rowB=%s (conv %s / %s)", rowA.Status, rowB.Status, rowA.ConversationID, rowB.ConversationID)
	}
	rolesOf := func() string {
		msgs, _ := e.st.ListMessages(ctx, rowB.ConversationID, 0, 0)
		var rl []string
		for _, m := range msgs {
			rl = append(rl, m.Role)
			if has(m.Content, "opA") {
				t.Fatalf("the superseded turn's batch was stored: %s", m.Content)
			}
		}
		return strings.Join(rl, " ")
	}
	if got := rolesOf(); got != "user user assistant" {
		t.Fatalf("stored roles after A's late batch = %q; want u(A) u(B) a(opB)", got)
	}
	status, body, _ = e.do("POST", "/v1/turns/"+turnA+"/tool-results", []byte(`{"siteId":"site-e2e","storeId":"store-e2e","results":[{"toolCallId":"opA","ok":true}]}`))
	if status != 409 || !strings.Contains(string(body), `"status":"superseded"`) {
		t.Fatalf("A's late results → %d %s, want 409 superseded", status, body)
	}
	// The browser applied B's op and reports it.
	status, body, _ = e.do("POST", "/v1/turns/"+turnB+"/tool-results", []byte(`{"siteId":"site-e2e","storeId":"store-e2e","results":[{"toolCallId":"opB","ok":true}]}`))
	if status != 200 || !strings.HasSuffix(kinds(parseSSE(t, body)), "turn.done") {
		t.Fatalf("the newer turn B's own results → %d %s", status, body)
	}
	if rowB, _ = e.st.GetTurn(ctx, turnB); rowB.Status != store.TurnDone {
		t.Fatalf("rowB=%s, want done", rowB.Status)
	}
	mark := e.logCount()
	status, body, _ = e.do("POST", "/v1/turns", e.turnBody("C text", ""))
	if status != 200 || !strings.HasSuffix(kinds(parseSSE(t, body)), "turn.done") {
		t.Fatalf("turn C → %d %s", status, body)
	}
	if e.logged(mark, "WARNING") {
		t.Fatal("turn C logged a history-repair WARNING: B's batch dangles mid-history")
	}
	if got := rolesOf(); got != "user user assistant user assistant user assistant" {
		t.Fatalf("final roles = %q", got)
	}
	reqs := e.fake.Requests()
	assertReplayable(t, reqs[len(reqs)-1].Body["messages"].([]any))
	if e.runner.Live() != 0 || e.runner.Parked() != 0 {
		t.Fatalf("leaked state: live=%d parked=%d", e.runner.Live(), e.runner.Parked())
	}
}

// assertReplayable fails when a request's messages would be rejected by the
// Messages API: a tool_use with no tool_result, or one answered twice.
func assertReplayable(t *testing.T, msgs []any) {
	t.Helper()
	open := map[string]bool{}
	answered := map[string]int{}
	for i, m := range msgs {
		blocks, _ := m.(map[string]any)["content"].([]any)
		for _, b := range blocks {
			blk, ok := b.(map[string]any)
			if !ok {
				continue
			}
			switch blk["type"] {
			case "tool_use":
				open[blk["id"].(string)] = true
			case "tool_result":
				id := blk["tool_use_id"].(string)
				if !open[id] {
					t.Errorf("messages[%d]: tool_result %q answers no open tool_use", i, id)
				}
				delete(open, id)
				answered[id]++
			}
		}
	}
	for id := range open {
		t.Errorf("request leaves tool_use %q unanswered", id)
	}
	for id, n := range answered {
		if n > 1 {
			t.Errorf("tool_use %q answered %d times", id, n)
		}
	}
}

func prefixBytes(t *testing.T, c modeltest.Captured) string {
	t.Helper()
	b, err := json.Marshal(map[string]any{"tools": c.Body["tools"], "system": c.Body["system"]})
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// has reports whether the compacted JSON of raw contains s (jsonb columns
// come back re-serialised with spaces after colons).
func has(raw json.RawMessage, s string) bool {
	var buf bytes.Buffer
	if err := json.Compact(&buf, raw); err != nil {
		return strings.Contains(string(raw), s)
	}
	return strings.Contains(buf.String(), s)
}
