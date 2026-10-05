package api

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/breadstack/sous-chef/internal/auth"
	"github.com/breadstack/sous-chef/internal/config"
)

const goodHash = "3e18656a3e18656a3e18656a3e18656a3e18656a3e18656a3e18656a3e18656a"

func newTestServer(t *testing.T, deps Deps, opts ...func(*Options)) *httptest.Server {
	t.Helper()
	o := Options{BuildID: "test-build", Routes: []string{"default", "opus"}, DefaultRouteConfigured: true, Heartbeat: -1, Logf: func(string, ...any) {}}
	for _, f := range opts {
		f(&o)
	}
	srv := New(auth.NewKeys([]string{"key-a", "key-b"}), o, deps)
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	return ts
}

func do(t *testing.T, ts *httptest.Server, method, path, key, body string, hdr ...string) (*http.Response, string) {
	t.Helper()
	var rd io.Reader
	if body != "" {
		rd = strings.NewReader(body)
	}
	req, _ := http.NewRequest(method, ts.URL+path, rd)
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	for i := 0; i+1 < len(hdr); i += 2 {
		req.Header.Set(hdr[i], hdr[i+1])
	}
	res, err := ts.Client().Do(req)
	if err != nil {
		t.Fatalf("%s %s: %v", method, path, err)
	}
	b, _ := io.ReadAll(res.Body)
	res.Body.Close()
	return res, string(b)
}

func errCode(t *testing.T, body string) string {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal([]byte(body), &m); err != nil {
		t.Fatalf("not a JSON error envelope: %q", body)
	}
	c, _ := m["error"].(string)
	return c
}

func TestOpenRoutes(t *testing.T) {
	ts := newTestServer(t, Deps{DB: pinger{}})
	res, body := do(t, ts, "GET", "/healthz", "", "")
	if res.StatusCode != 200 || !strings.Contains(body, `"status":"ok"`) {
		t.Errorf("healthz = %d %s", res.StatusCode, body)
	}
	res, body = do(t, ts, "GET", "/readyz", "", "")
	if res.StatusCode != 200 || !strings.Contains(body, `"status":"ready"`) || !strings.Contains(body, `"db":"ok"`) {
		t.Errorf("readyz = %d %s", res.StatusCode, body)
	}
	res, body = do(t, ts, "GET", "/version", "", "")
	if res.StatusCode != 200 {
		t.Fatalf("version = %d", res.StatusCode)
	}
	var v struct {
		BuildID string   `json:"build_id"`
		Go      string   `json:"go"`
		Routes  []string `json:"routes"`
	}
	if err := json.Unmarshal([]byte(body), &v); err != nil || v.BuildID != "test-build" || !strings.HasPrefix(v.Go, "go") || len(v.Routes) != 2 {
		t.Errorf("version body = %s (%v)", body, err)
	}
	// Without route info the body is what it always was.
	if strings.Contains(body, "route_info") || strings.Contains(body, "default_route") {
		t.Errorf("version body = %s", body)
	}
	// With it: the existing members, plus each route's kind, effective
	// fields_context, fallback and report_as (null when unset), and the
	// default route.
	backstop, muffin := "default", "blueberry-muffin"
	ts2 := newTestServer(t, Deps{DB: pinger{}}, func(o *Options) {
		o.Routes = []string{"default", "student"}
		o.DefaultRoute = "student"
		o.RouteInfo = map[string]config.RouteInfo{
			"default": {Kind: config.RouteAnthropic},
			"student": {Kind: config.RouteMessagesCompatible, FieldsContext: true, Fallback: &backstop, ReportAs: &muffin},
		}
	})
	res, body = do(t, ts2, "GET", "/version", "", "")
	var doc map[string]json.RawMessage
	if err := json.Unmarshal([]byte(body), &doc); err != nil || res.StatusCode != 200 {
		t.Fatalf("version = %d %s (%v)", res.StatusCode, body, err)
	}
	if string(doc["build_id"]) != `"test-build"` || string(doc["routes"]) != `["default","student"]` || doc["go"] == nil || string(doc["default_route"]) != `"student"` ||
		string(doc["route_info"]) != `{"default":{"kind":"anthropic","fields_context":false,"fallback":null,"report_as":null},"student":{"kind":"messages-compatible","fields_context":true,"fallback":"default","report_as":"blueberry-muffin"}}` {
		t.Errorf("version body = %s", body)
	}
	// Method mismatch on an open route is a 405 from the mux.
	res, _ = do(t, ts, "POST", "/healthz", "", "")
	if res.StatusCode != http.StatusMethodNotAllowed {
		t.Errorf("POST /healthz = %d", res.StatusCode)
	}
}

type pinger struct{ err error }

func (p pinger) Ping(context.Context) error { return p.err }

func TestReadyzNotReady(t *testing.T) {
	ts := newTestServer(t, Deps{}) // no DB
	res, body := do(t, ts, "GET", "/readyz", "", "")
	if res.StatusCode != 503 || !strings.Contains(body, `"db":"unconfigured"`) {
		t.Errorf("no db: %d %s", res.StatusCode, body)
	}
	ts = newTestServer(t, Deps{DB: pinger{err: errors.New("conn refused")}})
	res, body = do(t, ts, "GET", "/readyz", "", "")
	if res.StatusCode != 503 || !strings.Contains(body, "conn refused") {
		t.Errorf("db error: %d %s", res.StatusCode, body)
	}
	ts = newTestServer(t, Deps{DB: pinger{}}, func(o *Options) { o.DefaultRouteConfigured = false })
	res, body = do(t, ts, "GET", "/readyz", "", "")
	if res.StatusCode != 503 || !strings.Contains(body, `"default_route":"missing"`) {
		t.Errorf("no default route: %d %s", res.StatusCode, body)
	}
}

func TestAuthOnV1(t *testing.T) {
	ts := newTestServer(t, Deps{})
	for _, k := range []string{"", "wrong", "key-a-", "KEY-A"} {
		res, body := do(t, ts, "GET", "/v1/manifests/"+goodHash, k, "")
		if res.StatusCode != 401 || errCode(t, body) != "unauthorized" {
			t.Errorf("key %q: %d %s", k, res.StatusCode, body)
		}
	}
	// Both live keys pass auth; with no registry wired the stub answers 501.
	for _, k := range []string{"key-a", "key-b"} {
		res, body := do(t, ts, "GET", "/v1/manifests/"+goodHash, k, "")
		if res.StatusCode != 501 || errCode(t, body) != "not_implemented" {
			t.Errorf("key %q: %d %s", k, res.StatusCode, body)
		}
	}
	// Unknown /v1 route is a JSON 404 — but only after auth.
	res, body := do(t, ts, "GET", "/v1/nope", "", "")
	if res.StatusCode != 401 {
		t.Errorf("unauth unknown route = %d %s", res.StatusCode, body)
	}
	res, body = do(t, ts, "GET", "/v1/nope", "key-a", "")
	if res.StatusCode != 404 || errCode(t, body) != "not_found" {
		t.Errorf("unknown route = %d %s", res.StatusCode, body)
	}
}

func TestStubsAnswer501(t *testing.T) {
	ts := newTestServer(t, Deps{})
	cases := []struct{ method, path, body string }{
		{"PUT", "/v1/manifests/" + goodHash, `{}`},
		{"GET", "/v1/manifests/" + goodHash, ""},
		{"POST", "/v1/turns", `{"siteId":"s","manifestHash":"` + goodHash + `","buildId":"b","text":"hi"}`},
		{"POST", "/v1/turns/t1/tool-results", `{"results":[{"toolCallId":"c1","ok":true}]}`},
		{"GET", "/v1/conversations/site-1", ""},
		{"DELETE", "/v1/conversations/site-1", ""},
		{"POST", "/v1/feedback", `{"turnId":"t1"}`},
		{"PUT", "/v1/stores/st/trace-optout", ""},
		{"DELETE", "/v1/stores/st/trace-optout", ""},
	}
	for _, c := range cases {
		res, body := do(t, ts, c.method, c.path, "key-a", c.body)
		if res.StatusCode != 501 || errCode(t, body) != "not_implemented" {
			t.Errorf("%s %s = %d %s", c.method, c.path, res.StatusCode, body)
		}
	}
}

// ---- fakes for the port-backed handlers ----

type fakeRegistry struct {
	seen  map[string][]byte
	build string
}

func (f *fakeRegistry) Register(_ context.Context, hash string, body []byte, build string) (RegisterResult, bool, error) {
	if strings.Contains(string(body), "mismatch") {
		return RegisterResult{}, false, ErrHashMismatch(hash, "deadbeef")
	}
	f.build = build
	_, existed := f.seen[hash]
	if f.seen == nil {
		f.seen = map[string][]byte{}
	}
	f.seen[hash] = body
	return RegisterResult{Hash: hash, Fields: 2094, Lists: 69}, !existed, nil
}

func (f *fakeRegistry) Get(_ context.Context, hash string, withBody bool) (ManifestInfo, error) {
	b, ok := f.seen[hash]
	if !ok {
		return ManifestInfo{}, ErrNotFound("unknown_manifest", "not registered")
	}
	info := ManifestInfo{Hash: hash, Counts: map[string]int{"fields": 2094}, FirstSeen: time.Unix(0, 0).UTC(), Index: "s:hero f=2"}
	if withBody {
		info.Body = b
	}
	return info, nil
}

func TestManifestRoutes(t *testing.T) {
	reg := &fakeRegistry{}
	ts := newTestServer(t, Deps{Manifests: reg})

	res, body := do(t, ts, "PUT", "/v1/manifests/not-a-hash", "key-a", `{"schema":1}`)
	if res.StatusCode != 400 || errCode(t, body) != "bad_request" {
		t.Errorf("bad hash = %d %s", res.StatusCode, body)
	}
	res, body = do(t, ts, "PUT", "/v1/manifests/"+goodHash, "key-a", `{"schema":1}`, "X-Launcher-Build", "v3.0.90")
	if res.StatusCode != 201 || !strings.Contains(body, `"fields":2094`) || reg.build != "v3.0.90" {
		t.Errorf("first PUT = %d %s build=%q", res.StatusCode, body, reg.build)
	}
	res, _ = do(t, ts, "PUT", "/v1/manifests/"+goodHash, "key-a", `{"schema":1}`)
	if res.StatusCode != 200 {
		t.Errorf("second PUT = %d (idempotent → 200)", res.StatusCode)
	}
	res, body = do(t, ts, "PUT", "/v1/manifests/"+goodHash, "key-a", `{"mismatch":true}`)
	if res.StatusCode != 400 || errCode(t, body) != "hash_mismatch" || !strings.Contains(body, `"computed":"deadbeef"`) {
		t.Errorf("mismatch = %d %s", res.StatusCode, body)
	}
	// Empty body / wrong content type.
	req, _ := http.NewRequest("PUT", ts.URL+"/v1/manifests/"+goodHash, nil)
	req.Header.Set("Authorization", "Bearer key-a")
	r2, _ := ts.Client().Do(req)
	if r2.StatusCode != 400 {
		t.Errorf("empty PUT = %d", r2.StatusCode)
	}
	r2.Body.Close()
	res, _ = do(t, ts, "PUT", "/v1/manifests/"+goodHash, "key-a", `{"x":1}`, "Content-Type", "text/plain")
	if res.StatusCode != 400 {
		t.Errorf("text/plain PUT = %d", res.StatusCode)
	}

	res, body = do(t, ts, "GET", "/v1/manifests/"+goodHash, "key-a", "")
	if res.StatusCode != 200 || strings.Contains(body, `"body"`) || !strings.Contains(body, `"index":"s:hero f=2"`) {
		t.Errorf("GET = %d %s", res.StatusCode, body)
	}
	res, body = do(t, ts, "GET", "/v1/manifests/"+goodHash+"?body=1", "key-a", "")
	if res.StatusCode != 200 || !strings.Contains(body, `"body":{"schema":1}`) {
		t.Errorf("GET ?body=1 = %d %s", res.StatusCode, body)
	}
	other := strings.Repeat("ab", 32)
	res, body = do(t, ts, "GET", "/v1/manifests/"+other, "key-a", "")
	if res.StatusCode != 404 || errCode(t, body) != "unknown_manifest" {
		t.Errorf("GET unknown = %d %s", res.StatusCode, body)
	}
}

func TestManifestBodyLimit(t *testing.T) {
	ts := newTestServer(t, Deps{Manifests: &fakeRegistry{}}, func(o *Options) { o.MaxManifestBytes = 64 })
	res, body := do(t, ts, "PUT", "/v1/manifests/"+goodHash, "key-a", `{"pad":"`+strings.Repeat("x", 100)+`"}`)
	if res.StatusCode != http.StatusRequestEntityTooLarge || !strings.Contains(body, "too large") {
		t.Errorf("oversized = %d %s", res.StatusCode, body)
	}
}

// fakeRunner scripts Start/Continue and the events Run emits.
type fakeRunner struct {
	startErr    error
	contErr     error
	lastReq     TurnRequest
	lastResults ToolResultsRequest
	lastTurnID  string
	events      []Event
	runErr      error
}

type fakeTurn struct {
	id     string
	events []Event
	runErr error
}

func (f *fakeTurn) ID() string { return f.id }
func (f *fakeTurn) Run(ctx context.Context, em Emitter) error {
	for i, ev := range f.events {
		ev.TurnID = f.id
		ev.Seq = i + 1
		if err := em.Emit(ev); err != nil {
			return err
		}
	}
	return f.runErr
}

func (f *fakeRunner) Start(_ context.Context, req TurnRequest) (Turn, error) {
	f.lastReq = req
	if f.startErr != nil {
		return nil, f.startErr
	}
	return &fakeTurn{id: "turn-1", events: f.events, runErr: f.runErr}, nil
}

func (f *fakeRunner) Continue(_ context.Context, turnID string, res ToolResultsRequest) (Turn, error) {
	f.lastTurnID, f.lastResults = turnID, res
	if f.contErr != nil {
		return nil, f.contErr
	}
	return &fakeTurn{id: turnID, events: f.events, runErr: f.runErr}, nil
}

func turnBody(extra string) string {
	return `{"siteId":"site-1","storeId":"st-1","manifestHash":"` + goodHash + `","buildId":"launcher:abc","context":{"page":"home"},"text":"make the hero bigger"` + extra + `}`
}

func TestPostTurnValidation(t *testing.T) {
	fr := &fakeRunner{}
	ts := newTestServer(t, Deps{Turns: fr})
	cases := map[string]string{
		"empty body": "",
		"not json":   `{`,
		"trailing":   turnBody("") + ` {}`,
		"no siteId":  `{"manifestHash":"` + goodHash + `","buildId":"b","text":"x"}`,
		"bad hash":   `{"siteId":"s","manifestHash":"abc","buildId":"b","text":"x"}`,
		"no buildId": `{"siteId":"s","manifestHash":"` + goodHash + `","text":"x"}`,
		"no text":    `{"siteId":"s","manifestHash":"` + goodHash + `","buildId":"b","text":"  "}`,
	}
	for name, body := range cases {
		req, _ := http.NewRequest("POST", ts.URL+"/v1/turns", strings.NewReader(body))
		req.Header.Set("Authorization", "Bearer key-a")
		req.Header.Set("Content-Type", "application/json")
		res, err := ts.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		b, _ := io.ReadAll(res.Body)
		res.Body.Close()
		if res.StatusCode != 400 || errCode(t, string(b)) != "bad_request" {
			t.Errorf("%s: %d %s", name, res.StatusCode, b)
		}
	}
}

func TestPostTurnPreStreamErrors(t *testing.T) {
	fr := &fakeRunner{startErr: ErrUnknownManifest(goodHash)}
	ts := newTestServer(t, Deps{Turns: fr})
	res, body := do(t, ts, "POST", "/v1/turns", "key-a", turnBody(""))
	if res.StatusCode != 409 || errCode(t, body) != "unknown_manifest" || !strings.Contains(body, goodHash) {
		t.Errorf("unknown manifest = %d %s", res.StatusCode, body)
	}
	if ct := res.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Errorf("pre-stream error must be JSON, got %q", ct)
	}
	fr.startErr = ErrBusy()
	res, body = do(t, ts, "POST", "/v1/turns", "key-a", turnBody(""))
	if res.StatusCode != 503 || errCode(t, body) != "busy" {
		t.Errorf("busy = %d %s", res.StatusCode, body)
	}
	fr.startErr = errors.New("pg: connection reset")
	res, body = do(t, ts, "POST", "/v1/turns", "key-a", turnBody(""))
	if res.StatusCode != 500 || errCode(t, body) != "internal" || strings.Contains(body, "pg:") {
		t.Errorf("internal = %d %s (must not leak)", res.StatusCode, body)
	}
}

func TestPostTurnSSE(t *testing.T) {
	fr := &fakeRunner{events: []Event{
		{Type: EventTextDelta, Data: map[string]string{"text": "Sure."}},
		{Type: EventToolCall, Data: map[string]any{"id": "c1", "name": "set_fields", "input": map[string]any{"target": map[string]any{"kind": "section", "page": "home", "sectionId": "hero-1"}, "values": map[string]any{"titleSize": "xl"}}}},
		{Type: EventTurnWaiting, Data: map[string]any{"toolCallIds": []string{"c1"}}},
	}}
	ts := newTestServer(t, Deps{Turns: fr})
	res, body := do(t, ts, "POST", "/v1/turns", "key-a", turnBody(`,"model":"opus"`), "X-User-Email", "Owner@Shop.com")
	if res.StatusCode != 200 {
		t.Fatalf("status = %d %s", res.StatusCode, body)
	}
	if ct := res.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Errorf("Content-Type = %q", ct)
	}
	if fr.lastReq.UserEmail != "owner@shop.com" || fr.lastReq.Model != "opus" || fr.lastReq.SiteID != "site-1" || string(fr.lastReq.Context) != `{"page":"home"}` {
		t.Errorf("runner saw %+v", fr.lastReq)
	}
	want := []string{
		"event: text.delta\ndata: {\"seq\":1,\"text\":\"Sure.\",\"turnId\":\"turn-1\"}\n\n",
		"event: tool.call\ndata: {\"id\":\"c1\",\"input\":{\"target\":{\"kind\":\"section\",\"page\":\"home\",\"sectionId\":\"hero-1\"},\"values\":{\"titleSize\":\"xl\"}},\"name\":\"set_fields\",\"seq\":2,\"turnId\":\"turn-1\"}\n\n",
		"event: turn.waiting\ndata: {\"seq\":3,\"toolCallIds\":[\"c1\"],\"turnId\":\"turn-1\"}\n\n",
	}
	if body != strings.Join(want, "") {
		t.Errorf("stream =\n%s\nwant\n%s", body, strings.Join(want, ""))
	}
}

func TestPostTurnJSONFallback(t *testing.T) {
	fr := &fakeRunner{events: []Event{
		{Type: EventTextDelta, Data: map[string]string{"text": "Done."}},
		{Type: EventTurnDone, Data: map[string]any{"stop": "end_turn", "usage": map[string]int{"input_tokens": 10, "output_tokens": 5}}},
	}}
	ts := newTestServer(t, Deps{Turns: fr})
	res, body := do(t, ts, "POST", "/v1/turns", "key-a", turnBody(""), "Accept", "application/json")
	if res.StatusCode != 200 || !strings.HasPrefix(res.Header.Get("Content-Type"), "application/json") {
		t.Fatalf("status = %d ct=%q body=%s", res.StatusCode, res.Header.Get("Content-Type"), body)
	}
	var env struct {
		TurnID string `json:"turnId"`
		Events []struct {
			Event string          `json:"event"`
			Data  json.RawMessage `json:"data"`
		} `json:"events"`
	}
	if err := json.Unmarshal([]byte(body), &env); err != nil {
		t.Fatal(err)
	}
	if env.TurnID != "turn-1" || len(env.Events) != 2 || env.Events[0].Event != "text.delta" || env.Events[1].Event != "turn.done" {
		t.Errorf("envelope = %s", body)
	}
	if !strings.Contains(string(env.Events[1].Data), `"seq":2`) || !strings.Contains(string(env.Events[1].Data), `"turnId":"turn-1"`) {
		t.Errorf("event data = %s", env.Events[1].Data)
	}
}

func TestToolResults(t *testing.T) {
	fr := &fakeRunner{events: []Event{{Type: EventTurnDone, Data: map[string]string{"stop": "end_turn"}}}}
	ts := newTestServer(t, Deps{Turns: fr})

	for name, body := range map[string]string{
		"empty results": `{"results":[]}`,
		"no id":         `{"results":[{"ok":true}]}`,
		"fail no error": `{"results":[{"toolCallId":"c1","ok":false}]}`,
	} {
		res, b := do(t, ts, "POST", "/v1/turns/turn-9/tool-results", "key-a", body)
		if res.StatusCode != 400 || errCode(t, b) != "bad_request" {
			t.Errorf("%s: %d %s", name, res.StatusCode, b)
		}
	}

	good := `{"siteId":"site-9","storeId":"store-9","results":[{"toolCallId":"c1","ok":true,"written":["titleSize"],"sectionId":"hero-1"},{"toolCallId":"c2","ok":false,"error":{"code":"E_UNKNOWN_KEY","message":"no such key","key":"foo"}},{"toolCallId":"c3","ok":true,"asset":{"public_url":"https://cdn/x.jpg","alt":"a cookie"}}]}`
	res, body := do(t, ts, "POST", "/v1/turns/turn-9/tool-results", "key-a", good)
	if res.StatusCode != 200 || !strings.HasPrefix(body, "event: turn.done\n") || !strings.Contains(body, `"turnId":"turn-9"`) {
		t.Errorf("continue = %d %s", res.StatusCode, body)
	}
	if fr.lastTurnID != "turn-9" || len(fr.lastResults.Results) != 3 || fr.lastResults.SiteID != "site-9" || fr.lastResults.StoreID != "store-9" {
		t.Fatalf("runner saw %q %+v", fr.lastTurnID, fr.lastResults)
	}
	r := fr.lastResults.Results
	if r[1].Error == nil || r[1].Error.Code != "E_UNKNOWN_KEY" || r[1].Error.Key != "foo" || r[2].Asset == nil || r[2].Asset.PublicURL != "https://cdn/x.jpg" || r[0].Written[0] != "titleSize" {
		t.Errorf("results decoded wrong: %+v", r)
	}

	fr.contErr = ErrTurnNotWaiting("turn-9", "done")
	res, body = do(t, ts, "POST", "/v1/turns/turn-9/tool-results", "key-a", good)
	if res.StatusCode != 409 || errCode(t, body) != "turn_not_waiting" || !strings.Contains(body, `"status":"done"`) {
		t.Errorf("not waiting = %d %s", res.StatusCode, body)
	}
	fr.contErr = ErrTurnNotFound("turn-9")
	res, body = do(t, ts, "POST", "/v1/turns/turn-9/tool-results", "key-a", good)
	if res.StatusCode != 404 || errCode(t, body) != "turn_not_found" {
		t.Errorf("not found = %d %s", res.StatusCode, body)
	}
}

type fakeConvs struct {
	ended []string
	conv  Conversation
	err   error
}

func (f *fakeConvs) Get(_ context.Context, siteID string, before int) (Conversation, error) {
	if f.err != nil {
		return Conversation{}, f.err
	}
	c := f.conv
	c.StoreID = siteID + ":" + strconv.Itoa(before)
	return c, nil
}
func (f *fakeConvs) End(_ context.Context, siteID string) error {
	f.ended = append(f.ended, siteID)
	return f.err
}

type fakeFeedback struct {
	got []Feedback
	err error
}

func (f *fakeFeedback) Record(_ context.Context, fb Feedback) error {
	if f.err != nil {
		return f.err
	}
	f.got = append(f.got, fb)
	return nil
}

type fakeOptOut struct{ set, cleared []string }

func (f *fakeOptOut) Set(_ context.Context, id, by string) error {
	f.set = append(f.set, id+"/"+by)
	return nil
}
func (f *fakeOptOut) Clear(_ context.Context, id string) error {
	f.cleared = append(f.cleared, id)
	return nil
}

func TestConversationFeedbackOptOut(t *testing.T) {
	convs := &fakeConvs{conv: Conversation{ConversationID: "conv-1"}}
	fb := &fakeFeedback{}
	oo := &fakeOptOut{}
	ts := newTestServer(t, Deps{Conversations: convs, Feedback: fb, OptOut: oo})

	res, body := do(t, ts, "GET", "/v1/conversations/site-1?before=7", "key-a", "")
	if res.StatusCode != 200 || !strings.Contains(body, `"conversationId":"conv-1"`) || !strings.Contains(body, `"storeId":"site-1:7"`) || !strings.Contains(body, `"messages":[]`) {
		t.Errorf("GET conv = %d %s", res.StatusCode, body)
	}
	res, body = do(t, ts, "GET", "/v1/conversations/site-1?before=x", "key-a", "")
	if res.StatusCode != 400 {
		t.Errorf("bad before = %d %s", res.StatusCode, body)
	}
	res, _ = do(t, ts, "DELETE", "/v1/conversations/site-1", "key-a", "")
	if res.StatusCode != 204 || len(convs.ended) != 1 || convs.ended[0] != "site-1" {
		t.Errorf("DELETE conv = %d ended=%v", res.StatusCode, convs.ended)
	}

	res, body = do(t, ts, "POST", "/v1/feedback", "key-a", `{"kept_paths":["a"]}`)
	if res.StatusCode != 400 {
		t.Errorf("feedback without turnId = %d %s", res.StatusCode, body)
	}
	res, _ = do(t, ts, "POST", "/v1/feedback", "key-a", `{"turnId":"t1","kept_paths":["hero.title"],"undone":false,"saved":true,"note":"nice"}`)
	if res.StatusCode != 204 || len(fb.got) != 1 || fb.got[0].TurnID != "t1" || fb.got[0].Saved == nil || !*fb.got[0].Saved || fb.got[0].Undone == nil || *fb.got[0].Undone || fb.got[0].SiteID != "" {
		t.Errorf("feedback = %d %+v", res.StatusCode, fb.got)
	}
	// The proxy's tenant assertions reach the sink; a sink refusing them is
	// the 404 the README promises.
	res, _ = do(t, ts, "POST", "/v1/feedback", "key-a", `{"turnId":"t1","siteId":"site-7","storeId":"store-7","saved":true}`)
	if res.StatusCode != 204 || len(fb.got) != 2 || fb.got[1].SiteID != "site-7" || fb.got[1].StoreID != "store-7" {
		t.Errorf("feedback with siteId = %d %+v", res.StatusCode, fb.got)
	}
	fb.err = ErrTurnNotFound("t1")
	res, body = do(t, ts, "POST", "/v1/feedback", "key-a", `{"turnId":"t1","siteId":"site-8","saved":true}`)
	if res.StatusCode != 404 || errCode(t, body) != "turn_not_found" {
		t.Errorf("feedback site mismatch = %d %s", res.StatusCode, body)
	}
	fb.err = nil

	res, _ = do(t, ts, "PUT", "/v1/stores/st-9/trace-optout", "key-b", "", "X-User-Email", "ops@shop.com")
	if res.StatusCode != 204 || len(oo.set) != 1 || oo.set[0] != "st-9/ops@shop.com" {
		t.Errorf("PUT optout = %d set=%v", res.StatusCode, oo.set)
	}
	res, _ = do(t, ts, "DELETE", "/v1/stores/st-9/trace-optout", "key-b", "")
	if res.StatusCode != 204 || len(oo.cleared) != 1 || oo.cleared[0] != "st-9" {
		t.Errorf("DELETE optout = %d cleared=%v", res.StatusCode, oo.cleared)
	}
}

type panicky struct{}

func (panicky) Get(context.Context, string, int) (Conversation, error) { panic("boom") }
func (panicky) End(context.Context, string) error                      { panic("boom") }

func TestRecoverer(t *testing.T) {
	ts := newTestServer(t, Deps{Conversations: panicky{}})
	res, body := do(t, ts, "GET", "/v1/conversations/s", "key-a", "")
	if res.StatusCode != 500 || errCode(t, body) != "internal" {
		t.Errorf("panic = %d %s", res.StatusCode, body)
	}
}

func TestLoggerLine(t *testing.T) {
	var lines []string
	ts := newTestServer(t, Deps{}, func(o *Options) {
		o.Logf = func(f string, a ...any) {
			lines = append(lines, strings.TrimSpace(strings.ReplaceAll(f, "%", ""))+" "+joinAny(a))
		}
	})
	do(t, ts, "GET", "/healthz", "", "")
	do(t, ts, "GET", "/version", "", "")
	do(t, ts, "GET", "/v1/nope", "", "")
	if len(lines) != 2 {
		t.Fatalf("probes must not be logged; got %d lines: %v", len(lines), lines)
	}
	if !strings.Contains(lines[0], "/version") || !strings.Contains(lines[0], "200") || !strings.Contains(lines[1], "401") {
		t.Errorf("lines = %v", lines)
	}
}

func joinAny(a []any) string {
	parts := make([]string, len(a))
	for i, v := range a {
		switch x := v.(type) {
		case string:
			parts[i] = x
		case int:
			parts[i] = strconv.Itoa(x)
		default:
			parts[i] = "?"
		}
	}
	return strings.Join(parts, " ")
}

// ---- S3: storeId is required on POST /v1/turns ----

func TestPostTurnRequiresStoreID(t *testing.T) {
	fr := &fakeRunner{events: []Event{{Type: EventTurnDone, Data: map[string]string{"stop": "end_turn"}}}}
	ts := newTestServer(t, Deps{Turns: fr})
	for name, body := range map[string]string{
		"absent": `{"siteId":"site-1","manifestHash":"` + goodHash + `","buildId":"b","context":{},"text":"x"}`,
		"blank":  `{"siteId":"site-1","storeId":"  ","manifestHash":"` + goodHash + `","buildId":"b","context":{},"text":"x"}`,
	} {
		fr.lastReq = TurnRequest{}
		res, b := do(t, ts, "POST", "/v1/turns", "key-a", body)
		if res.StatusCode != 400 || errCode(t, b) != "bad_request" || !strings.Contains(b, "storeId is required") {
			t.Errorf("%s storeId: %d %s", name, res.StatusCode, b)
		}
		if fr.lastReq.SiteID != "" {
			t.Fatalf("%s storeId: runner must not be reached", name)
		}
	}
	res, b := do(t, ts, "POST", "/v1/turns", "key-a", turnBody(""))
	if res.StatusCode != 200 || fr.lastReq.StoreID != "st-1" {
		t.Errorf("with storeId: %d %s runner saw %+v", res.StatusCode, b, fr.lastReq)
	}
}

// ---- S1: 256 KB body cap → 413 ----

func TestDefaultBodyLimitIs256KB(t *testing.T) {
	if DefaultMaxBodyBytes != 256*1024 {
		t.Fatalf("DefaultMaxBodyBytes = %d, want 262144", DefaultMaxBodyBytes)
	}
	s := New(auth.NewKeys([]string{"k"}), Options{}, Deps{})
	if s.opts.MaxBodyBytes != 256*1024 || s.opts.MaxManifestBytes != 8<<20 {
		t.Fatalf("defaults = body %d manifest %d", s.opts.MaxBodyBytes, s.opts.MaxManifestBytes)
	}
}

func TestPostTurnBodyLimit413(t *testing.T) {
	fr := &fakeRunner{events: []Event{{Type: EventTurnDone, Data: map[string]string{"stop": "end_turn"}}}}
	fb := &fakeFeedback{}
	ts := newTestServer(t, Deps{Turns: fr, Feedback: fb}) // default MaxBodyBytes
	// 300 KB of text: over 256 KB, under the previous 1 MiB cap.
	big := turnBody(`,"pad":"` + strings.Repeat("x", 300*1024) + `"`)
	if len(big) <= 256*1024 || len(big) >= 1<<20 {
		t.Fatalf("fixture size %d must sit between 256 KB and 1 MiB", len(big))
	}
	res, body := do(t, ts, "POST", "/v1/turns", "key-a", big)
	if res.StatusCode != http.StatusRequestEntityTooLarge || errCode(t, body) != "bad_request" || !strings.Contains(body, "too large") {
		t.Errorf("oversized turn = %d %s", res.StatusCode, body)
	}
	if fr.lastReq.SiteID != "" {
		t.Fatal("oversized body must not reach the runner")
	}
	// Just under the cap passes.
	ok := turnBody(`,"pad":"` + strings.Repeat("x", 200*1024) + `"`)
	res, body = do(t, ts, "POST", "/v1/turns", "key-a", ok)
	if res.StatusCode != 200 || fr.lastReq.SiteID != "site-1" {
		t.Errorf("200 KB turn = %d %s", res.StatusCode, body)
	}
	// The same cap applies to tool-results and feedback.
	res, body = do(t, ts, "POST", "/v1/turns/turn-1/tool-results", "key-a", `{"results":[{"toolCallId":"c1","ok":true,"warnings":["`+strings.Repeat("w", 300*1024)+`"]}]}`)
	if res.StatusCode != http.StatusRequestEntityTooLarge {
		t.Errorf("oversized tool-results = %d %s", res.StatusCode, body)
	}
	res, body = do(t, ts, "POST", "/v1/feedback", "key-a", `{"turnId":"turn-1","note":"`+strings.Repeat("n", 300*1024)+`"}`)
	if res.StatusCode != http.StatusRequestEntityTooLarge || len(fb.got) != 0 {
		t.Errorf("oversized feedback = %d %s", res.StatusCode, body)
	}
}

// ---- P2: asset.srcset is the browser's {width: url} object ----

func TestToolResultsSrcsetObject(t *testing.T) {
	fr := &fakeRunner{events: []Event{{Type: EventTurnDone, Data: map[string]string{"stop": "end_turn"}}}}
	ts := newTestServer(t, Deps{Turns: fr})

	srcset := `{"480":"https://cdn/x-480.jpg","960":"https://cdn/x.jpg"}`
	res, body := do(t, ts, "POST", "/v1/turns/turn-9/tool-results", "key-a",
		`{"results":[{"toolCallId":"pa1","ok":true,"asset":{"public_url":"https://cdn/x.jpg","alt":"a cookie","srcset":`+srcset+`}}]}`)
	if res.StatusCode != 200 {
		t.Fatalf("object srcset = %d %s", res.StatusCode, body)
	}
	a := fr.lastResults.Results[0].Asset
	if a == nil || a.PublicURL != "https://cdn/x.jpg" || string(a.Srcset) != srcset {
		t.Fatalf("asset decoded = %+v", a)
	}
	if m := a.SrcsetMap(); len(m) != 2 || m["960"] != "https://cdn/x.jpg" {
		t.Errorf("SrcsetMap = %v", m)
	}
	// Re-marshalling (what the loop sends the model as tool_result) keeps the object.
	out, _ := json.Marshal(a)
	if string(out) != `{"public_url":"https://cdn/x.jpg","alt":"a cookie","srcset":`+srcset+`}` {
		t.Errorf("round trip = %s", out)
	}

	// Legacy string srcset and JSON null are still accepted; null → absent.
	// (Ordered slice, not a map: the per-case assertions read the last result.)
	for _, c := range []struct{ name, raw, want string }{
		{"string", `"https://cdn/x.jpg 1x"`, `"https://cdn/x.jpg 1x"`},
		{"null", `null`, ``},
	} {
		res, body = do(t, ts, "POST", "/v1/turns/turn-9/tool-results", "key-a",
			`{"results":[{"toolCallId":"pa1","ok":true,"asset":{"public_url":"https://cdn/x.jpg","srcset":`+c.raw+`}}]}`)
		if res.StatusCode != 200 {
			t.Errorf("%s srcset = %d %s", c.name, res.StatusCode, body)
			continue
		}
		if got := string(fr.lastResults.Results[0].Asset.Srcset); got != c.want {
			t.Errorf("%s srcset decoded = %q, want %q", c.name, got, c.want)
		}
		out, _ = json.Marshal(fr.lastResults.Results[0].Asset)
		if c.want == "" && strings.Contains(string(out), "srcset") {
			t.Errorf("absent srcset must be omitted, got %s", out)
		}
	}

	// Anything else is a 400, not a model-facing 400 later.
	for name, s := range map[string]string{
		"number":           `12`,
		"array":            `["a"]`,
		"non-string value": `{"480":480}`,
	} {
		res, body = do(t, ts, "POST", "/v1/turns/turn-9/tool-results", "key-a",
			`{"results":[{"toolCallId":"pa1","ok":true,"asset":{"public_url":"https://cdn/x.jpg","srcset":`+s+`}}]}`)
		if res.StatusCode != 400 || errCode(t, body) != "bad_request" || !strings.Contains(body, "results[0].asset.srcset") {
			t.Errorf("%s srcset = %d %s", name, res.StatusCode, body)
		}
	}
}

// ---- P5: POST /v1/manifests computes the hash server-side ----

type fakeComputeRegistry struct {
	fakeRegistry
	computed []string
}

func (f *fakeComputeRegistry) ComputeAndRegister(ctx context.Context, body []byte, build string) (RegisterResult, bool, error) {
	h := strings.Repeat("c", 64)
	f.computed = append(f.computed, h)
	return f.Register(ctx, h, body, build)
}

func testHasher(body []byte) (string, error) {
	var v map[string]any
	if err := json.Unmarshal(body, &v); err != nil {
		return "", err
	}
	sum := sha256.Sum256(body)
	return hex.EncodeToString(sum[:]), nil
}

func TestPostManifest(t *testing.T) {
	// Fallback path: plain registry + Deps.HashManifest.
	reg := &fakeRegistry{}
	ts := newTestServer(t, Deps{Manifests: reg, HashManifest: testHasher})
	body := `{"schema":1,"generated_from":{"x":1}}`
	want, _ := testHasher([]byte(body))

	res, out := do(t, ts, "POST", "/v1/manifests", "key-a", body, "X-Launcher-Build", "v3.0.91")
	if res.StatusCode != 201 || !strings.Contains(out, `"hash":"`+want+`"`) || !strings.Contains(out, `"fields":2094`) || reg.build != "v3.0.91" {
		t.Errorf("first POST = %d %s build=%q", res.StatusCode, out, reg.build)
	}
	if _, ok := reg.seen[want]; !ok {
		t.Errorf("registered under %v, want %s", reg.seen, want)
	}
	res, _ = do(t, ts, "POST", "/v1/manifests", "key-a", body)
	if res.StatusCode != 200 {
		t.Errorf("second POST = %d (idempotent → 200)", res.StatusCode)
	}
	// A PUT with the hash POST returned is the same registration.
	res, _ = do(t, ts, "PUT", "/v1/manifests/"+want, "key-a", body)
	if res.StatusCode != 200 {
		t.Errorf("PUT after POST = %d", res.StatusCode)
	}
	res, out = do(t, ts, "POST", "/v1/manifests", "key-a", `not json`)
	if res.StatusCode != 400 || errCode(t, out) != "bad_request" {
		t.Errorf("bad JSON = %d %s", res.StatusCode, out)
	}
	req, _ := http.NewRequest("POST", ts.URL+"/v1/manifests", nil)
	req.Header.Set("Authorization", "Bearer key-a")
	r2, _ := ts.Client().Do(req)
	if r2.StatusCode != 400 {
		t.Errorf("empty POST = %d", r2.StatusCode)
	}
	r2.Body.Close()
	// Registry errors pass through unchanged.
	res, out = do(t, ts, "POST", "/v1/manifests", "key-a", `{"mismatch":true}`)
	if res.StatusCode != 400 || errCode(t, out) != "hash_mismatch" {
		t.Errorf("registry error = %d %s", res.StatusCode, out)
	}

	// Preferred path: the registry's own ComputeAndRegister.
	creg := &fakeComputeRegistry{}
	ts = newTestServer(t, Deps{Manifests: creg, HashManifest: testHasher})
	res, out = do(t, ts, "POST", "/v1/manifests", "key-a", body, "X-Launcher-Build", "v3.0.92")
	if res.StatusCode != 201 || len(creg.computed) != 1 || !strings.Contains(out, `"hash":"`+strings.Repeat("c", 64)+`"`) || creg.build != "v3.0.92" {
		t.Errorf("ComputeAndRegister path = %d %s computed=%v", res.StatusCode, out, creg.computed)
	}

	// Neither → 501, nothing read.
	ts = newTestServer(t, Deps{Manifests: &fakeRegistry{}})
	res, out = do(t, ts, "POST", "/v1/manifests", "key-a", body)
	if res.StatusCode != 501 || errCode(t, out) != "not_implemented" {
		t.Errorf("no hasher = %d %s", res.StatusCode, out)
	}
	// No registry at all → 501 like PUT.
	ts = newTestServer(t, Deps{HashManifest: testHasher})
	res, out = do(t, ts, "POST", "/v1/manifests", "key-a", body)
	if res.StatusCode != 501 {
		t.Errorf("no registry = %d %s", res.StatusCode, out)
	}
	// Manifest cap applies (8 MiB default; 64 here).
	ts = newTestServer(t, Deps{Manifests: &fakeRegistry{}, HashManifest: testHasher}, func(o *Options) { o.MaxManifestBytes = 64 })
	res, out = do(t, ts, "POST", "/v1/manifests", "key-a", `{"pad":"`+strings.Repeat("x", 100)+`"}`)
	if res.StatusCode != http.StatusRequestEntityTooLarge {
		t.Errorf("oversized POST = %d %s", res.StatusCode, out)
	}
}
