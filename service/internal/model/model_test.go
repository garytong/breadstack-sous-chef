package model

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/param"

	"github.com/breadstack/sous-chef/internal/config"
	"github.com/breadstack/sous-chef/internal/model/modeltest"
)

func boolp(b bool) *bool { return &b }
func intp(n int) *int    { return &n }

// anthropicTable builds a one-route "anthropic" table pointed at the fake.
func anthropicTable(t *testing.T, fake *modeltest.Server, mutate func(*Options)) *Table {
	t.Helper()
	o := Options{
		Routes: map[string]config.Route{
			"default": {Kind: config.RouteAnthropic, Auth: config.AuthAPIKey, Model: "claude-sonnet-5-5", MaxTokens: 1024, Effort: "medium"},
		},
		Default: "default", Allowlist: []string{"default"},
		APIKey: "sk-test", BaseURL: fake.URL, CallTimeout: 5 * time.Second,
	}
	if mutate != nil {
		mutate(&o)
	}
	tbl, err := NewTable(o)
	if err != nil {
		t.Fatalf("NewTable: %v", err)
	}
	return tbl
}

// compatTable builds a one-route "messages-compatible" table.
func compatTable(t *testing.T, fake *modeltest.Server, auth config.RouteAuth, mutate func(*Options)) *Table {
	t.Helper()
	o := Options{
		Routes: map[string]config.Route{
			"lab": {Kind: config.RouteMessagesCompatible, URL: fake.URL, Auth: auth, Model: "sourdough", MaxTokens: 512},
		},
		Default: "lab", Allowlist: []string{"lab"}, CallTimeout: 5 * time.Second,
		RouteAPIKeys: map[string]string{"lab": "lab-key"},
	}
	if mutate != nil {
		mutate(&o)
	}
	tbl, err := NewTable(o)
	if err != nil {
		t.Fatalf("NewTable: %v", err)
	}
	return tbl
}

func sampleRequest() Request {
	return Request{
		System: []anthropic.BetaTextBlockParam{
			{Text: "RULES", CacheControl: anthropic.NewBetaCacheControlEphemeralParam()},
			{Text: "INDEX", CacheControl: anthropic.NewBetaCacheControlEphemeralParam()},
		},
		Tools: []anthropic.BetaToolUnionParam{{OfTool: ptr(Strict(anthropic.BetaToolParam{
			Name: "set_fields", Description: param.NewOpt("set"),
			InputSchema: anthropic.BetaToolInputSchemaParam{Properties: map[string]any{"values": map[string]any{"type": "object"}}, Required: []string{"values"}},
		}))}},
		Messages: []anthropic.BetaMessageParam{UserTextTurn("make the hero title bigger")},
	}
}

func ptr[T any](v T) *T { return &v }

var toolReply = modeltest.Message{
	ID: "msg_1", Model: "claude-sonnet-5-5",
	Content: []modeltest.Block{
		modeltest.Text("Making the title bigger."),
		modeltest.ToolUse("toolu_1", "set_fields", map[string]any{"target": map[string]any{"kind": "section", "page": "home", "sectionId": "hero-1"}, "values": map[string]any{"headlineFontSize": 56}}),
	},
	Usage: modeltest.Usage{InputTokens: 120, OutputTokens: 40, CacheReadInputTokens: 3000, CacheCreationInputTokens: 10},
}

// --- route table -----------------------------------------------------------

func TestNewTable_RejectsMisconfiguration(t *testing.T) {
	base := func() Options {
		return Options{
			Routes:  map[string]config.Route{"default": {Kind: config.RouteAnthropic, Auth: config.AuthAPIKey, Model: "claude-sonnet-5-5", MaxTokens: 10}},
			Default: "default", Allowlist: []string{"default"}, APIKey: "k",
		}
	}
	cases := map[string]func(*Options){
		"no routes":          func(o *Options) { o.Routes = nil },
		"default missing":    func(o *Options) { o.Default = "nope" },
		"allowlist unknown":  func(o *Options) { o.Allowlist = []string{"ghost"} },
		"no api key":         func(o *Options) { o.APIKey = "" },
		"bad base url":       func(o *Options) { o.BaseURL = "not a url" },
		"negative retries":   func(o *Options) { o.MaxRetries = intp(-1) },
		"anthropic id-token": func(o *Options) { r := o.Routes["default"]; r.Auth = config.AuthIDToken; o.Routes["default"] = r },
		"compat without url": func(o *Options) {
			o.Routes["default"] = config.Route{Kind: config.RouteMessagesCompatible, Auth: config.AuthNone, Model: "m", MaxTokens: 10}
		},
		"compat api-key without key": func(o *Options) {
			o.Routes["default"] = config.Route{Kind: config.RouteMessagesCompatible, URL: "http://x", Auth: config.AuthAPIKey, Model: "m", MaxTokens: 10}
		},
		"unknown kind": func(o *Options) { r := o.Routes["default"]; r.Kind = "grpc"; o.Routes["default"] = r },
		"no model":     func(o *Options) { r := o.Routes["default"]; r.Model = ""; o.Routes["default"] = r },
	}
	for name, mutate := range cases {
		o := base()
		mutate(&o)
		if _, err := NewTable(o); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
	if _, err := NewTable(base()); err != nil {
		t.Fatalf("baseline should build: %v", err)
	}
}

func TestTable_RouteSelection(t *testing.T) {
	o := Options{
		Routes: map[string]config.Route{
			"default": {Kind: config.RouteAnthropic, Auth: config.AuthAPIKey, Model: "claude-sonnet-5-5", MaxTokens: 10},
			"opus":    {Kind: config.RouteAnthropic, Auth: config.AuthAPIKey, Model: "claude-opus-5-5", MaxTokens: 10},
			"hidden":  {Kind: config.RouteAnthropic, Auth: config.AuthAPIKey, Model: "claude-haiku-4-5", MaxTokens: 10},
		},
		Default: "default", Allowlist: []string{"default", "opus"}, APIKey: "k",
	}
	tbl, err := NewTable(o)
	if err != nil {
		t.Fatal(err)
	}
	if b, _ := tbl.Route(""); b.Name() != "default" || b.Model() != "claude-sonnet-5-5" {
		t.Errorf("empty name should pick the default route, got %q", b.Name())
	}
	if b, err := tbl.Route("opus"); err != nil || b.Model() != "claude-opus-5-5" {
		t.Errorf("opus: %v %v", b, err)
	}
	if _, err := tbl.Route("hidden"); !errors.Is(err, ErrRouteNotAllowed) {
		t.Errorf("hidden should be refused as not allowlisted, got %v", err)
	}
	if _, err := tbl.Route("ghost"); !errors.Is(err, ErrUnknownRoute) {
		t.Errorf("ghost should be unknown, got %v", err)
	}
	if got := tbl.Names(); strings.Join(got, ",") != "default,hidden,opus" {
		t.Errorf("Names = %v", got)
	}
	if tbl.Default() != "default" {
		t.Errorf("Default = %q", tbl.Default())
	}
}

func TestStreamDefaultsByKind(t *testing.T) {
	fake := modeltest.New()
	defer fake.Close()
	a := anthropicTable(t, fake, nil)
	if b, _ := a.Route(""); !b.Streams() || b.Kind() != config.RouteAnthropic || b.Timeout() != 5*time.Second {
		t.Errorf("anthropic routes stream by default")
	}
	c := compatTable(t, fake, config.AuthNone, nil)
	if b, _ := c.Route(""); b.Streams() {
		t.Errorf("messages-compatible routes do not stream by default")
	}
	c2 := compatTable(t, fake, config.AuthNone, func(o *Options) { r := o.Routes["lab"]; r.Stream = boolp(true); o.Routes["lab"] = r })
	if b, _ := c2.Route(""); !b.Streams() {
		t.Errorf("an explicit stream:true wins")
	}
}

func TestFromConfigAndRouteKeys(t *testing.T) {
	cfg := &config.Config{
		ModelRoutes: map[string]config.Route{
			"default": {Kind: config.RouteAnthropic, Auth: config.AuthAPIKey, Model: "claude-sonnet-5-5", MaxTokens: 10},
			"lab-gpu": {Kind: config.RouteMessagesCompatible, URL: "http://gpu", Auth: config.AuthAPIKey, Model: "m", MaxTokens: 10},
		},
		DefaultRoute: "default", RouteAllowlist: []string{"default"},
		AnthropicAPIKey: "sk", AnthropicBaseURL: "http://base", ModelCallTimeout: 7 * time.Second,
	}
	env := map[string]string{"MODEL_ROUTE_LAB_GPU_API_KEY": "gpu-key"}
	o := FromConfig(cfg, func(k string) string { return env[k] })
	if o.APIKey != "sk" || o.BaseURL != "http://base" || o.CallTimeout != 7*time.Second || o.Default != "default" {
		t.Errorf("FromConfig did not copy the config: %+v", o)
	}
	if o.RouteAPIKeys["lab-gpu"] != "gpu-key" {
		t.Errorf("route key not read: %v", o.RouteAPIKeys)
	}
	if RouteAPIKeyEnv("lab.gpu-2") != "MODEL_ROUTE_LAB_GPU_2_API_KEY" {
		t.Errorf("RouteAPIKeyEnv = %q", RouteAPIKeyEnv("lab.gpu-2"))
	}
	if _, err := NewTable(o); err != nil {
		t.Fatalf("table from config: %v", err)
	}
	if o2 := FromConfig(cfg, nil); len(o2.RouteAPIKeys) != 0 {
		t.Errorf("nil getenv reads no keys")
	}
}

// --- anthropic route: request shape -----------------------------------------

func TestAnthropic_NonStreaming_RequestShapeAndReply(t *testing.T) {
	fake := modeltest.New(modeltest.Reply(toolReply))
	defer fake.Close()
	tbl := anthropicTable(t, fake, func(o *Options) {
		r := o.Routes["default"]
		r.Stream = boolp(false)
		o.Routes["default"] = r
	})
	b, _ := tbl.Route("")
	var texts []string
	resp, err := b.Message(context.Background(), sampleRequest(), func(s string) { texts = append(texts, s) })
	if err != nil {
		t.Fatalf("Message: %v", err)
	}
	reqs := fake.Requests()
	if len(reqs) != 1 {
		t.Fatalf("expected 1 request, got %d", len(reqs))
	}
	r := reqs[0]
	if r.Path != "/v1/messages" {
		t.Errorf("path = %s", r.Path)
	}
	if r.Stream {
		t.Errorf("stream must be absent on a non-streaming route")
	}
	if r.Header.Get("X-Api-Key") != "sk-test" {
		t.Errorf("x-api-key not sent")
	}
	if !strings.Contains(r.Header.Get("Anthropic-Beta"), "server-side-fallback-2026-07-01") {
		t.Errorf("anthropic-beta header = %q", r.Header.Get("Anthropic-Beta"))
	}
	body := r.Body
	if body["model"] != "claude-sonnet-5-5" || body["max_tokens"] != float64(1024) {
		t.Errorf("model/max_tokens: %v %v", body["model"], body["max_tokens"])
	}
	if body["fallbacks"] != "default" {
		t.Errorf("fallbacks = %v", body["fallbacks"])
	}
	if _, has := body["thinking"]; has {
		t.Errorf("thinking must be omitted (adaptive)")
	}
	if oc, _ := body["output_config"].(map[string]any); oc["effort"] != "medium" {
		t.Errorf("output_config = %v", body["output_config"])
	}
	if tc, _ := body["tool_choice"].(map[string]any); tc["type"] != "auto" {
		t.Errorf("tool_choice = %v", body["tool_choice"])
	}
	if cc, _ := body["cache_control"].(map[string]any); cc["type"] != "ephemeral" {
		t.Errorf("top-level cache_control = %v", body["cache_control"])
	}
	sys := body["system"].([]any)
	for i, s := range sys {
		if cc, _ := s.(map[string]any)["cache_control"].(map[string]any); cc["type"] != "ephemeral" {
			t.Errorf("system[%d] lost its cache_control: %v", i, s)
		}
	}
	tools := body["tools"].([]any)
	tool := tools[0].(map[string]any)
	if tool["name"] != "set_fields" || tool["strict"] != true {
		t.Errorf("tool = %v", tool)
	}
	if schema := tool["input_schema"].(map[string]any); schema["additionalProperties"] != false || schema["type"] != "object" {
		t.Errorf("input_schema = %v", schema)
	}
	// reply
	if resp.Streamed {
		t.Errorf("Streamed should be false")
	}
	if resp.StopReason != "tool_use" || resp.ID != "msg_1" || resp.Model != "claude-sonnet-5-5" {
		t.Errorf("resp = %+v", resp)
	}
	if resp.Text() != "Making the title bigger." || strings.Join(texts, "|") != "Making the title bigger." {
		t.Errorf("text = %q, onText = %v", resp.Text(), texts)
	}
	uses := resp.ToolUses()
	if len(uses) != 1 || uses[0].Name != "set_fields" || uses[0].ID != "toolu_1" {
		t.Fatalf("tool uses = %+v", uses)
	}
	var in struct {
		Values map[string]any `json:"values"`
	}
	if err := json.Unmarshal(uses[0].Input, &in); err != nil || in.Values["headlineFontSize"] != float64(56) {
		t.Errorf("tool input = %s (%v)", uses[0].Input, err)
	}
	if resp.Usage != (Usage{InputTokens: 120, OutputTokens: 40, CacheReadInputTokens: 3000, CacheCreationInputTokens: 10}) {
		t.Errorf("usage = %+v", resp.Usage)
	}
	if resp.Latency <= 0 {
		t.Errorf("latency not measured")
	}
}

func TestAnthropic_Streaming_DeltasToolUseAndUsage(t *testing.T) {
	fake := modeltest.New(modeltest.Reply(toolReply))
	defer fake.Close()
	tbl := anthropicTable(t, fake, nil)
	b, _ := tbl.Route("")
	var texts []string
	resp, err := b.Message(context.Background(), sampleRequest(), func(s string) { texts = append(texts, s) })
	if err != nil {
		t.Fatalf("Message: %v", err)
	}
	if !fake.Requests()[0].Stream {
		t.Fatalf("stream:true expected on the wire")
	}
	if !resp.Streamed {
		t.Errorf("Streamed should be true")
	}
	if len(texts) < 2 || strings.Join(texts, "") != "Making the title bigger." {
		t.Errorf("text deltas = %q", texts)
	}
	if resp.Text() != "Making the title bigger." {
		t.Errorf("accumulated text = %q", resp.Text())
	}
	uses := resp.ToolUses()
	if len(uses) != 1 {
		t.Fatalf("tool uses = %+v", uses)
	}
	var in map[string]any
	if err := json.Unmarshal(uses[0].Input, &in); err != nil {
		t.Fatalf("input assembled from input_json_delta is not JSON: %s", uses[0].Input)
	}
	if tgt := in["target"].(map[string]any); tgt["sectionId"] != "hero-1" {
		t.Errorf("input = %v", in)
	}
	// message_start carried input + cache tokens, message_delta the output tokens
	want := Usage{InputTokens: 120, OutputTokens: 40, CacheReadInputTokens: 3000, CacheCreationInputTokens: 10}
	if resp.Usage != want {
		t.Errorf("usage = %+v, want %+v", resp.Usage, want)
	}
	if resp.StopReason != "tool_use" {
		t.Errorf("stop = %q", resp.StopReason)
	}
}

func TestStreaming_ThinkingStrippedFromContentKeptInTurn(t *testing.T) {
	fake := modeltest.New(modeltest.Reply(modeltest.Message{
		Content: []modeltest.Block{modeltest.Thinking("let me see"), modeltest.Text("Done.")},
		Usage:   modeltest.Usage{InputTokens: 5, OutputTokens: 6},
	}))
	defer fake.Close()
	b, _ := anthropicTable(t, fake, nil).Route("")
	resp, err := b.Message(context.Background(), sampleRequest(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Content) != 1 || resp.Content[0].Type != "text" {
		t.Errorf("thinking must not reach Content: %+v", resp.Content)
	}
	js, _ := json.Marshal(resp.AssistantTurn())
	if !strings.Contains(string(js), `"thinking"`) || !strings.Contains(string(js), `"let me see"`) {
		t.Errorf("AssistantTurn must replay the thinking block: %s", js)
	}
	if resp.StopReason != "end_turn" {
		t.Errorf("stop = %q", resp.StopReason)
	}
}

func TestStopReasons_RefusalCategoryAndMaxTokens(t *testing.T) {
	fake := modeltest.New(
		modeltest.Reply(modeltest.Message{StopReason: "refusal", StopCategory: "cyber", Content: []modeltest.Block{modeltest.Text("")}}),
		modeltest.Reply(modeltest.Message{StopReason: "max_tokens", Content: []modeltest.Block{modeltest.Text("cut")}}),
	)
	defer fake.Close()
	b, _ := anthropicTable(t, fake, nil).Route("")
	r1, err := b.Message(context.Background(), sampleRequest(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if r1.StopReason != "refusal" || r1.StopCategory != "cyber" {
		t.Errorf("refusal = %q/%q", r1.StopReason, r1.StopCategory)
	}
	r2, err := b.Message(context.Background(), sampleRequest(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if r2.StopReason != "max_tokens" || r2.StopCategory != "" || r2.Text() != "cut" {
		t.Errorf("max_tokens = %+v", r2)
	}
}

func TestFallbackServedModelIsReported(t *testing.T) {
	fake := modeltest.New(modeltest.Reply(modeltest.Message{Model: "claude-opus-4-8", Content: []modeltest.Block{modeltest.Text("ok")}}))
	defer fake.Close()
	b, _ := anthropicTable(t, fake, nil).Route("")
	resp, err := b.Message(context.Background(), sampleRequest(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Model != "claude-opus-4-8" {
		t.Errorf("the model that answered must be reported, got %q", resp.Model)
	}
}

func TestRequestIDHeaderCaptured(t *testing.T) {
	fake := modeltest.New(
		modeltest.Step{Headers: map[string]string{"request-id": "req_abc"}, Message: &modeltest.Message{Content: []modeltest.Block{modeltest.Text("a")}}},
		modeltest.Step{Headers: map[string]string{"request-id": "req_def"}, Message: &modeltest.Message{Content: []modeltest.Block{modeltest.Text("b")}}},
	)
	defer fake.Close()
	b, _ := anthropicTable(t, fake, nil).Route("")
	r1, err := b.Message(context.Background(), sampleRequest(), nil) // streaming
	if err != nil {
		t.Fatal(err)
	}
	if r1.RequestID != "req_abc" {
		t.Errorf("streamed request id = %q", r1.RequestID)
	}
	tbl := anthropicTable(t, fake, func(o *Options) { r := o.Routes["default"]; r.Stream = boolp(false); o.Routes["default"] = r })
	b2, _ := tbl.Route("")
	r2, err := b2.Message(context.Background(), sampleRequest(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if r2.RequestID != "req_def" {
		t.Errorf("plain request id = %q", r2.RequestID)
	}
}

func TestNoFallbacksForModelsWithoutClassifiers(t *testing.T) {
	fake := modeltest.New(modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("hi")}}))
	defer fake.Close()
	tbl := anthropicTable(t, fake, func(o *Options) {
		o.Routes["default"] = config.Route{Kind: config.RouteAnthropic, Auth: config.AuthAPIKey, Model: "claude-haiku-4-5", MaxTokens: 100}
	})
	b, _ := tbl.Route("")
	if _, err := b.Message(context.Background(), sampleRequest(), nil); err != nil {
		t.Fatal(err)
	}
	r := fake.Requests()[0]
	if _, has := r.Body["fallbacks"]; has {
		t.Errorf("haiku must not request fallbacks")
	}
	if r.Header.Get("Anthropic-Beta") != "" {
		t.Errorf("no beta header expected, got %q", r.Header.Get("Anthropic-Beta"))
	}
	if _, has := r.Body["output_config"]; has {
		t.Errorf("no effort configured → no output_config")
	}
	for _, m := range []string{"claude-sonnet-5-5", "claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-mythos-5-1"} {
		if !wantsFallbacks(m) {
			t.Errorf("%s should want fallbacks", m)
		}
	}
	for _, m := range []string{"claude-haiku-4-5", "claude-sonnet-5", "claude-opus-4-8", "sourdough"} {
		if wantsFallbacks(m) {
			t.Errorf("%s should not want fallbacks", m)
		}
	}
}

// --- retries, errors, timeout ----------------------------------------------

func TestRetry_429ThenSuccess(t *testing.T) {
	fake := modeltest.New(
		modeltest.Error(429, "rate_limit_error", "slow down", map[string]string{"Retry-After-Ms": "1"}),
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("after retry")}}),
	)
	defer fake.Close()
	b, _ := anthropicTable(t, fake, nil).Route("")
	resp, err := b.Message(context.Background(), sampleRequest(), nil)
	if err != nil {
		t.Fatalf("expected the retry to succeed: %v", err)
	}
	if resp.Text() != "after retry" {
		t.Errorf("text = %q", resp.Text())
	}
	reqs := fake.Requests()
	if len(reqs) != 2 {
		t.Fatalf("expected 2 requests, got %d", len(reqs))
	}
	if reqs[0].Header.Get("X-Stainless-Retry-Count") != "0" || reqs[1].Header.Get("X-Stainless-Retry-Count") != "1" {
		t.Errorf("retry count headers = %q %q", reqs[0].Header.Get("X-Stainless-Retry-Count"), reqs[1].Header.Get("X-Stainless-Retry-Count"))
	}
}

func TestRetry_500ThenSuccess_NonStreaming(t *testing.T) {
	fake := modeltest.New(
		modeltest.Error(500, "api_error", "boom", map[string]string{"Retry-After-Ms": "1"}),
		modeltest.Error(503, "overloaded_error", "busy", map[string]string{"Retry-After-Ms": "1"}),
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("third time")}}),
	)
	defer fake.Close()
	tbl := anthropicTable(t, fake, func(o *Options) { r := o.Routes["default"]; r.Stream = boolp(false); o.Routes["default"] = r })
	b, _ := tbl.Route("")
	resp, err := b.Message(context.Background(), sampleRequest(), nil)
	if err != nil {
		t.Fatalf("expected success after two 5xx: %v", err)
	}
	if resp.Text() != "third time" || len(fake.Requests()) != 3 {
		t.Errorf("text=%q requests=%d", resp.Text(), len(fake.Requests()))
	}
}

func TestRetry_ExhaustedIsUnavailable(t *testing.T) {
	fake := modeltest.New(
		modeltest.Error(429, "rate_limit_error", "1", map[string]string{"Retry-After-Ms": "1"}),
		modeltest.Error(429, "rate_limit_error", "2", map[string]string{"Retry-After-Ms": "1"}),
		modeltest.Error(529, "overloaded_error", "3", map[string]string{"Retry-After-Ms": "1"}),
	)
	defer fake.Close()
	b, _ := anthropicTable(t, fake, nil).Route("")
	_, err := b.Message(context.Background(), sampleRequest(), nil)
	if err == nil {
		t.Fatal("expected an error")
	}
	if !IsUnavailable(err) {
		t.Errorf("exhausted retries should be unavailable: %v", err)
	}
	if StatusCode(err) != 529 {
		t.Errorf("status = %d", StatusCode(err))
	}
	if n := len(fake.Requests()); n != 3 {
		t.Errorf("DefaultMaxRetries=2 → 3 attempts, got %d", n)
	}
}

func TestMaxRetriesZero(t *testing.T) {
	fake := modeltest.New(modeltest.Error(500, "api_error", "boom", nil), modeltest.Reply(modeltest.Message{}))
	defer fake.Close()
	b, _ := anthropicTable(t, fake, func(o *Options) { o.MaxRetries = intp(0) }).Route("")
	if _, err := b.Message(context.Background(), sampleRequest(), nil); err == nil || len(fake.Requests()) != 1 {
		t.Errorf("no retry expected: err=%v requests=%d", err, len(fake.Requests()))
	}
}

func TestBadRequestIsNotRetriedNorUnavailable(t *testing.T) {
	fake := modeltest.New(modeltest.Error(400, "invalid_request_error", "bad tool schema", nil))
	defer fake.Close()
	b, _ := anthropicTable(t, fake, nil).Route("")
	_, err := b.Message(context.Background(), sampleRequest(), nil)
	if err == nil {
		t.Fatal("expected an error")
	}
	var apiErr *anthropic.Error
	if !errors.As(err, &apiErr) || apiErr.StatusCode != 400 {
		t.Errorf("expected *anthropic.Error 400, got %T %v", err, err)
	}
	if IsUnavailable(err) {
		t.Errorf("a 400 is a bug, not an outage")
	}
	if len(fake.Requests()) != 1 {
		t.Errorf("400 must not be retried, got %d requests", len(fake.Requests()))
	}
}

func TestCallTimeout(t *testing.T) {
	fake := modeltest.New(modeltest.Step{Delay: 2 * time.Second, Message: &modeltest.Message{Content: []modeltest.Block{modeltest.Text("late")}}})
	defer fake.Close()
	b, _ := anthropicTable(t, fake, func(o *Options) { o.CallTimeout = 150 * time.Millisecond; o.MaxRetries = intp(0) }).Route("")
	started := time.Now()
	_, err := b.Message(context.Background(), sampleRequest(), nil)
	if err == nil {
		t.Fatal("expected a timeout")
	}
	if !IsUnavailable(err) {
		t.Errorf("a timeout is unavailable: %v", err)
	}
	if el := time.Since(started); el > 1500*time.Millisecond {
		t.Errorf("timeout took %s, want ~150ms", el)
	}
}

func TestConnectionRefusedIsUnavailable(t *testing.T) {
	fake := modeltest.New()
	url := fake.URL
	fake.Close()
	o := Options{
		Routes:  map[string]config.Route{"default": {Kind: config.RouteAnthropic, Auth: config.AuthAPIKey, Model: "claude-sonnet-5-5", MaxTokens: 10}},
		Default: "default", Allowlist: []string{"default"}, APIKey: "k", BaseURL: url, MaxRetries: intp(0), CallTimeout: time.Second,
	}
	tbl, err := NewTable(o)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := tbl.Route("")
	_, err = b.Message(context.Background(), sampleRequest(), nil)
	if err == nil || !IsUnavailable(err) {
		t.Errorf("connection refused should be unavailable, got %v", err)
	}
	if IsUnavailable(errors.New("plain")) || IsUnavailable(nil) {
		t.Errorf("an ordinary error is not unavailable")
	}
}

func TestEmptyStreamIsAnError(t *testing.T) {
	fake := modeltest.New(modeltest.Step{Headers: map[string]string{"Content-Type": "text/event-stream"}, Body: ""})
	defer fake.Close()
	b, _ := anthropicTable(t, fake, nil).Route("")
	if _, err := b.Message(context.Background(), sampleRequest(), nil); err == nil {
		t.Errorf("an empty stream must not pass as a reply")
	}
}

// --- messages-compatible routes ----------------------------------------------

func TestCompatible_PlainRequest_APIKey(t *testing.T) {
	fake := modeltest.New(modeltest.Reply(modeltest.Message{Model: "sourdough:latest", Content: []modeltest.Block{modeltest.Text("plain")}}))
	defer fake.Close()
	b, _ := compatTable(t, fake, config.AuthAPIKey, nil).Route("")
	req := sampleRequest()
	resp, err := b.Message(context.Background(), req, nil)
	if err != nil {
		t.Fatal(err)
	}
	r := fake.Requests()[0]
	if r.Stream {
		t.Errorf("compatible routes do not stream by default")
	}
	for _, k := range []string{"output_config", "cache_control", "fallbacks"} {
		if _, has := r.Body[k]; has {
			t.Errorf("plainRequest must drop %s", k)
		}
	}
	if th, _ := r.Body["thinking"].(map[string]any); len(th) != 1 || th["type"] != "disabled" {
		t.Errorf("thinking must be stated disabled by default, got %v", r.Body["thinking"])
	}
	if r.Header.Get("Anthropic-Beta") != "" {
		t.Errorf("no beta header on a compatible route")
	}
	for _, s := range r.Body["system"].([]any) {
		if _, has := s.(map[string]any)["cache_control"]; has {
			t.Errorf("system block kept cache_control: %v", s)
		}
	}
	if r.Header.Get("Authorization") != "Bearer lab-key" || r.Header.Get("X-Api-Key") != "lab-key" {
		t.Errorf("api-key auth headers = %q / %q", r.Header.Get("Authorization"), r.Header.Get("X-Api-Key"))
	}
	if r.Header.Get("X-Serverless-Authorization") != "" {
		t.Errorf("no ID token on an api-key route")
	}
	if tc, _ := r.Body["tool_choice"].(map[string]any); tc["type"] != "auto" {
		t.Errorf("tool_choice kept: %v", r.Body["tool_choice"])
	}
	if r.Body["model"] != "sourdough" || r.Body["max_tokens"] != float64(512) {
		t.Errorf("model/max_tokens = %v/%v", r.Body["model"], r.Body["max_tokens"])
	}
	// the caller's prefix keeps its breakpoints (plainRequest copies)
	if req.System[0].CacheControl.Type == "" {
		t.Errorf("plainRequest mutated the caller's system blocks")
	}
	if resp.Model != "sourdough:latest" || resp.Text() != "plain" {
		t.Errorf("resp = %+v", resp)
	}
}

func TestCompatible_AuthNone_SendsNoCredential(t *testing.T) {
	fake := modeltest.New(modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("x")}}))
	defer fake.Close()
	b, _ := compatTable(t, fake, config.AuthNone, func(o *Options) { o.APIKey = "sk-must-not-leak" }).Route("")
	if _, err := b.Message(context.Background(), sampleRequest(), nil); err != nil {
		t.Fatal(err)
	}
	h := fake.Requests()[0].Header
	for _, k := range []string{"Authorization", "X-Api-Key", "X-Serverless-Authorization"} {
		if h.Get(k) != "" {
			t.Errorf("%s = %q, want none", k, h.Get(k))
		}
	}
}

// fakeTokens counts IDToken calls.
type fakeTokens struct {
	calls    atomic.Int32
	audience string
}

func (f *fakeTokens) IDToken(_ context.Context, audience string) (string, error) {
	f.calls.Add(1)
	f.audience = audience
	return "id-token-" + fmt.Sprint(f.calls.Load()), nil
}

func TestCompatible_IDToken_HeaderAndAudience(t *testing.T) {
	fake := modeltest.New(
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("a")}}),
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("b")}}),
	)
	defer fake.Close()
	tokens := &fakeTokens{}
	b, _ := compatTable(t, fake, config.AuthIDToken, func(o *Options) { o.Tokens = tokens }).Route("")
	for i := 0; i < 2; i++ {
		if _, err := b.Message(context.Background(), sampleRequest(), nil); err != nil {
			t.Fatal(err)
		}
	}
	reqs := fake.Requests()
	if reqs[0].Header.Get("X-Serverless-Authorization") != "Bearer id-token-1" || reqs[1].Header.Get("X-Serverless-Authorization") != "Bearer id-token-2" {
		t.Errorf("ID token headers = %q %q", reqs[0].Header.Get("X-Serverless-Authorization"), reqs[1].Header.Get("X-Serverless-Authorization"))
	}
	if reqs[0].Header.Get("Authorization") != "" || reqs[0].Header.Get("X-Api-Key") != "" {
		t.Errorf("an id-token route sends no other credential")
	}
	if tokens.audience != fake.URL {
		t.Errorf("audience = %q, want the route URL %q", tokens.audience, fake.URL)
	}
}

type failingTokens struct{}

func (failingTokens) IDToken(context.Context, string) (string, error) {
	return "", errors.New("no metadata")
}

func TestCompatible_IDTokenFailureFailsTheCall(t *testing.T) {
	fake := modeltest.New(modeltest.Reply(modeltest.Message{}))
	defer fake.Close()
	b, _ := compatTable(t, fake, config.AuthIDToken, func(o *Options) { o.Tokens = failingTokens{} }).Route("")
	_, err := b.Message(context.Background(), sampleRequest(), nil)
	if err == nil || !strings.Contains(err.Error(), "ID token") {
		t.Errorf("expected the token error, got %v", err)
	}
	if len(fake.Requests()) != 0 {
		t.Errorf("nothing must reach the server without a token")
	}
}

func TestCompatible_StreamingFallsBackToPlainOn4xx(t *testing.T) {
	fake := modeltest.New(
		modeltest.Error(400, "invalid_request_error", "streaming not supported", nil),
		modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("plain ok")}}),
	)
	defer fake.Close()
	tbl := compatTable(t, fake, config.AuthNone, func(o *Options) { r := o.Routes["lab"]; r.Stream = boolp(true); o.Routes["lab"] = r })
	b, _ := tbl.Route("")
	var texts []string
	resp, err := b.Message(context.Background(), sampleRequest(), func(s string) { texts = append(texts, s) })
	if err != nil {
		t.Fatalf("fallback should succeed: %v", err)
	}
	reqs := fake.Requests()
	if len(reqs) != 2 || !reqs[0].Stream || reqs[1].Stream {
		t.Fatalf("expected a streaming attempt then a plain one, got %d requests (%v)", len(reqs), reqs)
	}
	if resp.Streamed || resp.Text() != "plain ok" || strings.Join(texts, "") != "plain ok" {
		t.Errorf("resp = %+v texts=%v", resp, texts)
	}
}

func TestCompatible_StreamingWorksWhenSupported(t *testing.T) {
	fake := modeltest.New(modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("streamed"), modeltest.ToolUse("t1", "lookup_fields", map[string]any{"query": "hero"})}}))
	defer fake.Close()
	tbl := compatTable(t, fake, config.AuthNone, func(o *Options) { r := o.Routes["lab"]; r.Stream = boolp(true); o.Routes["lab"] = r })
	b, _ := tbl.Route("")
	resp, err := b.Message(context.Background(), sampleRequest(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if !resp.Streamed || resp.Text() != "streamed" || len(resp.ToolUses()) != 1 {
		t.Errorf("resp = %+v", resp)
	}
}

func TestAnthropic_StreamingDoesNotFallBackOn4xx(t *testing.T) {
	fake := modeltest.New(modeltest.Error(400, "invalid_request_error", "bad", nil), modeltest.Reply(modeltest.Message{}))
	defer fake.Close()
	b, _ := anthropicTable(t, fake, nil).Route("")
	if _, err := b.Message(context.Background(), sampleRequest(), nil); err == nil || len(fake.Requests()) != 1 {
		t.Errorf("Claude's 400 is final: err=%v requests=%d", err, len(fake.Requests()))
	}
}

// --- thinking / sampling fields per route -------------------------------------

// historyWithThinking is a stored conversation as the loop replays it: an
// assistant turn with a thinking block, a redacted one, text and a tool_use,
// then its tool_result (the shape `messages.content` holds).
func historyWithThinking(t *testing.T) []anthropic.BetaMessageParam {
	t.Helper()
	msgs, err := UnmarshalMessages([]byte(`[
	 {"role":"user","content":[{"type":"text","text":"make the hero title bigger"}]},
	 {"role":"assistant","content":[
	   {"type":"thinking","thinking":"let me see","signature":"sig"},
	   {"type":"redacted_thinking","data":"xyz"},
	   {"type":"text","text":"Making it bigger."},
	   {"type":"tool_use","id":"toolu_1","name":"set_fields","input":{"values":{"headlineFontSize":56}}}]},
	 {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"{\"ok\":true}"}]},
	 {"role":"user","content":[{"type":"text","text":"and bolder"}]}
	]`))
	if err != nil {
		t.Fatal(err)
	}
	return msgs
}

// blockTypes lists the content block types of the i-th message on the wire.
func blockTypes(t *testing.T, body map[string]any, i int) []string {
	t.Helper()
	msgs, _ := body["messages"].([]any)
	if i >= len(msgs) {
		t.Fatalf("message %d missing in %v", i, body["messages"])
	}
	var out []string
	for _, b := range msgs[i].(map[string]any)["content"].([]any) {
		out = append(out, b.(map[string]any)["type"].(string))
	}
	return out
}

func TestCompatible_ThinkingDisabledByDefault_ExactBody(t *testing.T) {
	fake := modeltest.New(modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("ok")}}))
	defer fake.Close()
	b, _ := compatTable(t, fake, config.AuthNone, nil).Route("")
	req := sampleRequest()
	req.Messages = historyWithThinking(t)
	if _, err := b.Message(context.Background(), req, nil); err != nil {
		t.Fatal(err)
	}
	r := fake.Requests()[0]
	// exact top-level keys: nothing Claude-only, thinking stated, no sampling fields unless set
	var keys []string
	for k := range r.Body {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	if got := strings.Join(keys, ","); got != "max_tokens,messages,model,system,thinking,tool_choice,tools" {
		t.Errorf("body keys = %s", got)
	}
	if string(mustJSON(t, r.Body["thinking"])) != `{"type":"disabled"}` {
		t.Errorf("thinking = %s", mustJSON(t, r.Body["thinking"]))
	}
	// the stored thinking blocks are not re-sent; everything else is, in order
	if got := strings.Join(blockTypes(t, r.Body, 1), ","); got != "text,tool_use" {
		t.Errorf("assistant history on the wire = %s, want thinking dropped", got)
	}
	if len(r.Body["messages"].([]any)) != 4 {
		t.Errorf("message count changed: %d", len(r.Body["messages"].([]any)))
	}
	if strings.Contains(string(r.Raw), "let me see") || strings.Contains(string(r.Raw), `"redacted_thinking"`) {
		t.Errorf("thinking text reached the wire: %s", r.Raw)
	}
	// the caller's history is untouched (it is the persisted one)
	if got := strings.Join(blockTypesOf(req.Messages[1]), ","); got != "thinking,redacted_thinking,text,tool_use" {
		t.Errorf("caller's history mutated: %s", got)
	}
}

func TestCompatible_ThinkingEnabled_ReplaysBlocksAndSamplingFields(t *testing.T) {
	fake := modeltest.New(modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("ok")}}))
	defer fake.Close()
	temp, topP := 0.2, 0.9
	tbl := compatTable(t, fake, config.AuthNone, func(o *Options) {
		r := o.Routes["lab"]
		r.Thinking = config.ThinkingEnabled
		r.Temperature = &temp
		r.TopP = &topP
		o.Routes["lab"] = r
	})
	b, _ := tbl.Route("")
	req := sampleRequest()
	req.Messages = historyWithThinking(t)
	if _, err := b.Message(context.Background(), req, nil); err != nil {
		t.Fatal(err)
	}
	r := fake.Requests()[0]
	if string(mustJSON(t, r.Body["thinking"])) != `{"budget_tokens":256,"type":"enabled"}` {
		t.Errorf("thinking = %s", mustJSON(t, r.Body["thinking"]))
	}
	if r.Body["temperature"] != 0.2 || r.Body["top_p"] != 0.9 {
		t.Errorf("temperature/top_p = %v/%v", r.Body["temperature"], r.Body["top_p"])
	}
	if got := strings.Join(blockTypes(t, r.Body, 1), ","); got != "thinking,redacted_thinking,text,tool_use" {
		t.Errorf("enabled must replay the thinking blocks, got %s", got)
	}
	for _, k := range []string{"output_config", "cache_control", "fallbacks"} {
		if _, has := r.Body[k]; has {
			t.Errorf("plainRequest must still drop %s", k)
		}
	}
}

func TestCompatible_TemperatureOnlyWhenSet(t *testing.T) {
	fake := modeltest.New(modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("ok")}}))
	defer fake.Close()
	zero := 0.0
	tbl := compatTable(t, fake, config.AuthNone, func(o *Options) { r := o.Routes["lab"]; r.Temperature = &zero; o.Routes["lab"] = r })
	b, _ := tbl.Route("")
	if _, err := b.Message(context.Background(), sampleRequest(), nil); err != nil {
		t.Fatal(err)
	}
	r := fake.Requests()[0]
	if v, has := r.Body["temperature"]; !has || v != 0.0 {
		t.Errorf("an explicit temperature 0 must be sent, got %v (present=%v)", v, has)
	}
	if _, has := r.Body["top_p"]; has {
		t.Errorf("top_p must be absent when unset")
	}
}

func TestAnthropic_UnaffectedByThinkingFields(t *testing.T) {
	fake := modeltest.New(modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("ok")}}))
	defer fake.Close()
	b, _ := anthropicTable(t, fake, nil).Route("")
	req := sampleRequest()
	req.Messages = historyWithThinking(t)
	if _, err := b.Message(context.Background(), req, nil); err != nil {
		t.Fatal(err)
	}
	r := fake.Requests()[0]
	for _, k := range []string{"thinking", "temperature", "top_p"} {
		if _, has := r.Body[k]; has {
			t.Errorf("anthropic route must not send %s (adaptive thinking), got %v", k, r.Body[k])
		}
	}
	if got := strings.Join(blockTypes(t, r.Body, 1), ","); got != "thinking,redacted_thinking,text,tool_use" {
		t.Errorf("anthropic route must replay preserved thinking, got %s", got)
	}
}

func TestDropThinking_EdgeCases(t *testing.T) {
	// no thinking anywhere → the same slice back, untouched
	plain := []anthropic.BetaMessageParam{UserTextTurn("a"), {Role: anthropic.BetaMessageParamRoleAssistant, Content: []anthropic.BetaContentBlockParamUnion{anthropic.NewBetaTextBlock("b")}}}
	if got := dropThinking(plain); len(got) != 2 || &got[0] != &plain[0] {
		t.Errorf("a history without thinking must be returned as is")
	}
	// a thinking-only assistant turn is dropped whole, not sent empty
	msgs, err := UnmarshalMessages([]byte(`[
	 {"role":"user","content":[{"type":"text","text":"a"}]},
	 {"role":"assistant","content":[{"type":"thinking","thinking":"x","signature":"s"}]},
	 {"role":"user","content":[{"type":"text","text":"b"}]}]`))
	if err != nil {
		t.Fatal(err)
	}
	got := dropThinking(msgs)
	if len(got) != 2 || got[0].Role != "user" || got[1].Role != "user" {
		t.Errorf("thinking-only turn should be dropped: %d messages", len(got))
	}
	// a user turn carrying a thinking-typed block (never produced, but) is left alone
	user, _ := UnmarshalMessages([]byte(`[{"role":"user","content":[{"type":"thinking","thinking":"x","signature":"s"},{"type":"text","text":"a"}]}]`))
	if got := dropThinking(user); len(got[0].Content) != 2 {
		t.Errorf("only assistant turns are filtered")
	}
}

func blockTypesOf(m anthropic.BetaMessageParam) []string {
	var out []string
	for _, c := range m.Content {
		switch {
		case c.OfThinking != nil:
			out = append(out, "thinking")
		case c.OfRedactedThinking != nil:
			out = append(out, "redacted_thinking")
		case c.OfText != nil:
			out = append(out, "text")
		case c.OfToolUse != nil:
			out = append(out, "tool_use")
		default:
			out = append(out, "?")
		}
	}
	return out
}

func mustJSON(t *testing.T, v any) []byte {
	t.Helper()
	js, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return js
}

// --- non-object tool input guard ---------------------------------------------

func TestNonObjectToolInputBecomesEmptyObject(t *testing.T) {
	fake := modeltest.New(modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.ToolUse("t1", "describe_target", "just a string")}}))
	defer fake.Close()
	b, _ := compatTable(t, fake, config.AuthNone, nil).Route("")
	resp, err := b.Message(context.Background(), sampleRequest(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if string(resp.ToolUses()[0].Input) != "{}" {
		t.Errorf("input = %s", resp.ToolUses()[0].Input)
	}
	js, _ := json.Marshal(resp.AssistantTurn())
	if !strings.Contains(string(js), `"input":{}`) {
		t.Errorf("AssistantTurn must send {} back: %s", js)
	}
}

// --- history helpers -----------------------------------------------------------

func TestToolResultsTurn_OneMessageAllResults(t *testing.T) {
	turn := ToolResultsTurn([]ToolResult{
		{ID: "t1", Content: `{"ok":true}`},
		{ID: "t2", Content: `{"code":"E_FIELD","message":"unknown key"}`, IsError: true},
	})
	js, err := json.Marshal(turn)
	if err != nil {
		t.Fatal(err)
	}
	var m struct {
		Role    string `json:"role"`
		Content []struct {
			Type      string `json:"type"`
			ToolUseID string `json:"tool_use_id"`
			IsError   bool   `json:"is_error"`
			Content   []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"content"`
	}
	if err := json.Unmarshal(js, &m); err != nil {
		t.Fatal(err)
	}
	if m.Role != "user" || len(m.Content) != 2 {
		t.Fatalf("turn = %s", js)
	}
	if m.Content[0].Type != "tool_result" || m.Content[0].ToolUseID != "t1" || m.Content[0].IsError {
		t.Errorf("result 1 = %+v", m.Content[0])
	}
	if m.Content[1].ToolUseID != "t2" || !m.Content[1].IsError || m.Content[1].Content[0].Text != `{"code":"E_FIELD","message":"unknown key"}` {
		t.Errorf("result 2 = %+v", m.Content[1])
	}
}

func TestMessagesRoundTripThroughJSON(t *testing.T) {
	fake := modeltest.New(modeltest.Reply(toolReply))
	defer fake.Close()
	b, _ := anthropicTable(t, fake, nil).Route("")
	resp, err := b.Message(context.Background(), sampleRequest(), nil)
	if err != nil {
		t.Fatal(err)
	}
	history := []anthropic.BetaMessageParam{
		UserTextTurn("make the hero title bigger"),
		resp.AssistantTurn(),
		ToolResultsTurn([]ToolResult{{ID: "toolu_1", Content: `{"ok":true,"written":["settings.slides[0].headlineFontSize"]}`}}),
	}
	data, err := MarshalMessages(history)
	if err != nil {
		t.Fatal(err)
	}
	back, err := UnmarshalMessages(data)
	if err != nil {
		t.Fatalf("UnmarshalMessages: %v", err)
	}
	again, err := MarshalMessages(back)
	if err != nil {
		t.Fatal(err)
	}
	if string(again) != string(data) {
		t.Errorf("round trip changed the history:\n%s\n%s", data, again)
	}
	// and the replayed history is accepted on the wire as sent
	fake.Append(modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("done")}}))
	req := sampleRequest()
	req.Messages = back
	if _, err := b.Message(context.Background(), req, nil); err != nil {
		t.Fatal(err)
	}
	sent := fake.Requests()[1].Body["messages"].([]any)
	if len(sent) != 3 {
		t.Fatalf("sent %d messages", len(sent))
	}
	asst := sent[1].(map[string]any)
	blocks := asst["content"].([]any)
	if asst["role"] != "assistant" || len(blocks) != 2 || blocks[1].(map[string]any)["type"] != "tool_use" {
		t.Errorf("assistant turn on the wire = %v", asst)
	}
	if tr := sent[2].(map[string]any)["content"].([]any)[0].(map[string]any); tr["type"] != "tool_result" || tr["tool_use_id"] != "toolu_1" {
		t.Errorf("tool_result on the wire = %v", tr)
	}
}

func TestStrict(t *testing.T) {
	tool := Strict(anthropic.BetaToolParam{Name: "x", InputSchema: anthropic.BetaToolInputSchemaParam{Properties: map[string]any{}}})
	js, _ := json.Marshal(tool)
	if !strings.Contains(string(js), `"strict":true`) || !strings.Contains(string(js), `"additionalProperties":false`) {
		t.Errorf("Strict = %s", js)
	}
}

// --- id tokens ---------------------------------------------------------------

func jwt(exp int64) string {
	payload, _ := json.Marshal(map[string]any{"exp": exp, "aud": "x"})
	return "h." + base64.RawURLEncoding.EncodeToString(payload) + ".s"
}

func TestMetadataTokens_CachesUntilMargin(t *testing.T) {
	now := time.Unix(1_000_000, 0)
	var served atomic.Int32
	var lastAudience string
	meta := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Metadata-Flavor") != "Google" {
			http.Error(w, "flavor", 403)
			return
		}
		lastAudience = r.URL.Query().Get("audience")
		n := served.Add(1)
		_, _ = fmt.Fprint(w, jwt(now.Unix()+3600*int64(n)))
	}))
	defer meta.Close()
	m := newMetadataTokens(meta.URL)
	m.now = func() time.Time { return now }
	tok1, err := m.IDToken(context.Background(), "https://gpu.run.app")
	if err != nil {
		t.Fatal(err)
	}
	tok2, _ := m.IDToken(context.Background(), "https://gpu.run.app")
	if tok1 != tok2 || served.Load() != 1 {
		t.Errorf("second call should hit the cache (served=%d)", served.Load())
	}
	if lastAudience != "https://gpu.run.app" {
		t.Errorf("audience = %q", lastAudience)
	}
	// 56 minutes later: inside the 5-minute margin → refetch
	now = now.Add(56 * time.Minute)
	tok3, _ := m.IDToken(context.Background(), "https://gpu.run.app")
	if tok3 == tok1 || served.Load() != 2 {
		t.Errorf("token within the margin must be refreshed (served=%d)", served.Load())
	}
	// a different audience is never served from the cache
	if _, err := m.IDToken(context.Background(), "https://other.run.app"); err != nil || served.Load() != 3 {
		t.Errorf("other audience: err=%v served=%d", err, served.Load())
	}
}

func TestMetadataTokens_Errors(t *testing.T) {
	bad := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { http.Error(w, "nope", 500) }))
	defer bad.Close()
	if _, err := newMetadataTokens(bad.URL).IDToken(context.Background(), "a"); err == nil || !strings.Contains(err.Error(), "500") {
		t.Errorf("expected the status in the error, got %v", err)
	}
	notJWT := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { _, _ = fmt.Fprint(w, "garbage") }))
	defer notJWT.Close()
	if _, err := newMetadataTokens(notJWT.URL).IDToken(context.Background(), "a"); err == nil {
		t.Errorf("a non-JWT reply must fail")
	}
}

func TestJWTExpiry(t *testing.T) {
	exp, err := jwtExpiry(jwt(1_700_000_000))
	if err != nil || exp.Unix() != 1_700_000_000 {
		t.Errorf("exp = %v %v", exp, err)
	}
	for _, bad := range []string{"", "a.b", "a.!!!.c", "a." + base64.RawURLEncoding.EncodeToString([]byte(`{"exp":0}`)) + ".c", "a." + base64.RawURLEncoding.EncodeToString([]byte(`not json`)) + ".c"} {
		if _, err := jwtExpiry(bad); err == nil {
			t.Errorf("%q should fail", bad)
		}
	}
	if NewMetadataTokens() == nil {
		t.Error("NewMetadataTokens")
	}
}

func TestRouteFallbackLinked(t *testing.T) {
	fake := modeltest.New()
	defer fake.Close()
	routes := func() map[string]config.Route {
		return map[string]config.Route{
			"default": {Kind: config.RouteAnthropic, Auth: config.AuthAPIKey, Model: "claude-sonnet-5-5", MaxTokens: 1024},
			"student": {Kind: config.RouteMessagesCompatible, URL: fake.URL, Auth: config.AuthNone, Model: "student", MaxTokens: 512, Fallback: "default"},
		}
	}
	opts := func(r map[string]config.Route) Options {
		return Options{Routes: r, Default: "default", Allowlist: []string{"student"}, APIKey: "sk-test", BaseURL: fake.URL}
	}
	tbl, err := NewTable(opts(routes()))
	if err != nil {
		t.Fatalf("NewTable: %v", err)
	}
	b, err := tbl.Route("student")
	if err != nil {
		t.Fatal(err)
	}
	// The fallback is reachable from its route although it is not allowlisted.
	if fb := b.Fallback(); fb == nil || fb.Name() != "default" || fb.Kind() != config.RouteAnthropic || fb.Fallback() != nil {
		t.Fatalf("fallback = %+v", fb)
	}
	if _, err := tbl.Route("default"); !errors.Is(err, ErrRouteNotAllowed) {
		t.Errorf("default by name: %v", err)
	}
	for name, mutate := range map[string]func(map[string]config.Route){
		"unknown": func(r map[string]config.Route) { x := r["student"]; x.Fallback = "ghost"; r["student"] = x },
		"itself":  func(r map[string]config.Route) { x := r["student"]; x.Fallback = "student"; r["student"] = x },
		"chained": func(r map[string]config.Route) { x := r["default"]; x.Fallback = "student"; r["default"] = x },
	} {
		r := routes()
		mutate(r)
		if _, err := NewTable(opts(r)); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}

// WithoutThinkingBefore strips the thinking blocks of the first n messages
// only (a hybrid turn's pre-escalation history) and never touches the
// caller's slices.
func TestWithoutThinkingBefore(t *testing.T) {
	thinking := func(s string) anthropic.BetaContentBlockParamUnion {
		return anthropic.BetaContentBlockParamUnion{OfThinking: &anthropic.BetaThinkingBlockParam{Thinking: s, Signature: "sig"}}
	}
	asst := func(blocks ...anthropic.BetaContentBlockParamUnion) anthropic.BetaMessageParam {
		return anthropic.BetaMessageParam{Role: anthropic.BetaMessageParamRoleAssistant, Content: blocks}
	}
	msgs := []anthropic.BetaMessageParam{
		UserTextTurn("one"),
		asst(thinking("student"), anthropic.NewBetaTextBlock("a")),
		UserTextTurn("two"),
		asst(thinking("only thinking")),
		UserTextTurn("three"),
		asst(thinking("claude"), anthropic.NewBetaTextBlock("b")),
	}
	got := WithoutThinkingBefore(msgs, 5)
	// The thinking-only turn is dropped whole; the last assistant turn,
	// past the boundary, keeps its block.
	if len(got) != 5 || len(got[1].Content) != 1 || got[1].Content[0].OfText == nil || got[3].Role != anthropic.BetaMessageParamRoleUser ||
		len(got[4].Content) != 2 || got[4].Content[0].OfThinking == nil {
		t.Fatalf("got %d messages: %+v", len(got), got)
	}
	if len(msgs) != 6 || len(msgs[1].Content) != 2 || len(msgs[3].Content) != 1 {
		t.Fatal("the caller's history was modified")
	}
	// Nothing to drop, n ≤ 0, n past the end: the input itself / all of it.
	if same := WithoutThinkingBefore(msgs, 1); len(same) != 6 || &same[0] != &msgs[0] {
		t.Error("no-op call copied")
	}
	if same := WithoutThinkingBefore(msgs, 0); &same[0] != &msgs[0] {
		t.Error("n=0 copied")
	}
	if all := WithoutThinkingBefore(msgs, 99); len(all) != 5 || hasThinking(all[1].Content) || hasThinking(all[4].Content) {
		t.Errorf("n past the end: %d messages", len(all))
	}
}
