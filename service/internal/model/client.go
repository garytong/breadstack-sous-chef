package model

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"

	"github.com/breadstack/sous-chef/internal/config"
)

// Request is one Messages call's content: the prompt builder's cached
// prefix (system blocks with their cache_control breakpoints, the tool
// list in its fixed order) and the conversation so far, the new user turn
// last. The route adds model, max_tokens, tool_choice, effort, caching and
// fallbacks (buildParams).
type Request struct {
	System   []anthropic.BetaTextBlockParam
	Tools    []anthropic.BetaToolUnionParam
	Messages []anthropic.BetaMessageParam
}

// Usage is one call's token counts, cache tokens included (the wire names
// are the thermometer ingest schema's).
type Usage struct {
	InputTokens              int64 `json:"input_tokens"`
	OutputTokens             int64 `json:"output_tokens"`
	CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
}

// Block is one content block of the reply as traces and the loop see it:
// text or tool_use. Thinking blocks are never here (they stay in the raw
// message for AssistantTurn): thinking text is not written to traces
// (thermometer rule, README §3).
type Block struct {
	Type  string          `json:"type"` // "text" | "tool_use"
	Text  string          `json:"text,omitempty"`
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"` // always a JSON object (toolInput)
}

// Response is one model call's outcome.
type Response struct {
	ID           string // the message id
	RequestID    string // the request-id response header, "" when the server sent none
	Model        string // the model that answered (a fallback may have served the turn)
	StopReason   string // end_turn | tool_use | max_tokens | refusal | …
	StopCategory string // stop_details.category when StopReason is refusal
	Content      []Block
	Usage        Usage
	Latency      time.Duration
	Streamed     bool // the reply was read from the model's SSE stream
	raw          *anthropic.BetaMessage
}

// Text is the reply's text blocks joined.
func (r *Response) Text() string {
	var sb strings.Builder
	for _, b := range r.Content {
		if b.Type == "text" {
			sb.WriteString(b.Text)
		}
	}
	return sb.String()
}

// ToolUses lists the reply's tool_use blocks in order.
func (r *Response) ToolUses() []Block {
	var out []Block
	for _, b := range r.Content {
		if b.Type == "tool_use" {
			out = append(out, b)
		}
	}
	return out
}

// AssistantTurn is the reply as the next request's assistant message
// (thermometer assistantTurn): thinking blocks included — the API wants
// them back unchanged on the same model, other models ignore them — and
// each tool_use input that was not an object sent back as {} (the input
// the tool ran with, and the only shape the API accepts there).
func (r *Response) AssistantTurn() anthropic.BetaMessageParam {
	turn := r.raw.ToParam()
	for i, block := range r.raw.Content {
		b, ok := block.AsAny().(anthropic.BetaToolUseBlock)
		if ok && i < len(turn.Content) && turn.Content[i].OfToolUse != nil && !isJSONObject([]byte(strings.TrimSpace(b.JSON.Input.Raw()))) {
			turn.Content[i].OfToolUse.Input = map[string]any{}
		}
	}
	return turn
}

// ToolResult is one tool's outcome, the browser's or a server tool's.
type ToolResult struct {
	ID      string // the tool_use id answered
	Content string // JSON or plain text the model reads
	IsError bool
}

// ToolResultsTurn is the user message that answers one batch of tool
// calls: every result in ONE message, in call order (splitting them
// teaches the model to stop calling tools in parallel).
func ToolResultsTurn(results []ToolResult) anthropic.BetaMessageParam {
	blocks := make([]anthropic.BetaContentBlockParamUnion, 0, len(results))
	for _, r := range results {
		blocks = append(blocks, anthropic.NewBetaToolResultBlock(r.ID, r.Content, r.IsError))
	}
	return anthropic.NewBetaUserMessage(blocks...)
}

// UserTextTurn is a user message of one text block.
func UserTextTurn(text string) anthropic.BetaMessageParam {
	return anthropic.NewBetaUserMessage(anthropic.NewBetaTextBlock(text))
}

// MarshalMessages / UnmarshalMessages round-trip the history through JSON
// (the `messages.content` column): the SDK's param types marshal to the
// wire shape and read it back.
func MarshalMessages(msgs []anthropic.BetaMessageParam) ([]byte, error) { return json.Marshal(msgs) }

func UnmarshalMessages(data []byte) ([]anthropic.BetaMessageParam, error) {
	var out []anthropic.BetaMessageParam
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// Message makes one call to the route under the call timeout (README §5:
// 60 s) and returns the reply. On a streaming route onText receives each
// text delta as the model writes it (text.delta); on a non-streaming route
// it is called once with the whole text after the reply arrives, so the
// caller's loop is the same either way. onText may be nil.
//
// Retries (429, 408, 409, 5xx, connection errors) are the SDK's, with its
// backoff and Retry-After handling. A messages-compatible route whose
// server rejects stream=true outright (a 4xx before the first event) is
// retried once without streaming.
func (b *Backend) Message(ctx context.Context, req Request, onText func(string)) (*Response, error) {
	ctx, cancel := context.WithTimeout(ctx, b.timeout)
	defer cancel()
	params := b.buildParams(req)
	started := b.now()
	var (
		resp *Response
		err  error
	)
	if b.stream {
		var events int
		resp, events, err = b.streamed(ctx, params, onText)
		if err != nil && events == 0 && b.route.Kind == config.RouteMessagesCompatible && isHardClientError(err) {
			resp, err = b.plain(ctx, params, onText)
		}
	} else {
		resp, err = b.plain(ctx, params, onText)
	}
	if err != nil {
		return nil, err
	}
	resp.Latency = b.now().Sub(started)
	if resp.Model == "" {
		resp.Model = b.route.Model
	}
	return resp, nil
}

// buildParams is the request for req on this route (README §5 request
// rules): thinking omitted (adaptive on current Claude models), effort set
// explicitly from the route, tool_choice auto (forced any/tool is rejected
// by Sonnet 5.5 / Opus 5.5), the system blocks' cache_control as the prompt
// builder set them plus the top-level breakpoint that auto-caches the
// growing message tail (thermometer chat.go), server-side fallbacks on the
// models that run safety classifiers. A messages-compatible route gets the
// plain request.
func (b *Backend) buildParams(req Request) anthropic.BetaMessageNewParams {
	params := anthropic.BetaMessageNewParams{
		Model:        anthropic.Model(b.route.Model),
		MaxTokens:    int64(b.route.MaxTokens),
		System:       req.System,
		Tools:        req.Tools,
		Messages:     req.Messages,
		ToolChoice:   anthropic.BetaToolChoiceUnionParam{OfAuto: &anthropic.BetaToolChoiceAutoParam{}},
		CacheControl: anthropic.NewBetaCacheControlEphemeralParam(),
	}
	if b.route.Effort != "" {
		params.OutputConfig = anthropic.BetaOutputConfigParam{Effort: anthropic.BetaOutputConfigEffort(b.route.Effort)}
	}
	if b.route.Kind == config.RouteAnthropic && wantsFallbacks(b.route.Model) {
		// A safety-classifier refusal is re-served by a fallback model
		// inside the same call instead of ending the turn with nothing.
		params.Betas = []anthropic.AnthropicBeta{anthropic.AnthropicBetaServerSideFallback2026_07_01}
		params.Fallbacks = anthropic.BetaFallbacksParamUnion{OfDefault: constant.ValueOf[constant.Default]()}
	}
	if b.route.Kind == config.RouteMessagesCompatible {
		plainRequest(&params, b.route)
	}
	return params
}

// plainRequest (thermometer chat.go) drops every Claude-only field from
// params — effort (output_config), the cache_control breakpoints on the
// system blocks and at the top level, the betas (the anthropic-beta header)
// and server-side fallbacks — leaving system + tools + messages +
// max_tokens + tool_choice, then adds the route's own thinking and sampling
// fields (README §5 table). A compatible server either rejects fields it
// does not know or ignores them silently, and a fallback to a Claude model
// is never what a local run wants. The system blocks are copied first so
// the caller's prefix keeps its breakpoints.
//
// Thinking is always stated, never left to the server: Ollama's anthropic.go
// (v0.34.4 FromMessagesRequest) maps {"type":"disabled"} → think=false and
// {"type":"enabled"} → think=true, while an absent field leaves think=nil
// and the renderer's default (true for qwen3.5) — which puts an OPEN <think>
// at the end of the prompt where every SFT row has the empty think block.
// With thinking disabled (the default) the stored thinking blocks are also
// dropped from the history sent, so the rendered conversation matches
// training; enabled replays them. budget_tokens is half of max_tokens: Ollama
// reads only type, and the value keeps the request valid for a Claude-shaped
// server (≥1024 and < max_tokens when max_tokens allows).
func plainRequest(params *anthropic.BetaMessageNewParams, route config.Route) {
	if route.ThinkingEnabled() {
		params.Thinking = anthropic.BetaThinkingConfigParamUnion{
			OfEnabled: &anthropic.BetaThinkingConfigEnabledParam{BudgetTokens: int64(route.MaxTokens / 2)},
		}
	} else {
		params.Thinking = anthropic.BetaThinkingConfigParamUnion{OfDisabled: &anthropic.BetaThinkingConfigDisabledParam{}}
		params.Messages = dropThinking(params.Messages)
	}
	if route.Temperature != nil {
		params.Temperature = param.NewOpt(*route.Temperature)
	}
	if route.TopP != nil {
		params.TopP = param.NewOpt(*route.TopP)
	}
	params.OutputConfig = anthropic.BetaOutputConfigParam{}
	params.CacheControl = anthropic.BetaCacheControlEphemeralParam{}
	system := make([]anthropic.BetaTextBlockParam, len(params.System))
	copy(system, params.System)
	for i := range system {
		system[i].CacheControl = anthropic.BetaCacheControlEphemeralParam{}
	}
	params.System = system
	params.Betas = nil
	params.Fallbacks = anthropic.BetaFallbacksParamUnion{}
}

// dropThinking returns msgs with every thinking / redacted_thinking block
// removed from the assistant turns. The caller's slices are not touched
// (the loop's history is the persisted one, replayed as stored): messages
// are copied only when a block is dropped. A message left with no content
// (a thinking-only assistant turn, which the loop never stores) is dropped
// whole rather than sent as an empty turn.
func dropThinking(msgs []anthropic.BetaMessageParam) []anthropic.BetaMessageParam {
	var out []anthropic.BetaMessageParam
	for i, m := range msgs {
		if m.Role != anthropic.BetaMessageParamRoleAssistant || !hasThinking(m.Content) {
			if out != nil {
				out = append(out, m)
			}
			continue
		}
		if out == nil {
			out = make([]anthropic.BetaMessageParam, 0, len(msgs))
			out = append(out, msgs[:i]...)
		}
		kept := make([]anthropic.BetaContentBlockParamUnion, 0, len(m.Content))
		for _, c := range m.Content {
			if c.OfThinking == nil && c.OfRedactedThinking == nil {
				kept = append(kept, c)
			}
		}
		if len(kept) == 0 {
			continue
		}
		m.Content = kept
		out = append(out, m)
	}
	if out == nil {
		return msgs
	}
	return out
}

// WithoutThinkingBefore returns msgs with the thinking blocks of its first n
// messages removed (dropThinking) and the rest as they are. A hybrid turn
// that escalated from a messages-compatible route to an anthropic one sends
// its history through here (README §4 "Hybrid"): the blocks written before
// the escalation are another model's — Claude rejects a thinking block whose
// signature it did not issue — while the blocks Claude writes afterwards go
// back to it unchanged, as its tool loop needs. Thinking blocks of earlier
// turns may always be left out of a request. The caller's slices are not
// touched.
func WithoutThinkingBefore(msgs []anthropic.BetaMessageParam, n int) []anthropic.BetaMessageParam {
	if n <= 0 {
		return msgs
	}
	if n > len(msgs) {
		n = len(msgs)
	}
	head := dropThinking(msgs[:n:n])
	if len(head) == n && (n == 0 || &head[0] == &msgs[0]) {
		return msgs // nothing dropped
	}
	out := make([]anthropic.BetaMessageParam, 0, len(head)+len(msgs)-n)
	out = append(out, head...)
	return append(out, msgs[n:]...)
}

func hasThinking(blocks []anthropic.BetaContentBlockParamUnion) bool {
	for _, c := range blocks {
		if c.OfThinking != nil || c.OfRedactedThinking != nil {
			return true
		}
	}
	return false
}

// wantsFallbacks (thermometer chat.go, Sonnet 5.5 added) reports whether
// the model runs safety classifiers that can end a turn with stop_reason
// "refusal": the Fable/Mythos tier, Opus 5 / 5.5 and Sonnet 5.5, which is
// when `fallbacks: "default"` is worth requesting (and accepted).
func wantsFallbacks(model string) bool {
	for _, p := range []string{"claude-opus-5", "claude-sonnet-5-5", "claude-fable", "claude-mythos"} {
		if strings.HasPrefix(model, p) {
			return true
		}
	}
	return false
}

func (b *Backend) plain(ctx context.Context, params anthropic.BetaMessageNewParams, onText func(string)) (*Response, error) {
	var http *http.Response
	msg, err := b.client.Beta.Messages.New(ctx, params, option.WithRequestTimeout(b.timeout), option.WithResponseInto(&http))
	if err != nil {
		return nil, err
	}
	resp := fromMessage(msg)
	if http != nil {
		resp.RequestID = http.Header.Get("request-id")
	}
	if onText != nil {
		if t := resp.Text(); t != "" {
			onText(t)
		}
	}
	return resp, nil
}

// streamed reads the model's SSE, accumulating the message (the SDK's
// Accumulate merges message_start usage with message_delta's output
// tokens and stop reason, and assembles tool_use input from
// input_json_delta). events counts stream events received, so a caller
// can tell a rejected request from a stream that broke midway.
func (b *Backend) streamed(ctx context.Context, params anthropic.BetaMessageNewParams, onText func(string)) (*Response, int, error) {
	var http *http.Response
	stream := b.client.Beta.Messages.NewStreaming(ctx, params, option.WithRequestTimeout(b.timeout), option.WithResponseInto(&http))
	var acc anthropic.BetaMessage
	events := 0
	for stream.Next() {
		ev := stream.Current()
		events++
		if err := acc.Accumulate(ev); err != nil {
			return nil, events, err
		}
		if ev.Type == "content_block_delta" && onText != nil {
			if td, ok := ev.AsContentBlockDelta().Delta.AsAny().(anthropic.BetaTextDelta); ok && td.Text != "" {
				onText(td.Text)
			}
		}
	}
	if err := stream.Err(); err != nil {
		return nil, events, err
	}
	if events == 0 {
		return nil, 0, errors.New("model: empty stream")
	}
	resp := fromMessage(&acc)
	resp.Streamed = true
	if http != nil {
		resp.RequestID = http.Header.Get("request-id")
	}
	return resp, events, nil
}

func fromMessage(msg *anthropic.BetaMessage) *Response {
	resp := &Response{
		ID:         msg.ID,
		Model:      string(msg.Model),
		StopReason: string(msg.StopReason),
		Usage: Usage{
			InputTokens:              msg.Usage.InputTokens,
			OutputTokens:             msg.Usage.OutputTokens,
			CacheReadInputTokens:     msg.Usage.CacheReadInputTokens,
			CacheCreationInputTokens: msg.Usage.CacheCreationInputTokens,
		},
		raw: msg,
	}
	if msg.StopReason == anthropic.BetaStopReasonRefusal {
		resp.StopCategory = string(msg.StopDetails.Category)
	}
	for _, block := range msg.Content {
		switch v := block.AsAny().(type) {
		case anthropic.BetaTextBlock:
			resp.Content = append(resp.Content, Block{Type: "text", Text: v.Text})
		case anthropic.BetaToolUseBlock:
			resp.Content = append(resp.Content, Block{Type: "tool_use", ID: v.ID, Name: v.Name, Input: toolInput(v)})
		}
	}
	return resp
}

// toolInput (thermometer chat.go) is the input a tool runs with: the
// tool_use block's raw JSON when it is an object, else {}. Claude always
// sends an object; an open model behind a compatible server may send none,
// null or a string for a call without arguments.
func toolInput(b anthropic.BetaToolUseBlock) json.RawMessage {
	if raw := []byte(strings.TrimSpace(b.JSON.Input.Raw())); isJSONObject(raw) {
		return json.RawMessage(raw)
	}
	return json.RawMessage("{}")
}

func isJSONObject(raw []byte) bool {
	return len(raw) > 0 && raw[0] == '{' && json.Valid(raw)
}

// IsUnavailable reports whether err is a transient upstream condition —
// rate limit, overload, outage, a timeout or a connection failure — that
// the turn should report as 503 model_unavailable (README §5), as opposed
// to a bug or misconfiguration (a 400).
func IsUnavailable(err error) bool {
	var apiErr *anthropic.Error
	if errors.As(err, &apiErr) {
		return apiErr.StatusCode == 429 || apiErr.StatusCode == 408 || apiErr.StatusCode >= 500
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	var netErr net.Error
	if errors.As(err, &netErr) {
		return true
	}
	var urlErr *url.Error
	return errors.As(err, &urlErr)
}

// StatusCode is the HTTP status of an API error, 0 for anything else.
func StatusCode(err error) int {
	var apiErr *anthropic.Error
	if errors.As(err, &apiErr) {
		return apiErr.StatusCode
	}
	return 0
}

// isHardClientError: a 4xx that is not a retryable one — the server refused
// the request's shape, e.g. stream=true on a server that cannot stream.
func isHardClientError(err error) bool {
	code := StatusCode(err)
	return code >= 400 && code < 500 && code != 408 && code != 429
}

// Strict marks a tool definition strict (schema-valid arguments) with
// additionalProperties:false on its input schema, the shape the README §2
// data-op tools use. It is here, beside the request builder, so toolgen
// and the prompt builder share one spelling.
func Strict(tool anthropic.BetaToolParam) anthropic.BetaToolParam {
	tool.Strict = param.NewOpt(true)
	if tool.InputSchema.ExtraFields == nil {
		tool.InputSchema.ExtraFields = map[string]any{}
	}
	tool.InputSchema.ExtraFields["additionalProperties"] = false
	return tool
}
