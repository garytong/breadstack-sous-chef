// Package modeltest is a scripted fake of POST /v1/messages for tests: the
// model package's own and the service's end-to-end test. Each request
// consumes the next Step of the script; a Step is an HTTP status with a
// body, or a Message the fake renders either as a JSON reply or — when the
// request says "stream": true — as the Messages API's SSE event sequence
// (message_start → content_block_* → message_delta → message_stop), text
// and tool_use inputs split over several deltas so a client must
// accumulate them.
//
// Every request is captured (headers + parsed body) for assertions. The
// fake never speaks the SDK's types: scripts are plain structs/JSON.
package modeltest

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"time"
)

// Usage is the reply's token counts.
type Usage struct {
	InputTokens              int64 `json:"input_tokens"`
	OutputTokens             int64 `json:"output_tokens"`
	CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
}

// Block is one content block of a scripted reply.
type Block struct {
	Type     string // "text" | "tool_use" | "thinking"
	Text     string // text blocks
	ID       string // tool_use id
	Name     string // tool_use name
	Input    any    // tool_use input; marshalled to JSON (a non-object is sent as-is, to test the {} guard)
	Thinking string // thinking blocks
}

// Message is a scripted reply.
type Message struct {
	ID           string
	Model        string
	StopReason   string // default "end_turn"; "tool_use" when a tool_use block is present and unset
	StopCategory string // stop_details.category for a refusal
	Content      []Block
	Usage        Usage
}

// Step is one scripted response. Zero Status = 200. With Message nil the
// Body is written as-is (an error body, say); otherwise the Message is
// rendered. Delay sleeps before answering (timeouts).
type Step struct {
	Status  int
	Headers map[string]string
	Body    string
	Message *Message
	Delay   time.Duration
}

// Captured is one request the fake saw.
type Captured struct {
	Header http.Header
	Body   map[string]any // the parsed request body
	Raw    []byte
	Stream bool // body said "stream": true
	Path   string
}

// Server is the fake.
type Server struct {
	*httptest.Server
	mu       sync.Mutex
	steps    []Step
	next     int
	captured []Captured
}

// New starts a fake answering the steps in order. Past the script every
// request gets 500 with a note. Close it with Close.
func New(steps ...Step) *Server {
	s := &Server{steps: steps}
	s.Server = httptest.NewServer(http.HandlerFunc(s.handle))
	return s
}

// Requests returns the requests seen so far, in order.
func (s *Server) Requests() []Captured {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]Captured, len(s.captured))
	copy(out, s.captured)
	return out
}

// Append adds steps to the script (for tests that script turn by turn).
func (s *Server) Append(steps ...Step) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.steps = append(s.steps, steps...)
}

func (s *Server) handle(w http.ResponseWriter, r *http.Request) {
	raw, _ := io.ReadAll(r.Body)
	var body map[string]any
	_ = json.Unmarshal(raw, &body)
	stream, _ := body["stream"].(bool)
	s.mu.Lock()
	s.captured = append(s.captured, Captured{Header: r.Header.Clone(), Body: body, Raw: raw, Stream: stream, Path: r.URL.Path})
	var step Step
	scripted := s.next < len(s.steps)
	if scripted {
		step = s.steps[s.next]
		s.next++
	}
	s.mu.Unlock()
	if !scripted {
		http.Error(w, `{"type":"error","error":{"type":"api_error","message":"modeltest: script exhausted"}}`, http.StatusInternalServerError)
		return
	}
	if step.Delay > 0 {
		select {
		case <-time.After(step.Delay):
		case <-r.Context().Done():
			return
		}
	}
	for k, v := range step.Headers {
		w.Header().Set(k, v)
	}
	status := step.Status
	if status == 0 {
		status = http.StatusOK
	}
	if step.Message == nil {
		if w.Header().Get("Content-Type") == "" {
			w.Header().Set("Content-Type", "application/json")
		}
		w.WriteHeader(status)
		_, _ = io.WriteString(w, step.Body)
		return
	}
	msg := *step.Message
	fillDefaults(&msg)
	if stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(status)
		writeSSE(w, msg)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(messageJSON(msg))
}

func fillDefaults(m *Message) {
	if m.ID == "" {
		m.ID = "msg_fake"
	}
	if m.Model == "" {
		m.Model = "fake-model"
	}
	if m.StopReason == "" {
		m.StopReason = "end_turn"
		for _, b := range m.Content {
			if b.Type == "tool_use" {
				m.StopReason = "tool_use"
			}
		}
	}
}

func blockJSON(b Block) map[string]any {
	switch b.Type {
	case "tool_use":
		in := b.Input
		if in == nil {
			in = map[string]any{}
		}
		return map[string]any{"type": "tool_use", "id": b.ID, "name": b.Name, "input": in}
	case "thinking":
		return map[string]any{"type": "thinking", "thinking": b.Thinking, "signature": "sig"}
	default:
		return map[string]any{"type": "text", "text": b.Text}
	}
}

func usageJSON(u Usage) map[string]any {
	return map[string]any{
		"input_tokens": u.InputTokens, "output_tokens": u.OutputTokens,
		"cache_read_input_tokens": u.CacheReadInputTokens, "cache_creation_input_tokens": u.CacheCreationInputTokens,
	}
}

func stopJSON(m Message) (string, any) {
	if m.StopReason == "refusal" {
		return m.StopReason, map[string]any{"type": "refusal", "category": m.StopCategory, "explanation": "scripted refusal"}
	}
	return m.StopReason, nil
}

func messageJSON(m Message) map[string]any {
	content := make([]any, 0, len(m.Content))
	for _, b := range m.Content {
		content = append(content, blockJSON(b))
	}
	stop, details := stopJSON(m)
	return map[string]any{
		"id": m.ID, "type": "message", "role": "assistant", "model": m.Model,
		"content": content, "stop_reason": stop, "stop_sequence": nil, "stop_details": details,
		"usage": usageJSON(m.Usage),
	}
}

func writeEvent(w io.Writer, name string, data any) {
	js, _ := json.Marshal(data)
	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", name, js)
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

// writeSSE renders m as the stream a real server sends. Usage: message_start
// carries input/cache tokens with output_tokens 1; message_delta carries
// the final output_tokens (and repeats the cache counts), so a client that
// only reads one of them gets the wrong numbers.
func writeSSE(w io.Writer, m Message) {
	start := messageJSON(m)
	start["content"] = []any{}
	start["stop_reason"] = nil
	start["stop_details"] = nil
	start["usage"] = usageJSON(Usage{InputTokens: m.Usage.InputTokens, OutputTokens: 1,
		CacheReadInputTokens: m.Usage.CacheReadInputTokens, CacheCreationInputTokens: m.Usage.CacheCreationInputTokens})
	writeEvent(w, "message_start", map[string]any{"type": "message_start", "message": start})
	for i, b := range m.Content {
		switch b.Type {
		case "tool_use":
			writeEvent(w, "content_block_start", map[string]any{"type": "content_block_start", "index": i,
				"content_block": map[string]any{"type": "tool_use", "id": b.ID, "name": b.Name, "input": map[string]any{}}})
			in := b.Input
			if in == nil {
				in = map[string]any{}
			}
			js, _ := json.Marshal(in)
			for _, part := range split(string(js), 3) {
				writeEvent(w, "content_block_delta", map[string]any{"type": "content_block_delta", "index": i,
					"delta": map[string]any{"type": "input_json_delta", "partial_json": part}})
			}
		case "thinking":
			writeEvent(w, "content_block_start", map[string]any{"type": "content_block_start", "index": i,
				"content_block": map[string]any{"type": "thinking", "thinking": "", "signature": ""}})
			writeEvent(w, "content_block_delta", map[string]any{"type": "content_block_delta", "index": i,
				"delta": map[string]any{"type": "thinking_delta", "thinking": b.Thinking}})
			writeEvent(w, "content_block_delta", map[string]any{"type": "content_block_delta", "index": i,
				"delta": map[string]any{"type": "signature_delta", "signature": "sig"}})
		default:
			writeEvent(w, "content_block_start", map[string]any{"type": "content_block_start", "index": i,
				"content_block": map[string]any{"type": "text", "text": ""}})
			for _, part := range split(b.Text, 2) {
				writeEvent(w, "content_block_delta", map[string]any{"type": "content_block_delta", "index": i,
					"delta": map[string]any{"type": "text_delta", "text": part}})
			}
		}
		writeEvent(w, "content_block_stop", map[string]any{"type": "content_block_stop", "index": i})
	}
	stop, details := stopJSON(m)
	writeEvent(w, "message_delta", map[string]any{"type": "message_delta",
		"delta": map[string]any{"stop_reason": stop, "stop_sequence": nil, "stop_details": details},
		"usage": usageJSON(m.Usage)})
	writeEvent(w, "message_stop", map[string]any{"type": "message_stop"})
}

// split cuts s into n roughly equal non-empty pieces (fewer when s is
// short), on rune boundaries.
func split(s string, n int) []string {
	runes := []rune(s)
	if len(runes) == 0 {
		return []string{""}
	}
	if n > len(runes) {
		n = len(runes)
	}
	var out []string
	for i := 0; i < n; i++ {
		lo, hi := len(runes)*i/n, len(runes)*(i+1)/n
		out = append(out, string(runes[lo:hi]))
	}
	return out
}

// Text and ToolUse are Block shorthands for scripts.
func Text(s string) Block { return Block{Type: "text", Text: s} }
func ToolUse(id, name string, input any) Block {
	return Block{Type: "tool_use", ID: id, Name: name, Input: input}
}
func Thinking(s string) Block { return Block{Type: "thinking", Thinking: s} }

// Reply is a Step answering 200 with m.
func Reply(m Message) Step { return Step{Message: &m} }

// Error is a Step answering status with an Anthropic-shaped error body and
// the given headers (e.g. "Retry-After-Ms": "1").
func Error(status int, errType, message string, headers map[string]string) Step {
	body := fmt.Sprintf(`{"type":"error","error":{"type":%q,"message":%q}}`, errType, message)
	return Step{Status: status, Body: body, Headers: headers}
}

// SystemText returns the text of the request's system blocks joined, for
// prefix-stability assertions.
func (c Captured) SystemText() string {
	var sb strings.Builder
	if blocks, ok := c.Body["system"].([]any); ok {
		for _, b := range blocks {
			if m, ok := b.(map[string]any); ok {
				sb.WriteString(fmt.Sprint(m["text"]))
			}
		}
	} else if s, ok := c.Body["system"].(string); ok {
		sb.WriteString(s)
	}
	return sb.String()
}
