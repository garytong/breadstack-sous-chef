// Package trace records production turns for the lab (README §4, §6):
// what the model was sent and what it said (ops), what the stream emitted
// and the browser reported back (events), the per-turn context, and later
// the merchant's feedback. Traces are kept 90 days unless the store opted
// out; the opt-out only disables trace storage, never a turn.
//
// Pattern copied from breadstack-thermometer server/internal/chat/trace.go
// (traceRecord / transcript): every method is a no-op on a nil Recorder,
// which is what Begin returns for an opted-out store, so the loop never
// branches on tracing. Two rules carried over verbatim: thinking text is
// never written anywhere (StripThinking), and the model's raw tool calls are
// stored apart from the browser's outcomes so training never learns a
// correction as model output.
package trace

import (
	"bytes"
	"context"
	"encoding/json"
	"log"
	"sync"
	"time"

	"github.com/breadstack/sous-chef/internal/store"
)

// Store is the slice of *store.Store the recorder needs.
type Store interface {
	IsOptedOut(ctx context.Context, storeID string) (bool, error)
	InsertTrace(ctx context.Context, t store.Trace) error
}

// Op is one raw model tool call as the model wrote it, plus which model
// call of the turn produced it. Input is the tool_use block's input,
// untouched.
type Op struct {
	Call      int             `json:"call"` // 1-based model call within the turn
	ToolUseID string          `json:"tool_use_id"`
	Name      string          `json:"name"`
	Input     json.RawMessage `json:"input"`
	Server    bool            `json:"server,omitempty"` // answered server-side (lookup_fields…)
}

// ModelCall summarises one Messages API call of the turn: what came back
// apart from the tool calls (which are Ops). Text is the model's prose for
// that call; thinking is never recorded.
type ModelCall struct {
	Call       int    `json:"call"`
	Model      string `json:"model"`
	StopReason string `json:"stop_reason"`
	Text       string `json:"text,omitempty"`
	Usage      Usage  `json:"usage"`
	LatencyMS  int64  `json:"latency_ms"`
	RequestID  string `json:"request_id,omitempty"`
}

// Usage mirrors the Messages API usage block (input_tokens is the uncached
// input only; the cache fields hold the rest).
type Usage struct {
	InputTokens              int64 `json:"input_tokens"`
	OutputTokens             int64 `json:"output_tokens"`
	CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
}

// Event is one SSE event the turn emitted, or one tool result the browser
// posted back (Kind "tool.result"). Seq is the stream's sequence number;
// browser results carry the seq of the tool.call they answer: the loop
// records them via Event(seq, "tool.result", data) with the turn's own seq
// and data.for_seq (README §4); Recorder.ToolResult is the legacy helper that
// numbers from the event count and sets the top-level ForSeq instead.
type Event struct {
	Seq    int             `json:"seq"`
	Kind   string          `json:"event"` // text.delta | tool.call | turn.suspended | turn.waiting | turn.done | turn.error | tool.result
	Data   json.RawMessage `json:"data,omitempty"`
	At     time.Time       `json:"at"`
	ForSeq int             `json:"for_seq,omitempty"`
}

// Outcome is how the turn ended.
type Outcome struct {
	Status string `json:"status"` // done | error | expired
	Stop   string `json:"stop,omitempty"`
	Error  string `json:"error,omitempty"`
	Calls  int    `json:"calls"`
	Usage  Usage  `json:"usage"`
}

// opsDoc is the traces.ops column.
type opsDoc struct {
	ManifestHash string      `json:"manifest_hash"`
	BuildID      string      `json:"build_id"`
	Route        string      `json:"route"`
	UserEmail    string      `json:"user_email,omitempty"`
	Text         string      `json:"text"` // the merchant's message
	Calls        []ModelCall `json:"calls"`
	Ops          []Op        `json:"ops"`
	Outcome      Outcome     `json:"outcome"`
	StartedAt    time.Time   `json:"started_at"`
	EndedAt      time.Time   `json:"ended_at"`
}

// Begin is what Recorder needs to start.
type Begin struct {
	TurnID       string
	SiteID       string
	StoreID      string
	ManifestHash string
	BuildID      string
	Route        string
	UserEmail    string
	Text         string          // merchant text
	Context      json.RawMessage // the turn request's context object
	StartedAt    time.Time
}

// Recorder accumulates one turn's trace. Safe for concurrent use (the
// stream writer and the loop run on different goroutines). A nil *Recorder
// is valid and does nothing.
type Recorder struct {
	st   Store
	now  func() time.Time
	mu   sync.Mutex
	rec  store.Trace
	ops  opsDoc
	evs  []Event
	done bool
}

// New starts a recorder unless the store opted out. A failed opt-out lookup
// FAILS CLOSED: it is logged and the turn is not traced. A missed trace is
// recoverable; a trace of an opted-out merchant is not, and the turn itself
// is never affected either way.
func New(ctx context.Context, st Store, b Begin) *Recorder {
	if st == nil {
		return nil
	}
	out, err := st.IsOptedOut(ctx, b.StoreID)
	if err != nil {
		log.Printf("trace: WARNING opt-out lookup for store %q failed, skipping trace for turn %s: %v", b.StoreID, b.TurnID, err)
		return nil
	}
	if out {
		return nil
	}
	started := b.StartedAt
	if started.IsZero() {
		started = time.Now().UTC()
	}
	return &Recorder{
		st:  st,
		now: func() time.Time { return time.Now().UTC() },
		rec: store.Trace{TurnID: b.TurnID, SiteID: b.SiteID, StoreID: b.StoreID, Context: b.Context},
		ops: opsDoc{ManifestHash: b.ManifestHash, BuildID: b.BuildID, Route: b.Route, UserEmail: b.UserEmail,
			Text: b.Text, Calls: []ModelCall{}, Ops: []Op{}, StartedAt: started},
		evs: []Event{},
	}
}

// Enabled reports whether this recorder writes anything.
func (r *Recorder) Enabled() bool { return r != nil }

// SetClock replaces the recorder's clock (tests).
func (r *Recorder) SetClock(now func() time.Time) {
	if r != nil {
		r.now = now
	}
}

// ModelCall records one Messages API response: its prose and usage here,
// its tool_use blocks as Ops. blocks is the response content array; any
// thinking/redacted_thinking block is dropped before anything is read.
func (r *Recorder) ModelCall(mc ModelCall, blocks json.RawMessage) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if mc.Call == 0 {
		mc.Call = len(r.ops.Calls) + 1
	}
	var text string
	for _, b := range parseBlocks(StripThinking(blocks)) {
		switch b.Type {
		case "text":
			text += b.Text
		case "tool_use":
			input := b.Input
			if len(input) == 0 {
				input = json.RawMessage(`{}`)
			}
			r.ops.Ops = append(r.ops.Ops, Op{Call: mc.Call, ToolUseID: b.ID, Name: b.Name, Input: input})
		}
	}
	if mc.Text == "" {
		mc.Text = text
	}
	r.ops.Calls = append(r.ops.Calls, mc)
	r.ops.Outcome.Calls = len(r.ops.Calls)
	r.ops.Outcome.Usage.InputTokens += mc.Usage.InputTokens
	r.ops.Outcome.Usage.OutputTokens += mc.Usage.OutputTokens
	r.ops.Outcome.Usage.CacheReadInputTokens += mc.Usage.CacheReadInputTokens
	r.ops.Outcome.Usage.CacheCreationInputTokens += mc.Usage.CacheCreationInputTokens
}

// MarkServer flags the op with toolUseID as answered server-side.
func (r *Recorder) MarkServer(toolUseID string) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	for i := range r.ops.Ops {
		if r.ops.Ops[i].ToolUseID == toolUseID {
			r.ops.Ops[i].Server = true
		}
	}
}

// Event records one emitted SSE event. text.delta events are coalesced:
// consecutive deltas become one event so a streamed paragraph is not 200
// rows of three characters.
func (r *Recorder) Event(seq int, kind string, data json.RawMessage) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if kind == "text.delta" && len(r.evs) > 0 && r.evs[len(r.evs)-1].Kind == "text.delta" {
		last := &r.evs[len(r.evs)-1]
		if merged, ok := mergeTextDelta(last.Data, data); ok {
			last.Data = merged
			return
		}
	}
	r.evs = append(r.evs, Event{Seq: seq, Kind: kind, Data: data, At: r.now()})
}

// ToolResult records what the browser (or the server-tool dispatcher)
// answered for the tool.call at forSeq. result is the tool-results body
// item verbatim ({toolCallId, ok, error?, warnings?, written?, asset?}).
func (r *Recorder) ToolResult(forSeq int, result json.RawMessage) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.evs = append(r.evs, Event{Seq: len(r.evs) + 1, Kind: "tool.result", Data: result, At: r.now(), ForSeq: forSeq})
}

// Finish closes the trace with its outcome and writes it. It is safe to
// call once; later calls are no-ops. A write that fails is logged, never the
// turn's failure (thermometer finish). The context should be
// context.WithoutCancel(request ctx) so a client hanging up does not lose
// the row.
func (r *Recorder) Finish(ctx context.Context, status, stop, errMsg string) error {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	if r.done {
		r.mu.Unlock()
		return nil
	}
	r.done = true
	r.ops.Outcome.Status = status
	r.ops.Outcome.Stop = stop
	r.ops.Outcome.Error = errMsg
	r.ops.EndedAt = r.now()
	ops, err := marshal(r.ops)
	if err != nil {
		r.mu.Unlock()
		return err
	}
	evs, err := marshal(r.evs)
	if err != nil {
		r.mu.Unlock()
		return err
	}
	rec := r.rec
	rec.Ops = ops
	rec.Events = evs
	rec.Created = r.ops.StartedAt
	r.mu.Unlock()
	if err := r.st.InsertTrace(ctx, rec); err != nil {
		log.Printf("trace: WARNING trace for turn %s not recorded: %v", rec.TurnID, err)
		return err
	}
	return nil
}

// Snapshot returns the ops and events documents as they stand (tests and
// the debugging endpoint); nil recorder → nil, nil.
func (r *Recorder) Snapshot() (ops, events json.RawMessage) {
	if r == nil {
		return nil, nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	ops, _ = marshal(r.ops)
	events, _ = marshal(r.evs)
	return ops, events
}

// --- content blocks ---

// block is the subset of a Messages API content block the trace reads.
type block struct {
	Type  string          `json:"type"`
	Text  string          `json:"text,omitempty"`
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
}

func parseBlocks(raw json.RawMessage) []block {
	var blocks []block
	if len(raw) == 0 || json.Unmarshal(raw, &blocks) != nil {
		return nil
	}
	return blocks
}

// StripThinking removes every thinking and redacted_thinking block from a
// Messages API content array and returns the rest byte-for-byte (each
// surviving block is re-emitted from its raw bytes, so unknown block types
// and field order survive). Non-array input (a plain string content) is
// returned unchanged. Thinking is never stored: not in traces, not in
// messages (README §3; thermometer trace.go traceThinking rule, applied to
// every model here, Claude or in-house).
func StripThinking(content json.RawMessage) json.RawMessage {
	var items []json.RawMessage
	if len(content) == 0 || json.Unmarshal(content, &items) != nil {
		return content
	}
	kept := make([]json.RawMessage, 0, len(items))
	dropped := false
	for _, it := range items {
		var head struct {
			Type string `json:"type"`
		}
		if json.Unmarshal(it, &head) == nil && (head.Type == "thinking" || head.Type == "redacted_thinking") {
			dropped = true
			continue
		}
		kept = append(kept, it)
	}
	if !dropped {
		return content
	}
	out, err := json.Marshal(kept)
	if err != nil {
		return content
	}
	return out
}

// HasThinking reports whether a content array carries a thinking block
// (the messages repository's guard).
func HasThinking(content json.RawMessage) bool {
	for _, b := range parseBlocks(content) {
		if b.Type == "thinking" || b.Type == "redacted_thinking" {
			return true
		}
	}
	return false
}

// mergeTextDelta concatenates two {text} payloads.
func mergeTextDelta(a, b json.RawMessage) (json.RawMessage, bool) {
	var x, y struct {
		Text string `json:"text"`
	}
	if json.Unmarshal(a, &x) != nil || json.Unmarshal(b, &y) != nil {
		return nil, false
	}
	out, err := json.Marshal(struct {
		Text string `json:"text"`
	}{x.Text + y.Text})
	return out, err == nil
}

// marshal is json.Marshal without HTML escaping (the prompts' <tags> and
// "&" stay readable; thermometer write()).
func marshal(v any) (json.RawMessage, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return json.RawMessage(bytes.TrimRight(buf.Bytes(), "\n")), nil
}
