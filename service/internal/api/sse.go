package api

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"sync"
	"time"
)

// DefaultHeartbeat is the README §1.2 interval for `heartbeat` events.
const DefaultHeartbeat = 15 * time.Second

// ErrClientGone is returned by Emit after a write failed or the sink was
// closed; the turn should stop streaming.
var ErrClientGone = errors.New("sse: client gone")

// encodeEventData marshals ev.Data and injects turnId and seq. Data must be
// a JSON object (or nil); anything else is an error so a bug surfaces in
// tests rather than as an unparseable stream.
func encodeEventData(ev Event) ([]byte, error) {
	obj := map[string]json.RawMessage{}
	if ev.Data != nil {
		raw, err := json.Marshal(ev.Data)
		if err != nil {
			return nil, fmt.Errorf("event %s: %w", ev.Type, err)
		}
		if err := json.Unmarshal(raw, &obj); err != nil {
			return nil, fmt.Errorf("event %s: data must be a JSON object: %w", ev.Type, err)
		}
	}
	if ev.TurnID != "" {
		obj["turnId"] = json.RawMessage(strconv.Quote(ev.TurnID))
	}
	obj["seq"] = json.RawMessage(strconv.Itoa(ev.Seq))
	return json.Marshal(obj)
}

// SSEWriter streams events as text/event-stream and keeps the connection
// alive with `heartbeat` events while nothing else is being sent (model
// calls can take tens of seconds; Cloud Run's HTTP/1 idle cut-off and the
// browser's fetch both prefer traffic).
type SSEWriter struct {
	w       io.Writer
	flush   func()
	mu      sync.Mutex
	closed  bool
	err     error
	stop    chan struct{}
	done    chan struct{}
	timeout time.Duration
	written int
}

// NewSSEWriter sets the stream headers, writes them (200) and starts the
// heartbeat ticker. heartbeat <= 0 disables heartbeats. The caller must
// Close it before returning from the handler.
func NewSSEWriter(w http.ResponseWriter, heartbeat time.Duration) (*SSEWriter, error) {
	fl, ok := w.(http.Flusher)
	if !ok {
		return nil, errors.New("sse: ResponseWriter does not support flushing")
	}
	h := w.Header()
	h.Set("Content-Type", "text/event-stream; charset=utf-8")
	h.Set("Cache-Control", "no-cache, no-store")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no") // nginx-style proxies: do not buffer
	w.WriteHeader(http.StatusOK)
	fl.Flush()

	s := &SSEWriter{w: w, flush: fl.Flush, stop: make(chan struct{}), done: make(chan struct{}), timeout: heartbeat}
	go s.heartbeatLoop()
	return s, nil
}

func (s *SSEWriter) heartbeatLoop() {
	defer close(s.done)
	if s.timeout <= 0 {
		<-s.stop
		return
	}
	t := time.NewTicker(s.timeout)
	defer t.Stop()
	for {
		select {
		case <-s.stop:
			return
		case <-t.C:
			s.mu.Lock()
			if !s.closed {
				// Heartbeats carry no turnId/seq: they are transport, not
				// turn events, and must not advance the sequence.
				s.writeLocked(EventHeartbeat, []byte("{}"))
			}
			s.mu.Unlock()
		}
	}
}

// Emit writes one event frame and flushes. The ticker is not reset on
// Emit on purpose: a heartbeat every interval even during bursts is
// harmless and keeps the logic trivially correct.
func (s *SSEWriter) Emit(ev Event) error {
	data, err := encodeEventData(ev)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return ErrClientGone
	}
	return s.writeLocked(ev.Type, data)
}

// writeLocked writes `event: <type>\ndata: <json>\n\n`. The JSON encoder
// never emits raw newlines, so one data line is always enough.
func (s *SSEWriter) writeLocked(typ string, data []byte) error {
	if s.err != nil {
		return s.err
	}
	var b bytes.Buffer
	b.Grow(len(typ) + len(data) + 16)
	b.WriteString("event: ")
	b.WriteString(typ)
	b.WriteString("\ndata: ")
	b.Write(data)
	b.WriteString("\n\n")
	n, err := s.w.Write(b.Bytes())
	s.written += n
	if err != nil {
		s.err = fmt.Errorf("%w: %v", ErrClientGone, err)
		return s.err
	}
	s.flush()
	return nil
}

// Close stops the heartbeat and marks the writer closed; later Emits return
// ErrClientGone. Safe to call twice.
func (s *SSEWriter) Close() {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return
	}
	s.closed = true
	s.mu.Unlock()
	close(s.stop)
	<-s.done
}

// Written is the number of bytes written so far (tests, logging).
func (s *SSEWriter) Written() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.written
}

// Collector is the `Accept: application/json` fallback: it buffers events
// and the handler renders `{turnId, events:[{event, data}]}` at the end.
// Heartbeats are never collected (nothing to keep alive).
type Collector struct {
	mu     sync.Mutex
	events []CollectedEvent
}

// CollectedEvent is one buffered event in the JSON fallback envelope.
type CollectedEvent struct {
	Event string          `json:"event"`
	Data  json.RawMessage `json:"data"`
}

// Emit buffers the event.
func (c *Collector) Emit(ev Event) error {
	if ev.Type == EventHeartbeat {
		return nil
	}
	data, err := encodeEventData(ev)
	if err != nil {
		return err
	}
	c.mu.Lock()
	c.events = append(c.events, CollectedEvent{Event: ev.Type, Data: data})
	c.mu.Unlock()
	return nil
}

// Events returns the buffered events (never nil, so it marshals as []).
func (c *Collector) Events() []CollectedEvent {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]CollectedEvent, len(c.events))
	copy(out, c.events)
	return out
}

// Len is the number of buffered events.
func (c *Collector) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.events)
}

// wantsJSON implements the README §1.2 negotiation: SSE is the default;
// only an explicit `Accept: application/json` without text/event-stream
// selects the JSON envelope.
func wantsJSON(r *http.Request) bool {
	accept := r.Header.Get("Accept")
	if accept == "" {
		return false
	}
	hasJSON, hasSSE := false, false
	for _, part := range bytes.Split([]byte(accept), []byte(",")) {
		mt := string(bytes.TrimSpace(bytes.SplitN(part, []byte(";"), 2)[0]))
		switch mt {
		case "application/json":
			hasJSON = true
		case "text/event-stream":
			hasSSE = true
		}
	}
	return hasJSON && !hasSSE
}
