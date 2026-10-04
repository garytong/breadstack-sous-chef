package api

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestEncodeEventDataInjectsTurnIDAndSeq(t *testing.T) {
	b, err := encodeEventData(Event{Type: EventTextDelta, TurnID: "t1", Seq: 3, Data: map[string]any{"text": "hi"}})
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	if m["text"] != "hi" || m["turnId"] != "t1" || m["seq"] != float64(3) {
		t.Errorf("got %s", b)
	}

	// nil data → {"seq":0} only (no turnId when empty)
	b, err = encodeEventData(Event{Type: EventHeartbeat})
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != `{"seq":0}` {
		t.Errorf("nil data = %s", b)
	}

	// struct data works too
	type done struct {
		Stop string `json:"stop"`
	}
	b, _ = encodeEventData(Event{Type: EventTurnDone, TurnID: "t", Seq: 9, Data: done{"end_turn"}})
	if !strings.Contains(string(b), `"stop":"end_turn"`) || !strings.Contains(string(b), `"seq":9`) {
		t.Errorf("struct data = %s", b)
	}

	// non-object data is rejected
	if _, err := encodeEventData(Event{Type: "x", Data: []int{1}}); err == nil {
		t.Error("array data should be rejected")
	}
	if _, err := encodeEventData(Event{Type: "x", Data: "str"}); err == nil {
		t.Error("string data should be rejected")
	}
}

func TestSSEWriterFrames(t *testing.T) {
	rr := httptest.NewRecorder()
	s, err := NewSSEWriter(rr, -1) // heartbeat off
	if err != nil {
		t.Fatal(err)
	}
	if ct := rr.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Errorf("Content-Type = %q", ct)
	}
	if rr.Code != http.StatusOK {
		t.Errorf("status = %d", rr.Code)
	}
	if !rr.Flushed {
		t.Error("headers should be flushed on open")
	}
	must := func(ev Event) {
		if err := s.Emit(ev); err != nil {
			t.Fatalf("Emit %s: %v", ev.Type, err)
		}
	}
	must(Event{Type: EventTextDelta, TurnID: "t1", Seq: 1, Data: map[string]string{"text": "Hel"}})
	must(Event{Type: EventToolCall, TurnID: "t1", Seq: 2, Data: map[string]any{"id": "c1", "name": "set_fields", "input": map[string]any{"values": map[string]any{"title": "x"}}}})
	must(Event{Type: EventTurnWaiting, TurnID: "t1", Seq: 3, Data: map[string]any{"toolCallIds": []string{"c1"}}})
	s.Close()

	body := rr.Body.String()
	frames := strings.Split(strings.TrimSuffix(body, "\n\n"), "\n\n")
	if len(frames) != 3 {
		t.Fatalf("got %d frames:\n%s", len(frames), body)
	}
	wantTypes := []string{EventTextDelta, EventToolCall, EventTurnWaiting}
	for i, f := range frames {
		lines := strings.Split(f, "\n")
		if len(lines) != 2 {
			t.Errorf("frame %d has %d lines: %q", i, len(lines), f)
			continue
		}
		if lines[0] != "event: "+wantTypes[i] {
			t.Errorf("frame %d event line = %q", i, lines[0])
		}
		if !strings.HasPrefix(lines[1], "data: {") || !strings.Contains(lines[1], `"turnId":"t1"`) {
			t.Errorf("frame %d data line = %q", i, lines[1])
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(strings.TrimPrefix(lines[1], "data: ")), &m); err != nil {
			t.Errorf("frame %d data not JSON: %v", i, err)
		} else if m["seq"] != float64(i+1) {
			t.Errorf("frame %d seq = %v", i, m["seq"])
		}
	}
	if s.Written() != len(body) {
		t.Errorf("Written = %d, body = %d", s.Written(), len(body))
	}

	// After Close, Emit fails with ErrClientGone and Close is idempotent.
	if err := s.Emit(Event{Type: EventTurnDone}); !errors.Is(err, ErrClientGone) {
		t.Errorf("Emit after Close = %v", err)
	}
	s.Close()
}

// syncRecorder is an httptest.ResponseRecorder whose body can be read
// while the heartbeat goroutine is writing to it.
type syncRecorder struct {
	mu sync.Mutex
	rr *httptest.ResponseRecorder
}

func (s *syncRecorder) Header() http.Header { return s.rr.Header() }
func (s *syncRecorder) WriteHeader(c int)   { s.mu.Lock(); s.rr.WriteHeader(c); s.mu.Unlock() }
func (s *syncRecorder) Write(b []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.rr.Write(b)
}
func (s *syncRecorder) Flush() { s.mu.Lock(); s.rr.Flush(); s.mu.Unlock() }
func (s *syncRecorder) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.rr.Body.String()
}

func TestSSEWriterHeartbeat(t *testing.T) {
	rec := &syncRecorder{rr: httptest.NewRecorder()}
	s, err := NewSSEWriter(rec, 20*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	// Wait for at least two heartbeats without emitting anything.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) && strings.Count(rec.String(), "event: heartbeat") < 2 {
		time.Sleep(5 * time.Millisecond)
	}
	s.Close()
	body := rec.String()
	n := strings.Count(body, "event: heartbeat\ndata: {}\n\n")
	if n < 2 {
		t.Fatalf("expected >= 2 heartbeat frames, got %d:\n%s", n, body)
	}
	// Heartbeats carry no seq / turnId.
	if strings.Contains(body, `"seq"`) {
		t.Errorf("heartbeat must not carry seq: %s", body)
	}
	// No heartbeats after Close.
	before := len(rec.String())
	time.Sleep(60 * time.Millisecond)
	if after := len(rec.String()); after != before {
		t.Errorf("writer kept writing after Close (%d -> %d bytes)", before, after)
	}
}

// failingWriter reports a broken connection on the first write.
type failingWriter struct {
	http.ResponseWriter
	fail bool
}

func (f *failingWriter) Write(b []byte) (int, error) {
	if f.fail {
		return 0, errors.New("broken pipe")
	}
	return f.ResponseWriter.Write(b)
}
func (f *failingWriter) Flush() {}

func TestSSEWriterClientGone(t *testing.T) {
	fw := &failingWriter{ResponseWriter: httptest.NewRecorder()}
	s, err := NewSSEWriter(fw, -1)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	fw.fail = true
	err = s.Emit(Event{Type: EventTextDelta, Data: map[string]string{"text": "x"}})
	if !errors.Is(err, ErrClientGone) {
		t.Fatalf("Emit on broken writer = %v, want ErrClientGone", err)
	}
	// sticky
	if err := s.Emit(Event{Type: EventTextDelta}); !errors.Is(err, ErrClientGone) {
		t.Errorf("second Emit = %v", err)
	}
}

type noFlush struct{ http.ResponseWriter }

func TestSSEWriterRequiresFlusher(t *testing.T) {
	// Hide Flush by wrapping in a struct that only embeds the interface
	// (method set is the interface's; httptest.ResponseRecorder's Flush is not promoted).
	var w http.ResponseWriter = struct{ http.ResponseWriter }{noFlush{httptest.NewRecorder()}}
	if _, err := NewSSEWriter(w, -1); err == nil {
		t.Error("expected error for non-Flusher writer")
	}
}

func TestCollector(t *testing.T) {
	c := &Collector{}
	if c.Len() != 0 || len(c.Events()) != 0 || c.Events() == nil {
		t.Fatal("empty collector should return a non-nil empty slice")
	}
	_ = c.Emit(Event{Type: EventHeartbeat})
	_ = c.Emit(Event{Type: EventTextDelta, TurnID: "t", Seq: 1, Data: map[string]string{"text": "a"}})
	_ = c.Emit(Event{Type: EventTurnDone, TurnID: "t", Seq: 2, Data: map[string]string{"stop": "end_turn"}})
	evs := c.Events()
	if len(evs) != 2 {
		t.Fatalf("heartbeats must be dropped; got %d events", len(evs))
	}
	if evs[0].Event != EventTextDelta || !strings.Contains(string(evs[0].Data), `"seq":1`) {
		t.Errorf("evs[0] = %+v", evs[0])
	}
	if err := c.Emit(Event{Type: "x", Data: 42}); err == nil {
		t.Error("non-object data should error")
	}
	b, _ := json.Marshal(map[string]any{"turnId": "t", "events": c.Events()})
	if !strings.Contains(string(b), `"events":[{"event":"text.delta","data":{`) {
		t.Errorf("envelope = %s", b)
	}
}

func TestWantsJSON(t *testing.T) {
	cases := map[string]bool{
		"":                                    false,
		"text/event-stream":                   false,
		"application/json":                    true,
		"application/json; charset=utf-8":     true,
		"text/event-stream, application/json": false,
		"application/json, text/event-stream": false,
		"*/*":                                 false,
		"text/html,application/json;q=0.9":    true,
	}
	for accept, want := range cases {
		r := httptest.NewRequest(http.MethodPost, "/v1/turns", nil)
		if accept != "" {
			r.Header.Set("Accept", accept)
		}
		if got := wantsJSON(r); got != want {
			t.Errorf("wantsJSON(%q) = %v, want %v", accept, got, want)
		}
	}
}
