package api

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/breadstack/sous-chef/internal/auth"
)

type fakePrimer struct {
	mu       sync.Mutex
	calls    []string
	dispatch PrimeDispatch
}

func (f *fakePrimer) record(s string) {
	f.mu.Lock()
	f.calls = append(f.calls, s)
	f.mu.Unlock()
}
func (f *fakePrimer) ManifestRegistered(hash string) PrimeDispatch {
	f.record("manifest " + hash[:8])
	return f.dispatch
}
func (f *fakePrimer) ConversationOpened(site, conv, hash string) PrimeDispatch {
	f.record("opened " + site + " " + conv + " " + hash[:8])
	return f.dispatch
}
func (f *fakePrimer) ConversationEnded(site, hash string) PrimeDispatch {
	f.record("ended " + site + " " + hash[:8])
	return f.dispatch
}

func (f *fakePrimer) got() string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return strings.Join(f.calls, "|")
}

// The handlers tell the Primer about manifest uploads, the panel reading a
// conversation (with X-Manifest-Hash) and a conversation end — after the
// response, and only on success (a 404 conversation_not_found read is the
// panel opening on a site with none: the floor).
func TestPrimerTriggers(t *testing.T) {
	pr := &fakePrimer{}
	convs := &fakeConvs{conv: Conversation{ConversationID: "conv-1"}}
	ts := newTestServer(t, Deps{Manifests: &fakeRegistry{}, HashManifest: testHasher, Conversations: convs, Primer: pr})
	h := "X-Manifest-Hash"

	do(t, ts, "PUT", "/v1/manifests/"+goodHash, "key-a", `{"schema":1}`)
	do(t, ts, "PUT", "/v1/manifests/"+goodHash, "key-a", `{"mismatch":1}`) // refused: no prime
	do(t, ts, "GET", "/v1/conversations/site-1", "key-a", "", h, goodHash)
	do(t, ts, "GET", "/v1/conversations/site-1?before=5", "key-a", "", h, goodHash) // paging: no prime
	do(t, ts, "GET", "/v1/conversations/site-1", "key-a", "")                       // no hash: no prime
	do(t, ts, "GET", "/v1/conversations/site-1", "key-a", "", h, "not-a-hash")      // bad hash: no prime
	do(t, ts, "DELETE", "/v1/conversations/site-1", "key-a", "", h, goodHash)
	want := "manifest " + goodHash[:8] + "|opened site-1 conv-1 " + goodHash[:8] + "|ended site-1 " + goodHash[:8]
	if got := pr.got(); got != want {
		t.Errorf("primer calls = %q, want %q", got, want)
	}

	convs.err = ErrNotFound("conversation_not_found", "none open")
	res, _ := do(t, ts, "GET", "/v1/conversations/site-2", "key-a", "", h, goodHash)
	if res.StatusCode != 404 || !strings.HasSuffix(pr.got(), "|opened site-2  "+goodHash[:8]) {
		t.Errorf("404 read: %d, calls %q", res.StatusCode, pr.got())
	}
	convs.err = ErrStoreUnavailable()
	do(t, ts, "GET", "/v1/conversations/site-3", "key-a", "", h, goodHash)
	do(t, ts, "DELETE", "/v1/conversations/site-3", "key-a", "", h, goodHash)
	if strings.Contains(pr.got(), "site-3") {
		t.Errorf("a failed read/end primed: %q", pr.got())
	}

	// No Primer: the routes answer as before.
	ts2 := newTestServer(t, Deps{Conversations: &fakeConvs{conv: Conversation{ConversationID: "c"}}})
	if res, _ := do(t, ts2, "GET", "/v1/conversations/site-1", "key-a", "", h, goodHash); res.StatusCode != 200 {
		t.Errorf("without a Primer: %d", res.StatusCode)
	}
}

// The handler sends its complete response (sized, flushed) before it waits
// on the prime's dispatch, so the client is not held while the instance
// keeps its CPU for the prime; the handler returns once the dispatch closes.
func TestPrimerHold_ResponseCompleteBeforeTheHandlerReturns(t *testing.T) {
	ch := make(chan struct{})
	pr := &fakePrimer{dispatch: ch}
	convs := &fakeConvs{conv: Conversation{ConversationID: "conv-1"}}
	returned := make(chan time.Time, 1)
	srv := New(auth.NewKeys([]string{"key-a"}), Options{BuildID: "b", Routes: []string{"default"}, DefaultRouteConfigured: true, Heartbeat: -1, Logf: func(string, ...any) {}},
		Deps{Conversations: convs, Primer: pr})
	h := srv.Handler()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h.ServeHTTP(w, r)
		returned <- time.Now()
	}))
	t.Cleanup(ts.Close)
	go func() { time.Sleep(400 * time.Millisecond); close(ch) }()
	started := time.Now()
	res, body := do(t, ts, "GET", "/v1/conversations/site-1", "key-a", "", "X-Manifest-Hash", goodHash)
	got := time.Since(started)
	if res.StatusCode != 200 || !strings.Contains(body, "conv-1") || res.ContentLength <= 0 {
		t.Fatalf("status %d, length %d, body %q", res.StatusCode, res.ContentLength, body)
	}
	if got > 300*time.Millisecond {
		t.Errorf("the client waited %s for the prime", got)
	}
	if end := (<-returned).Sub(started); end < 350*time.Millisecond {
		t.Errorf("the handler returned after %s, before the dispatch closed", end)
	}
}
