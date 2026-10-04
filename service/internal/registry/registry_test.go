package registry

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/breadstack/sous-chef/internal/api"
	"github.com/breadstack/sous-chef/internal/manifest"
	"github.com/breadstack/sous-chef/internal/store"
)

// memStore is an in-memory Store.
type memStore struct {
	mu   sync.Mutex
	rows map[string]store.Manifest
	seq  int
	gets int
}

func newMemStore() *memStore { return &memStore{rows: map[string]store.Manifest{}} }

func (s *memStore) UpsertManifest(_ context.Context, m store.Manifest) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.rows[m.Hash]; ok {
		return false, nil
	}
	s.seq++
	m.FirstSeen = time.Unix(int64(s.seq), 0)
	s.rows[m.Hash] = m
	return true, nil
}

func (s *memStore) GetManifest(_ context.Context, hash string, body bool) (*store.Manifest, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.gets++
	m, ok := s.rows[hash]
	if !ok {
		return nil, store.ErrNotFound
	}
	out := m
	if !body {
		out.Body = nil
	}
	return &out, nil
}

func (s *memStore) PreviousManifest(_ context.Context, buildLine, exclude string) (*store.Manifest, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var best *store.Manifest
	for h, m := range s.rows {
		if h == exclude || m.BuildLine != buildLine {
			continue
		}
		mm := m
		if best == nil || mm.FirstSeen.After(best.FirstSeen) {
			best = &mm
		}
	}
	if best == nil {
		return nil, store.ErrNotFound
	}
	return best, nil
}

var (
	snapOnce sync.Once
	snap     []byte
)

func snapshot(t *testing.T) []byte {
	t.Helper()
	snapOnce.Do(func() {
		b, err := os.ReadFile("../../testdata/manifest.snapshot.json")
		if err != nil {
			t.Fatalf("snapshot: %v", err)
		}
		snap = b
	})
	return snap
}

func hashOf(t *testing.T, body []byte) string {
	t.Helper()
	h, err := manifest.HashBytes(body)
	if err != nil {
		t.Fatal(err)
	}
	return h
}

// smallManifest is a tiny valid manifest; mutate changes one field so a
// second hash and a diff exist.
func smallManifest(t *testing.T, extraKey string) []byte {
	t.Helper()
	doc := map[string]any{
		"version":        1,
		"generated_from": map[string]any{"branch": "x", "sha": "abc", "at": "2026-10-01T00:00:00Z"},
		"sections": map[string]any{
			"hero": map[string]any{"editorKey": "hero", "pages": []string{"home"},
				"fields": map[string]any{"title": map[string]any{"path": "title", "label": "Title", "type": "text"}},
				"lists":  map[string]any{}},
		},
		"globalPanels": map[string]any{},
		"branding":     map[string]any{"fields": map[string]any{}},
		"pages":        map[string]any{"home": map[string]any{"fields": map[string]any{}}},
		"details":      map[string]any{},
		"coverage":     map[string]any{"probes": 1},
	}
	if extraKey != "" {
		doc["sections"].(map[string]any)["hero"].(map[string]any)["fields"].(map[string]any)[extraKey] = map[string]any{"path": extraKey, "label": extraKey, "type": "text"}
	}
	b, _ := json.Marshal(doc)
	return b
}

func TestRegister_SnapshotCountsAndIdempotence(t *testing.T) {
	st := newMemStore()
	r := New(st)
	ctx := context.Background()
	body := snapshot(t)
	hash := hashOf(t, body)

	res, created, err := r.Register(ctx, hash, body, "abc123")
	if err != nil || !created {
		t.Fatalf("register: created=%v err=%v", created, err)
	}
	if res.Fields != 2094 || res.Lists != 69 || res.Sections != 19 || res.GlobalPanels != 12 || res.Pages != 17 || res.Details != 11 {
		t.Errorf("counts = %+v", res)
	}
	if res.Diff != nil {
		t.Errorf("first manifest of the line must have a null diff, got %+v", res.Diff)
	}
	if len(res.GeneratedFrom) == 0 || !strings.Contains(string(res.GeneratedFrom), `"sha"`) {
		t.Errorf("generated_from = %s", res.GeneratedFrom)
	}
	if !r.Cached(hash) {
		t.Error("register must prime the cache")
	}
	// Stored row: build line int, index text present, counts.
	row := st.rows[hash]
	if row.BuildLine != "int" {
		t.Errorf("build line = %q", row.BuildLine)
	}
	var idx indexDoc
	if err := json.Unmarshal(row.Index, &idx); err != nil || !strings.HasPrefix(idx.Text, "# Manifest index") || idx.Bytes != len(idx.Text) {
		t.Errorf("index column = %.80s… (%v)", row.Index, err)
	}

	_, created, err = r.Register(ctx, hash, body, "abc123")
	if err != nil || created {
		t.Errorf("second register: created=%v err=%v (want 200 path)", created, err)
	}
}

func TestRegister_HashMismatchAndBadBody(t *testing.T) {
	r := New(newMemStore())
	ctx := context.Background()
	body := smallManifest(t, "")
	wrong := strings.Repeat("0", 64)
	_, _, err := r.Register(ctx, wrong, body, "")
	var apiErr *api.Error
	if !errors.As(err, &apiErr) || apiErr.Code != "hash_mismatch" || apiErr.Status != 400 || apiErr.Extra["computed"] != hashOf(t, body) {
		t.Errorf("mismatch err = %v", err)
	}
	_, _, err = r.Register(ctx, wrong, []byte(`{not json`), "")
	if !errors.As(err, &apiErr) || apiErr.Code != "bad_request" {
		t.Errorf("bad json err = %v", err)
	}
	// Right hash, wrong schema version.
	v2 := []byte(`{"version":2,"sections":{},"pages":{}}`)
	_, _, err = r.Register(ctx, hashOf(t, v2), v2, "")
	if !errors.As(err, &apiErr) || apiErr.Code != "bad_request" || !strings.Contains(apiErr.Message, "version") {
		t.Errorf("version err = %v", err)
	}
}

func TestRegister_DiffAgainstPreviousOfSameBuildLine(t *testing.T) {
	r := New(newMemStore())
	ctx := context.Background()
	a := smallManifest(t, "")
	b := smallManifest(t, "subtitle")
	if _, _, err := r.Register(ctx, hashOf(t, a), a, "v3.0.90"); err != nil {
		t.Fatal(err)
	}
	// Same body on the int line: a different line, so no diff.
	res, _, err := r.Register(ctx, hashOf(t, b), b, "deadbeef")
	if err != nil || res.Diff != nil {
		t.Errorf("int line must not diff against prod: %+v %v", res.Diff, err)
	}
	res, created, err := r.Register(ctx, hashOf(t, b), b, "v3.0.91")
	if err != nil {
		t.Fatal(err)
	}
	if created {
		t.Error("b was already registered (by the int line); content-addressed → not created")
	}
	if res.Diff == nil || res.Diff.PrevHash != hashOf(t, a) || len(res.Diff.AddedFields) != 1 || res.Diff.AddedFields[0] != "sections.hero.subtitle" || len(res.Diff.RemovedFields) != 0 {
		t.Errorf("diff = %+v", res.Diff)
	}
}

func TestGetAndLoad(t *testing.T) {
	st := newMemStore()
	r := New(st)
	ctx := context.Background()
	body := snapshot(t)
	hash := hashOf(t, body)

	// Unknown everywhere.
	_, err := r.Get(ctx, hash, false)
	var apiErr *api.Error
	if !errors.As(err, &apiErr) || apiErr.Status != 404 {
		t.Errorf("get unknown = %v", err)
	}
	_, err = r.Load(ctx, hash)
	if !errors.As(err, &apiErr) || apiErr.Status != 409 || apiErr.Code != "unknown_manifest" || apiErr.Extra["hash"] != hash {
		t.Errorf("load unknown = %v", err)
	}

	if _, _, err := r.Register(ctx, hash, body, "v3.0.90"); err != nil {
		t.Fatal(err)
	}
	info, err := r.Get(ctx, hash, false)
	if err != nil || info.Hash != hash || info.Counts["fields"] != 2094 || len(info.Body) != 0 || !strings.Contains(info.Index, "s:heroBanner") {
		t.Errorf("get = %+v %v", info.Counts, err)
	}
	info, err = r.Get(ctx, hash, true)
	if err != nil || len(info.Body) != len(body) {
		t.Errorf("get with body = %d bytes %v", len(info.Body), err)
	}

	// Load from cache (no store read), then from the store after eviction.
	gets := st.gets
	c, err := r.Load(ctx, hash)
	if err != nil || c.Hash != hash || len(c.Tools) != 10 || len(c.Prefix.Tools) != 10 || c.Prefix.Index != c.Index.Text {
		t.Fatalf("load = %v %v", c, err)
	}
	if st.gets != gets {
		t.Error("cached load hit the store")
	}
	r.mu.Lock()
	delete(r.cache, hash)
	r.mu.Unlock()
	c2, err := r.Load(ctx, hash)
	if err != nil || c2.Hash != hash {
		t.Fatal(err)
	}
	p1, _ := c.Prefix.Bytes()
	p2, _ := c2.Prefix.Bytes()
	if string(p1) != string(p2) {
		t.Error("prefix differs between a cached and a reloaded compile")
	}
}

func TestCacheEviction(t *testing.T) {
	r := New(newMemStore())
	r.cacheSize = 2
	ctx := context.Background()
	var hashes []string
	for _, k := range []string{"", "a", "b"} {
		body := smallManifest(t, k)
		h := hashOf(t, body)
		hashes = append(hashes, h)
		if _, _, err := r.Register(ctx, h, body, ""); err != nil {
			t.Fatal(err)
		}
		time.Sleep(2 * time.Millisecond)
	}
	if r.Cached(hashes[0]) || !r.Cached(hashes[1]) || !r.Cached(hashes[2]) {
		t.Errorf("eviction kept the wrong entries: %v %v %v", r.Cached(hashes[0]), r.Cached(hashes[1]), r.Cached(hashes[2]))
	}
}

// TestRegisterComputed_SameResultAsPut covers POST /v1/manifests: the server
// computes the hash and the result is byte-identical to PUT with that hash,
// idempotence included (201 then 200), and the snapshot hashes to the §1.1
// test vector 727d151b5d6b04cd68b73199ab0b3bdf061263f9623f4f6eda00a79137c06f38.
func TestRegisterComputed_SameResultAsPut(t *testing.T) {
	ctx := context.Background()
	body := snapshot(t)
	const vector = "727d151b5d6b04cd68b73199ab0b3bdf061263f9623f4f6eda00a79137c06f38"

	// Fresh store: POST first, then PUT with the returned hash must be a 200.
	st := newMemStore()
	r := New(st)
	res, created, err := r.RegisterComputed(ctx, body, "abc123")
	if err != nil || !created {
		t.Fatalf("RegisterComputed: created=%v err=%v", created, err)
	}
	if res.Hash != vector {
		t.Fatalf("hash = %s, want %s", res.Hash, vector)
	}
	if res.Fields != 2094 || res.Lists != 69 || res.Sections != 19 || res.GlobalPanels != 12 || res.Pages != 17 || res.Details != 11 {
		t.Errorf("counts = %+v", res)
	}
	if !r.Cached(vector) {
		t.Error("POST must compile into the cache like PUT")
	}
	again, created, err := r.RegisterComputed(ctx, body, "abc123")
	if err != nil || created {
		t.Fatalf("second POST: created=%v err=%v", created, err)
	}
	put, created, err := r.Register(ctx, vector, body, "abc123")
	if err != nil || created {
		t.Fatalf("PUT after POST: created=%v err=%v", created, err)
	}
	a, _ := json.Marshal(res)
	b, _ := json.Marshal(again)
	c, _ := json.Marshal(put)
	if string(a) != string(b) || string(a) != string(c) {
		t.Errorf("POST, repeat POST and PUT differ:\n%s\n%s\n%s", a, b, c)
	}

	// Other order on a fresh store: PUT first, POST second → same body, 200.
	st2 := newMemStore()
	r2 := New(st2)
	put2, created, err := r2.Register(ctx, vector, body, "abc123")
	if err != nil || !created {
		t.Fatalf("PUT: created=%v err=%v", created, err)
	}
	post2, created, err := r2.RegisterComputed(ctx, body, "abc123")
	if err != nil || created {
		t.Fatalf("POST after PUT: created=%v err=%v", created, err)
	}
	d, _ := json.Marshal(put2)
	e, _ := json.Marshal(post2)
	if string(d) != string(e) {
		t.Errorf("PUT then POST differ:\n%s\n%s", d, e)
	}
}

func TestRegisterComputed_DiffAndBadBody(t *testing.T) {
	ctx := context.Background()
	st := newMemStore()
	r := New(st)
	first := smallManifest(t, "")
	if _, _, err := r.RegisterComputed(ctx, first, "v1.0.0"); err != nil {
		t.Fatal(err)
	}
	second := smallManifest(t, "subtitle")
	res, created, err := r.RegisterComputed(ctx, second, "v1.0.1")
	if err != nil || !created {
		t.Fatalf("second: created=%v err=%v", created, err)
	}
	if res.Hash != hashOf(t, second) {
		t.Errorf("hash = %s, want %s", res.Hash, hashOf(t, second))
	}
	if res.Diff == nil || res.Diff.PrevHash != hashOf(t, first) || len(res.Diff.AddedFields) != 1 {
		t.Errorf("diff = %+v", res.Diff)
	}

	_, _, err = r.RegisterComputed(ctx, []byte(`{not json`), "abc")
	var apiErr *api.Error
	if !errors.As(err, &apiErr) || apiErr.Code != "bad_request" {
		t.Fatalf("bad body: err = %v", err)
	}
	_, _, err = r.RegisterComputed(ctx, []byte(`{"version":99}`), "abc")
	if !errors.As(err, &apiErr) || apiErr.Code != "bad_request" {
		t.Fatalf("invalid manifest: err = %v", err)
	}
	if st.seq != 2 {
		t.Errorf("stored rows = %d, want 2", st.seq)
	}
}
