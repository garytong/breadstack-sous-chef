// Package registry is the content-addressed manifest registry (README §1,
// §1.1): PUT/GET /v1/manifests/{hash} and POST /v1/manifests (server
// computes the hash) on top of the store, and the compiled
// per-hash prefix the loop sends with every model call — parsed manifest,
// rendered index, tool definitions — cached in memory so a turn never
// re-parses the 759 KB document.
package registry

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/breadstack/sous-chef/internal/api"
	"github.com/breadstack/sous-chef/internal/manifest"
	"github.com/breadstack/sous-chef/internal/prompt"
	"github.com/breadstack/sous-chef/internal/store"
	"github.com/breadstack/sous-chef/internal/toolgen"
)

// Store is the slice of *store.Store the registry needs.
type Store interface {
	UpsertManifest(ctx context.Context, m store.Manifest) (created bool, err error)
	GetManifest(ctx context.Context, hash string, body bool) (*store.Manifest, error)
	PreviousManifest(ctx context.Context, buildLine, excludeHash string) (*store.Manifest, error)
}

// RegistrationRecorder is the optional Store extension that records every
// registration with its X-Launcher-Build (*store.Store implements it).
// Without it Register behaves as before and GET /v1/manifests/deployed can
// rank by real turns only.
type RegistrationRecorder interface {
	RecordRegistration(ctx context.Context, hash, build, source string, at time.Time) error
}

// DeployedLister is the optional Store extension behind
// GET /v1/manifests/deployed (*store.Store implements it).
type DeployedLister interface {
	DeployedManifest(ctx context.Context, line string, since time.Time, limit int) ([]store.DeployedCandidate, error)
}

// Compiled is everything a turn needs for one manifest hash.
type Compiled struct {
	Hash     string
	Manifest *manifest.Manifest
	Index    prompt.Index
	Tools    []toolgen.Tool
	Prefix   prompt.Prefix
	LoadedAt time.Time
}

// DefaultCacheSize bounds the compiled cache: int and prod run one manifest
// each, plus a few during a rollout.
const DefaultCacheSize = 8

// Registry implements api.ManifestRegistry and the loop's Load.
type Registry struct {
	st        Store
	cacheSize int
	mu        sync.Mutex
	cache     map[string]*Compiled
	now       func() time.Time
	logf      func(format string, a ...any)
}

// New builds a registry over st.
func New(st Store) *Registry {
	return &Registry{st: st, cacheSize: DefaultCacheSize, cache: map[string]*Compiled{},
		now: func() time.Time { return time.Now().UTC() }, logf: log.Printf}
}

// indexDoc is the manifests.index column: the rendered system[1] text and
// its size, so GET can serve it without re-rendering.
type indexDoc struct {
	Text  string `json:"text"`
	Bytes int    `json:"bytes"`
}

// Register implements api.ManifestRegistry.Register: recompute the hash,
// compile (which validates the document), store, and diff against the
// previous manifest of the build line.
func (r *Registry) Register(ctx context.Context, hash string, body []byte, launcherBuild string) (api.RegisterResult, bool, error) {
	computed, err := manifest.HashBytes(body)
	if err != nil {
		return api.RegisterResult{}, false, api.ErrBadRequest("manifest: " + err.Error())
	}
	if computed != hash {
		return api.RegisterResult{}, false, api.ErrHashMismatch(hash, computed)
	}
	c, err := compile(body)
	if err != nil {
		return api.RegisterResult{}, false, api.ErrBadRequest(err.Error())
	}
	counts := c.Manifest.Counts()
	countsJSON, _ := json.Marshal(counts)
	idx, _ := json.Marshal(indexDoc{Text: c.Index.Text, Bytes: len(c.Index.Text)})
	var generated json.RawMessage
	if g, ok := c.Manifest.Raw()["generated_from"]; ok && g != nil {
		generated, _ = json.Marshal(g)
	}
	buildLine := store.BuildLine(launcherBuild)
	created, err := r.st.UpsertManifest(ctx, store.Manifest{
		Hash: hash, Body: body, Index: idx, GeneratedFrom: generated, BuildLine: buildLine, Counts: countsJSON,
	})
	if err != nil {
		return api.RegisterResult{}, false, err
	}
	r.put(c)
	r.recordRegistration(ctx, hash, launcherBuild)

	res := api.RegisterResult{
		Hash: hash, Fields: counts.Fields, Lists: counts.Lists, Sections: counts.Sections,
		GlobalPanels: counts.GlobalPanels, Pages: counts.Pages, Details: counts.Details,
		GeneratedFrom: generated,
	}
	prev, err := r.st.PreviousManifest(ctx, buildLine, hash)
	switch {
	case errors.Is(err, store.ErrNotFound):
		// first manifest of this build line: diff stays null
	case err != nil:
		return api.RegisterResult{}, false, err
	default:
		pm, perr := manifest.Parse(prev.Body)
		if perr != nil {
			// A stored manifest that no longer parses is a bug worth seeing,
			// but it must not block registering a good one.
			break
		}
		d := manifest.ComputeDiff(pm, c.Manifest)
		res.Diff = &api.ManifestDiff{
			PrevHash: prev.Hash, AddedFields: d.AddedFields, RemovedFields: d.RemovedFields,
			AddedLists: d.AddedLists, RemovedLists: d.RemovedLists,
		}
	}
	return res, created, nil
}

// recordRegistration notes the registration for GET /v1/manifests/deployed.
// Best effort: a failure is logged and never fails the registration (the
// launcher's boot loop must not retry over bookkeeping).
func (r *Registry) recordRegistration(ctx context.Context, hash, launcherBuild string) {
	rec, ok := r.st.(RegistrationRecorder)
	if !ok {
		return
	}
	if err := rec.RecordRegistration(ctx, hash, launcherBuild, store.RegistrationSource(launcherBuild), r.now()); err != nil {
		r.logf("registry: record registration of %s (build %q): %v", short(hash), truncate(launcherBuild, 40), err)
	}
}

func short(h string) string {
	if len(h) > 12 {
		return h[:12]
	}
	return h
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n] + "…"
	}
	return s
}

// RegisterComputed is Register for POST /v1/manifests: the caller sends
// only the body and the server computes the hash (README §1.1), for a
// launcher that cannot reproduce the canonicalisation (JavaScript orders
// integer-like keys first; JSON.stringify leaves U+2028/U+2029 raw). The
// result, created flag and side effects are exactly those of
// Register(ctx, hash, body, launcherBuild) with the computed hash — the
// returned RegisterResult.Hash is the hash to send as X-Manifest-Hash. An
// unparsable body is a 400 bad_request.
func (r *Registry) RegisterComputed(ctx context.Context, body []byte, launcherBuild string) (api.RegisterResult, bool, error) {
	hash, err := manifest.HashBytes(body)
	if err != nil {
		return api.RegisterResult{}, false, api.ErrBadRequest("manifest: " + err.Error())
	}
	return r.Register(ctx, hash, body, launcherBuild)
}

// ComputeAndRegister implements api.ComputeRegistrar (the optional interface
// the POST /v1/manifests handler prefers over its Deps.HashManifest
// fallback); it is RegisterComputed under the interface's name.
func (r *Registry) ComputeAndRegister(ctx context.Context, body []byte, launcherBuild string) (api.RegisterResult, bool, error) {
	return r.RegisterComputed(ctx, body, launcherBuild)
}

var _ api.ComputeRegistrar = (*Registry)(nil)

// Get implements api.ManifestRegistry.Get.
func (r *Registry) Get(ctx context.Context, hash string, withBody bool) (api.ManifestInfo, error) {
	m, err := r.st.GetManifest(ctx, hash, withBody)
	if errors.Is(err, store.ErrNotFound) {
		return api.ManifestInfo{}, api.ErrNotFound("unknown_manifest", "manifest hash not registered")
	}
	if err != nil {
		return api.ManifestInfo{}, err
	}
	info := api.ManifestInfo{Hash: m.Hash, GeneratedFrom: m.GeneratedFrom, FirstSeen: m.FirstSeen, Counts: map[string]int{}}
	_ = json.Unmarshal(m.Counts, &info.Counts)
	var idx indexDoc
	if json.Unmarshal(m.Index, &idx) == nil {
		info.Index = idx.Text
	}
	if withBody {
		info.Body = m.Body
	}
	return info, nil
}

// DeployedCandidates is how many candidates GET /v1/manifests/deployed
// lists.
const DeployedCandidates = 5

// Deployed implements api.DeployedFinder: the manifest the deployed
// launcher of q.Line serves (store.DeployedManifest ranks), with its
// metadata and, when q.Body, the raw body.
func (r *Registry) Deployed(ctx context.Context, q api.DeployedQuery) (api.DeployedInfo, error) {
	lister, ok := r.st.(DeployedLister)
	if !ok {
		return api.DeployedInfo{}, api.ErrNotImplemented("deployed manifest")
	}
	since := r.now().Add(-time.Duration(q.Days) * 24 * time.Hour)
	cands, err := lister.DeployedManifest(ctx, q.Line, since, DeployedCandidates)
	if err != nil {
		return api.DeployedInfo{}, err
	}
	if len(cands) == 0 {
		return api.DeployedInfo{}, api.ErrNoLauncherManifest(q.Line, q.Days)
	}
	top := cands[0]
	m, err := r.st.GetManifest(ctx, top.Hash, q.Body)
	if err != nil {
		// The query joins manifests, so a miss here is a race with nothing
		// that deletes manifests: report it as the internal error it is.
		return api.DeployedInfo{}, fmt.Errorf("registry: deployed manifest %s: %w", short(top.Hash), err)
	}
	info := api.DeployedInfo{
		Hash: top.Hash, BuildLine: q.Line, Source: store.SourceLauncher, Basis: api.DeployedBasisTurns,
		LauncherBuild: top.LauncherBuild, GeneratedFrom: m.GeneratedFrom, FirstSeen: m.FirstSeen.UTC(),
		FirstRegisteredAt: utc(top.FirstRegisteredAt), LastRegisteredAt: utc(top.LastRegisteredAt),
		Registrations: top.Registrations, LastRealTurnAt: utc(top.LastRealTurnAt), RealTurns7d: top.RealTurns7d,
		WindowDays: q.Days, Counts: map[string]int{}, Candidates: make([]api.DeployedCandidate, 0, len(cands)),
	}
	if top.Registered() {
		info.Basis = api.DeployedBasisRegistration
	}
	_ = json.Unmarshal(m.Counts, &info.Counts)
	if q.Body {
		info.Body = m.Body
	}
	for _, c := range cands {
		info.Candidates = append(info.Candidates, api.DeployedCandidate{
			Hash: c.Hash, LauncherBuild: c.LauncherBuild, LastRegisteredAt: utc(c.LastRegisteredAt),
			LastRealTurnAt: utc(c.LastRealTurnAt), RealTurns7d: c.RealTurns7d,
		})
	}
	return info, nil
}

var _ api.DeployedFinder = (*Registry)(nil)

func utc(t *time.Time) *time.Time {
	if t == nil {
		return nil
	}
	u := t.UTC()
	return &u
}

// Load returns the compiled prefix for hash, from the cache or the store.
// An unknown hash is api.ErrUnknownManifest (409): the proxy uploads the
// manifest and retries.
func (r *Registry) Load(ctx context.Context, hash string) (*Compiled, error) {
	r.mu.Lock()
	c, ok := r.cache[hash]
	r.mu.Unlock()
	if ok {
		return c, nil
	}
	m, err := r.st.GetManifest(ctx, hash, true)
	if errors.Is(err, store.ErrNotFound) {
		return nil, api.ErrUnknownManifest(hash)
	}
	if err != nil {
		return nil, err
	}
	c, err = compile(m.Body)
	if err != nil {
		return nil, fmt.Errorf("registry: stored manifest %s: %w", hash, err)
	}
	if c.Hash != hash {
		return nil, fmt.Errorf("registry: stored manifest %s hashes to %s", hash, c.Hash)
	}
	r.put(c)
	return c, nil
}

// Cached reports whether hash is compiled in memory (tests, /version).
func (r *Registry) Cached(hash string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	_, ok := r.cache[hash]
	return ok
}

func (r *Registry) put(c *Compiled) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.cache[c.Hash]; !ok && len(r.cache) >= r.cacheSize {
		// Evict the oldest load.
		var oldest string
		var at time.Time
		for h, e := range r.cache {
			if oldest == "" || e.LoadedAt.Before(at) {
				oldest, at = h, e.LoadedAt
			}
		}
		delete(r.cache, oldest)
	}
	r.cache[c.Hash] = c
}

// compile parses, renders the index and builds the tools; deterministic for
// a given document.
func compile(body []byte) (*Compiled, error) {
	m, err := manifest.Parse(body)
	if err != nil {
		return nil, err
	}
	idx, err := prompt.RenderIndex(body)
	if err != nil {
		return nil, err
	}
	tools := toolgen.Tools(m)
	sdk, err := prompt.SDKTools(tools)
	if err != nil {
		return nil, err
	}
	return &Compiled{
		Hash: m.Hash(), Manifest: m, Index: idx, Tools: tools,
		Prefix: prompt.NewPrefix(idx, sdk), LoadedAt: time.Now(),
	}, nil
}

// Compile is compile for tests and tools (the lab's replay harness).
func Compile(body []byte) (*Compiled, error) { return compile(body) }
