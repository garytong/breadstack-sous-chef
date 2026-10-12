package api

import (
	"context"
	"encoding/json"
	"errors"
	"time"
)

// This file defines the interfaces the HTTP layer depends on. Other
// packages (internal/manifest, internal/loop, internal/store, internal/trace)
// implement them; the api package never imports those packages, so it can
// be tested with fakes and the other packages can be built in parallel.
//
// Any method may return *Error (errors.go) to pick the HTTP status and
// error code; any other error becomes 500 `internal`.

// Emitter receives the events of one turn (README §1.2). The SSE writer and
// the JSON collector both implement it. Emit returns an error once the
// client has gone away; callers should stop the turn then.
type Emitter interface {
	Emit(ev Event) error
}

// Event is one turn event. TurnID and Seq are added to the data object on
// the wire (`{...data, "turnId": ..., "seq": ...}`); Data must marshal to a
// JSON object (a struct or map), or nil for `{}`.
type Event struct {
	Type   string
	TurnID string
	Seq    int
	Data   any
}

// Event type names (README §1.2).
const (
	EventTextDelta     = "text.delta"
	EventToolCall      = "tool.call"
	EventTurnSuspended = "turn.suspended"
	EventTurnWaiting   = "turn.waiting"
	// EventTurnEscalated is informational: the turn moved from its route to
	// the route's fallback (`{from, to, reason}`); a client may ignore it.
	EventTurnEscalated = "turn.escalated"
	EventTurnDone      = "turn.done"
	EventTurnError     = "turn.error"
	EventHeartbeat     = "heartbeat"
)

// TextSourceService is a text.delta's `source` when the service wrote the
// text itself rather than the model: the fixed closing line of a turn that
// ended without its closing model call (turn.done `fast_finish:true`,
// README §4 "Fast finish"). A model's text.delta carries no `source`. The
// launcher shows both alike; the lab's collector must not train on it.
const TextSourceService = "service"

// ManifestRegistry is the content-addressed manifest store (README §1, §1.1).
//
// POST /v1/manifests (hash computed server-side) uses ComputeRegistrar when
// the registry implements it, and otherwise Deps.HashManifest + Register.
type ManifestRegistry interface {
	// Register stores body under hash. It recomputes the hash and returns
	// *Error{Code: "hash_mismatch"} on disagreement. created is true the
	// first time a hash is seen (201 vs 200).
	Register(ctx context.Context, hash string, body []byte, launcherBuild string) (res RegisterResult, created bool, err error)
	// Get returns the manifest's metadata, plus the raw body when withBody.
	// Unknown hash → *Error{Code: "unknown_manifest", Status: 404}.
	Get(ctx context.Context, hash string, withBody bool) (ManifestInfo, error)
}

// ComputeRegistrar is the optional registry extension behind POST
// /v1/manifests: compute the hash of body, then register it exactly as
// Register does. created is true the first time the hash is seen.
type ComputeRegistrar interface {
	ComputeAndRegister(ctx context.Context, body []byte, launcherBuild string) (res RegisterResult, created bool, err error)
}

// DeployedFinder is the optional registry extension behind GET
// /v1/manifests/deployed: which manifest the deployed launcher of a build
// line serves, judged from launcher registrations and real (non-lab)
// turns. No launcher registration and no real turn in the window →
// *Error{Code: "no_launcher_manifest", Status: 404}. A registry without it
// makes the route answer 501.
type DeployedFinder interface {
	Deployed(ctx context.Context, q DeployedQuery) (DeployedInfo, error)
}

// DeployedQuery is the parsed GET /v1/manifests/deployed query.
type DeployedQuery struct {
	Line string // "int" | "prod"
	Days int    // real-turn window, 1..365
	Body bool   // include the raw manifest
}

// Deployed-manifest bases: what decided DeployedInfo.Hash.
const (
	DeployedBasisRegistration = "registration" // newest launcher registration of the line
	DeployedBasisTurns        = "turns"        // no launcher registration recorded: newest real turn
)

// DeployedInfo is the GET /v1/manifests/deployed response body. Timestamps
// are null when the signal is absent.
type DeployedInfo struct {
	Hash              string              `json:"hash"`
	BuildLine         string              `json:"build_line"`
	Source            string              `json:"source"` // always "launcher"
	Basis             string              `json:"basis"`
	LauncherBuild     string              `json:"launcher_build"`
	GeneratedFrom     json.RawMessage     `json:"generated_from"`
	FirstSeen         time.Time           `json:"first_seen"`
	FirstRegisteredAt *time.Time          `json:"first_registered_at"`
	LastRegisteredAt  *time.Time          `json:"last_registered_at"`
	Registrations     int64               `json:"registrations"`
	LastRealTurnAt    *time.Time          `json:"last_real_turn_at"`
	RealTurns7d       int64               `json:"real_turns_7d"`
	WindowDays        int                 `json:"window_days"`
	Counts            map[string]int      `json:"counts"`
	Candidates        []DeployedCandidate `json:"candidates"`
	Body              json.RawMessage     `json:"body,omitempty"`
}

// DeployedCandidate is one entry of DeployedInfo.Candidates (newest first,
// at most 5, the deployed hash first).
type DeployedCandidate struct {
	Hash             string     `json:"hash"`
	LauncherBuild    string     `json:"launcher_build"`
	LastRegisteredAt *time.Time `json:"last_registered_at"`
	LastRealTurnAt   *time.Time `json:"last_real_turn_at"`
	RealTurns7d      int64      `json:"real_turns_7d"`
}

// ManifestHasher computes the manifest hash of a raw body (the canonical
// recipe in internal/manifest: HashBytes). Deps.HashManifest is the fallback
// for POST /v1/manifests when the registry is not a ComputeRegistrar.
type ManifestHasher func(body []byte) (string, error)

// RegisterResult is the PUT /v1/manifests/{hash} and POST /v1/manifests
// response body.
type RegisterResult struct {
	Hash          string          `json:"hash"`
	Fields        int             `json:"fields"`
	Lists         int             `json:"lists"`
	Sections      int             `json:"sections"`
	GlobalPanels  int             `json:"globalPanels"`
	Pages         int             `json:"pages"`
	Details       int             `json:"details"`
	GeneratedFrom json.RawMessage `json:"generated_from,omitempty"`
	Diff          *ManifestDiff   `json:"diff"`
}

// ManifestDiff compares a manifest with the previous one of its build line.
type ManifestDiff struct {
	PrevHash      string   `json:"prev_hash"`
	AddedFields   []string `json:"added_fields"`
	RemovedFields []string `json:"removed_fields"`
	AddedLists    []string `json:"added_lists"`
	RemovedLists  []string `json:"removed_lists"`
}

// ManifestInfo is the GET /v1/manifests/{hash} response body.
type ManifestInfo struct {
	Hash          string          `json:"hash"`
	GeneratedFrom json.RawMessage `json:"generated_from,omitempty"`
	Counts        map[string]int  `json:"counts"`
	FirstSeen     time.Time       `json:"first_seen"`
	Index         string          `json:"index"`
	Body          json.RawMessage `json:"body,omitempty"`
}

// TurnRequest is the POST /v1/turns body (README §1.2). Context is kept
// opaque here; internal/prompt parses it. StoreID is required: the trace
// opt-out is keyed by store, so a turn without one would silently defeat it.
type TurnRequest struct {
	ConversationID string          `json:"conversationId,omitempty"`
	SiteID         string          `json:"siteId"`
	StoreID        string          `json:"storeId"`
	ManifestHash   string          `json:"manifestHash"`
	BuildID        string          `json:"buildId"`
	Context        json.RawMessage `json:"context"`
	Text           string          `json:"text"`
	Model          string          `json:"model,omitempty"`
	// UserEmail is filled by the HTTP layer from X-User-Email (attribution).
	UserEmail string `json:"-"`
}

// ToolResult is one entry of POST /v1/turns/{turnId}/tool-results.
type ToolResult struct {
	ToolCallID string     `json:"toolCallId"`
	OK         bool       `json:"ok"`
	Error      *ToolError `json:"error,omitempty"`
	Warnings   []string   `json:"warnings,omitempty"`
	Written    []string   `json:"written,omitempty"`
	SectionID  string     `json:"sectionId,omitempty"`
	Asset      *Asset     `json:"asset,omitempty"`
	// Hidden is the launcher's `hidden: [{key, by, byValue}]` next to the
	// unchanged `W_HIDDEN:<key>` warnings (shared contract 4): which setting
	// gates a written field and that setting's post-op value. Kept raw — a
	// malformed member never refuses the body; the loop reads it leniently,
	// forwards it to the model and judges hidden writes from it.
	Hidden json.RawMessage `json:"hidden,omitempty"`
	Raw    json.RawMessage `json:"-"`
}

// ToolError is the ops engine's `{code, message, key}` failure shape.
type ToolError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Key     string `json:"key,omitempty"`
}

// Asset is the pick_asset resume payload. It is the single source of the
// asset shape: the loop marshals it verbatim as the model's tool_result, and
// the model copies alt/srcset into set_fields `assets[key]`.
//
// Srcset is passed through as raw JSON because the browser picker emits an
// OBJECT — `{"<width>": "<url>", ...}` (launcher compactSrcsetObject), which
// the ops engine reads back as a map — not an HTML srcset string. Validate
// accepts that object (string values) or, for older callers, a plain string.
type Asset struct {
	PublicURL string          `json:"public_url"`
	Alt       string          `json:"alt,omitempty"`
	Srcset    json.RawMessage `json:"srcset,omitempty"`
}

// Validate checks the shape of a decoded asset; the message is suitable for
// a 400 bad_request. A JSON null srcset is normalised to absent.
func (a *Asset) Validate() error {
	if a == nil || len(a.Srcset) == 0 {
		return nil
	}
	var v any
	if err := json.Unmarshal(a.Srcset, &v); err != nil {
		return errors.New("asset.srcset: invalid JSON")
	}
	switch s := v.(type) {
	case nil:
		a.Srcset = nil
		return nil
	case string:
		return nil
	case map[string]any:
		for k, val := range s {
			if _, ok := val.(string); !ok {
				return errors.New("asset.srcset." + k + " must be a URL string")
			}
		}
		return nil
	default:
		return errors.New("asset.srcset must be an object {width: url} or a string")
	}
}

// SrcsetMap returns the srcset as a width → URL map when it is an object,
// or nil when absent or a string.
func (a *Asset) SrcsetMap() map[string]string {
	if a == nil || len(a.Srcset) == 0 {
		return nil
	}
	var m map[string]string
	if err := json.Unmarshal(a.Srcset, &m); err != nil {
		return nil
	}
	return m
}

// ToolResultsRequest is the POST /v1/turns/{turnId}/tool-results body.
// SiteID/StoreID are optional tenant assertions from the launcher proxy
// (README §1, §11): when present, the runner verifies that the turn's
// conversation belongs to them and answers 404 turn_not_found otherwise,
// without revealing whether the turn exists.
type ToolResultsRequest struct {
	SiteID  string       `json:"siteId,omitempty"`
	StoreID string       `json:"storeId,omitempty"`
	Results []ToolResult `json:"results"`
}

// TurnRunner is the agent loop (README §4), split in two phases so the HTTP
// layer can answer cheap pre-checks with a status code and only then open
// the event stream (where heartbeats start):
//
//   - Start / Continue validate and persist (manifest known? turn waiting?
//     instance busy?) and return a Turn or an *Error — 409 unknown_manifest,
//     409 turn_not_waiting, 404 turn_not_found, 503 busy, 503
//     store_unavailable, 400 bad_request.
//     No events are emitted in this phase.
//   - Turn.Run drives the model until this request's stream is complete
//     (turn.waiting / turn.suspended / turn.done / turn.error emitted). It
//     must emit turn.error itself on failure; a returned error is only
//     logged. It must stop when ctx is cancelled or Emit returns an error.
type TurnRunner interface {
	Start(ctx context.Context, req TurnRequest) (Turn, error)
	Continue(ctx context.Context, turnID string, results ToolResultsRequest) (Turn, error)
}

// Turn is one started (or resumed) turn ready to stream.
type Turn interface {
	ID() string
	Run(ctx context.Context, em Emitter) error
}

// Conversation is the GET /v1/conversations/{siteId} body.
type Conversation struct {
	ConversationID string            `json:"conversationId"`
	StoreID        string            `json:"storeId,omitempty"`
	Messages       []StoredMessage   `json:"messages"`
	Turns          []json.RawMessage `json:"turns"`
}

// StoredMessage is one row of `messages`.
type StoredMessage struct {
	Seq     int             `json:"seq"`
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// ConversationStore reads and ends the per-site conversation.
type ConversationStore interface {
	// Get returns the site's open conversation with its last 200 messages,
	// or those before `before` when before > 0.
	Get(ctx context.Context, siteID string, before int) (Conversation, error)
	// End sets ended_at on the open conversation; no-op when none is open.
	End(ctx context.Context, siteID string) error
}

// Feedback is the POST /v1/feedback body. SiteID/StoreID are the same
// optional tenant assertions as on tool-results: when present, the sink
// verifies the turn's conversation matches and answers 404 turn_not_found
// otherwise.
type Feedback struct {
	TurnID    string   `json:"turnId"`
	SiteID    string   `json:"siteId,omitempty"`
	StoreID   string   `json:"storeId,omitempty"`
	KeptPaths []string `json:"kept_paths,omitempty"`
	Undone    *bool    `json:"undone,omitempty"`
	Saved     *bool    `json:"saved,omitempty"`
	Note      string   `json:"note,omitempty"`
}

// FeedbackSink merges feedback into traces.feedback.
type FeedbackSink interface {
	Record(ctx context.Context, fb Feedback) error
}

// OptOutStore is the per-store trace opt-out flag.
type OptOutStore interface {
	Set(ctx context.Context, storeID, byEmail string) error
	Clear(ctx context.Context, storeID string) error
}

// Pinger is anything /readyz must be able to reach (the Postgres pool).
type Pinger interface {
	Ping(ctx context.Context) error
}

// Primer warms the in-house model server's prompt cache ahead of a
// merchant's turn (README §4 "Prompt-cache priming"; loop.Runner implements
// it). Every method returns at once — the work runs in the background and
// never changes the response. Optional: a nil Deps.Primer is off.
//
// The PrimeDispatch each returns is closed once the prime's requests are on
// the wire (or it was skipped or deferred); nil when nothing was scheduled.
// The handler, having sent its complete response, waits on it (bounded by
// PrimeHold) before returning: Cloud Run with request-based CPU throttles a
// goroutine once no request is in flight.
type Primer interface {
	// ManifestRegistered: PUT/POST /v1/manifests succeeded for hash.
	ManifestRegistered(hash string) PrimeDispatch
	// ConversationOpened: GET /v1/conversations/{siteId} (the panel opening)
	// with X-Manifest-Hash; conversationID "" when the site has none open.
	ConversationOpened(siteID, conversationID, manifestHash string) PrimeDispatch
	// ConversationEnded: DELETE /v1/conversations/{siteId} succeeded.
	ConversationEnded(siteID, manifestHash string) PrimeDispatch
}

// PrimeDispatch is closed when a scheduled prime no longer needs the
// instance's CPU (model.Dispatch); nil = nothing scheduled.
type PrimeDispatch <-chan struct{}

// PrimeHold bounds a handler's wait on a PrimeDispatch.
const PrimeHold = 3 * time.Second
