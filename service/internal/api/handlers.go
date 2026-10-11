package api

import (
	"errors"
	"net/http"
	"strconv"
	"strings"

	"github.com/breadstack/sous-chef/internal/auth"
)

// ---- manifests ----

func (s *Server) putManifest(w http.ResponseWriter, r *http.Request) {
	if s.deps.Manifests == nil {
		writeError(w, r, ErrNotImplemented("manifest registry"))
		return
	}
	hash := r.PathValue("hash")
	if !isHexHash(hash) {
		writeError(w, r, ErrBadRequest("hash must be 64 lower-case hex characters"))
		return
	}
	body, err := readAll(w, r, s.opts.MaxManifestBytes)
	if err != nil {
		writeError(w, r, err)
		return
	}
	res, created, err := s.deps.Manifests.Register(r.Context(), hash, body, r.Header.Get("X-Launcher-Build"))
	if err != nil {
		writeError(w, r, err)
		return
	}
	status := http.StatusOK
	if created {
		status = http.StatusCreated
	}
	writeJSON(w, status, res)
	if s.deps.Primer != nil {
		holdForPrime(w, s.deps.Primer.ManifestRegistered(res.Hash))
	}
}

// postManifest is PUT /v1/manifests/{hash} without the hash in the URL: the
// server computes it (the launcher's JS cannot reproduce the Go
// canonicalisation) and answers exactly like PUT — 201 on first sight, 200
// after, body RegisterResult carrying the computed hash. It prefers the
// registry's ComputeAndRegister and falls back to Deps.HashManifest +
// Register.
func (s *Server) postManifest(w http.ResponseWriter, r *http.Request) {
	if s.deps.Manifests == nil {
		writeError(w, r, ErrNotImplemented("manifest registry"))
		return
	}
	cr, hasCR := s.deps.Manifests.(ComputeRegistrar)
	if !hasCR && s.deps.HashManifest == nil {
		writeError(w, r, ErrNotImplemented("manifest hashing"))
		return
	}
	body, err := readAll(w, r, s.opts.MaxManifestBytes)
	if err != nil {
		writeError(w, r, err)
		return
	}
	build := r.Header.Get("X-Launcher-Build")
	var (
		res     RegisterResult
		created bool
	)
	if hasCR {
		res, created, err = cr.ComputeAndRegister(r.Context(), body, build)
	} else {
		hash, herr := s.deps.HashManifest(body)
		if herr != nil {
			writeError(w, r, ErrBadRequest("manifest is not a JSON object: "+herr.Error()))
			return
		}
		res, created, err = s.deps.Manifests.Register(r.Context(), hash, body, build)
	}
	if err != nil {
		writeError(w, r, err)
		return
	}
	status := http.StatusOK
	if created {
		status = http.StatusCreated
	}
	writeJSON(w, status, res)
	if s.deps.Primer != nil {
		holdForPrime(w, s.deps.Primer.ManifestRegistered(res.Hash))
	}
}

func (s *Server) getManifest(w http.ResponseWriter, r *http.Request) {
	if s.deps.Manifests == nil {
		writeError(w, r, ErrNotImplemented("manifest registry"))
		return
	}
	hash := r.PathValue("hash")
	if !isHexHash(hash) {
		writeError(w, r, ErrBadRequest("hash must be 64 lower-case hex characters"))
		return
	}
	withBody := r.URL.Query().Get("body") == "1"
	info, err := s.deps.Manifests.Get(r.Context(), hash, withBody)
	if err != nil {
		writeError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, info)
}

// ---- turns ----

func (s *Server) postTurn(w http.ResponseWriter, r *http.Request) {
	if s.deps.Turns == nil {
		writeError(w, r, ErrNotImplemented("turn runner"))
		return
	}
	var req TurnRequest
	if err := decodeJSON(w, r, s.opts.MaxBodyBytes, &req); err != nil {
		writeError(w, r, err)
		return
	}
	switch {
	case strings.TrimSpace(req.SiteID) == "":
		writeError(w, r, ErrBadRequest("siteId is required"))
		return
	case strings.TrimSpace(req.StoreID) == "":
		// The trace opt-out is keyed by store; a turn without one would
		// bypass it, so storeId is not optional.
		writeError(w, r, ErrBadRequest("storeId is required"))
		return
	case !isHexHash(req.ManifestHash):
		writeError(w, r, ErrBadRequest("manifestHash must be 64 lower-case hex characters"))
		return
	case strings.TrimSpace(req.BuildID) == "":
		writeError(w, r, ErrBadRequest("buildId is required"))
		return
	case strings.TrimSpace(req.Text) == "":
		writeError(w, r, ErrBadRequest("text is required"))
		return
	}
	req.UserEmail = auth.UserEmail(r.Context())

	turn, err := s.deps.Turns.Start(r.Context(), req)
	if err != nil {
		writeError(w, r, err)
		return
	}
	s.stream(w, r, turn)
}

func (s *Server) postToolResults(w http.ResponseWriter, r *http.Request) {
	if s.deps.Turns == nil {
		writeError(w, r, ErrNotImplemented("turn runner"))
		return
	}
	turnID := r.PathValue("turnId")
	var req ToolResultsRequest
	if err := decodeJSON(w, r, s.opts.MaxBodyBytes, &req); err != nil {
		writeError(w, r, err)
		return
	}
	if len(req.Results) == 0 {
		writeError(w, r, ErrBadRequest("results must not be empty"))
		return
	}
	for i, res := range req.Results {
		if res.ToolCallID == "" {
			writeError(w, r, ErrBadRequest("results["+strconv.Itoa(i)+"].toolCallId is required"))
			return
		}
		if !res.OK && res.Error == nil {
			writeError(w, r, ErrBadRequest("results["+strconv.Itoa(i)+"]: ok=false requires error"))
			return
		}
		if err := res.Asset.Validate(); err != nil {
			writeError(w, r, ErrBadRequest("results["+strconv.Itoa(i)+"]."+err.Error()))
			return
		}
	}
	turn, err := s.deps.Turns.Continue(r.Context(), turnID, req)
	if err != nil {
		writeError(w, r, err)
		return
	}
	s.stream(w, r, turn)
}

// stream runs the turn against the negotiated emitter: SSE (default) or the
// JSON envelope `{turnId, events}` for `Accept: application/json`.
func (s *Server) stream(w http.ResponseWriter, r *http.Request, turn Turn) {
	if wantsJSON(r) {
		col := &Collector{}
		if err := turn.Run(r.Context(), col); err != nil {
			s.opts.Logf("api: turn %s: %v", turn.ID(), err)
		}
		writeJSON(w, http.StatusOK, map[string]any{"turnId": turn.ID(), "events": col.Events()})
		return
	}
	sse, err := NewSSEWriter(w, s.opts.Heartbeat)
	if err != nil {
		writeError(w, r, err)
		return
	}
	defer sse.Close()
	if err := turn.Run(r.Context(), sse); err != nil {
		s.opts.Logf("api: turn %s: %v", turn.ID(), err)
	}
}

// ---- conversations ----

func (s *Server) getConversation(w http.ResponseWriter, r *http.Request) {
	if s.deps.Conversations == nil {
		writeError(w, r, ErrNotImplemented("conversation store"))
		return
	}
	before := 0
	if b := r.URL.Query().Get("before"); b != "" {
		n, err := strconv.Atoi(b)
		if err != nil || n < 0 {
			writeError(w, r, ErrBadRequest("before must be a non-negative integer"))
			return
		}
		before = n
	}
	conv, err := s.deps.Conversations.Get(r.Context(), r.PathValue("siteId"), before)
	if err != nil {
		writeError(w, r, err)
		var apiErr *Error
		if before == 0 && errors.As(err, &apiErr) && apiErr.Status == http.StatusNotFound {
			s.primeOpened(w, r, "") // no conversation open: the floor
		}
		return
	}
	if conv.Messages == nil {
		conv.Messages = []StoredMessage{}
	}
	writeJSON(w, http.StatusOK, conv)
	if before == 0 {
		s.primeOpened(w, r, conv.ConversationID)
	}
}

// primeOpened tells the Primer the launcher read a site's conversation —
// the panel opening — when the proxy sent the manifest hash it runs on.
func (s *Server) primeOpened(w http.ResponseWriter, r *http.Request, conversationID string) {
	if hash := r.Header.Get("X-Manifest-Hash"); s.deps.Primer != nil && isHexHash(hash) {
		holdForPrime(w, s.deps.Primer.ConversationOpened(r.PathValue("siteId"), conversationID, hash))
	}
}

func (s *Server) deleteConversation(w http.ResponseWriter, r *http.Request) {
	if s.deps.Conversations == nil {
		writeError(w, r, ErrNotImplemented("conversation store"))
		return
	}
	if err := s.deps.Conversations.End(r.Context(), r.PathValue("siteId")); err != nil {
		writeError(w, r, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
	if hash := r.Header.Get("X-Manifest-Hash"); s.deps.Primer != nil && isHexHash(hash) {
		holdForPrime(w, s.deps.Primer.ConversationEnded(r.PathValue("siteId"), hash))
	}
}

// ---- feedback ----

func (s *Server) postFeedback(w http.ResponseWriter, r *http.Request) {
	if s.deps.Feedback == nil {
		writeError(w, r, ErrNotImplemented("feedback sink"))
		return
	}
	var fb Feedback
	if err := decodeJSON(w, r, s.opts.MaxBodyBytes, &fb); err != nil {
		writeError(w, r, err)
		return
	}
	if fb.TurnID == "" {
		writeError(w, r, ErrBadRequest("turnId is required"))
		return
	}
	if err := s.deps.Feedback.Record(r.Context(), fb); err != nil {
		writeError(w, r, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// ---- trace opt-out ----

func (s *Server) putOptOut(w http.ResponseWriter, r *http.Request) {
	if s.deps.OptOut == nil {
		writeError(w, r, ErrNotImplemented("opt-out store"))
		return
	}
	if err := s.deps.OptOut.Set(r.Context(), r.PathValue("storeId"), auth.UserEmail(r.Context())); err != nil {
		writeError(w, r, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) deleteOptOut(w http.ResponseWriter, r *http.Request) {
	if s.deps.OptOut == nil {
		writeError(w, r, ErrNotImplemented("opt-out store"))
		return
	}
	if err := s.deps.OptOut.Clear(r.Context(), r.PathValue("storeId")); err != nil {
		writeError(w, r, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}
