package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
)

// Error is a typed API error: `{error, message, ...extra}` with an HTTP
// status (README §1 error list). Ports return *Error to choose the status
// and code; everything else is rendered as 500 `internal`.
type Error struct {
	Status  int
	Code    string
	Message string
	// Extra fields are merged into the envelope (e.g. `hash` on
	// unknown_manifest).
	Extra map[string]any
}

func (e *Error) Error() string { return fmt.Sprintf("%s: %s", e.Code, e.Message) }

// Constructors for the README's error codes.
func ErrUnauthorized() *Error {
	return &Error{Status: http.StatusUnauthorized, Code: "unauthorized", Message: "missing or invalid service key"}
}
func ErrBadRequest(msg string) *Error {
	return &Error{Status: http.StatusBadRequest, Code: "bad_request", Message: msg}
}
func ErrNotFound(code, msg string) *Error {
	return &Error{Status: http.StatusNotFound, Code: code, Message: msg}
}
func ErrUnknownManifest(hash string) *Error {
	return &Error{Status: http.StatusConflict, Code: "unknown_manifest",
		Message: "manifest hash not registered; upload it then retry", Extra: map[string]any{"hash": hash}}
}
func ErrHashMismatch(claimed, computed string) *Error {
	return &Error{Status: http.StatusBadRequest, Code: "hash_mismatch",
		Message: "body hash does not match the URL hash", Extra: map[string]any{"claimed": claimed, "computed": computed}}
}
func ErrTurnNotFound(turnID string) *Error {
	return &Error{Status: http.StatusNotFound, Code: "turn_not_found", Message: "turn not found or expired", Extra: map[string]any{"turnId": turnID}}
}
func ErrTurnNotWaiting(turnID, status string) *Error {
	return &Error{Status: http.StatusConflict, Code: "turn_not_waiting",
		Message: "turn is not waiting for tool results", Extra: map[string]any{"turnId": turnID, "status": status}}
}
func ErrModelUnavailable(msg string) *Error {
	return &Error{Status: http.StatusServiceUnavailable, Code: "model_unavailable", Message: msg}
}
func ErrBusy() *Error {
	return &Error{Status: http.StatusServiceUnavailable, Code: "busy", Message: "too many concurrent turns on this instance"}
}

// ErrStoreUnavailable is the 503 a Start answers when storage fails while
// it is deciding whether its row is still its own (README §4): nothing of
// the turn is stored and the caller may retry.
func ErrStoreUnavailable() *Error {
	return &Error{Status: http.StatusServiceUnavailable, Code: "store_unavailable", Message: "storage is temporarily unavailable; try again in a moment"}
}
func ErrNotImplemented(what string) *Error {
	return &Error{Status: http.StatusNotImplemented, Code: "not_implemented", Message: what + " is not wired yet"}
}

// envelope renders the wire shape.
func (e *Error) envelope() map[string]any {
	m := map[string]any{"error": e.Code, "message": e.Message}
	for k, v := range e.Extra {
		if k != "error" && k != "message" {
			m[k] = v
		}
	}
	return m
}

// writeError renders err as the JSON error envelope. Non-*Error values are
// logged and rendered as 500 `internal` without leaking their text.
func writeError(w http.ResponseWriter, r *http.Request, err error) {
	var apiErr *Error
	if !errors.As(err, &apiErr) {
		log.Printf("api: %s %s: internal error: %v", r.Method, r.URL.Path, err)
		apiErr = &Error{Status: http.StatusInternalServerError, Code: "internal", Message: "internal error"}
	}
	writeJSON(w, apiErr.Status, apiErr.envelope())
}

// writeJSON writes v as JSON with the given status.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	if v == nil {
		return
	}
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(v)
}
