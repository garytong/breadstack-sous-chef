package api

import (
	"fmt"
	"net/http"
	"strconv"
	"strings"
)

// DefaultDeployedDays is GET /v1/manifests/deployed's real-turn window.
const DefaultDeployedDays = 30

// MaxDeployedDays bounds ?days= (turns older than the 90-day conversation
// retention are gone anyway).
const MaxDeployedDays = 365

// ErrNoLauncherManifest is the 404 of GET /v1/manifests/deployed when the
// line has no launcher registration and no real turn in the window.
func ErrNoLauncherManifest(line string, days int) *Error {
	return &Error{Status: http.StatusNotFound, Code: "no_launcher_manifest",
		Message: fmt.Sprintf("no launcher registration and no real turn on line %s in the last %d days", line, days),
		Extra:   map[string]any{"line": line, "days": days}}
}

// getDeployedManifest is GET /v1/manifests/deployed?line=int|prod[&body=1]
// [&days=N]: the manifest the deployed launcher of the line serves (README
// §1). line defaults to int; days (the real-turn window) to 30.
func (s *Server) getDeployedManifest(w http.ResponseWriter, r *http.Request) {
	finder, ok := s.deps.Manifests.(DeployedFinder)
	if s.deps.Manifests == nil || !ok {
		writeError(w, r, ErrNotImplemented("deployed manifest"))
		return
	}
	q := r.URL.Query()
	line := strings.TrimSpace(q.Get("line"))
	if line == "" {
		line = "int"
	}
	if line != "int" && line != "prod" {
		writeError(w, r, ErrBadRequest("line must be int or prod"))
		return
	}
	days := DefaultDeployedDays
	if v := strings.TrimSpace(q.Get("days")); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 || n > MaxDeployedDays {
			writeError(w, r, ErrBadRequest(fmt.Sprintf("days must be an integer from 1 to %d", MaxDeployedDays)))
			return
		}
		days = n
	}
	info, err := finder.Deployed(r.Context(), DeployedQuery{Line: line, Days: days, Body: q.Get("body") == "1"})
	if err != nil {
		writeError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, info)
}
