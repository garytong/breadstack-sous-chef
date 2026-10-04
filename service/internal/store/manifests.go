package store

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// Manifest is one row of manifests (README §6). Body is the raw launcher
// manifest, Index the rendered prompt index the loop sends as system[1],
// Counts the {fields, lists, sections, globalPanels, pages, details} map
// the register response echoes.
type Manifest struct {
	Hash          string
	Body          json.RawMessage
	Index         json.RawMessage
	GeneratedFrom json.RawMessage // may be nil
	BuildLine     string          // "int" | "prod"
	Counts        json.RawMessage
	FirstSeen     time.Time
}

// BuildLine maps an X-Launcher-Build value onto the manifest's build line:
// a build starting with "v" (v3.0.91) is prod, anything else (a git sha,
// "dev", empty) is int (README §1).
func BuildLine(launcherBuild string) string {
	if strings.HasPrefix(strings.TrimSpace(launcherBuild), "v") {
		return "prod"
	}
	return "int"
}

// UpsertManifest registers m. The registry is content-addressed, so a hash
// already present is left exactly as it was (first_seen included) and
// created is false; the register handler turns that into 200 vs 201.
func (s *Store) UpsertManifest(ctx context.Context, m Manifest) (created bool, err error) {
	if len(m.Hash) != 64 {
		return false, fmt.Errorf("store: manifest hash must be 64 hex chars, got %d", len(m.Hash))
	}
	if m.BuildLine == "" {
		m.BuildLine = "int"
	}
	if len(m.Counts) == 0 {
		m.Counts = json.RawMessage(`{}`)
	}
	if len(m.Index) == 0 {
		m.Index = json.RawMessage(`{}`)
	}
	tag, err := s.pool.Exec(ctx, `
		INSERT INTO manifests (hash, body, index, generated_from, build_line, counts, first_seen)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
		ON CONFLICT (hash) DO NOTHING`,
		m.Hash, m.Body, m.Index, nullableJSON(m.GeneratedFrom), m.BuildLine, m.Counts, s.now())
	if err != nil {
		return false, fmt.Errorf("store: upsert manifest: %w", err)
	}
	return tag.RowsAffected() == 1, nil
}

// GetManifest returns the manifest for hash. With body=false Body is nil
// (the index and counts are what a turn needs; the body is 759 KB).
func (s *Store) GetManifest(ctx context.Context, hash string, body bool) (*Manifest, error) {
	bodyExpr := "NULL::jsonb"
	if body {
		bodyExpr = "body"
	}
	row := s.pool.QueryRow(ctx, `
		SELECT hash, `+bodyExpr+`, index, generated_from, build_line, counts, first_seen
		FROM manifests WHERE hash = $1`, hash)
	var m Manifest
	if err := row.Scan(&m.Hash, &m.Body, &m.Index, &m.GeneratedFrom, &m.BuildLine, &m.Counts, &m.FirstSeen); err != nil {
		return nil, notFound(err)
	}
	return &m, nil
}

// ManifestExists is the cheap check every turn makes before anything else;
// a miss is the 409 unknown_manifest the proxy retries after uploading.
func (s *Store) ManifestExists(ctx context.Context, hash string) (bool, error) {
	var one int
	err := s.pool.QueryRow(ctx, `SELECT 1 FROM manifests WHERE hash = $1`, hash).Scan(&one)
	if err != nil {
		if notFound(err) == ErrNotFound {
			return false, nil
		}
		return false, fmt.Errorf("store: manifest exists: %w", err)
	}
	return true, nil
}

// PreviousManifest returns the newest manifest of buildLine other than
// excludeHash — the one the register diff is computed against — with its
// body. ErrNotFound when the line has no other manifest.
func (s *Store) PreviousManifest(ctx context.Context, buildLine, excludeHash string) (*Manifest, error) {
	row := s.pool.QueryRow(ctx, `
		SELECT hash, body, index, generated_from, build_line, counts, first_seen
		FROM manifests
		WHERE build_line = $1 AND hash <> $2
		ORDER BY first_seen DESC
		LIMIT 1`, buildLine, excludeHash)
	var m Manifest
	if err := row.Scan(&m.Hash, &m.Body, &m.Index, &m.GeneratedFrom, &m.BuildLine, &m.Counts, &m.FirstSeen); err != nil {
		return nil, notFound(err)
	}
	return &m, nil
}

// ListManifests returns every manifest newest first, without bodies.
func (s *Store) ListManifests(ctx context.Context) ([]Manifest, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT hash, NULL::jsonb, index, generated_from, build_line, counts, first_seen
		FROM manifests ORDER BY first_seen DESC`)
	if err != nil {
		return nil, fmt.Errorf("store: list manifests: %w", err)
	}
	defer rows.Close()
	var out []Manifest
	for rows.Next() {
		var m Manifest
		if err := rows.Scan(&m.Hash, &m.Body, &m.Index, &m.GeneratedFrom, &m.BuildLine, &m.Counts, &m.FirstSeen); err != nil {
			return nil, err
		}
		out = append(out, m)
	}
	return out, rows.Err()
}

// nullableJSON turns an empty RawMessage into SQL NULL; pgx would otherwise
// send "" which jsonb rejects.
func nullableJSON(raw json.RawMessage) any {
	if len(raw) == 0 {
		return nil
	}
	return raw
}
