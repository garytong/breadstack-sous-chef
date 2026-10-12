package store

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"time"
)

// Registration sources (manifest_registrations.source).
const (
	SourceLauncher = "launcher"
	SourceLab      = "lab"
)

// MaxLauncherBuildLen caps the stored X-Launcher-Build; longer values are
// cut (a header is not a place for an essay, and the primary key holds it).
const MaxLauncherBuildLen = 128

var labBuild = regexp.MustCompile(`(?i)^lab([-_.:].*)?$`)

// RegistrationSource classifies an X-Launcher-Build (or a turn's buildId):
// "lab" when the trimmed value is empty, `lab`, `lab-…`, `lab_…`, `lab.…`,
// `lab:…` (the lab's collector, r4collect and exam runner), or `dev` (a
// launcher built without LAUNCHER_BUILD_ID: a developer's local run, never
// a deploy); otherwise "launcher". sqlNonLauncher is the same rule in SQL.
func RegistrationSource(build string) string {
	b := strings.TrimSpace(build)
	if b == "" || labBuild.MatchString(b) || strings.EqualFold(b, "dev") {
		return SourceLab
	}
	return SourceLauncher
}

// sqlNonLauncher(col) is RegistrationSource(col) == "lab" in SQL, for the
// turns.build_id side of the deployed query (TestRegistrationSourceSQLParity
// keeps the two in step).
func sqlNonLauncher(col string) string {
	return `(btrim(` + col + `) = '' OR btrim(` + col + `) ~* '^lab([-_.:].*)?$' OR lower(btrim(` + col + `)) = 'dev')`
}

// sqlBuildLine(col) is BuildLine(col) in SQL.
func sqlBuildLine(col string) string {
	return `(CASE WHEN btrim(` + col + `) LIKE 'v%' THEN 'prod' ELSE 'int' END)`
}

// RecordRegistration notes one registration of hash by build (the
// X-Launcher-Build as sent, cut to MaxLauncherBuildLen): the first one
// inserts a row, later ones bump last_at and registrations. source ""
// means RegistrationSource(build). A zero at means the store's clock. The
// manifest row must exist (foreign key): call after UpsertManifest.
func (s *Store) RecordRegistration(ctx context.Context, hash, build, source string, at time.Time) error {
	if len(build) > MaxLauncherBuildLen {
		build = build[:MaxLauncherBuildLen]
	}
	if source == "" {
		source = RegistrationSource(build)
	}
	if at.IsZero() {
		at = s.now()
	}
	_, err := s.pool.Exec(ctx, `
		INSERT INTO manifest_registrations (hash, launcher_build, source, first_at, last_at, registrations)
		VALUES ($1, $2, $3, $4, $4, 1)
		ON CONFLICT (hash, launcher_build) DO UPDATE SET
			last_at       = GREATEST(manifest_registrations.last_at, EXCLUDED.last_at),
			registrations = manifest_registrations.registrations + 1`,
		hash, build, source, at)
	if err != nil {
		return fmt.Errorf("store: record registration: %w", err)
	}
	return nil
}

// DeployedCandidate is one manifest a launcher of the line may be serving,
// with both signals the deployed query reads.
type DeployedCandidate struct {
	Hash string
	// LauncherBuild is the X-Launcher-Build of the hash's newest launcher
	// registration, or, for a turns-only hash, the buildId of its newest
	// real turn.
	LauncherBuild     string
	FirstRegisteredAt *time.Time // nil: no launcher registration recorded
	LastRegisteredAt  *time.Time
	Registrations     int64      // launcher registrations, all builds
	LastRealTurnAt    *time.Time // newest real turn since the window start; nil: none
	RealTurns7d       int64      // real turns in the 7 days before now
}

// Registered reports whether the candidate has a launcher registration.
func (c DeployedCandidate) Registered() bool { return c.LastRegisteredAt != nil }

// DeployedManifest returns up to limit manifests a launcher of line ("int"
// | "prod") registered or used for a real turn since `since`, the deployed
// one first.
//
// Ranking: a hash with a launcher registration ranks above every hash
// without one, and registered hashes rank by their newest registration.
// A launcher registers at every boot, so this follows deploys and
// rollbacks; real turns alone would not, since a merchant tab opened
// before a deploy keeps sending the old hash for hours. Turns decide only
// when the line has no launcher registration at all (rows older than
// migration 0006), newest real turn first. A real turn is one whose
// conversation's site_id does not start `lab-` and whose buildId is a
// launcher build (RegistrationSource); its line is BuildLine(buildId).
func (s *Store) DeployedManifest(ctx context.Context, line string, since time.Time, limit int) ([]DeployedCandidate, error) {
	if line != "int" && line != "prod" {
		return nil, fmt.Errorf("store: deployed manifest: line %q", line)
	}
	if limit < 1 {
		limit = 1
	}
	week := s.now().Add(-7 * 24 * time.Hour)
	rows, err := s.pool.Query(ctx, `
		WITH reg AS (
			SELECT hash,
			       min(first_at)       AS first_at,
			       max(last_at)        AS last_at,
			       sum(registrations)  AS n,
			       (array_agg(launcher_build ORDER BY last_at DESC, launcher_build))[1] AS build
			FROM manifest_registrations
			WHERE source = 'launcher' AND `+sqlBuildLine("launcher_build")+` = $1
			GROUP BY hash
		), trn AS (
			SELECT t.manifest_hash AS hash,
			       max(t.started) FILTER (WHERE t.started >= $2) AS last_at,
			       count(*)       FILTER (WHERE t.started >= $3) AS n7,
			       (array_agg(t.build_id ORDER BY t.started DESC))[1] AS build
			FROM turns t JOIN conversations c ON c.id = t.conversation_id
			WHERE t.started >= LEAST($2::timestamptz, $3::timestamptz)
			  AND c.site_id NOT LIKE 'lab-%'
			  AND NOT `+sqlNonLauncher("t.build_id")+`
			  AND `+sqlBuildLine("t.build_id")+` = $1
			GROUP BY t.manifest_hash
		)
		SELECT m.hash, COALESCE(reg.build, trn.build, ''),
		       reg.first_at, reg.last_at, COALESCE(reg.n, 0)::bigint,
		       trn.last_at, COALESCE(trn.n7, 0)::bigint
		FROM reg FULL OUTER JOIN trn ON trn.hash = reg.hash
		JOIN manifests m ON m.hash = COALESCE(reg.hash, trn.hash)
		WHERE reg.hash IS NOT NULL OR trn.last_at IS NOT NULL
		ORDER BY (reg.last_at IS NULL), reg.last_at DESC NULLS LAST, trn.last_at DESC NULLS LAST, m.hash
		LIMIT $4`, line, since, week, limit)
	if err != nil {
		return nil, fmt.Errorf("store: deployed manifest: %w", err)
	}
	defer rows.Close()
	var out []DeployedCandidate
	for rows.Next() {
		var c DeployedCandidate
		if err := rows.Scan(&c.Hash, &c.LauncherBuild, &c.FirstRegisteredAt, &c.LastRegisteredAt,
			&c.Registrations, &c.LastRealTurnAt, &c.RealTurns7d); err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	return out, rows.Err()
}
