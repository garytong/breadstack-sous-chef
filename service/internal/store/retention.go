package store

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// RetentionPolicy is what the daily job applies (README §6): traces older
// than TraceDays go; ended conversations (with their messages and turns, via
// ON DELETE CASCADE) older than ConversationDays go; usage rows the
// thermometer has accepted go after ShippedUsageDays. Unshipped usage rows
// are never pruned here — they are what the shipper still owes.
//
// InactivityDays closes the gap for conversations nobody ever ended or
// rotated: an open conversation whose `updated` is older than the cutoff
// is ended by the job and then pruned in the same run, so merchant text
// cannot outlive the window just because the site stopped using Sous Chef.
//
// RegistrationDays prunes manifest_registrations rows whose last_at is
// older than the cutoff, except the newest launcher row of each build line
// (GET /v1/manifests/deployed must always have a registration to rank by
// once one was recorded). 0 means DefaultRegistrationDays, so policies
// built before the field existed keep working.
type RetentionPolicy struct {
	TraceDays        int
	ConversationDays int
	ShippedUsageDays int
	InactivityDays   int
	RegistrationDays int
}

// DefaultRegistrationDays is the manifest_registrations window.
const DefaultRegistrationDays = 180

// DefaultRetention is the production policy.
var DefaultRetention = RetentionPolicy{TraceDays: 90, ConversationDays: 90, ShippedUsageDays: 30, InactivityDays: 90, RegistrationDays: DefaultRegistrationDays}

// Validate rejects a policy that would delete everything or nothing sensible.
func (p RetentionPolicy) Validate() error {
	if p.TraceDays < 1 || p.ConversationDays < 1 || p.ShippedUsageDays < 1 || p.InactivityDays < 1 {
		return errors.New("store: retention days must all be >= 1")
	}
	if p.RegistrationDays < 0 {
		return errors.New("store: retention registration days must be >= 0 (0 = default)")
	}
	return nil
}

// RegistrationCutoff is the manifest_registrations threshold from now.
func (p RetentionPolicy) RegistrationCutoff(now time.Time) time.Time {
	d := p.RegistrationDays
	if d == 0 {
		d = DefaultRegistrationDays
	}
	return now.Add(-time.Duration(d) * 24 * time.Hour)
}

// Cutoffs computes the deletion thresholds from now: a row created (or, for
// inactivity, last updated) strictly before its cutoff is affected.
func (p RetentionPolicy) Cutoffs(now time.Time) (traces, conversations, usage, inactivity time.Time) {
	day := 24 * time.Hour
	return now.Add(-time.Duration(p.TraceDays) * day),
		now.Add(-time.Duration(p.ConversationDays) * day),
		now.Add(-time.Duration(p.ShippedUsageDays) * day),
		now.Add(-time.Duration(p.InactivityDays) * day)
}

// RetentionResult counts what one run removed.
type RetentionResult struct {
	Traces        int64
	Ended         int64 // open conversations ended for inactivity (also counted in Conversations once pruned)
	Conversations int64 // messages and turns cascade and are not counted separately
	UsageRows     int64
	Registrations int64 // manifest_registrations rows pruned
	RanAt         time.Time
}

// Retention applies p once. Each table is pruned in its own statement so a
// failure on one leaves the others' work committed; the error names the
// table. Live conversations are never touched unless they have been idle
// past InactivityDays (then they are ended and pruned); neither are
// unshipped usage rows.
func (s *Store) Retention(ctx context.Context, p RetentionPolicy) (RetentionResult, error) {
	if err := p.Validate(); err != nil {
		return RetentionResult{}, err
	}
	now := s.now()
	res := RetentionResult{RanAt: now}
	traceCut, convCut, usageCut, idleCut := p.Cutoffs(now)

	tag, err := s.pool.Exec(ctx, `DELETE FROM traces WHERE created < $1`, traceCut)
	if err != nil {
		return res, fmt.Errorf("store: retention traces: %w", err)
	}
	res.Traces = tag.RowsAffected()

	// Inactivity: end open conversations idle past the cutoff. `updated` is
	// deliberately left alone so the prune below (updated < idleCut) takes
	// them in this same run.
	tag, err = s.pool.Exec(ctx, `UPDATE conversations SET ended_at = $2 WHERE ended_at IS NULL AND updated < $1`, idleCut, now)
	if err != nil {
		return res, fmt.Errorf("store: retention inactive conversations: %w", err)
	}
	res.Ended = tag.RowsAffected()

	tag, err = s.pool.Exec(ctx, `
		DELETE FROM conversations
		WHERE ended_at IS NOT NULL AND (ended_at < $1 OR updated < $2)`, convCut, idleCut)
	if err != nil {
		return res, fmt.Errorf("store: retention conversations: %w", err)
	}
	res.Conversations = tag.RowsAffected()

	tag, err = s.pool.Exec(ctx, `DELETE FROM usage_rows WHERE shipped AND occurred_at < $1`, usageCut)
	if err != nil {
		return res, fmt.Errorf("store: retention usage_rows: %w", err)
	}
	res.UsageRows = tag.RowsAffected()

	// Registrations: idle rows go, but the newest launcher row of each
	// build line always stays.
	tag, err = s.pool.Exec(ctx, `
		DELETE FROM manifest_registrations r
		WHERE r.last_at < $1
		  AND (r.hash, r.launcher_build) NOT IN (
			SELECT DISTINCT ON (line) hash, launcher_build FROM (
				SELECT hash, launcher_build, last_at, `+sqlBuildLine("launcher_build")+` AS line
				FROM manifest_registrations WHERE source = 'launcher'
			) x
			ORDER BY line, last_at DESC, hash, launcher_build)`, p.RegistrationCutoff(now))
	if err != nil {
		return res, fmt.Errorf("store: retention manifest_registrations: %w", err)
	}
	res.Registrations = tag.RowsAffected()
	return res, nil
}
