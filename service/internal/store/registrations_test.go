package store_test

import (
	"context"
	"encoding/json"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/breadstack/sous-chef/internal/store"
	"github.com/breadstack/sous-chef/internal/store/storetest"
)

const hashD = "3e18656a000000000000000000000000000000000000000000000000000000d4"
const hashE = "3e18656a000000000000000000000000000000000000000000000000000000e5"

var day0 = time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)

func at(h int) time.Time { return day0.Add(time.Duration(h) * time.Hour) }

func putManifest(t *testing.T, s *store.Store, hash, line string) {
	t.Helper()
	if _, err := s.UpsertManifest(context.Background(), store.Manifest{Hash: hash, Body: json.RawMessage(`{"v":1}`),
		BuildLine: line, Counts: json.RawMessage(`{"fields":3}`)}); err != nil {
		t.Fatal(err)
	}
}

func register(t *testing.T, s *store.Store, hash, build string, when time.Time) {
	t.Helper()
	if err := s.RecordRegistration(context.Background(), hash, build, "", when); err != nil {
		t.Fatalf("RecordRegistration(%s, %q): %v", hash[len(hash)-2:], build, err)
	}
}

var siteSeq atomic.Int64

// turnAt records a finished turn on hash for site (a fresh conversation
// each time, so sites may repeat) with the launcher's buildId.
func turnAt(t *testing.T, s *store.Store, site, hash, build string, when time.Time) {
	t.Helper()
	ctx := context.Background()
	s.SetClock(func() time.Time { return when })
	defer s.SetClock(func() time.Time { return day0.AddDate(0, 0, 3) })
	c, err := s.CreateConversation(ctx, fmt.Sprintf("%s%d", site, siteSeq.Add(1)), "store-1")
	if err != nil {
		t.Fatal(err)
	}
	tr, err := s.CreateTurn(ctx, store.NewTurn{ConversationID: c.ID, ManifestHash: hash, BuildID: build})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.FinishTurn(ctx, tr.ID, store.TurnDone, nil); err != nil {
		t.Fatal(err)
	}
}

func hashes(cs []store.DeployedCandidate) []string {
	out := make([]string, len(cs))
	for i, c := range cs {
		out[i] = c.Hash[len(c.Hash)-2:]
	}
	return out
}

func TestRecordRegistration(t *testing.T) {
	s := storetest.Open(t)
	ctx := ctxT(t)
	putManifest(t, s, hashA, "int")

	register(t, s, hashA, "1a2b3c4", at(0))
	register(t, s, hashA, "1a2b3c4", at(5))
	register(t, s, hashA, "1a2b3c4", at(2)) // out of order: last_at never moves back
	register(t, s, hashA, "lab-r4", at(1))

	var first, last time.Time
	var n int64
	var source string
	err := s.Pool().QueryRow(ctx, `SELECT first_at, last_at, registrations, source FROM manifest_registrations
		WHERE hash = $1 AND launcher_build = '1a2b3c4'`, hashA).Scan(&first, &last, &n, &source)
	if err != nil {
		t.Fatal(err)
	}
	if !first.Equal(at(0)) || !last.Equal(at(5)) || n != 3 || source != store.SourceLauncher {
		t.Errorf("launcher row = first %s last %s n %d source %s", first, last, n, source)
	}
	if err := s.Pool().QueryRow(ctx, `SELECT source FROM manifest_registrations WHERE hash = $1 AND launcher_build = 'lab-r4'`, hashA).Scan(&source); err != nil || source != store.SourceLab {
		t.Errorf("lab-r4 row source = %q, %v", source, err)
	}
	// Unknown hash: the foreign key refuses it.
	if err := s.RecordRegistration(ctx, hashB, "1a2b3c4", "", at(0)); err == nil {
		t.Error("registration of an unknown hash accepted")
	}
	// Explicit bad source: the CHECK refuses it.
	if err := s.RecordRegistration(ctx, hashA, "x", "staging", at(0)); err == nil {
		t.Error("source 'staging' accepted")
	}
	// A huge header is cut, not refused.
	long := make([]byte, 1000)
	for i := range long {
		long[i] = 'z'
	}
	if err := s.RecordRegistration(ctx, hashA, string(long), "", time.Time{}); err != nil {
		t.Fatalf("long build: %v", err)
	}
	var stored string
	if err := s.Pool().QueryRow(ctx, `SELECT launcher_build FROM manifest_registrations WHERE launcher_build LIKE 'zzz%'`).Scan(&stored); err != nil || len(stored) != store.MaxLauncherBuildLen {
		t.Errorf("long build stored as %d chars, %v", len(stored), err)
	}
}

func TestDeployedManifest_TurnsOnly(t *testing.T) {
	// No registration rows (the state right after migration 0006): the
	// newest real turn decides; lab sites and lab/dev builds never count.
	s := storetest.Open(t)
	ctx := ctxT(t)
	for _, h := range []string{hashA, hashB, hashC, hashD} {
		putManifest(t, s, h, "int")
	}
	turnAt(t, s, "site-", hashA, "1a2b3c4", at(0))
	turnAt(t, s, "site-", hashA, "1a2b3c4", at(1))
	turnAt(t, s, "site-", hashB, "9f8e7d6", at(2))
	// Newer, but not real: lab site, lab builds, a dev launcher, empty build.
	turnAt(t, s, "lab-r4p1-", hashC, "1a2b3c4", at(10))
	turnAt(t, s, "site-", hashC, "lab-r4", at(11))
	turnAt(t, s, "site-", hashC, "lab", at(11))
	turnAt(t, s, "site-", hashD, "dev", at(12))
	turnAt(t, s, "site-", hashD, "", at(12))
	// A prod launcher's turn is on the prod line.
	turnAt(t, s, "site-", hashD, "v3.0.91", at(13))

	s.SetClock(func() time.Time { return at(24) })
	got, err := s.DeployedManifest(ctx, "int", at(-24*30), 5)
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprint(hashes(got)) != "[b2 a1]" {
		t.Fatalf("int candidates = %v, want [b2 a1]", hashes(got))
	}
	b := got[0]
	if b.Registered() || b.LauncherBuild != "9f8e7d6" || b.LastRealTurnAt == nil || !b.LastRealTurnAt.Equal(at(2)) || b.RealTurns7d != 1 {
		t.Errorf("top = %+v", b)
	}
	if got[1].RealTurns7d != 2 {
		t.Errorf("A real_turns_7d = %d, want 2", got[1].RealTurns7d)
	}
	prod, err := s.DeployedManifest(ctx, "prod", at(-24*30), 5)
	if err != nil || fmt.Sprint(hashes(prod)) != "[d4]" {
		t.Errorf("prod candidates = %v, %v; want [d4]", hashes(prod), err)
	}
	// The window: turns before `since` do not count.
	got, err = s.DeployedManifest(ctx, "int", at(3), 5)
	if err != nil || len(got) != 0 {
		t.Errorf("window after every real turn = %v, %v; want none", hashes(got), err)
	}
	// limit
	got, _ = s.DeployedManifest(ctx, "int", at(-24*30), 1)
	if len(got) != 1 {
		t.Errorf("limit 1 = %d rows", len(got))
	}
	if _, err := s.DeployedManifest(ctx, "staging", at(0), 5); err == nil {
		t.Error("line 'staging' accepted")
	}
}

func TestDeployedManifest_RegistrationsAndRollback(t *testing.T) {
	s := storetest.Open(t)
	ctx := ctxT(t)
	for _, h := range []string{hashA, hashB, hashC, hashE} {
		putManifest(t, s, h, "int")
	}
	since := at(-24 * 30)
	deployed := func(want string) store.DeployedCandidate {
		t.Helper()
		got, err := s.DeployedManifest(ctx, "int", since, 5)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) == 0 || got[0].Hash[62:] != want {
			t.Fatalf("deployed = %v, want %s first", hashes(got), want)
		}
		return got[0]
	}

	register(t, s, hashA, "aaaa111", at(0))
	turnAt(t, s, "site-", hashA, "aaaa111", at(1))
	deployed("a1")

	// Deploy B: its boot registration wins even though stale merchant tabs
	// keep sending A afterwards.
	register(t, s, hashB, "bbbb222", at(2))
	turnAt(t, s, "site-", hashA, "aaaa111", at(3))
	top := deployed("b2")
	if !top.Registered() || top.LauncherBuild != "bbbb222" || top.Registrations != 1 || top.LastRealTurnAt != nil {
		t.Errorf("B = %+v", top)
	}

	// Lab, dev and empty-header registrations never count, however new.
	register(t, s, hashC, "lab-exam", at(4))
	register(t, s, hashC, "lab", at(4))
	register(t, s, hashC, "dev", at(4))
	register(t, s, hashC, "", at(4))
	// A prod launcher registration is on the other line.
	register(t, s, hashE, "v3.0.91", at(4))
	deployed("b2")

	// Rollback: the older A is re-registered by the rolled-back revision.
	register(t, s, hashA, "aaaa111", at(5))
	top = deployed("a1")
	if top.Registrations != 2 || !top.FirstRegisteredAt.Equal(at(0)) || !top.LastRegisteredAt.Equal(at(5)) {
		t.Errorf("A after rollback = %+v", top)
	}
	// The same hash under two launcher builds (a rebuild with no manifest
	// change): counted together, the newest build named.
	register(t, s, hashA, "aaaa999", at(6))
	top = deployed("a1")
	if top.LauncherBuild != "aaaa999" || top.Registrations != 3 {
		t.Errorf("A rebuilt = %+v", top)
	}
	// A turns-only hash ranks below every registered one.
	putManifest(t, s, hashD, "int")
	turnAt(t, s, "site-", hashD, "dddd444", at(7))
	got, _ := s.DeployedManifest(ctx, "int", since, 5)
	if fmt.Sprint(hashes(got)) != "[a1 b2 d4]" {
		t.Errorf("candidates = %v, want [a1 b2 d4]", hashes(got))
	}
	prod, _ := s.DeployedManifest(ctx, "prod", since, 5)
	if fmt.Sprint(hashes(prod)) != "[e5]" {
		t.Errorf("prod candidates = %v, want [e5]", hashes(prod))
	}
}

func TestRetentionPrunesRegistrations(t *testing.T) {
	s := storetest.Open(t)
	ctx := ctxT(t)
	now := time.Date(2026, 10, 1, 0, 0, 0, 0, time.UTC)
	old := now.AddDate(0, 0, -200)
	for _, h := range []string{hashA, hashB, hashC} {
		putManifest(t, s, h, "int")
	}
	register(t, s, hashA, "aaaa111", old)                 // old int launcher, newest of its line: kept
	register(t, s, hashB, "bbbb222", old.Add(-time.Hour)) // older int launcher: pruned
	register(t, s, hashB, "lab-r4", old)                  // old lab: pruned
	register(t, s, hashC, "v3.0.90", old)                 // old prod launcher, newest of its line: kept
	register(t, s, hashC, "lab", now.AddDate(0, 0, -1))   // young: kept

	s.SetClock(func() time.Time { return now })
	res, err := s.Retention(ctx, store.DefaultRetention)
	if err != nil {
		t.Fatal(err)
	}
	if res.Registrations != 2 {
		t.Errorf("pruned %d registrations, want 2", res.Registrations)
	}
	var builds []string
	rows, _ := s.Pool().Query(ctx, `SELECT launcher_build FROM manifest_registrations ORDER BY launcher_build`)
	for rows.Next() {
		var b string
		_ = rows.Scan(&b)
		builds = append(builds, b)
	}
	rows.Close()
	if fmt.Sprint(builds) != "[aaaa111 lab v3.0.90]" {
		t.Errorf("kept %v", builds)
	}
	// The kept int row still answers the deployed query (registrations are
	// not windowed by days).
	got, err := s.DeployedManifest(ctx, "int", now.AddDate(0, 0, -30), 5)
	if err != nil || fmt.Sprint(hashes(got)) != "[a1]" {
		t.Errorf("deployed after retention = %v, %v", hashes(got), err)
	}
	// A zero RegistrationDays (a policy built before the field) is the
	// 180-day default, not "prune everything".
	res, err = s.Retention(ctx, store.RetentionPolicy{TraceDays: 90, ConversationDays: 90, ShippedUsageDays: 30, InactivityDays: 90})
	if err != nil || res.Registrations != 0 {
		t.Errorf("second run = %+v, %v", res, err)
	}
}

func TestMigration0006RoundTrip(t *testing.T) {
	dsn := storetest.OpenDSN(t)
	ctx := ctxT(t)
	exists := func() bool {
		t.Helper()
		s, err := store.Open(ctx, dsn)
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		var reg *string
		if err := s.Pool().QueryRow(ctx, `SELECT to_regclass('manifest_registrations')::text`).Scan(&reg); err != nil {
			t.Fatal(err)
		}
		var idx *string
		if err := s.Pool().QueryRow(ctx, `SELECT to_regclass('turns_manifest_started')::text`).Scan(&idx); err != nil {
			t.Fatal(err)
		}
		if (reg == nil) != (idx == nil) {
			t.Fatalf("table %v and index %v disagree", reg, idx)
		}
		return reg != nil
	}
	if !exists() {
		t.Fatal("manifest_registrations missing after up")
	}
	if err := store.MigrateDown(dsn); err != nil {
		t.Fatal(err)
	}
	// Down to 0 drops turns too; re-up and step 6 again is the round trip.
	if _, err := store.Migrate(dsn); err != nil {
		t.Fatal(err)
	}
	if !exists() {
		t.Fatal("manifest_registrations missing after re-up")
	}
}
