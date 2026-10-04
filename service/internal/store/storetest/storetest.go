// Package storetest gives store, trace and usage tests a migrated Postgres.
//
// Resolution order:
//  1. TEST_DATABASE_URL set → that server (docker-compose's local Postgres,
//     or CI's service container). Every test still gets its own database.
//  2. `docker` on PATH and usable → one postgres:16-alpine container per test
//     binary, started on first use and removed by Main.
//  3. Neither → every test that calls Open skips with the reason.
//
// Each Open creates a fresh database on that server, applies the embedded
// migrations and drops the database when the test ends, so tests are
// isolated and may run in parallel. No third-party test dependency: the
// container is driven through os/exec.
package storetest

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"net"
	"net/url"
	"os"
	"os/exec"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"

	"github.com/breadstack/sous-chef/internal/store"
)

const image = "postgres:16-alpine"

var (
	once        sync.Once
	adminDSN    string // server-level DSN pointing at the postgres database
	unavailable string // non-empty = reason to skip
	containerID string
	// Source says where the server came from after the first Open:
	// "env", "docker" or "".
	Source string
)

// Main wraps testing.M so the container (if one was started) is removed
// after the package's tests: `func TestMain(m *testing.M) { os.Exit(storetest.Main(m)) }`.
func Main(m *testing.M) int {
	code := m.Run()
	Shutdown()
	return code
}

// Shutdown removes the container started by this process, if any.
func Shutdown() {
	if containerID != "" {
		_ = exec.Command("docker", "rm", "-f", "-v", containerID).Run()
		containerID = ""
	}
}

// Open returns a Store on a fresh, migrated database, or skips t. The
// database is dropped when t ends.
func Open(t testing.TB) *store.Store {
	t.Helper()
	dsn := OpenDSN(t)
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	s, err := store.Open(ctx, dsn)
	if err != nil {
		t.Fatalf("storetest: open store: %v", err)
	}
	t.Cleanup(s.Close)
	return s
}

// OpenDSN is Open without the Store: a DSN to a fresh migrated database
// (for tests that build their own pool or exercise Migrate directly).
func OpenDSN(t testing.TB) string {
	t.Helper()
	once.Do(start)
	if unavailable != "" {
		t.Skip("storetest: " + unavailable)
	}
	name := "t_" + randomHex(6)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	admin, err := pgx.Connect(ctx, adminDSN)
	if err != nil {
		t.Fatalf("storetest: admin connect: %v", err)
	}
	if _, err := admin.Exec(ctx, `CREATE DATABASE `+name); err != nil {
		admin.Close(ctx)
		t.Fatalf("storetest: create database: %v", err)
	}
	admin.Close(ctx)
	dsn := withDatabase(adminDSN, name)
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		admin, err := pgx.Connect(ctx, adminDSN)
		if err != nil {
			return
		}
		defer admin.Close(ctx)
		_, _ = admin.Exec(ctx, `DROP DATABASE IF EXISTS `+name+` WITH (FORCE)`)
	})
	if _, err := store.Migrate(dsn); err != nil {
		t.Fatalf("storetest: migrate: %v", err)
	}
	return dsn
}

// start resolves the server once per process.
func start() {
	if env := strings.TrimSpace(os.Getenv("TEST_DATABASE_URL")); env != "" {
		if err := waitReady(env, 15*time.Second); err != nil {
			unavailable = "TEST_DATABASE_URL set but not reachable: " + err.Error()
			return
		}
		adminDSN = env
		Source = "env"
		return
	}
	if _, err := exec.LookPath("docker"); err != nil {
		unavailable = "no TEST_DATABASE_URL and docker not on PATH"
		return
	}
	if err := exec.Command("docker", "info").Run(); err != nil {
		unavailable = "no TEST_DATABASE_URL and docker daemon not usable"
		return
	}
	port, err := freePort()
	if err != nil {
		unavailable = "no free port: " + err.Error()
		return
	}
	password := randomHex(12)
	out, err := exec.Command("docker", "run", "-d", "--rm",
		"-e", "POSTGRES_PASSWORD="+password,
		"-e", "POSTGRES_DB=postgres",
		"-p", fmt.Sprintf("127.0.0.1:%d:5432", port),
		"--label", "sous-chef-storetest=1",
		image, "-c", "fsync=off", "-c", "synchronous_commit=off", "-c", "full_page_writes=off").Output()
	if err != nil {
		msg := err.Error()
		if ee, ok := err.(*exec.ExitError); ok {
			msg += ": " + strings.TrimSpace(string(ee.Stderr))
		}
		unavailable = "docker run " + image + " failed: " + msg
		return
	}
	containerID = strings.TrimSpace(string(out))
	dsn := fmt.Sprintf("postgres://postgres:%s@127.0.0.1:%d/postgres?sslmode=disable", password, port)
	if err := waitReady(dsn, 60*time.Second); err != nil {
		Shutdown()
		unavailable = "postgres container never became ready: " + err.Error()
		return
	}
	adminDSN = dsn
	Source = "docker"
}

// waitReady polls until a connection succeeds or the deadline passes.
func waitReady(dsn string, within time.Duration) error {
	deadline := time.Now().Add(within)
	var last error
	for time.Now().Before(deadline) {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		conn, err := pgx.Connect(ctx, dsn)
		if err == nil {
			err = conn.Ping(ctx)
			conn.Close(ctx)
		}
		cancel()
		if err == nil {
			return nil
		}
		last = err
		time.Sleep(250 * time.Millisecond)
	}
	return last
}

// withDatabase swaps the database name in a postgres:// URL.
func withDatabase(dsn, name string) string {
	u, err := url.Parse(dsn)
	if err != nil || u.Scheme == "" {
		// key=value DSN: append/replace dbname.
		parts := []string{}
		for _, kv := range strings.Fields(dsn) {
			if strings.HasPrefix(kv, "dbname=") {
				continue
			}
			parts = append(parts, kv)
		}
		return strings.Join(append(parts, "dbname="+name), " ")
	}
	u.Path = "/" + name
	return u.String()
}

func freePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}

func randomHex(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}
