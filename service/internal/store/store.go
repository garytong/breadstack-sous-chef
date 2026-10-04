// Package store is the Postgres layer: a pgx/v5 pool, the embedded
// golang-migrate migrations, and one repository file per table group
// (manifests, conversations+messages+turns, traces+store_optout,
// usage_rows) with hand-written SQL. Nothing here knows about HTTP or the
// model; the loop, trace and usage packages call it.
package store

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

// ErrNotFound is returned by every Get* when no row matches.
var ErrNotFound = errors.New("store: not found")

// ErrConflict is returned by AppendMessageAfter when the conversation's
// tail is no longer the seq the caller built on (someone else appended).
var ErrConflict = errors.New("store: conflict")

// ErrConversationEnded is returned by every append into a conversation whose
// ended_at is set (rotated away or DELETE'd: nothing is replayed from it
// again, so nothing may be written into it), and by StartTurn for a named
// conversation that has ended. It is an ErrConflict too: to the loop's
// compare-and-set it means the same thing — this writer no longer owns the
// conversation's tail.
var ErrConversationEnded = fmt.Errorf("%w: conversation has ended", ErrConflict)

// Store wraps the pool. Zero value is unusable; use Open.
type Store struct {
	pool *pgxpool.Pool
	now  func() time.Time
}

// Open connects with dsn (a pgx URL or key=value DSN; Cloud Run passes the
// /cloudsql/<conn> socket in host=) and pings once so a bad DSN fails at
// boot, not on the first turn.
func Open(ctx context.Context, dsn string) (*Store, error) {
	cfg, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		return nil, fmt.Errorf("store: parse DATABASE_URL: %w", err)
	}
	if cfg.MaxConns < 4 {
		cfg.MaxConns = 4
	}
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("store: connect: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("store: ping: %w", err)
	}
	return &Store{pool: pool, now: func() time.Time { return time.Now().UTC() }}, nil
}

// NewFromPool wraps an existing pool (tests).
func NewFromPool(pool *pgxpool.Pool) *Store {
	return &Store{pool: pool, now: func() time.Time { return time.Now().UTC() }}
}

// SetClock replaces the clock used for `created`/`occurred_at` defaults
// the store sets itself (retention cutoffs, turn expiry). Tests only.
func (s *Store) SetClock(now func() time.Time) { s.now = now }

// Pool exposes the underlying pool for callers that need a transaction
// spanning several repositories (the loop's tool-results step).
func (s *Store) Pool() *pgxpool.Pool { return s.pool }

// Ping is what /readyz calls.
func (s *Store) Ping(ctx context.Context) error { return s.pool.Ping(ctx) }

// Close releases the pool.
func (s *Store) Close() {
	if s.pool != nil {
		s.pool.Close()
	}
}

// isUniqueViolation reports whether err is Postgres 23505.
func isUniqueViolation(err error) bool {
	var pgErr *pgconn.PgError
	return errors.As(err, &pgErr) && pgErr.Code == "23505"
}

// notFound maps pgx.ErrNoRows onto ErrNotFound and leaves other errors.
func notFound(err error) error {
	if errors.Is(err, pgx.ErrNoRows) {
		return ErrNotFound
	}
	return err
}
