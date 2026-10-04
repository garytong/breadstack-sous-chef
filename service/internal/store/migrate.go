package store

import (
	"database/sql"
	"errors"
	"fmt"

	"github.com/golang-migrate/migrate/v4"
	pgxmigrate "github.com/golang-migrate/migrate/v4/database/pgx/v5"
	"github.com/golang-migrate/migrate/v4/source/iofs"
	"github.com/jackc/pgx/v5/stdlib"

	"github.com/breadstack/sous-chef/migrations"
)

// Migrate applies every pending embedded migration to the database at dsn
// and returns the schema version afterwards. It opens its own short-lived
// database/sql connection (golang-migrate's pgx/v5 driver wants one) so it
// works before the pool exists — `sous-chef -migrate`, MIGRATE_AT_BOOT=on
// and the test helper all call it. A database already at the newest
// version is not an error.
func Migrate(dsn string) (version uint, err error) {
	m, closeFn, err := newMigrator(dsn)
	if err != nil {
		return 0, err
	}
	defer closeFn()
	if err := m.Up(); err != nil && !errors.Is(err, migrate.ErrNoChange) {
		return 0, fmt.Errorf("store: migrate up: %w", err)
	}
	v, dirty, err := m.Version()
	if err != nil {
		return 0, fmt.Errorf("store: migrate version: %w", err)
	}
	if dirty {
		return v, fmt.Errorf("store: schema version %d is dirty", v)
	}
	return v, nil
}

// MigrateDown rolls back every migration (tests and local resets only).
func MigrateDown(dsn string) error {
	m, closeFn, err := newMigrator(dsn)
	if err != nil {
		return err
	}
	defer closeFn()
	if err := m.Down(); err != nil && !errors.Is(err, migrate.ErrNoChange) {
		return fmt.Errorf("store: migrate down: %w", err)
	}
	return nil
}

// MigrationVersion reports the applied schema version (0 when none).
func MigrationVersion(dsn string) (uint, bool, error) {
	m, closeFn, err := newMigrator(dsn)
	if err != nil {
		return 0, false, err
	}
	defer closeFn()
	v, dirty, err := m.Version()
	if errors.Is(err, migrate.ErrNilVersion) {
		return 0, false, nil
	}
	return v, dirty, err
}

func newMigrator(dsn string) (*migrate.Migrate, func(), error) {
	src, err := iofs.New(migrations.FS, migrations.Dir)
	if err != nil {
		return nil, nil, fmt.Errorf("store: migrations fs: %w", err)
	}
	db, err := sql.Open("pgx", dsn)
	if err != nil {
		return nil, nil, fmt.Errorf("store: open for migrate: %w", err)
	}
	driver, err := pgxmigrate.WithInstance(db, &pgxmigrate.Config{})
	if err != nil {
		db.Close()
		return nil, nil, fmt.Errorf("store: migrate driver: %w", err)
	}
	m, err := migrate.NewWithInstance("iofs", src, "pgx5", driver)
	if err != nil {
		db.Close()
		return nil, nil, fmt.Errorf("store: migrator: %w", err)
	}
	closeFn := func() {
		// migrate.Close closes the driver, which closes db.
		_, _ = m.Close()
	}
	return m, closeFn, nil
}

// pgx stdlib registration: sql.Open("pgx", ...) above needs the driver
// imported; the blank identifier keeps the import when nothing else uses it.
var _ = stdlib.OpenDB
