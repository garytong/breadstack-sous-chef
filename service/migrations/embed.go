// Package migrations embeds the numbered golang-migrate SQL files
// (NNNN_name.up.sql / NNNN_name.down.sql) so the binary carries its own
// schema: `sous-chef -migrate`, MIGRATE_AT_BOOT=on and the test helper all
// apply the same files through store.Migrate.
package migrations

import "embed"

// FS holds every *.sql file in this directory.
//
//go:embed *.sql
var FS embed.FS

// Dir is the path inside FS that the migrate iofs source reads.
const Dir = "."
