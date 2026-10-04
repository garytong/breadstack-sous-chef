-- manifests: one row per launcher manifest, keyed by the content hash
-- (sha256 of canonical JSON minus generated_from and coverage; README §1.1).
-- int and prod launchers register different manifests concurrently, so the
-- registry is content-addressed and never updated in place. build_line is
-- 'prod' when the registering X-Launcher-Build starts with 'v', else 'int';
-- it only serves the register diff ("previous manifest of the same line").
CREATE TABLE manifests (
    hash           text        PRIMARY KEY,
    body           jsonb       NOT NULL,
    index          jsonb       NOT NULL,
    generated_from jsonb,
    build_line     text        NOT NULL DEFAULT 'int',
    counts         jsonb       NOT NULL DEFAULT '{}'::jsonb,
    first_seen     timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT manifests_hash_hex CHECK (hash ~ '^[0-9a-f]{64}$'),
    CONSTRAINT manifests_build_line CHECK (build_line IN ('int', 'prod'))
);

CREATE INDEX manifests_build_line_first_seen ON manifests (build_line, first_seen DESC);
