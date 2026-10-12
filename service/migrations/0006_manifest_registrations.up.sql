-- manifest_registrations: every PUT/POST /v1/manifests, whether it created
-- the hash (201) or not (200), keyed by (hash, X-Launcher-Build). The
-- manifests table is content-addressed and keeps neither the registering
-- build nor any later registration, so on its own it cannot tell which
-- manifest the deployed launcher serves (README §1, GET
-- /v1/manifests/deployed). A launcher registers at every boot, so the newest
-- launcher registration follows deploys and rollbacks alike.
--
-- source is 'lab' for registrations that are not a deployed launcher
-- (store.RegistrationSource: an empty header, `lab`, `lab-…`/`lab_…`/
-- `lab.…`/`lab:…`, or a local launcher's default `dev`), else 'launcher'.
-- Retention (sous-chef -retention) prunes rows idle for 180 days but always
-- keeps the newest launcher row of each build line.
CREATE TABLE manifest_registrations (
    hash           text        NOT NULL REFERENCES manifests (hash) ON DELETE CASCADE,
    launcher_build text        NOT NULL,
    source         text        NOT NULL,
    first_at       timestamptz NOT NULL DEFAULT now(),
    last_at        timestamptz NOT NULL DEFAULT now(),
    registrations  bigint      NOT NULL DEFAULT 1,
    PRIMARY KEY (hash, launcher_build),
    CONSTRAINT manifest_registrations_source CHECK (source IN ('launcher', 'lab'))
);

CREATE INDEX manifest_registrations_source_last ON manifest_registrations (source, last_at DESC);

-- The deployed-manifest query's real-turn signal (newest non-lab turn per
-- manifest hash).
CREATE INDEX turns_manifest_started ON turns (manifest_hash, started DESC);
