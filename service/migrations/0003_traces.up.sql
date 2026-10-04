-- traces: one per turn, kept 90 days (TRACE_RETENTION_DAYS) unless the
-- store opted out. ops = the model's raw tool calls, events = what the
-- browser reported back and what the stream emitted; they are stored apart
-- so training never learns a browser correction as model output (README §4).
-- No foreign key to turns: a trace outlives nothing, but the retention job
-- prunes the two tables on different clocks and must not be ordered by it.
CREATE TABLE traces (
    turn_id   uuid        PRIMARY KEY,
    site_id   text        NOT NULL,
    store_id  text        NOT NULL DEFAULT '',
    context   jsonb,
    events    jsonb       NOT NULL DEFAULT '[]'::jsonb,
    ops       jsonb       NOT NULL DEFAULT '[]'::jsonb,
    feedback  jsonb,
    created   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX traces_created ON traces (created);
CREATE INDEX traces_site_created ON traces (site_id, created DESC);

-- store_optout: stores whose turns are never traced. Only trace storage is
-- affected; turns, messages and usage rows are unchanged.
CREATE TABLE store_optout (
    store_id text        PRIMARY KEY,
    created  timestamptz NOT NULL DEFAULT now(),
    by_email text        NOT NULL DEFAULT ''
);
