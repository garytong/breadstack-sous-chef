-- usage_rows: one row per model call (thermometer recordUsage pattern).
-- Column names follow the thermometer's /api/ingest/usage wire shape so a
-- row ships without translation; shipped flips when the thermometer
-- answered 2xx, and shipped rows older than 30 days are pruned.
CREATE TABLE usage_rows (
    id                 bigserial   PRIMARY KEY,
    turn_id            uuid,
    operation          text        NOT NULL DEFAULT 'chat',
    model              text        NOT NULL,
    site_id            text        NOT NULL DEFAULT '',
    site_host          text        NOT NULL DEFAULT '',
    store_id           text        NOT NULL DEFAULT '',
    user_email         text        NOT NULL DEFAULT '',
    input_tokens       bigint      NOT NULL DEFAULT 0,
    output_tokens      bigint      NOT NULL DEFAULT 0,
    cache_write_tokens bigint      NOT NULL DEFAULT 0,
    cache_read_tokens  bigint      NOT NULL DEFAULT 0,
    latency_ms         bigint,
    status             text        NOT NULL DEFAULT '',
    stop_reason        text        NOT NULL DEFAULT '',
    request_id         text        NOT NULL DEFAULT '',
    occurred_at        timestamptz NOT NULL DEFAULT now(),
    shipped            boolean     NOT NULL DEFAULT false,
    CONSTRAINT usage_rows_tokens_nonneg CHECK (
        input_tokens >= 0 AND output_tokens >= 0 AND cache_write_tokens >= 0 AND cache_read_tokens >= 0
    )
);

CREATE INDEX usage_rows_unshipped ON usage_rows (id) WHERE NOT shipped;
CREATE INDEX usage_rows_shipped_occurred ON usage_rows (occurred_at) WHERE shipped;
CREATE INDEX usage_rows_turn ON usage_rows (turn_id);
