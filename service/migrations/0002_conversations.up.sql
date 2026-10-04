-- conversations: one open conversation per site at a time (the partial
-- unique index). DELETE /v1/conversations/{siteId} sets ended_at; the rows
-- stay until retention prunes them (README §6).
CREATE TABLE conversations (
    id         uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    site_id    text        NOT NULL,
    store_id   text        NOT NULL DEFAULT '',
    created    timestamptz NOT NULL DEFAULT now(),
    updated    timestamptz NOT NULL DEFAULT now(),
    ended_at   timestamptz
);

CREATE UNIQUE INDEX conversations_open_site ON conversations (site_id) WHERE ended_at IS NULL;
CREATE INDEX conversations_ended_at ON conversations (ended_at) WHERE ended_at IS NOT NULL;

-- messages: the append-only Messages API history, replayed verbatim.
-- content is one Messages API content array (text / tool_use / tool_result /
-- thinking / redacted_thinking blocks) stored exactly as the model returned
-- it: thinking blocks ARE kept here because the API wants them back on the
-- next call of the same route (README §3). They are stripped only from
-- traces (internal/trace) and from messages-compatible requests at send time.
CREATE TABLE messages (
    conversation_id uuid        NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    seq             integer     NOT NULL,
    role            text        NOT NULL,
    content         jsonb       NOT NULL,
    created         timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (conversation_id, seq),
    CONSTRAINT messages_role CHECK (role IN ('user', 'assistant'))
);

-- turns: one per POST /v1/turns; the state machine of README §4.
-- pending_tool_calls holds the tool_use blocks the browser still owes
-- results for while status is waiting_tools or suspended.
CREATE TABLE turns (
    id                 uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id    uuid        NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    status             text        NOT NULL DEFAULT 'running',
    manifest_hash      text        NOT NULL,
    build_id           text        NOT NULL DEFAULT '',
    model              text        NOT NULL DEFAULT '',
    user_email         text        NOT NULL DEFAULT '',
    started            timestamptz NOT NULL DEFAULT now(),
    ended              timestamptz,
    usage              jsonb,
    pending_tool_calls jsonb,
    CONSTRAINT turns_status CHECK (status IN ('running', 'waiting_tools', 'suspended', 'done', 'error', 'expired'))
);

CREATE INDEX turns_conversation_started ON turns (conversation_id, started);
-- The TTL reaper scans only live turns.
CREATE INDEX turns_live_started ON turns (started) WHERE status IN ('running', 'waiting_tools', 'suspended');
