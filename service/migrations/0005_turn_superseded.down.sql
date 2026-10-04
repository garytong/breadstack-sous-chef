-- Rows already marked superseded would violate the old constraint; they are
-- terminal either way, so fold them into 'error' before narrowing.
UPDATE turns SET status = 'error' WHERE status = 'superseded';
ALTER TABLE turns DROP CONSTRAINT turns_status;
ALTER TABLE turns ADD CONSTRAINT turns_status
    CHECK (status IN ('running', 'waiting_tools', 'suspended', 'done', 'error', 'expired'));
