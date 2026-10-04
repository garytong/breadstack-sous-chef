-- turns.status gains 'superseded': a waiting_tools/suspended turn that a
-- newer POST /v1/turns on the same conversation closed before the browser
-- answered it (README §4). Terminal, like done/error/expired; the live
-- partial index is unchanged.
ALTER TABLE turns DROP CONSTRAINT turns_status;
ALTER TABLE turns ADD CONSTRAINT turns_status
    CHECK (status IN ('running', 'waiting_tools', 'suspended', 'done', 'error', 'expired', 'superseded'));
