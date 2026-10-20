"""Shared test isolation for the Mistral arm's tests (as lab/tests/conftest.py): mxarm's exam and eval adopt the
round's scoring policy into the process environment (r4score.adopt_trigger_policy: $SOUSCHEF_SCORE_POLICY,
$SOUSCHEF_FONT_POLICY), as the job's process would. Restore them after every test, so a run that collects these
beside lab/tests never scores another test's turns under this round's policy."""
from __future__ import annotations

import os

import pytest

POLICY_ENV = ("SOUSCHEF_SCORE_POLICY", "SOUSCHEF_FONT_POLICY")


@pytest.fixture(autouse=True)
def _restore_scoring_policy_env():
    before = {k: os.environ.get(k) for k in POLICY_ENV}
    yield
    for k, v in before.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
