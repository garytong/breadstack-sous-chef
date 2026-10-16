"""Shared test isolation.

A stage under test (collect_stage / eval_stage `run`) writes the round's scoring policy into the process
environment (r4score.adopt_trigger_policy: $SOUSCHEF_SCORE_POLICY, $SOUSCHEF_FONT_POLICY) the way the job's process
would. Restore them after every test so one test's round policy never scores another test's turns.
"""
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
