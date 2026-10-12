"""baseline: replicates, the noise model (σ_run, σ_Δ), the paired McNemar and
Skip's ack reference (DESIGN §1.4)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from souschef_lab import baseline as BL  # noqa: E402
from souschef_lab import pipeline_common as PC  # noqa: E402

TH = PC.load_thresholds()


def rows(passes: dict[str, int]) -> dict[str, int]:
    return dict(passes)


def base_with(reps: list[dict[str, int]], window: int | None = None) -> dict:
    b = BL.new(digest="sha256:x", tag="r4", training_manifest="bf", engine_sha="e7", service_build="f7")
    for i, r in enumerate(reps):
        BL.add_replicate(b, r, run=f"r{i}", at="2026-10-10T00:00:00Z", ghost_writes=i, window=window)
    return b


def test_row_p_per_set_and_window():
    r1 = {"a:1": 1, "a:2": 1, "b:1": 0}
    r2 = {"a:1": 1, "a:2": 0, "b:1": 0}
    r3 = {"a:1": 1, "a:2": 1, "b:1": 1}
    b = base_with([r1, r2, r3])
    assert b["row_p"] == {"a:1": 1.0, "a:2": 0.6667, "b:1": 0.3333}
    assert b["per_set"]["total"]["runs"] == [2, 1, 3] and b["per_set"]["a"]["mean"] == 1.667
    assert b["ghost_writes_mean"] == 1.0
    assert len(base_with([r1, r2, r3], window=2)["replicates"]) == 2


def test_sigma_prior_row_p_observed_and_floor():
    keys = [f"s:{i}" for i in range(100)]
    s, how = BL.sigma_run(keys, None, TH)
    assert how == "prior" and math.isclose(s, math.sqrt(100 * 0.08 * 0.92))
    flat = base_with([{k: 1 for k in keys}] * 3)
    s, how = BL.sigma_run(keys, flat, TH)
    assert s == TH["min_sigma_rows"] and how.endswith("floor")
    flip = base_with([{k: int(i < 10) for i, k in enumerate(keys)}, {k: int(i < 20) for i, k in enumerate(keys)},
                      {k: int(i < 30) for i, k in enumerate(keys)}])
    s, how = BL.sigma_run(keys, flip, TH)
    assert how == "observed" and math.isclose(s, 10.0)


def test_compare_delta_sigma_delta_and_mcnemar():
    keys = [f"s:{i}" for i in range(50)]
    b = base_with([{k: 1 for k in keys}, {k: 1 for k in keys}])
    arm = [{k: int(i >= 8) for i, k in enumerate(keys)}]
    c = BL.compare(b, arm, TH)
    t = c["total"]
    assert t["n"] == 50 and t["delta"] == 8.0 and t["b"] == 8 and t["c"] == 0
    assert math.isclose(t["sigma_delta"], round(1.0 * math.sqrt(1 + 1 / 2), 3))
    assert t["mcnemar_p"] < 0.01


def test_compare_uses_only_common_rows_and_the_ack_reference():
    b = base_with([{"s:1": 1, "s:2": 1, "s:3": 1}, {"s:1": 1, "s:2": 1, "s:3": 1}])
    arm = [{"s:1": 0, "s:2": 1}]  # s:3 was stale on today's manifest
    c = BL.compare(b, arm, TH)
    assert c["total"]["n"] == 2 and c["total"]["delta"] == 1.0
    ref, k = BL.ack_reference({"arm_D_sets": {"total": 1.0, "s": 1.0}, "arm_D_replicates": 2})
    c2 = BL.compare(b, arm, TH, ref_means=ref, ref_reps=k)
    assert c2["total"]["delta"] == 0.0 and c2["total"]["K"] == 2


def test_no_baseline_compares_nothing_but_reports_the_arm():
    c = BL.compare(None, [{"s:1": 1, "s:2": 0}], TH)
    assert c["total"]["arm_mean"] == 1.0 and c["total"]["delta"] is None and c["total"]["mcnemar_p"] is None
    assert "no baseline" in BL.summary_line(c)


def test_row_p_sigma_is_unbiased_with_few_replicates():
    """K/(K−1): two replicates that disagree on a row are p = ½, whose plug-in p(1−p) = ¼ under-states the
    row's variance by half; the corrected estimate doubles it (verifier 2: false retrains on a K = 2 baseline)."""
    keys = [f"s:{i}" for i in range(40)]
    two = base_with([{k: 1 for k in keys}, {k: int(i >= 10) for i, k in enumerate(keys)}])
    s, how = BL.sigma_run(keys, two, TH)
    assert how == "row_p" and math.isclose(s, math.sqrt(2 / 1 * 10 * 0.25))


def test_ack_rows_compare_on_the_rows_both_have():
    """An ack keeps arm D per row: a later check compares on the rows the ack, the baseline and today's arm all
    have, so a row that went stale since the Skip never shifts Δ (per-set means would: ~0.85 rows a row)."""
    keys = [f"s:{i}" for i in range(10)]
    b = base_with([{k: 1 for k in keys}, {k: 1 for k in keys}])
    ack = {"arm_D_rows": [{k: int(i >= 2) for i, k in enumerate(keys)}], "arm_D_sets": {"total": 8.0, "s": 8.0},
           "arm_D_replicates": 1}
    today = [{k: int(i >= 2) for i, k in enumerate(keys) if i != 9}]  # s:9 (a passing row) went stale
    c = BL.compare(b, today, TH, ref_rows=BL.ack_rows(ack))
    assert c["total"]["n"] == 9 and c["total"]["delta"] == 0.0 and c["total"]["K"] == 1
    means, k = BL.ack_reference(ack)
    coarse = BL.compare(b, today, TH, ref_means=means, ref_reps=k)
    assert coarse["total"]["delta"] == 1.0  # the old per-set comparison: one row of phantom drift
    assert BL.ack_rows({"arm_D_sets": {"total": 3}}) is None and BL.ack_rows(None) is None
