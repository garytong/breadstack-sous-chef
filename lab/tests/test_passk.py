"""passk.py on invented eval runs: the unbiased estimator against
hand-computed values, the read-out by kind and verdict, the learnable band,
the headroom ladder, the refusals and the CLI."""

import json
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from souschef_lab import passk  # noqa: E402

HASH = "727d151b5d6b04cd68b73199ab0b3bdf061263f9623f4f6eda00a79137c06f38"
SECRET = "Fresh from the oven"  # a reply's words never reach the output


def result(cid, kind, passed, verdict=None, **kw):
    r = {"id": cid, "kind": kind, "passed": passed, "verdict": verdict or ("pass" if passed else "extra_writes"),
         "error": None, "reply": SECRET}
    r.update(kw)
    return r


def run(i, results, **kw):
    return {"label": f"exam-s{i}", "route": "default", "manifest_hash": HASH, "rule_version": "verify-1",
            "results": results, **kw}


def four_runs():
    """q1 (set_field) passes every sample, q2 (clarify) one of four, q3
    (list_add) none — one sample an error, one no_answer."""
    runs = []
    for i in range(4):
        rs = [result("q1", "set_field", True),
              result("q2", "clarify", i == 0, "clarify_ok" if i == 0 else "no_answer"),
              result("q3", "list_add", False, "invalid_op" if i > 1 else ("no_answer" if i == 0 else "error"),
                     error="HTTP 503 model_unavailable" if i == 1 else None)]
        runs.append(run(i + 1, rs))
    return runs


def write(tmp_path, runs):
    out = []
    for i, r in enumerate(runs, 1):
        p = tmp_path / f"eval-exam-s{i}-2026100100000{i}Z.json"
        p.write_text(json.dumps(r))
        out.append(p)
    return out


def test_the_unbiased_estimator_against_hand_computed_values():
    assert passk.pass_at_k(8, 0, 4) == 0.0 and passk.pass_at_k(8, 8, 1) == 1.0
    assert passk.pass_at_k(8, 1, 1) == pytest.approx(1 / 8)
    assert passk.pass_at_k(8, 1, 4) == pytest.approx(1 - 35 / 70)  # C(7,4)/C(8,4)
    assert passk.pass_at_k(8, 2, 2) == pytest.approx(1 - 15 / 28)  # C(6,2)/C(8,2)
    assert passk.pass_at_k(8, 1, 8) == 1.0 and passk.pass_at_k(8, 5, 4) == 1.0  # fewer than k fails
    assert passk.pass_at_k(3, 1, 4) is None  # fewer samples than k
    assert passk.pass_at_k(4, 5, 1) is None and passk.pass_at_k(4, 1, 0) is None
    v, m = passk.mean_pass_at_k([(8, 1), (8, 0), (3, 3)], 4)
    assert m == 2 and v == pytest.approx(0.25)
    assert passk.mean_pass_at_k([], 1) == (None, 0)


def test_the_read_out_by_kind_verdict_format_and_band():
    t = passk.tally(four_runs())
    assert (t["runs"], t["cases"], t["answers"], t["complete"]) == (4, 3, 12, 3)
    o = t["overall"]
    assert o["pass@1"] == round(100 * (1 + 0.25 + 0) / 3, 1) and o["pass@4"] == round(100 * 2 / 3, 1)
    assert o["pass@2"] == round(100 * (1 + 0.5 + 0) / 3, 1) and o["pass@8"] is None and o["pass@8_cases"] == 0
    assert t["by_kind"]["set_field"]["pass@1"] == 100.0 and t["by_kind"]["clarify"]["pass@1"] == 25.0
    assert t["by_kind"]["list_add"]["pass@4"] == 0.0 and list(t["by_kind"]) == ["clarify", "list_add", "set_field"]
    assert t["by_verdict"]["pass"] == {"answers": 4, "share_pct": 33.3}
    assert list(t["by_verdict"]) == ["invalid_op", "clarify_ok", "no_answer", "pass", "error"]  # verdict order, then extras
    assert t["format"] == {"answers": 12, "error": 1, "no_answer": 4, "failed": 5, "rate_pct": 41.7}
    assert t["learnable"] == {"band": "1 to n-1 of n", "cases": 1, "always": 1, "never": 1, "share_pct": 33.3}
    assert t["headroom"]["points"] == round(o["pass@4"] - o["pass@1"], 1) and t["headroom"]["verdict"] == "sft-then-dpo"
    assert t["per_case"] == {"q1": [4, 4], "q2": [4, 1], "q3": [4, 0]}


@pytest.mark.parametrize("points, verdict", [(4.9, "stop"), (5.0, "sft-then-dpo"), (7.5, "sft-then-dpo"),
                                             (12.0, "sft-then-dpo"), (None, "unknown")])
def test_the_ladder_runs_sft_then_dpo_from_five_points(points, verdict):
    assert passk.ladder(points)["verdict"] == verdict
    assert passk.HEADROOM_MIN_POINTS == 5.0 and passk.HEADROOM_K == 4


def test_a_case_a_run_missed_counts_with_the_samples_it_has():
    runs = four_runs()
    runs[3]["results"] = runs[3]["results"][:2]  # the last run never answered q3
    t = passk.tally(runs)
    assert t["per_case"]["q3"] == [3, 0] and t["complete"] == 2
    assert t["overall"]["pass@4_cases"] == 2  # q3 has only three samples


def test_the_cli_prints_counts_and_writes_the_tally(tmp_path, capsys):
    files = write(tmp_path, four_runs())
    out = tmp_path / "runs" / "passk-default.json"
    assert passk.main([*map(str, files), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "sft-then-dpo" in printed and "learnable band" in printed and "format failures: 5 of 12" in printed
    assert "kind clarify" in printed and "verdict pass" in printed
    assert SECRET not in printed
    saved = json.loads(out.read_text())
    assert saved["kind"] == "passk" and saved["route"] == "default" and saved["manifest_hash"] == HASH
    assert saved["rule_version"] == "verify-1" and len(saved["files"]) == 4
    assert SECRET not in out.read_text()


def _mutate(runs, i, **kw):
    runs[i].update(kw)
    return runs


@pytest.mark.parametrize("change, what", [
    (lambda rs: _mutate(rs, 1, route="opus"), "route"),
    (lambda rs: _mutate(rs, 1, manifest_hash="0" * 64), "manifest hash"),
    (lambda rs: _mutate(rs, 1, rule_version="verify-2"), "rule version"),
])
def test_runs_of_different_routes_manifests_or_rules_are_refused(tmp_path, change, what):
    files = write(tmp_path, change(four_runs()))
    with pytest.raises(passk.RunsError, match=what):
        passk.load_runs(files)


def test_a_run_given_twice_a_non_run_and_a_repeated_case_are_refused(tmp_path, capsys):
    files = write(tmp_path, four_runs())
    with pytest.raises(passk.RunsError, match="given twice"):
        passk.load_runs([files[0], files[0]])
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"label": "x"}))
    with pytest.raises(passk.RunsError, match="not an eval run"):
        passk.load_runs([bad])
    dup = four_runs()[0]
    dup["results"].append(dup["results"][0])
    (tmp_path / "dup.json").write_text(json.dumps(dup))
    with pytest.raises(passk.RunsError, match="more than once"):
        passk.load_runs([tmp_path / "dup.json"])
    assert passk.main([str(files[0]), str(files[0])]) == 2
    assert "given twice" in capsys.readouterr().err


def test_a_model_key_is_accepted_as_the_route():
    runs = four_runs()
    for r in runs:
        r["model"] = r.pop("route")
    assert passk.route_of(runs[0]) == "default" and len(passk.load_runs.__code__.co_varnames) > 0
    t = passk.tally(runs)
    assert t["cases"] == 3
