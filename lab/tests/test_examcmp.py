"""examcmp: chunk merging, the Wilson interval, errors counted as fails, cost."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from souschef_lab import examcmp as M  # noqa: E402

U = {"input_tokens": 10, "output_tokens": 300, "cache_creation_input_tokens": 2000, "cache_read_input_tokens": 60000, "calls": 3}


def report(path: Path, set_name: str, route: str, rows: list[dict], *, finished: str, rng=None, of=None) -> None:
    doc = {"set": set_name, "route": route, "label": "x", "cases": of or len(rows), "fields_context": True,
           "started_at": "2026-10-10T00:00:00Z", "finished_at": finished, "results": rows}
    if rng:
        doc["rows_range"] = {"offset": rng[0], "end": rng[1], "of": of}
    path.write_text(json.dumps(doc))


def r(i, verdict="pass", status="ok", usage=None, escalated=None):
    return {"id": f"p{i}#1", "status": status, "verdict": verdict, "code": None if status == "ok" else "timeout",
            "usage": usage or {}, "escalated": escalated, "latency_ms": 1000 * i}


def test_wilson_matches_the_textbook_values():
    lo, hi = M.wilson(8, 10)
    assert (round(lo, 4), round(hi, 4)) == (0.4902, 0.9433)
    assert M.wilson(0, 0) is None and M.wilson(0, 5)[0] == 0.0 and M.wilson(5, 5)[1] == 1.0


def test_chunks_merge_later_wins_and_errors_fail(tmp_path):
    report(tmp_path / "a.json", "real", "default", [r(0), r(1, "wrong_target")], finished="2026-10-10T00:01:00Z", rng=(0, 2), of=4)
    report(tmp_path / "b.json", "real", "default", [r(2, status="error"), r(3)], finished="2026-10-10T00:02:00Z", rng=(2, 4), of=4)
    report(tmp_path / "c.json", "real", "default", [r(1)], finished="2026-10-10T00:03:00Z", rng=(1, 2), of=4)  # a re-run wins
    s = M.summarize(M.load([("cand", str(tmp_path / "*.json"))]), M.DEFAULT_ROUTE_MODELS, M.DEFAULT_FALLBACKS, 0, set())
    assert len(s) == 1
    g = s[0]
    assert (g["n"], g["of"], g["passed"], g["errors"], g["scored"]) == (4, 4, 3, 1, 3)
    assert g["rate"] == 0.75 and g["rate_scored"] == 1.0  # the error is a fail in `rate`, left out of `rate_scored`
    assert g["verdicts"] == {"error:timeout": 1, "pass": 3}
    assert "of 4!" not in M.format_table(s)


def test_cost_claude_route_and_escalated_hybrid_rows(tmp_path):
    report(tmp_path / "c.json", "real", "default", [r(0, usage=U)], finished="2026-10-10T00:01:00Z")
    report(tmp_path / "h.json", "real", "blueberry-muffin",
           [r(0, usage=U), r(1, usage=U, escalated={"from": "blueberry-muffin", "to": "default", "reason": "tool_error"})],
           finished="2026-10-10T01:00:00Z")
    groups = M.load([("cand", str(tmp_path / "c.json")), ("cand", str(tmp_path / "h.json"))])
    s = {g["route"]: g for g in M.summarize(groups, M.DEFAULT_ROUTE_MODELS, M.DEFAULT_FALLBACKS, 1.0, {"blueberry-muffin"})}
    one = (2.00 * 10 + 10.00 * 300 + 2.50 * 2000 + 0.20 * 60000) / 1e6
    assert s["default"]["cost_usd"] == pytest.approx(one, abs=1e-4) and s["default"]["cost_upper_rows"] == 0
    assert s["blueberry-muffin"]["cost_usd"] == pytest.approx(one, abs=1e-4)  # only the escalated row, at Sonnet rates
    assert s["blueberry-muffin"]["cost_upper_rows"] == 1 and s["blueberry-muffin"]["gpu_usd"] == pytest.approx(1.0)
    assert "≤" in M.format_table(list(s.values()))


def test_cli(tmp_path, capsys):
    report(tmp_path / "x.json", "out_of_scope", "default", [r(0, "unsupported_ok"), r(1, "wrong_where")],
           finished="2026-10-10T00:01:00Z")
    out = tmp_path / "s.json"
    assert M.main(["--run", f"base={tmp_path}/x.json", "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "out_of_scope" in text and "1/2" in text and "wrong_where 1" in text
    doc = json.loads(out.read_text())
    assert doc["groups"][0]["passed"] == 1 and doc["paired"] == []  # one revision: nothing to pair
    with pytest.raises(SystemExit):
        M.main(["--run", f"base={tmp_path}/nothing-*.json"])
    with pytest.raises(SystemExit):
        M.main(["--run", "base"])


def test_mcnemar_exact_p():
    assert M.mcnemar_p(0, 0) == 1.0
    assert M.mcnemar_p(0, 6) == pytest.approx(2 / 64)  # 2 × 0.5^6
    assert M.mcnemar_p(1, 9) == pytest.approx(2 * 11 / 1024)
    assert M.mcnemar_p(5, 5) == 1.0


def test_paired_counts_rows_both_revisions_ran(tmp_path):
    report(tmp_path / "exam-real-base.json", "real", "default",
           [r(0), r(1), r(2, "wrong_target"), r(3, "no_write"), r(4)], finished="2026-10-10T00:01:00Z")
    report(tmp_path / "exam-real-cand.json", "real", "default",
           [r(0), r(1, "wrong_target"), r(2), r(3), r(5)], finished="2026-10-10T00:01:00Z")  # p4 / p5 only one side
    groups = M.load([("base", str(tmp_path / "*-base.json")), ("cand", str(tmp_path / "*-cand.json"))])
    rows = M.paired(groups, ["base", "cand"])
    one = [x for x in rows if x["set"] == "real"][0]
    assert (one["n"], one["both"], one["only_a"], one["only_b"], one["neither"]) == (4, 1, 1, 2, 0)
    assert one["p"] == pytest.approx(1.0)
    assert [x for x in rows if x["set"] == "(all)"][0]["n"] == 4
    assert "a only" in M.format_paired(rows)


def test_keyword_rule_and_hidden_write_split(tmp_path):
    esc = {"from": "blueberry-muffin", "to": "default", "reason": "hidden_write"}
    rows = [dict(r(0, "unsupported_ok"), keyword_verdict="no_pointer", scored_by="where"),
            dict(r(1, "wrong_where"), keyword_verdict="unsupported_ok", scored_by="where"),
            dict(r(2, "pass", usage=U, escalated=esc))]
    report(tmp_path / "x.json", "out_of_scope", "blueberry-muffin", rows, finished="2026-10-10T00:01:00Z")
    s = M.summarize(M.load([("cand", str(tmp_path / "x.json"))]), M.DEFAULT_ROUTE_MODELS, M.DEFAULT_FALLBACKS, 0, set())[0]
    assert s["keyword_passed"] == 1 and s["passed"] == 2
    assert s["escalated_by_reason"] == {"hidden_write": 1}
    assert s["without_hidden_write"] == {"n": 2, "passed": 1, "cost_usd": 0.0}
    text = M.format_table([s])
    assert "kw (keyword rule for both) 1/3" in text and "without hidden_write 1/2" in text
