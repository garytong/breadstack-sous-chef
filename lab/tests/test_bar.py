"""retrain/bar.py: the paired eval's shipping bar (design §3.8)."""
from __future__ import annotations

from souschef_lab.retrain import bar as BAR

TH = {"prior_flip_rate": 0.08, "min_sigma_rows": 1.0}


def reps(set_name: str, n: int, passed: int, k: int = 2, offset: int = 0) -> list[dict[str, int]]:
    """k replicates of n rows, the first `passed` passing (rotated by `offset` per replicate)."""
    out = []
    for i in range(k):
        out.append({f"{set_name}:r{j:03d}": int(((j + offset * i) % n) < passed) for j in range(n)})
    return out


def arms_equal(**over) -> dict:
    a = {"paraphrase": reps("paraphrase", 40, 34), "out_of_scope": reps("out_of_scope", 13, 12),
         "F0H:unseen": reps("F0H:unseen", 30, 27, k=1), "F0H:seen": reps("F0H:seen", 10, 9, k=1)}
    cur = {k: [dict(r) for r in v] for k, v in a.items()}
    cand = {k: [dict(r) for r in v] for k, v in a.items()}
    for k, v in over.items():
        cand[k] = v
    return {"current": cur, "candidate": cand}


def where_all(arms) -> dict:
    return {a: [dict(r) for r in arms[a]["out_of_scope"]] for a in arms}


def rule(b: dict, name: str) -> dict:
    return next(r for r in b["rules"] if r["name"] == name)


def test_equal_arms_pass_and_the_table_marks_bar_sets():
    arms = arms_equal()
    b = BAR.evaluate(arms, baseline=None, th=TH, oos_where=where_all(arms), ghosts={"current": [1, 1], "candidate": [1, 0]})
    assert b["pass"], b["rules"]
    by = {r["set"]: r for r in b["sets"]}
    assert by["paraphrase"]["bar"] and by["out_of_scope"]["bar"] and by["F0H:unseen"]["bar"]
    assert not by["F0H:seen"]["bar"]
    assert by["paraphrase"]["tol"] >= 1 and by["paraphrase"]["n"] == 40
    assert "paraphrase" in BAR.table_text(b) and "F0H:seen" not in BAR.table_text(b)


def test_a_set_far_below_current_fails_and_the_pooled_mcnemar_too():
    arms = arms_equal(paraphrase=reps("paraphrase", 40, 20))
    b = BAR.evaluate(arms, baseline=None, th=TH, oos_where=where_all(arms))
    assert not b["pass"]
    assert not rule(b, "sets")["pass"] and "paraphrase" in rule(b, "sets")["detail"]
    assert not rule(b, "mcnemar")["pass"]


def test_tolerance_comes_from_the_current_baseline():
    # a noisy baseline (rows flipping) widens the tolerance enough to accept a 3-row dip
    noisy = {"replicates": [{"rows": {f"paraphrase:r{j:03d}": int((j + i) % 2) for j in range(40)}} for i in range(3)]}
    arms = arms_equal(paraphrase=reps("paraphrase", 40, 31))
    b = BAR.evaluate(arms, baseline=noisy, th=TH, oos_where=where_all(arms))
    assert rule(b, "sets")["pass"], [r for r in b["sets"] if r["set"] == "paraphrase"]
    steady = {"replicates": [{"rows": {f"paraphrase:r{j:03d}": 1 for j in range(40)}} for _ in range(3)]}
    b2 = BAR.evaluate(arms_equal(paraphrase=reps("paraphrase", 40, 29)), baseline=steady, th=TH, oos_where=where_all(arms))
    assert not rule(b2, "sets")["pass"]


def test_out_of_scope_needs_eleven_of_thirteen_by_where():
    arms = arms_equal()
    w = where_all(arms)
    w["candidate"] = reps("out_of_scope", 13, 10)
    b = BAR.evaluate(arms, baseline=None, th=TH, oos_where=w)
    assert not rule(b, "oos_where")["pass"] and "need 11" in rule(b, "oos_where")["detail"]
    w["candidate"] = reps("out_of_scope", 13, 11)
    assert rule(BAR.evaluate(arms, baseline=None, th=TH, oos_where=w), "oos_where")["pass"]


def test_f0h_unseen_regression_fails():
    arms = arms_equal(**{"F0H:unseen": reps("F0H:unseen", 30, 18, k=1)})
    b = BAR.evaluate(arms, baseline=None, th=TH, oos_where=where_all(arms))
    assert not rule(b, "f0h_unseen")["pass"]


def test_trigger_dodged_keys_need_75_percent_and_better_than_current():
    arms = arms_equal()
    key = "globalPanels.ButtonsSection.buttonShadowColor"
    probe = {key: {"current": [{"probe:a": 0, "probe:b": 0, "probe:c": 0, "probe:d": 1}],
                   "candidate": [{"probe:a": 1, "probe:b": 1, "probe:c": 1, "probe:d": 1}]}}
    b = BAR.evaluate(arms, baseline=None, th=TH, oos_where=where_all(arms), trigger_rules=["d"], dodged_keys=[key],
                     probe_by_key=probe)
    assert rule(b, "trigger")["pass"], rule(b, "trigger")
    probe[key]["candidate"] = [{"probe:a": 1, "probe:b": 1, "probe:c": 0, "probe:d": 0}]
    b = BAR.evaluate(arms, baseline=None, th=TH, oos_where=where_all(arms), trigger_rules=["d"], dodged_keys=[key],
                     probe_by_key=probe)
    assert not rule(b, "trigger")["pass"]


def test_trigger_exam_drift_needs_an_improvement_of_one_sigma():
    arms = arms_equal()
    b = BAR.evaluate(arms, baseline=None, th=TH, oos_where=where_all(arms), trigger_rules=["a"])
    assert not rule(b, "trigger")["pass"]  # no better than the live model on the drifted exam
    better = arms_equal(paraphrase=reps("paraphrase", 40, 40))
    b = BAR.evaluate(better, baseline=None, th=TH, oos_where=where_all(better), trigger_rules=["a"])
    assert rule(b, "trigger")["pass"], rule(b, "trigger")
    other = BAR.evaluate(arms, baseline=None, th=TH, oos_where=where_all(arms), trigger_rules=["e"])
    assert rule(other, "trigger")["pass"] and "not applicable" in rule(other, "trigger")["detail"]


def test_more_ghost_writes_fail():
    arms = arms_equal()
    b = BAR.evaluate(arms, baseline=None, th=TH, oos_where=where_all(arms), ghosts={"current": [0, 1], "candidate": [2, 3]})
    assert not rule(b, "ghosts")["pass"] and not b["pass"]


def test_f9h_is_a_bar_set_from_ten_rows_and_recorded_sets_never_are():
    arms = arms_equal(F9H=reps("F9H", 12, 5))
    arms["current"]["F9H"] = reps("F9H", 12, 5)
    arms["current"]["paraphrase:recorded"] = reps("paraphrase:recorded", 40, 30, k=1)
    arms["candidate"]["paraphrase:recorded"] = reps("paraphrase:recorded", 40, 10, k=1)
    b = BAR.evaluate(arms, baseline=None, th=TH, oos_where=where_all(arms))
    by = {r["set"]: r for r in b["sets"]}
    assert by["F9H"]["bar"] and not by["paraphrase:recorded"]["bar"]
    assert b["pass"]  # a recorded-context regression is reported, not a bar
    small = arms_equal(F9H=reps("F9H", 6, 1))
    small["current"]["F9H"] = reps("F9H", 6, 6)
    assert not {r["set"]: r for r in BAR.evaluate(small, baseline=None, th=TH, oos_where=where_all(small))["sets"]}["F9H"]["bar"]


def test_rows_an_arm_errored_on_are_left_out_of_both():
    arms = arms_equal()
    del arms["candidate"]["paraphrase"][0]["paraphrase:r000"]
    b = BAR.evaluate(arms, baseline=None, th=TH, oos_where=where_all(arms))
    assert {r["set"]: r for r in b["sets"]}["paraphrase"]["n"] == 39
