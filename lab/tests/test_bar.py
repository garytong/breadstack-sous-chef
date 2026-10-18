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


def test_a_measured_seed_spread_widens_only_its_own_set():
    # steady baseline: a 5-row dip fails on eval noise alone, passes once σ_seed (rows) is folded in
    steady = {"replicates": [{"rows": {f"paraphrase:r{j:03d}": 1 for j in range(40)}} for _ in range(3)]}
    arms = arms_equal(paraphrase=reps("paraphrase", 40, 29))
    b0 = BAR.evaluate(arms, baseline=steady, th=TH, oos_where=where_all(arms))
    p0 = next(r for r in b0["sets"] if r["set"] == "paraphrase")
    assert not rule(b0, "sets")["pass"] and "sigma_seed" not in p0
    th = {**TH, "sigma_seed": {"paraphrase": 2.5, "holdout": 9.0}}
    b1 = BAR.evaluate(arms, baseline=steady, th=th, oos_where=where_all(arms))
    by = {r["set"]: r for r in b1["sets"]}
    assert rule(b1, "sets")["pass"], by["paraphrase"]
    assert by["paraphrase"]["sigma_seed"] == 2.5 and by["paraphrase"]["tol_eval_only"] == p0["tol"]
    assert by["paraphrase"]["tol"] >= 5 and "sigma_seed" not in by["out_of_scope"]
    # both models are seed draws: σ_seed counts twice, ⌈2·sqrt(σ_Δ² + 2·σ_seed²)⌉
    import math
    sd = by["paraphrase"]["sigma_delta"]
    assert by["paraphrase"]["tol"] == max(1, math.ceil(2 * math.sqrt(sd ** 2 + 2 * 2.5 ** 2) - 1e-9))
    from souschef_lab.retrain import noise as N
    assert by["paraphrase"]["tol"] == N.seed_aware_tol(sd, 2.5)


def test_capability_rows_are_reported_never_paired():
    """r6-4 review: a row a new tool re-keyed (p137.k2 create_page) — the live model was never offered the tool — sits in
    the report-only `capability` set: a candidate's free win there moves no set Δ, no McNemar count, no drift total."""
    arms = arms_equal()
    arms["current"]["capability"] = [{"capability:p137.k2": 0}, {"capability:p137.k2": 0}]
    arms["candidate"]["capability"] = [{"capability:p137.k2": 1}, {"capability:p137.k2": 1}]
    b = BAR.evaluate(arms, baseline=None, th=TH, oos_where=where_all(arms), trigger_rules=["a"])
    cap = {r["set"]: r for r in b["sets"]}["capability"]
    assert cap["bar"] is False and cap["delta"] == 1.0 and b["pooled"]["c"] == 0
    base = BAR.evaluate(arms_equal(), baseline=None, th=TH, oos_where=where_all(arms), trigger_rules=["a"])
    assert rule(b, "trigger")["detail"] == rule(base, "trigger")["detail"]
    assert "capability" in BAR.table_text(b)


def test_out_of_scope_allows_two_misses_on_twelve_rows():
    arms = arms_equal(out_of_scope=reps("out_of_scope", 12, 12))
    arms["current"]["out_of_scope"] = reps("out_of_scope", 12, 12)
    w = where_all(arms)
    w["candidate"] = reps("out_of_scope", 12, 10)
    assert rule(BAR.evaluate(arms, baseline=None, th=TH, oos_where=w), "oos_where")["pass"]
    w["candidate"] = reps("out_of_scope", 12, 9)
    b = BAR.evaluate(arms, baseline=None, th=TH, oos_where=w)
    assert not rule(b, "oos_where")["pass"] and "need 10" in rule(b, "oos_where")["detail"]


# --- r7: the recorded r4 reference (add_reference, rule r4_floor) ----------------------------------------------------------

def ref_doc(sets_n: dict[str, tuple[int, int]], k: int = 3, extra: dict[str, int] | None = None) -> dict:
    """A baseline.json with k replicates: per set (n rows, passed), keys "<set>:r###" (plus `extra` rows)."""
    reps_ = []
    for i in range(k):
        rows: dict[str, int] = {}
        for s, (n, passed) in sets_n.items():
            rows.update(reps(s, n, passed, k=1)[0])
        rows.update(extra or {})
        reps_.append({"at": "2026-10-14T02:43:34Z", "run": f"cal#T{i + 1}", "rows": rows})
    return {"digest": "sha256:f657", "tag": "r4", "replicates": reps_}


def cand_of(sets_n: dict[str, tuple[int, int]], k: int = 2) -> dict[str, list[dict[str, int]]]:
    return {s: reps(s, n, passed, k=k) for s, (n, passed) in sets_n.items()}


def evaluated() -> dict:
    arms = arms_equal()
    return BAR.evaluate(arms, baseline=None, th=TH, oos_where=where_all(arms))


R4_SETS = {"paraphrase": (40, 34), "holdout": (50, 44), "real": (14, 12), "followup": (12, 9)}


def test_r4_floor_passes_within_tolerance_and_adds_reference_rows():
    doc = ref_doc(R4_SETS)
    cand = cand_of({**R4_SETS, "holdout": (50, 43)})  # one row down: within tol
    b = BAR.add_reference(evaluated(), cand, BAR.reference_replicates(doc, BAR.REFERENCE_SETS), doc, TH)
    r = rule(b, "r4_floor")
    assert r["pass"] and b["pass"] and "holdout -1.0" in r["detail"], r
    by = {x["set"]: x for x in b["sets"]}
    ho = by["holdout:vs-r4"]
    assert ho["variant"] == "reference" and ho["bar"] is False and ho["n"] == 50 and ho["m"] == [3, 2]
    assert ho["current"] == 44 and ho["candidate"] == 43 and ho["tol"] >= 1
    assert {"paraphrase:vs-r4", "real:vs-r4", "followup:vs-r4"} <= set(by)
    assert b["r4_reference"]["measured"] and b["r4_reference"]["pooled"]["b"] >= 2
    t = BAR.table_text(b)
    assert t.index("paraphrase:vs-r4") > t.index("paraphrase ") and "holdout:vs-r4" in t
    # the live bar is untouched: the reference rows never feed the `sets` / `mcnemar` rules
    assert rule(b, "sets") == rule(evaluated(), "sets") and b["pooled"] == evaluated()["pooled"]


def test_r4_floor_fails_four_holdout_rows_below_r4():
    doc = ref_doc(R4_SETS)
    cand = cand_of({**R4_SETS, "holdout": (50, 40)})
    b = BAR.add_reference(evaluated(), cand, BAR.reference_replicates(doc, BAR.REFERENCE_SETS), doc, TH)
    r = rule(b, "r4_floor")
    assert not r["pass"] and not b["pass"] and r["detail"].startswith("below holdout"), r
    assert all(x["pass"] for x in b["rules"] if x["name"] != "r4_floor")  # only the floor fails
    # `bar: false` measures the same and never gates
    b2 = BAR.add_reference(evaluated(), cand, BAR.reference_replicates(doc, BAR.REFERENCE_SETS), doc, TH, gate=False)
    assert rule(b2, "r4_floor")["pass"] and b2["pass"] and "REPORT ONLY (would fail)" in rule(b2, "r4_floor")["detail"]


def test_r4_floor_pooled_mcnemar_catches_small_losses_spread_over_sets():
    big = {"paraphrase": (150, 140), "holdout": (60, 55), "real": (14, 12), "followup": (12, 9)}
    doc = ref_doc(big)
    cand = cand_of({"paraphrase": (150, 135), "holdout": (60, 51), "real": (14, 12), "followup": (12, 9)})
    th = {**TH, "sigma_seed": {"paraphrase": 4.0, "holdout": 3.0}}  # wide set tolerances: only the pooled test fails
    b = BAR.add_reference(evaluated(), cand, BAR.reference_replicates(doc, BAR.REFERENCE_SETS), doc, th)
    r = rule(b, "r4_floor")
    by = {x["set"]: x for x in b["sets"]}
    assert all(by[f"{s}:vs-r4"]["delta"] >= -by[f"{s}:vs-r4"]["tol"] for s in big)
    assert by["paraphrase:vs-r4"]["sigma_seed"] == 4.0  # σ_seed is looked up by the set, not "paraphrase:vs-r4"
    assert not r["pass"] and "McNemar" in r["detail"], r


def test_r4_floor_leaves_amended_and_stale_rows_out_and_counts_them():
    doc = ref_doc(R4_SETS, extra={"holdout:h022": 1, "holdout:h006": 1, "holdout:h099": 1})
    cand = cand_of(R4_SETS)
    for rep in cand["holdout"]:
        rep.update({"holdout:h022.k3": 0, "holdout:h006.k3": 0, "holdout:h059.k2": 0, "holdout:h200": 1})
    b = BAR.add_reference(evaluated(), cand, BAR.reference_replicates(doc, BAR.REFERENCE_SETS), doc, TH)
    ho = {x["set"]: x for x in b["sets"]}["holdout:vs-r4"]
    assert ho["n"] == 50 and ho["delta"] == 0 and rule(b, "r4_floor")["pass"]  # the .kN losses never pair
    assert ho["excluded"] == {"amended": 3, "stale": 1, "unrecorded": 1}
    info = b["r4_reference"]
    assert info["excluded_amended"] == 3 and info["excluded_stale"] == 1
    ex = info["excluded"]["holdout"]
    assert ex["amended"] == ["holdout:h006.k3", "holdout:h022.k3", "holdout:h059.k2"]
    assert ex["amended_recorded"] == ["holdout:h006", "holdout:h022"] and ex["stale"] == ["holdout:h099"]


def test_a_missing_reference_is_not_measured_and_passes():
    b = BAR.add_reference(evaluated(), cand_of(R4_SETS), None, None, TH,
                          source={"looked_in": ["baselines/reference/r4/baseline.json"]})
    r = rule(b, "r4_floor")
    assert r["pass"] and b["pass"] and r["detail"].startswith("NOT MEASURED") and "baselines/reference/r4" in r["detail"]
    assert not any(x.get("variant") == "reference" for x in b["sets"])
    assert not b["r4_reference"]["measured"]
    assert BAR.reference_text(b["r4_reference"]).startswith("Against r4's recorded answers: NOT MEASURED")
    # idempotent: a second call replaces the rule and the rows, never duplicates them
    doc = ref_doc(R4_SETS)
    BAR.add_reference(b, cand_of(R4_SETS), BAR.reference_replicates(doc, BAR.REFERENCE_SETS), doc, TH)
    BAR.add_reference(b, cand_of(R4_SETS), BAR.reference_replicates(doc, BAR.REFERENCE_SETS), doc, TH)
    assert [x["name"] for x in b["rules"]].count("r4_floor") == 1
    assert [x["set"] for x in b["sets"]].count("holdout:vs-r4") == 1
    assert "holdout +0.0 (tol" in BAR.reference_text(b["r4_reference"])


def test_r4_floor_adds_one_seed_draw_not_two():
    """r7 review: r4 is ONE fixed recorded model, only the candidate is a seed draw — the reference rows add
    1·σ_seed², the live bar 2·σ_seed². With σ_seed 2.5 the live formula gives the holdout tolerance 8 (r6's −6.3 would
    pass); the reference's gives 6."""
    doc = ref_doc(R4_SETS)
    cand = cand_of(R4_SETS)
    th = {**TH, "sigma_seed": {"holdout": 2.5}}
    b = BAR.add_reference(evaluated(), cand, BAR.reference_replicates(doc, BAR.REFERENCE_SETS), doc, th)
    ho = {x["set"]: x for x in b["sets"]}["holdout:vs-r4"]
    import math
    sd = ho["sigma_delta"]
    assert ho["tol"] == max(1, math.ceil(2 * math.sqrt(sd ** 2 + 2.5 ** 2) - 1e-9))
    live = BAR.set_row("holdout", "rederived", cand["holdout"], cand["holdout"], doc, th, bar=True)
    assert live["tol"] == max(1, math.ceil(2 * math.sqrt(live["sigma_delta"] ** 2 + 2 * 2.5 ** 2) - 1e-9))
    assert ho["tol"] < live["tol"]


def test_a_holdout_only_r4_floor_failure_a_reroll_could_explain_says_so():
    doc = ref_doc(R4_SETS)
    near = BAR.add_reference(evaluated(), cand_of({**R4_SETS, "holdout": (50, 40)}),
                             BAR.reference_replicates(doc, BAR.REFERENCE_SETS), doc, TH)
    r = rule(near, "r4_floor")
    tol = {x["set"]: x for x in near["sets"]}["holdout:vs-r4"]["tol"]
    assert not r["pass"] and ("within a reroll's reach" in r["detail"]) == (4 <= tol + BAR.REROLL_ROWS["holdout"])
    far = BAR.add_reference(evaluated(), cand_of({**R4_SETS, "holdout": (50, 30)}),
                            BAR.reference_replicates(doc, BAR.REFERENCE_SETS), doc, TH)
    assert not rule(far, "r4_floor")["pass"] and "reroll" not in rule(far, "r4_floor")["detail"]
    # still a FAIL either way: the note only points at how the failure is read
    assert not near["pass"]
