"""retrain/dataset_stage.py and parity_gate.py: the training set's gates (design §3.4).
The full stage runs in tests/test_pipeline_cli.py's rehearsal; the parity gate against
the real service binary runs when SOUS_CHEF_BIN names it (plus node, the engine and
the Postgres tools)."""
from __future__ import annotations

import json
import os
import shutil
import tarfile
from pathlib import Path

import pytest

from souschef_lab import FIXTURES_DIR
from souschef_lab import engine as E
from souschef_lab.retrain import dataset_stage as DS
from souschef_lab.retrain import parity_gate as PG
from souschef_lab.retrain.common import ConfigError, open_store


def row(i, *, mode="exact", follow=0, drift=False, text="Done.", calls=None, group=None):
    a = {"role": "assistant", "content": text}
    if calls:
        a["tool_calls"] = calls
    return {"id": f"r{i}", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, a],
            "meta": {"mode": mode, "history_assistant_turns": follow, "block_drift": drift, "fields_context": True,
                     "group": group or f"g{i}"}}


def write(d: Path, train, val):
    d.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
        (d / name).write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_row_checks_find_preview_drift_missing_where_and_no_follow_ups(tmp_path):
    rep = {"type": "function", "function": {"name": "report_unsupported", "arguments": json.dumps({"reason": "x", "where": "domain"})}}
    bare = {"type": "function", "function": {"name": "report_unsupported", "arguments": json.dumps({"reason": "x"})}}
    write(tmp_path, [row(1, follow=1), row(2, mode="where", calls=[rep])], [row(3)])
    assert DS.row_checks(tmp_path, need_followups=True)["problems"] == []
    write(tmp_path, [row(1), row(2, mode="where", calls=[bare]), row(3, text="Check the preview to see it.")],
          [row(4, drift=True)])
    probs = DS.row_checks(tmp_path, need_followups=True)["problems"]
    assert any("preview" in p for p in probs) and any("block drift" in p for p in probs)
    assert any("without `where`" in p for p in probs) and "train: no follow-up rows" in probs
    assert "train: no follow-up rows" not in DS.row_checks(tmp_path, need_followups=False)["problems"]
    write(tmp_path, [row(1)], [])
    assert "val: no rows" in DS.row_checks(tmp_path, need_followups=False)["problems"]


def test_ensure_val_moves_one_group_when_val_is_empty(tmp_path):
    write(tmp_path, [row(1, group="a"), row(2, group="a"), row(3, group="b")], [])
    assert DS.ensure_val(tmp_path) == 2
    tr = [json.loads(x) for x in (tmp_path / "train.jsonl").read_text().splitlines()]
    va = [json.loads(x) for x in (tmp_path / "val.jsonl").read_text().splitlines()]
    assert [r["id"] for r in tr] == ["r3"] and [r["id"] for r in va] == ["r1", "r2"]
    assert DS.ensure_val(tmp_path) == 0  # val has rows now


def test_model_context_is_the_model_image_folder(tmp_path):
    p = DS.model_context_tgz(tmp_path / "context.tar.gz")
    with tarfile.open(p) as t:
        names = t.getnames()
    assert "model-image/Dockerfile" in names and "model-image/model-context.sh" in names
    assert all(n.startswith("model-image/") for n in names)


def test_the_stage_needs_a_finished_collection(tmp_path):
    s = open_store(tmp_path / "b")
    run = "rt-x"
    s.write_jsonl(f"runs/{run}/prepare/cases.jsonl", [])
    s.write_bytes(f"runs/{run}/prepare/manifest.json", (FIXTURES_DIR / "exam" / "manifest-real.json").read_bytes())
    s.write_json(f"runs/{run}/result-collect.json", {"status": "teacher_changed"})
    with pytest.raises(ConfigError, match="collect"):
        DS.run(s, run, mode="full", workdir=tmp_path / "w", parity="skip")


def test_parity_plan_falls_back_when_the_preferred_fields_are_gone():
    from souschef_lab.manifest import Manifest
    d = json.loads((FIXTURES_DIR / "exam" / "manifest-real.json").read_text())
    d["globalPanels"]["ButtonsSection"]["fields"].pop("borderRadius")
    d["globalPanels"]["HeaderSection"]["fields"].pop("announcementBar.message")
    m = Manifest(d)
    follow, oos, single = PG.plan(m)
    tkey = next(iter(follow[0]["expected"]["ops"][0]["values"]))
    nkey = next(iter(single[0]["expected"]["ops"][0]["values"]))
    assert tkey != "announcementBar.message" and nkey != "borderRadius"
    assert follow[0]["history"][0]["expect"] == "ask" and oos[0]["score"]["mode"] == "where"


def test_local_postgres_starts_and_stops(tmp_path):
    if not (shutil.which("initdb") or os.environ.get("PG_BIN")):
        pytest.skip("no Postgres server tools here")
    import socket
    with PG.local_postgres(tmp_path) as url:
        port = int(url.split(":")[2].split("/")[0])
        with socket.create_connection(("127.0.0.1", port), timeout=5):
            pass
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=2)


def test_parity_gate_against_the_real_binary(tmp_path):
    exe = os.environ.get("SOUS_CHEF_BIN")
    if not exe or not Path(exe).is_file() or not E.engine_available() or not shutil.which("initdb"):
        pytest.skip("SOUS_CHEF_BIN (the deployed service binary), node + the engine and initdb are needed")
    rep = PG.run(FIXTURES_DIR / "exam" / "manifest-real.json", tmp_path, binary=Path(exe))
    assert rep["ok"], rep["failures"]
    assert rep["calls_checked"] == 7


def test_drop_where_missing_moves_unlinked_refusals_to_rejects(tmp_path):
    # 2026-10-14: a teacher answer can pass without calling report_unsupported at all; the row
    # gate then refused the whole first full round. The dataset stage now drops those rows first.
    import json as _json
    from souschef_lab.retrain import dataset_stage as DS

    def row(i, mode, call=None):
        msgs = [{"role": "user", "content": "connect my domain"}]
        if call is not None:
            msgs.append({"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "report_unsupported", "arguments": _json.dumps(call)}}]})
        msgs.append({"role": "assistant", "content": "I can't do that here."})
        return {"id": f"r{i}", "meta": {"mode": mode, "case_id": f"c{i}", "family": "F7"}, "messages": msgs}

    d = tmp_path / "dataset"
    DS.write_jsonl(d / "train.jsonl", [row(1, "edit"), row(2, "where", {"reason": "x", "where": "domain"}), row(3, "where")])
    DS.write_jsonl(d / "val.jsonl", [row(4, "where", {"reason": "x"})])
    DS.write_jsonl(d / "rejects.jsonl", [{"id": "r0", "case_id": "c0", "family": "F1", "verdict": "fail", "reasons": ["not_pass"]}])

    dropped = DS.drop_where_missing(d)

    assert dropped == {"train": 1, "val": 1}
    train, val, rej = (DS.read_jsonl(d / f) for f in ("train.jsonl", "val.jsonl", "rejects.jsonl"))
    assert [r["id"] for r in train] == ["r1", "r2"] and val == []
    assert [r["id"] for r in rej] == ["r0", "r3", "r4"] and rej[1]["reasons"] == ["where_missing"]
    assert DS.where_missing(train) == 0
    assert not [p for p in DS.row_checks(d, need_followups=False)["problems"] if "where" in p]
    assert DS.drop_where_missing(d) == {"train": 0, "val": 0}  # idempotent


# --- r6: the new-key positives gate, the font-pair switch, stats.json after the drops -----------------------------

def _case(cid, ops, *, mode="exact", state=None, expect_hidden=None):
    return {"id": cid, "score": {"mode": mode}, "expected": {"ops": ops, "expect_hidden": expect_hidden},
            "site_state": state, "meta": {"family": "F7"}, "context": {"page": "home"}}


def _row(i, cid, *, family="F7", follow=0):
    return {"id": f"{cid}#p{i}", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                                               {"role": "assistant", "content": "Done."}],
            "meta": {"case_id": cid, "family": family, "mode": "exact", "history_turns": follow, "fields_context": True,
                     "group": cid}}


def test_new_keys_of_reads_the_wanted_list_and_skips_holdouts_and_dropped_keys():
    rep = {"targets": {"wanted_list": [
        {"address": "a.b.c", "reason": "added_field", "holdout": False},
        {"address": "a.b.d", "reason": "added_field", "holdout": True},
        {"address": "a.b.e", "reason": "cases_dropped", "holdout": False},
        {"address": "a.b.f", "reason": "retyped_field", "holdout": False}]},
        "target_list": [{"address": "zzz", "reason": "added_field", "holdout": False}]}
    assert [t["address"] for t in DS.new_keys_of(rep)] == ["a.b.c", "a.b.f"]
    del rep["targets"]["wanted_list"]  # a prepare report written before r6: the authored targets
    assert [t["address"] for t in DS.new_keys_of(rep)] == ["zzz"]
    assert DS.new_keys_of(None) == []


def test_new_key_positives_counts_visible_writes_by_kind_and_flags_short_keys():
    from souschef_lab.manifest import Manifest
    from tests.test_author import CB_GAP, NEW_ADDR, gated, grown
    from tests.test_rebase import BTN
    d = grown()
    d["sections"] = gated()["sections"]
    m = Manifest(d)
    st = {"homeSections": [{"id": "contentBlock-1", "settings": {"style": "image-text"}}]}
    tgt = {"kind": "section", "page": "home", "sectionId": "contentBlock-1"}
    shadow = [{"op": "set_fields", "target": BTN, "values": {"buttonShadowColor": "#111111"}}]
    cases = {f"s{i}": _case(f"s{i}", shadow, mode="keys" if i == 0 else "exact") for i in range(6)}
    cases["h1"] = _case("h1", [{"op": "set_fields", "target": tgt, "values": {"stackedImageGap": 20}}], state=st)
    cases["v1"] = _case("v1", [{"op": "set_fields", "target": tgt, "values": {"imagePosition": "top", "stackedImageGap": 20}}],
                        state=st)
    rows = [_row(i, c) for i, c in enumerate(cases)]
    keys = [{"address": NEW_ADDR, "reason": "added_field"}, {"address": CB_GAP, "reason": "added_field"}]
    nk = DS.new_key_positives(rows, cases, m, keys, minimum=6)
    assert nk["keys"][NEW_ADDR]["positives"] == 6 and nk["keys"][NEW_ADDR]["kinds"] == {"relative": 1, "set": 5}
    # the hidden write is reported, not counted
    assert nk["keys"][CB_GAP]["positives"] == 1 and nk["keys"][CB_GAP]["hidden"] == 1
    assert nk["short"] == {CB_GAP: 1} and nk["ok"] is False
    nk = DS.new_key_positives(rows, cases, m, keys, minimum=6, waive={CB_GAP})
    assert nk["ok"] and nk["keys"][CB_GAP]["exempt"] == "waived"


def test_new_key_positives_count_the_whole_pack_setting_and_a_mirrored_slide_write():
    """r7 four keys: a row writing both Pack filter kinds is a positive of the whole setting (one kind alone is not);
    a set of the hero's flat btnSize (= slide 0, the same engine paths) counts for slides[].btnSize."""
    from souschef_lab.manifest import Manifest
    from tests.test_author import EDIBLES, HERO, HERO_BTN, PACK, PRODUCTS, four
    m = Manifest(four())
    cases = {"both": _case("both", [{"op": "set_fields", "target": PRODUCTS, "values": {
                 "packFilterDisplay.prerolls": "grams", "packFilterDisplay.edibles": "grams"}}]),
             "one": _case("one", [{"op": "set_fields", "target": PRODUCTS, "values": {"packFilterDisplay.edibles": "grams"}}]),
             "reset": _case("reset", [{"op": "reset", "target": PRODUCTS, "keys": ["packFilterDisplay"]}]),
             "flat": _case("flat", [{"op": "set_fields", "target": HERO, "values": {"btnSize": 20}}])}
    rows = [_row(i, c) for i, c in enumerate(cases)]
    keys = [{"address": a, "reason": "reauthor"} for a in (PACK, EDIBLES, HERO_BTN)]
    nk = DS.new_key_positives(rows, cases, m, keys, minimum=6)
    assert nk["keys"][PACK]["positives"] == 2 and nk["keys"][PACK]["kinds"] == {"set": 1, "reset": 1}
    assert nk["keys"][EDIBLES]["positives"] == 2 and nk["keys"][HERO_BTN]["positives"] == 1


def test_refresh_stats_and_drops_rewrite_stats_from_the_files(tmp_path):
    """r5's bucket stats.json kept the pre-drop counts (2,465 / 2,679) after the where hotfix."""
    from souschef_lab.manifest import Manifest
    from tests.test_author import grown
    d = tmp_path / "dataset"
    font = {"kind": "global", "panel": "TypographyColorsSection"}
    cases = {"f1": _case("f1", [{"op": "set_fields", "target": font, "values": {"bodyFont": "Lora", "headingFont": "Lora"}}]),
             "f2": _case("f2", [{"op": "set_fields", "target": font, "values": {"bodyFont": "Lora"}}])}
    DS.write_jsonl(d / "train.jsonl", [_row(1, "f1"), _row(2, "f2", follow=1), _row(3, "x", family="F1")])
    DS.write_jsonl(d / "val.jsonl", [_row(4, "y", family="F1")])
    DS.write_jsonl(d / "rejects.jsonl", [])
    (d / "stats.json").write_text(json.dumps({"rows": 99, "train": 98, "val": 1, "tokens_train": {"counter": "estimate"}}))
    (d / "dataset-manifest.json").write_text(json.dumps({"counts": {"train": 98, "val": 1, "rejects": 0},
                                                          "fields_context": {"enabled": True, "rows": 99}}))
    m = Manifest(grown())
    rows = DS.read_jsonl(d / "train.jsonl")
    assert DS.font_pair_rows(rows, cases, m) == ["f1#p1"]
    assert DS.site_font_rows(rows, cases) == []
    cases_all = {**cases, "f1": {**cases["f1"], "expected": {**cases["f1"]["expected"], "site_fonts": {"value": "Lora"}}}}
    assert DS.site_font_rows(rows, cases_all) == ["f1#p1"]  # prepare's fix_site_fonts under font policy "all"
    assert DS.drop_rows(d, {"f1#p1"}, "font_pair_policy") == {"train": 1, "val": 0}
    st = DS.refresh_stats(d, json.loads((d / "stats.json").read_text()), font_pair={"policy": "drop", "rows": 1})
    disk = json.loads((d / "stats.json").read_text())
    assert disk == st and (st["rows"], st["train"], st["val"], st["rejects"]) == (3, 2, 1, 1)
    assert st["by_family"] == {"F1": 2, "F7": 1} and st["multi_turn_rows"] == 1 and st["font_pair"]["rows"] == 1
    assert st["tokens_train"]["counter"] == "estimate" and st["tokens_train"]["total"] > 0
    man = json.loads((d / "dataset-manifest.json").read_text())
    assert man["counts"] == {"train": 2, "val": 1, "rejects": 1} and man["fields_context"]["rows"] == 3
    assert DS.read_jsonl(d / "rejects.jsonl")[0]["reasons"] == ["font_pair_policy"]


def test_the_stage_fails_a_full_round_short_of_new_key_positives(tmp_path, monkeypatch):
    """Gate 5 on a real (DRY) round: the smoke round authored 4 cases for its one target, so a full-mode
    dataset run fails with the key named; SMOKE is exempt; stats.json matches the files after the drops."""
    from souschef_lab import pipeline as P
    from souschef_lab.retrain import dry as DRY
    if not E.engine_available():
        pytest.skip("node and the engine (COOKIE_CUTTER_DIR) are needed")
    bucket, run_id = DRY.seed(tmp_path, smoke=True)
    common = ["--run-id", run_id, "--bucket", bucket, "--mode", "smoke", "--workdir", str(tmp_path / "jobs"),
              "--engine-dir", os.environ["COOKIE_CUTTER_DIR"], "--dry"]
    assert P.main(["prepare", *common, "--workers", "2"]) == 0
    assert P.main(["collect", *common, "--workers", "4"]) == 0
    s = open_store(bucket)
    rep = s.read_json(f"runs/{run_id}/prepare/report.json")
    addr = rep["target_list"][0]["address"]
    rep["targets"]["wanted_list"] = [{"address": addr, "reason": "added_field", "holdout": False}]
    s.write_json(f"runs/{run_id}/prepare/report.json", rep)
    res = DS.run(s, run_id, mode="smoke", workdir=tmp_path / "w1", parity="skip", caps=False, progress=lambda x: None)
    assert res["status"] == "ok"
    gates = json.loads((tmp_path / "w1" / "data" / "gates.json").read_text())
    assert gates["new_key_positives"]["skipped"] == "smoke"
    data = tmp_path / "w1" / "data" / "dataset"
    st = json.loads((data / "stats.json").read_text())
    assert st["train"] == len(DS.read_jsonl(data / "train.jsonl")) and st["val"] == len(DS.read_jsonl(data / "val.jsonl"))
    assert addr in st["new_key_positives"]
    with pytest.raises(ConfigError) as e:
        DS.run(s, run_id, mode="full", workdir=tmp_path / "w2", parity="skip", caps=False, progress=lambda x: None)
    assert "new_key_positives: 1 new/retyped key(s) have fewer than 6 positive-write train rows" in str(e.value)
    assert addr in str(e.value)
    monkeypatch.setenv("SOUSCHEF_NEW_KEY_POSITIVES_WAIVE", addr)
    try:
        DS.run(s, run_id, mode="full", workdir=tmp_path / "w3", parity="skip", caps=False, progress=lambda x: None)
    except ConfigError as e2:  # other full-mode gates may still object to a smoke-sized set; not this one
        assert "new_key_positives" not in str(e2)
    # the same switches from the round's trigger (run.json), and trigger.train → train-config.json
    monkeypatch.delenv("SOUSCHEF_NEW_KEY_POSITIVES_WAIVE")
    run_file = s.read_json(f"runs/{run_id}/run.json") or {}
    run_file["trigger"] = {**(run_file.get("trigger") or {}), "new_key_positives_waive": [addr],
                           "train": {"seeds": "3407,2"}}
    s.write_json(f"runs/{run_id}/run.json", run_file)
    try:
        DS.run(s, run_id, mode="full", workdir=tmp_path / "w4", parity="skip", caps=False, progress=lambda x: None)
    except ConfigError as e3:
        assert "new_key_positives" not in str(e3)
    res = DS.run(s, run_id, mode="smoke", workdir=tmp_path / "w5", parity="skip", caps=False, progress=lambda x: None)
    assert s.read_json(f"runs/{run_id}/train-config.json") == {"seeds": "3407,2"}
    # written once (by w4's run when its other gates passed, else this one); a later run keeps it
    assert res["metrics"]["train_config"] in ({"seeds": "3407,2"}, {"kept_existing": True})
    win = res["metrics"]["train_window"]
    assert win["arms"] == ["base:3407", "base:2"] and win["fits"] is True and win["window_h"] == 12


def test_train_config_of_checks_the_trigger_keys():
    assert DS.train_config_of({"trigger": {}}) is None
    assert DS.train_config_of({"trigger": {"train": {"init_adapter": "rt-x", "init_lr": 5e-5}}}) == {
        "init_adapter": "rt-x", "init_lr": 5e-5}
    with pytest.raises(ConfigError):
        DS.train_config_of({"trigger": {"train": {"lr": 1}}})
    with pytest.raises(ConfigError):
        DS.train_config_of({"trigger": {"train": "3407"}})


@pytest.mark.parametrize("train,msg", [
    ({"seeds": "3407,x"}, "seed"), ({"seed": 2, "seeds": "3,4"}, "both set"), ({"init_lr": 5e-5}, "INIT_ADAPTER"),
    ({"init_adapter": "rt-x", "init_lr": 3}, "below 1"), ({"gen_check": "lots"}, "whole number"),
    ({"init_adapter": "gs://elsewhere/base/a"}, "reads only"), ({"init_adapter": "/home/me/adapter"}, "run id"),
])
def test_train_config_values_are_checked_before_any_gpu(train, msg):
    with pytest.raises(ConfigError, match=msg):
        DS.train_config_of({"trigger": {"train": train}})


def test_a_warm_start_adapter_must_exist_in_the_bucket(tmp_path):
    from souschef_lab.retrain.common import open_store
    s = open_store(tmp_path / "bucket")
    with pytest.raises(ConfigError, match="no adapter_config.json under base/adapters/r4-sft"):
        DS.train_config_of({"trigger": {"train": {"init_adapter": "base/adapters/r4-sft"}}}, s)
    s.write_json("base/adapters/r4-sft/adapter/adapter_config.json", R4_LORA)   # the trainer accepts …/adapter/ too
    assert DS.train_config_of({"trigger": {"train": {"init_adapter": "base/adapters/r4-sft"}}}, s)
    with pytest.raises(ConfigError, match="runs/rt-1/train/adapter/adapter"):
        DS.train_config_of({"trigger": {"train": {"init_adapter": "rt-1"}}}, s)
    s.write_json("runs/rt-1/train/adapter/adapter/adapter_config.json", R4_LORA)
    assert DS.train_config_of({"trigger": {"train": {"init_adapter": "rt-1", "seeds": "3407,2"}}}, s)


R4_LORA = {"peft_type": "LORA", "r": 16, "lora_alpha": 16, "base_model_name_or_path": "unsloth/Qwen3.5-9B"}


def test_trigger_train_arms_are_checked_before_any_gpu(tmp_path):
    """The user's r6 decision (2026-10-15): from-base seeds AND a warm start from r4 in one round."""
    from souschef_lab.retrain.common import open_store
    s = open_store(tmp_path / "bucket")
    both = {"arms": ["base:3407", "base:2", "warm:3407"], "init_adapter": "base/adapters/r4-sft"}
    with pytest.raises(ConfigError, match="no adapter_config.json under base/adapters/r4-sft"):
        DS.train_config_of({"trigger": {"train": both}}, s)
    s.write_json("base/adapters/r4-sft/adapter/adapter_config.json", R4_LORA)
    assert DS.train_config_of({"trigger": {"train": both}}, s) == both
    assert DS.train_config_of({"trigger": {"train": {"arms": "base:3407,base:2"}}}, s) == {"arms": "base:3407,base:2"}
    for train, msg in (({"arms": ["base:3407", "warm:3407"]}, "set INIT_ADAPTER"),
                       ({"arms": ["base:3407", "base:2"], "init_adapter": "base/adapters/r4-sft"}, "no arm is warm"),
                       ({"arms": ["base:3407"], "seeds": "3407,2"}, "both set"),
                       ({"arms": ["cold:1"]}, "base:<seed> or warm:<seed>"),
                       ({"arms": ["base:2", "base:2"]}, "listed twice"),
                       ({"arms": ["base:1", "base:2", "base:3", "base:4", "base:5"]}, "at most 4"),
                       ({"arms": ","}, "no arm")):
        with pytest.raises(ConfigError, match=msg):
            DS.train_config_of({"trigger": {"train": train}}, s)
    # the adapter's LoRA shape and base, read from the bucket: a mismatch fails here, not after the GPU's model load
    s.write_json("base/adapters/r8/adapter_config.json", {**R4_LORA, "r": 8, "lora_alpha": 8})
    with pytest.raises(ConfigError, match="r8/α8"):
        DS.train_config_of({"trigger": {"train": {**both, "init_adapter": "base/adapters/r8"}}}, s)
    s.write_json("base/adapters/q3/adapter_config.json", {**R4_LORA, "base_model_name_or_path": "unsloth/Qwen3-8B"})
    with pytest.raises(ConfigError, match="Qwen3-8B"):
        DS.train_config_of({"trigger": {"train": {**both, "init_adapter": "base/adapters/q3"}}}, s)


def test_capability_rows_count_the_new_families_kept_and_rejected():
    """r6-4: stats.json `capabilities` — the create_page / list_options cases' rows by group and mode."""
    def cap(cid, capability, group, mode):
        c = _case(cid, [], mode=mode)
        c["meta"].update(family="F8", capability=capability, group=group)
        return c
    cases = {"p1": cap("p1", "create_page", "page:create", "page"), "p2": cap("p2", "create_page", "page:exists", "page"),
             "r1": cap("r1", "create_page", "case:F3:r4-F3-0064", "page"),
             "l1": cap("l1", "list_options", "list:tags", "list"), "x": _case("x", [])}
    train = [_row(1, "p1"), _row(2, "r1", family="F3"), _row(3, "x")]
    val = [_row(4, "l1")]
    rejects = [{"id": "p2#p1", "case_id": "p2", "family": "F8", "reasons": ["not_pass"]}]
    got = DS.capability_rows(train, val, rejects, cases)
    assert got == {
        "create_page": {"train": 2, "val": 0, "rejected": 1, "by_mode": {"page": 2},
                        "by_group": {"case:F3:r4-F3-0064": {"train": 1, "val": 0, "rejected": 0},
                                     "page:create": {"train": 1, "val": 0, "rejected": 0},
                                     "page:exists": {"train": 0, "val": 0, "rejected": 1}}},
        "list_options": {"train": 0, "val": 1, "rejected": 0, "by_mode": {"list": 1},
                         "by_group": {"list:tags": {"train": 0, "val": 1, "rejected": 0}}}}
    assert DS.capability_rows(train, val, [], {"x": cases["x"]}) == {}


# --- r7: the ask rule on kept rows, the ask / write pairs, the ask cap -------------------------------------------------

def _ask_case(cid, text, *, family=None, twin=None, mode="ask", **meta):
    from tests.test_rebase import BTN
    m = {"family": "F2", "group": f"g:{cid}", **meta}
    if family:
        m["ask_family"] = family
    if twin:
        m["ask_pair"] = {"twin": twin, "role": "ask"}
    return {"id": cid, "request": text, "target": BTN, "fields": ["buttonShadowColor"], "site_state": {},
            "score": {"mode": mode}, "expected": {"ops": []}, "context": {"page": "home"}, "meta": m}


def _write_case(cid, ask_id=None, *, key="buttonShadowColor"):
    from tests.test_rebase import BTN
    op = {"op": "set_fields", "target": BTN, "values": {key: "#111111"}}
    meta = {"family": "F2", "group": f"g:{ask_id or cid}"}
    if ask_id:
        meta["ask_pair"] = {"twin": ask_id, "role": "write"}
    return {"id": cid, "request": "x", "target": BTN, "fields": [key], "score": {"mode": "keys"},
            "expected": {"ops": [op], "keys": [{"target": BTN, "key": key, "list": ""}],
                         "direction": {"key": key, "dir": "darker"}}, "meta": meta}


def _r(cid, *, mode, user="u", n=0):
    return {"id": f"{cid}#p{n}", "messages": [{"role": "system", "content": "s"},
                                              {"role": "user", "content": f'<context>\n{{"page":"home"}}\n</context>\n{user}'},
                                              {"role": "assistant", "content": "Which colour would you like?"}],
            "meta": {"case_id": cid, "family": "F2", "mode": mode, "group": f"g:{cid}", "fields_context": True,
                     "history_turns": 0}}


def test_a_carried_ask_row_with_a_value_is_rejected_and_unpaired_asks_drop(tmp_path):
    """r7 §1.2 / §2.3: the dataset re-checks every kept ask row on its rendered text (a carried row whose case predates
    the rule included), and trains an ask only next to its write twin."""
    from souschef_lab.manifest import Manifest
    from tests.test_author import grown
    m = Manifest(grown())
    cases = {"a1": _ask_case("a1", "can we rethink the button shadow", twin="w1"),
             "a2": _ask_case("a2", "the button shadow", twin="w2"),  # its twin was never kept
             "a3": _ask_case("a3", "the button shadow", twin="w3"),  # carried row: the text it trains has a value
             "c1": _ask_case("c1", "which shadow?", mode="clarify"),
             "p1": _ask_case("p1", "add a blank page", capability="create_page"),
             "w1": _write_case("w1", "a1"), "w3": _write_case("w3", "a3")}
    d = tmp_path / "dataset"
    DS.write_jsonl(d / "train.jsonl", [_r("a1", mode="ask", user="can we rethink the button shadow"),
                                       _r("a2", mode="ask", user="the button shadow"),
                                       _r("a3", mode="ask", user="make the button shadow #7a1f2b"),
                                       _r("c1", mode="clarify"), _r("p1", mode="ask", user="add a blank page"),
                                       _r("w1", mode="keys"), _r("w3", mode="keys")])
    DS.write_jsonl(d / "val.jsonl", [])
    DS.write_jsonl(d / "rejects.jsonl", [])
    assert DS.row_user_text(_r("a3", mode="ask", user="make the button shadow #7a1f2b")) == "make the button shadow #7a1f2b"
    st = DS.enforce_ask_rules(d, cases, m, share=0)
    rej = {r["id"]: r["reasons"] for r in DS.read_jsonl(d / "rejects.jsonl")}
    assert rej == {"a3#p0": ["ask_rule_violation"], "a2#p0": ["ask_unpaired"]}
    assert st["violations"]["ids"] == {"a3#p0": "value:V1:colour_literal"} and st["violations_kept"] == 0
    assert st["unpaired"]["ids"] == ["a2#p0"] and st["unpaired_kept"] == 0
    assert st["exempt"] == {"clarify": 1, "create_page": 1} and st["rows"] == 3  # a1, c1, p1
    assert st["pairs"]["buttonShadowColor"] == {"ask": 2, "twin_write": 2, "other_relative": 0}
    # idempotent: nothing more to drop
    st2 = DS.enforce_ask_rules(d, cases, m, share=0)
    assert st2["violations"]["ids"] == {} and st2["unpaired"]["ids"] == [] and st2["rows"] == 3


def _cap_set(n_rows=1000):
    """80 ask rows over the families (plus their 80 write twins) in a 1,000-row set."""
    fams = [("complaint", 4), ("which_one", 2), ("page_title_missing", 1), ("text_missing", 8),
            ("choice_missing", 8), ("rest", 57)]
    cases, rows = {}, []
    i = 0
    for fam, n in fams:
        for _ in range(n):
            i += 1
            a, w = f"a{i:03d}", f"w{i:03d}"
            key = f"field{i % 23}"
            mode = "clarify" if fam == "which_one" else "ask"
            cases[a] = _ask_case(a, "can we rethink it", family=fam, twin=w, mode=mode)
            cases[a]["fields"] = [key]
            cases[w] = _write_case(w, a, key=key)
            rows += [_r(a, mode=mode), _r(w, mode="keys")]
    for j in range(n_rows - len(rows)):
        cases[f"o{j}"] = {"id": f"o{j}", "score": {"mode": "exact"}, "expected": {"ops": []}, "meta": {}}
        rows.append(_r(f"o{j}", mode="exact"))
    return cases, rows


def test_the_ask_cap_keeps_the_floors_and_the_twins(tmp_path):
    from souschef_lab.manifest import Manifest
    from tests.test_author import grown
    m = Manifest(grown())
    cases, rows = _cap_set()
    assert len(rows) == 1000 and sum(1 for r in rows if r["meta"]["mode"] in DS.ASK_MODES) == 80
    drop, rep = DS.cap_ask_rows(rows, cases, m, 0.03)
    assert rep["cap"] == 30 and rep["ask_rows"] == 30 and len(drop) == 50 and rep["floors_kept"] == 23
    assert rep["by_family"] == {"choice_missing": 8, "complaint": 4, "page_title_missing": 1, "rest": 7,
                                "text_missing": 8, "which_one": 2}
    again, _ = DS.cap_ask_rows(list(reversed(rows)), cases, m, 0.03)
    assert again == drop  # deterministic: sha1 order, not the files' order
    kept_rest = [r for r in rows if r["meta"]["mode"] == "ask" and r["id"] not in drop
                 and cases[r["meta"]["case_id"]]["meta"]["ask_family"] == "rest"]
    assert len({cases[r["meta"]["case_id"]]["fields"][0] for r in kept_rest}) == 7  # one per field first
    assert DS.cap_ask_rows(rows, cases, m, 0)[0] == set()  # 0: off
    # on the files: the capped rows move to rejects, every twin stays, stats.json is rewritten after the drop
    d = tmp_path / "dataset"
    DS.write_jsonl(d / "train.jsonl", rows[:900])
    DS.write_jsonl(d / "val.jsonl", rows[900:])
    DS.write_jsonl(d / "rejects.jsonl", [])
    (d / "stats.json").write_text(json.dumps({"rows": 1000, "train": 900, "val": 100}))
    st = DS.enforce_ask_rules(d, cases, m, share=0.03)
    assert st["rows"] == 30 and sum(st["capped"].values()) == 50 and st["unpaired_kept"] == 0
    kept = DS.read_jsonl(d / "train.jsonl") + DS.read_jsonl(d / "val.jsonl")
    assert len(kept) == 950 and all(r["meta"]["mode"] == "keys" for r in kept if r["meta"]["case_id"].startswith("w"))
    assert sum(1 for r in kept if r["meta"]["case_id"].startswith("w")) == 80  # the capped asks' twins included
    assert {r["reasons"][0] for r in DS.read_jsonl(d / "rejects.jsonl")} == {"ask_cap"}
    new = DS.refresh_stats(d, json.loads((d / "stats.json").read_text()), ask=st)
    assert new["rows"] == 950 and new["ask"]["rows"] == 30 and json.loads((d / "stats.json").read_text()) == new
    # share 0 keeps every ask
    DS.write_jsonl(d / "train.jsonl", rows)
    DS.write_jsonl(d / "val.jsonl", [])
    DS.write_jsonl(d / "rejects.jsonl", [])
    assert DS.enforce_ask_rules(d, cases, m, share=0)["rows"] == 80


def test_ask_share_and_positives_required_switches(monkeypatch):
    monkeypatch.delenv("SOUSCHEF_ASK_SHARE_MAX", raising=False)
    monkeypatch.delenv("SOUSCHEF_POSITIVES_REQUIRED", raising=False)
    assert DS.ask_share_of({}) == 0.03 and DS.ask_share_of({"ask_share_max": 0}) == 0.0
    assert DS.ask_share_of({"ask_share_max": "0.05"}) == 0.05 and DS.ask_share_of({}, 0.01) == 0.01
    monkeypatch.setenv("SOUSCHEF_ASK_SHARE_MAX", "0.02")
    assert DS.ask_share_of({"ask_share_max": 0.05}) == 0.02
    with pytest.raises(ConfigError):
        DS.ask_share_of({}, 1.5)
    req = DS.positives_required_of({"positives_required": ["a.b.c", "a.b.d", "a.b.c"]})
    assert req == [{"address": "a.b.c", "reason": "positives_required", "holdout": False},
                   {"address": "a.b.d", "reason": "positives_required", "holdout": False}]
    assert [t["address"] for t in DS.positives_required_of({"positives_required": "x.y.z, a.b.c"})] == ["a.b.c", "x.y.z"]
    rep = {"targets": {"wanted_list": [{"address": "s.c.k2", "reason": "reauthor", "holdout": False}]}}
    assert [t["address"] for t in DS.new_keys_of(rep)] == ["s.c.k2"]  # a re-authored key is gated like a new one
