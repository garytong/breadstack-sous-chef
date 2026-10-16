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
