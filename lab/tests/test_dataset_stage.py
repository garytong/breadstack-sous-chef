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
