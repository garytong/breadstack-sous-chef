"""retrain/collect_stage.py: the teacher collection driver (design §3.3). r4collect is
replaced by a recording runner here; the real one against the lab's fake service runs
in tests/test_pipeline_cli.py's end-to-end rehearsal."""
from __future__ import annotations

import json

import pytest

from souschef_lab import FIXTURES_DIR
from souschef_lab import r4dataset
from souschef_lab.retrain import collect_stage as CS
from souschef_lab.retrain.common import ConfigError, open_store

RUN = "rt-20261012-0310-ccd1afab"
FAMS = ["F0", "F1", "F2", "F3", "F7", "F0H"]


def seeded(tmp_path):
    s = open_store(tmp_path / "bucket")
    cases = []
    for i, f in enumerate(FAMS * 2):
        c = {"id": f"c-{f}-{i}", "holdout": f == "F0H", "meta": {"family": f}, "history": []}
        if f == "F2" and i >= len(FAMS):
            c["history"] = [{"user": "x"}]
        cases.append(c)
    s.write_jsonl(f"runs/{RUN}/prepare/cases.jsonl", cases)
    s.write_bytes(f"runs/{RUN}/prepare/manifest.json", (FIXTURES_DIR / "exam" / "manifest-real.json").read_bytes())
    return s, cases


class Runner:
    def __init__(self, rcs=None):
        self.calls: list[list[str]] = []
        self.rcs = list(rcs or [])

    def __call__(self, argv):
        self.calls.append(argv)
        log = argv[argv.index("--log") + 1]
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"case_id": "x", "chunk": len(self.calls)}) + "\n")
        return self.rcs.pop(0) if self.rcs else 0


@pytest.fixture
def ok_render(monkeypatch):
    def fake(cases, logs, manifest):
        ids = [x for x in (logs[0].parent / "smoke.ids").read_text().split() if x]
        return {"rows": len(ids), "ok_rows": len(ids), "rendered": len(ids), "render_sources": {"conversation": len(ids)},
                "block_drift_rows": 0, "errors": {}, "by_case": [{"case_id": i} for i in ids]}
    monkeypatch.setattr(r4dataset, "render_check", fake)


def version_of(*builds):
    seq = list(builds)

    def v(url):
        b = seq.pop(0) if len(seq) > 1 else seq[0]
        return {"build_id": b, "route_info": {"default": {"fields_context": True}, "opus": {"fields_context": True}}}
    return v


def test_smoke_ids_one_trainable_case_per_family_and_a_follow_up(tmp_path):
    s, cases = seeded(tmp_path)
    p = tmp_path / "c.jsonl"
    p.write_text("".join(json.dumps(c) + "\n" for c in cases))
    ids = CS.smoke_ids(p)
    assert ids == ["c-F0-0", "c-F1-1", "c-F2-8", "c-F3-3", "c-F7-4"]  # F0H is a holdout; F2's single-turn case skipped


def test_full_run_passes_in_order_with_chunks_prefixes_and_the_teacher_pin(tmp_path, monkeypatch, ok_render):
    s, _ = seeded(tmp_path)
    monkeypatch.setenv("SOUS_CHEF_KEY", "k-secret-not-on-argv")
    plan3 = {}

    def fake_plan3(argv):
        out = argv[argv.index("--out-dir") + 1]
        with open(f"{out}/pass3-opus.ids", "w") as f:
            f.write("c-F1-1\n")
        plan3["argv"] = argv
        return 0
    monkeypatch.setattr("souschef_lab.r4collect.main", fake_plan3)
    runner = Runner([0, 3, 3, 0, 0, 0])  # smoke; pass1 three chunks; pass2; pass3-opus
    res = CS.run(s, RUN, mode="full", attempt=1, url="https://int.example", workdir=tmp_path / "w", expect_build="f79753b-speed-a",
                 runner=runner, version=version_of("f79753b-speed-a"), progress=lambda m: None, sync_s=3600)
    assert res["status"] == "ok" and res["metrics"]["passes"] == ["pass1", "pass2", "pass3-opus"]
    labels = [a[a.index("--label") + 1] for a in runner.calls]
    assert labels == ["smoke", "pass1", "pass1", "pass1", "pass2", "pass3-opus"]
    routes = [a[a.index("--route") + 1] for a in runner.calls]
    assert routes == ["default", "default", "default", "default", "opus", "opus"]
    for a in runner.calls:
        assert a[a.index("--site-prefix") + 1].startswith("lab-") and "--url" in a
        assert a[a.index("--build-id") + 1] == f"lab-{RUN}"
        assert "k-secret-not-on-argv" not in " ".join(a)
    p2 = runner.calls[4]
    assert p2[p2.index("--select") + 1] == "unusable" and p2[p2.index("--prior") + 1].endswith("a1/pass1.jsonl")
    # the logs are in the bucket (resume material) and the result names where
    assert res["logs_prefix"] == f"runs/{RUN}/collect/a1"
    assert any(k.endswith("a1/pass1.jsonl") for k in s.list(f"runs/{RUN}/collect/a1"))


def test_a_teacher_change_ends_the_attempt_cleanly(tmp_path, monkeypatch, ok_render):
    s, _ = seeded(tmp_path)
    monkeypatch.setenv("SOUS_CHEF_KEY", "k")
    runner = Runner([0, 3, 3])
    res = CS.run(s, RUN, mode="full", attempt=1, url="https://int.example", workdir=tmp_path / "w", expect_build="b1",
                 runner=runner, version=version_of("b1", "b1", "b1", "b2"), progress=lambda m: None, sync_s=3600)
    assert res["status"] == "teacher_changed" and "b1 → b2" in res["metrics"]["teacher_changed"]
    assert res["notification"]["details"]["reason"] == "teacher_changed"
    # attempt 2 starts with its own logs and site ids
    runner2 = Runner([0, 0, 0])
    monkeypatch.setattr("souschef_lab.r4collect.main", lambda argv: 0)
    res2 = CS.run(s, RUN, mode="full", attempt=2, url="https://int.example", workdir=tmp_path / "w", expect_build="b2",
                  runner=runner2, version=version_of("b2"), progress=lambda m: None, sync_s=3600)
    assert res2["status"] == "ok" and res2["logs_prefix"].endswith("/a2")
    assert all(a[a.index("--site-prefix") + 1].startswith(f"lab-{CS.run8(RUN)}2") for a in runner2.calls)


def test_a_failed_smoke_check_stops_before_the_passes(tmp_path, monkeypatch):
    s, _ = seeded(tmp_path)
    monkeypatch.setenv("SOUS_CHEF_KEY", "k")
    monkeypatch.setattr(r4dataset, "render_check", lambda c, l, m: {"rows": 1, "ok_rows": 1, "rendered": 1,
                                                                     "render_sources": {"conversation": 1},
                                                                     "block_drift_rows": 1, "errors": {}, "by_case": []})
    runner = Runner([0])
    with pytest.raises(ConfigError):
        CS.run(s, RUN, mode="full", attempt=1, url="https://int.example", workdir=tmp_path / "w", expect_build="b1",
               runner=runner, version=version_of("b1"), progress=lambda m: None, sync_s=3600)
    res = s.read_json(f"runs/{RUN}/result-collect.json")
    assert res["status"] == "smoke_failed" and "block drift" in res["notification"]["summary"]
    assert [a[a.index("--label") + 1] for a in runner.calls] == ["smoke"]


def test_preflight_refuses_without_a_key_or_with_fields_off(tmp_path, monkeypatch, ok_render):
    s, _ = seeded(tmp_path)
    monkeypatch.delenv("SOUS_CHEF_KEY", raising=False)
    with pytest.raises(ConfigError, match="SOUS_CHEF_KEY"):
        CS.run(s, RUN, mode="full", attempt=1, url="https://int.example", workdir=tmp_path / "w", expect_build="b1",
               runner=Runner(), version=version_of("b1"), progress=lambda m: None)
    monkeypatch.setenv("SOUS_CHEF_KEY", "k")

    def off(url):
        return {"build_id": "b1", "route_info": {"default": {"fields_context": False}, "opus": {"fields_context": True}}}
    with pytest.raises(ConfigError, match="fields_context"):
        CS.run(s, RUN, mode="full", attempt=1, url="https://int.example", workdir=tmp_path / "w", expect_build="b1",
               runner=Runner(), version=off, progress=lambda m: None)
    monkeypatch.setenv("SOUS_CHEF_BUILD_FILE", str(tmp_path / "missing"))
    with pytest.raises(ConfigError, match="BUILD_ID"):
        CS.run(s, RUN, mode="full", attempt=1, url="https://int.example", workdir=tmp_path / "w2", runner=Runner(),
               version=version_of("b1"), progress=lambda m: None)


def test_a_retried_task_pulls_its_logs_back_first(tmp_path, monkeypatch, ok_render):
    s, _ = seeded(tmp_path)
    monkeypatch.setenv("SOUS_CHEF_KEY", "k")
    s.write_bytes(f"runs/{RUN}/collect/a1/pass1.jsonl", b'{"case_id": "earlier"}\n')
    seen = {}

    def runner(argv):
        log = argv[argv.index("--log") + 1]
        if argv[argv.index("--label") + 1] == "pass1":
            seen["before"] = open(log).read()
        return 0
    CS.run(s, RUN, mode="smoke", attempt=1, url="https://int.example", workdir=tmp_path / "w", expect_build="b1",
           runner=runner, version=version_of("b1"), progress=lambda m: None, sync_s=3600)
    assert "earlier" in seen["before"]


def test_smoke_mode_runs_pass1_only(tmp_path, monkeypatch, ok_render):
    s, _ = seeded(tmp_path)
    monkeypatch.setenv("SOUS_CHEF_KEY", "k")
    runner = Runner()
    res = CS.run(s, RUN, mode="smoke", attempt=1, url="https://int.example", workdir=tmp_path / "w", expect_build="b1",
                 runner=runner, version=version_of("b1"), progress=lambda m: None, sync_s=3600)
    assert res["metrics"]["passes"] == ["pass1"]
    assert [a[a.index("--label") + 1] for a in runner.calls] == ["smoke", "pass1"]
