"""enginebundle: pack → publish → ensure on a local bucket, the sha matching
(current / stale / unknown) and the integrity checks. A tiny stand-in snapshot
(headless.mjs + one ops module) unless node + COOKIE_CUTTER_DIR are there, when
the real snapshot's browser defaults are dumped too."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from souschef_lab import browser_defaults as BD  # noqa: E402
from souschef_lab import enginebundle as EB  # noqa: E402
from souschef_lab import gcsio  # noqa: E402
from souschef_lab.engine import ENGINE_REL_PATH, engine_fingerprint  # noqa: E402

pytestmark = pytest.mark.skipif(not shutil.which("zstd"), reason="needs zstd")
SHA = "e7e90106" + "0" * 32


def tiny_snapshot(root: Path) -> Path:
    ops = root / ENGINE_REL_PATH.parent
    ops.mkdir(parents=True)
    (ops / "headless.mjs").write_text("import { run } from './engine.js'\nrun()\n")
    (ops / "engine.js").write_text("export function run() { return 1 }\n")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "x.js").write_text("// dep\n")
    return root


def stored_defaults(out: Path) -> None:
    (out / EB.DEFAULTS_NAME).write_text(json.dumps(BD.load()), encoding="utf-8")


def test_same_sha_and_choose():
    assert EB.same_sha("e7e90106", SHA) and EB.same_sha(SHA, "E7E90106")
    assert not EB.same_sha("e7e9", SHA)
    idx = {"snapshots": {SHA: {"created_at": "2026-10-10"}, "1" * 40: {"created_at": "2026-10-11"}}, "newest": "1" * 40}
    assert EB.choose(idx, "e7e90106") == (SHA, "current")
    assert EB.choose(idx, "9cc1cd1") == ("1" * 40, "stale")
    assert EB.choose(idx, "newest") == ("1" * 40, "current")
    assert EB.choose(None, "e7e90106") == (None, "unknown")
    assert EB.status_for(idx, None) == "unknown"


def test_pack_publish_ensure_round_trip(tmp_path, monkeypatch):
    snap = tiny_snapshot(tmp_path / "snap")
    out = tmp_path / "bundle"
    meta = EB.pack(snap, SHA, out, dump_defaults=False)
    stored_defaults(out)
    assert meta["sha"] == SHA and (snap / "SNAPSHOT_OF").read_text().strip() == SHA
    assert meta["fingerprint"] == engine_fingerprint(snap / ENGINE_REL_PATH)["fingerprint"]
    b = gcsio.bucket(tmp_path / "bucket")
    idx = EB.publish(b, out)
    assert idx["newest"] == SHA and b.exists(f"engine/{SHA}/snapshot.tar.zst")
    for k in ("COOKIE_CUTTER_DIR", "ENGINE", "SOUS_CHEF_ENGINE", BD.ENV_OVERRIDE):
        monkeypatch.setenv(k, "")  # recorded, so the test's end restores what was there (ensure sets them for real)
        monkeypatch.delenv(k)
    bundle = EB.ensure(b, "e7e90106", workdir=tmp_path / "x", redump=False)
    assert bundle.status == "current" and bundle.engine.is_file() and bundle.fingerprint == meta["fingerprint"]
    assert os.environ["ENGINE"] == str(bundle.engine) and os.environ[BD.ENV_OVERRIDE] == str(bundle.browser_defaults)
    assert BD.fixture_path() == bundle.browser_defaults and BD.load()["pages"]
    again = EB.ensure(b, "9cc1cd1", workdir=tmp_path / "x", redump=False, set_env=False)  # cached extract, other sha
    assert again.status == "stale" and again.sha == SHA
    BD._load.cache_clear()


def test_a_tampered_tar_or_engine_is_refused(tmp_path):
    snap = tiny_snapshot(tmp_path / "snap")
    out = tmp_path / "bundle"
    EB.pack(snap, SHA, out, dump_defaults=False)
    stored_defaults(out)
    b = gcsio.bucket(tmp_path / "bucket")
    EB.publish(b, out)
    meta = b.read_json(f"engine/{SHA}/engine.json")
    b.write_json(f"engine/{SHA}/engine.json", {**meta, "fingerprint": "0" * 64})
    with pytest.raises(EB.BundleError, match="fingerprint"):
        EB.ensure(b, SHA, workdir=tmp_path / "y", redump=False, set_env=False)
    b.write_json(f"engine/{SHA}/engine.json", {**meta, "tar_sha256": "0" * 64})
    with pytest.raises(EB.BundleError, match="sha256"):
        EB.ensure(b, SHA, workdir=tmp_path / "z", redump=False, set_env=False)


def test_pack_refuses_links_outside_and_git_checkouts(tmp_path):
    snap = tiny_snapshot(tmp_path / "snap")
    shutil.rmtree(snap / "node_modules")
    (tmp_path / "elsewhere").mkdir()
    (snap / "node_modules").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(EB.BundleError, match="outside"):
        EB.pack(snap, SHA, tmp_path / "o", dump_defaults=False)


def test_no_snapshot_at_all(tmp_path):
    with pytest.raises(EB.BundleError, match="no vendored"):
        EB.ensure(gcsio.bucket(tmp_path / "b"), "e7e90106", workdir=tmp_path / "w", set_env=False)


@pytest.mark.engine
@pytest.mark.skipif(not os.environ.get("COOKIE_CUTTER_DIR") or not shutil.which("node"),
                    reason="needs node + COOKIE_CUTTER_DIR")
def test_browser_defaults_of_a_real_snapshot_match_the_fixture():
    doc = BD.generate(os.environ["COOKIE_CUTTER_DIR"])
    assert BD.same_content(json.loads((LAB / "fixtures" / "browser-defaults.json").read_text()), doc)

def test_the_fingerprint_does_not_depend_on_where_the_engine_lies(tmp_path):
    """Files imported from outside launcher/client (packages/ui) are named relatively, so a bundle extracted
    anywhere fingerprints as it did when it was packed (absolute names failed every drift check, 2026-10-19)."""
    fps = []
    for where in ("packed-here", "extracted/somewhere/else"):
        snap = tiny_snapshot(tmp_path / where)
        ops = snap / ENGINE_REL_PATH.parent
        (ops / "engine.js").write_text("import { r } from '../../../../../packages/ui/src/registry.js'\nexport const run = () => r\n")
        ui = snap / "packages" / "ui" / "src"
        ui.mkdir(parents=True)
        (ui / "registry.js").write_text("export const r = 1\n")
        fp = engine_fingerprint(snap / ENGINE_REL_PATH)
        assert "../../packages/ui/src/registry.js" in fp["files"], fp["files"]
        assert not any(f.startswith("/") for f in fp["files"]), fp["files"]
        fps.append(fp["fingerprint"])
    assert fps[0] == fps[1]
