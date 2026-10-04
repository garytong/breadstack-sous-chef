"""engine.py — the headless.mjs wrapper.

Pure-Python tests always run. Tests marked `engine` need node >= 20 and the
cookie-cutter checkout (SOUS_CHEF_ENGINE / COOKIE_CUTTER_DIR); they skip
otherwise — unless SOUS_CHEF_LAB_REQUIRE_ENGINE=1 (CI checks cookie-cutter
out alongside and must fail on a skip). Each boot costs several seconds, so
the engine tests batch several ops per call and share one Engine.
"""

import json
import os
import time

import pytest

from souschef_lab import FIXTURES_DIR
from souschef_lab import engine as E
from souschef_lab import manifest as M
from souschef_lab import state as S

REQUIRE = os.environ.get("SOUS_CHEF_LAB_REQUIRE_ENGINE") == "1"
AVAILABLE = E.engine_available()
HERO = {"kind": "section", "page": "home", "sectionId": "heroBanner"}
CB1 = {"kind": "section", "page": "home", "sectionId": "contentBlock-1"}
GALLERY = {"kind": "section", "page": "home", "sectionId": "gallery"}


def _engine_or_skip():
    if not AVAILABLE:
        reason = E.unavailable_reason()
        if REQUIRE:
            pytest.fail(f"SOUS_CHEF_LAB_REQUIRE_ENGINE=1 but the engine is unavailable: {reason}")
        pytest.skip(f"engine unavailable: {reason}")


engine_test = pytest.mark.engine


# ── pure python ─────────────────────────────────────────────────────────────

def test_require_flag_contract():
    """With the flag set, an unavailable engine is a failure, never a skip."""
    if REQUIRE:
        assert AVAILABLE, E.unavailable_reason()


def test_find_engine_env_precedence(tmp_path, monkeypatch):
    fake = tmp_path / "launcher/client/src/sousChef/ops/headless.mjs"
    fake.parent.mkdir(parents=True)
    fake.write_text("// fake")
    monkeypatch.delenv("SOUS_CHEF_ENGINE", raising=False)
    monkeypatch.setenv("COOKIE_CUTTER_DIR", str(tmp_path))
    assert E.find_engine() == fake.resolve()
    explicit = tmp_path / "other.mjs"
    explicit.write_text("// x")
    assert E.find_engine(explicit) == explicit.resolve()
    monkeypatch.setenv("SOUS_CHEF_ENGINE", str(explicit))
    assert E.find_engine() == explicit.resolve()
    assert E.find_engine(tmp_path / "missing.mjs") in (fake.resolve(), explicit.resolve())


def test_unavailable_reason_names_the_problem(tmp_path, monkeypatch):
    monkeypatch.delenv("SOUS_CHEF_ENGINE", raising=False)
    monkeypatch.setenv("COOKIE_CUTTER_DIR", str(tmp_path / "nowhere"))
    monkeypatch.setattr(E, "DEFAULT_COOKIE_CUTTER_DIR", tmp_path / "nowhere")
    assert "not found" in E.unavailable_reason()
    with pytest.raises(E.EngineError):
        E.Engine()


def test_engine_result_helpers():
    out = {
        "ok": False,
        "results": [
            {"op": {"op": "set_fields"}, "ok": True, "warnings": ["W_HIDDEN:x"], "written": ["homeSections[0].settings.x"]},
            {"op": {"op": "set_fields"}, "ok": False, "error": {"code": "E_FIELD", "message": "m", "key": "y"}, "warnings": [], "written": []},
        ],
        "changes": [{"slice": "homeSections", "written": ["homeSections[0].settings.x"],
                     "diff": [{"path": "homeSections[0].settings.x", "before": 1, "after": 2}, {"path": "homeSections[0].settings.z", "before": 3}]}],
    }
    r = E.EngineResult.from_output(out, returncode=2, elapsed_s=1.5)
    assert not r.ok and r.returncode == 2
    assert r.error_codes == ["E_FIELD"] and r.warnings == ["W_HIDDEN:x"]
    assert r.written == ["homeSections[0].settings.x"]
    assert r.diff_set() == {("homeSections[0].settings.x", "2"), ("homeSections[0].settings.z", "null")}
    assert r.to_dict()["results"] == out["results"]


def test_request_hash_stable_and_sensitive():
    a = E.request_hash({"s": 1}, [{"op": "reset"}], E.GOLDEN_CTX, "m")
    assert a == E.request_hash({"s": 1}, [{"op": "reset"}], dict(E.GOLDEN_CTX), "m")
    assert a != E.request_hash({"s": 2}, [{"op": "reset"}], E.GOLDEN_CTX, "m")
    assert a != E.request_hash({"s": 1}, [{"op": "reset"}], E.GOLDEN_CTX, "m2")


def test_parse_rejects_exit_1_and_non_json():
    with pytest.raises(E.EngineError):
        E.Engine._parse(1, "", "boom")
    with pytest.raises(E.EngineError):
        E.Engine._parse(0, "not json", "")
    assert E.Engine._parse(2, '{"ok": false}', "")["ok"] is False


def test_write_request_files(tmp_path):
    s, o, c = E.write_request_files(tmp_path / "req", {"a": 1}, [{"op": "x"}], {"templateId": "editorial"})
    assert json.loads(s.read_text()) == {"a": 1} and json.loads(o.read_text()) == [{"op": "x"}] and c is not None
    _, _, none = E.write_request_files(tmp_path / "req2", {}, [])
    assert none is None


# ── engine-backed ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def eng(tmp_path_factory):
    _engine_or_skip()
    return E.Engine(cache_dir=tmp_path_factory.mktemp("engine-cache"))


@pytest.fixture(scope="module")
def manifest():
    return M.load(FIXTURES_DIR / "manifest.snapshot.json")


@pytest.fixture(scope="module")
def base(manifest):
    return S.build_state(manifest)


@engine_test
def test_stdin_mode_set_fields_lists_reset_and_mirror(eng, base, record_property):
    """One boot: set_fields, hero mirror, edit_list add/remove/reorder/update, reset, restore_default."""
    ops = [
        {"op": "set_fields", "target": CB1, "values": {"tagline": "Fresh daily", "paddingY": 40}},
        {"op": "set_fields", "target": HERO, "values": {"headline": "<p>Yo</p>"}},                       # mirror key → slides[0] + flat
        {"op": "edit_list", "target": HERO, "list": "slides", "action": "update", "index": 0, "item": {"tagline": "Hi"}},
        {"op": "edit_list", "target": GALLERY, "list": "images", "action": "add"},
        {"op": "edit_list", "target": GALLERY, "list": "images", "action": "remove", "index": 0},
        {"op": "edit_list", "target": CB1, "list": "items", "action": "reorder", "order": [1, 0, 2]},
        {"op": "edit_list", "target": HERO, "list": "slides[].bullets", "action": "update", "index": [0, 1], "item": {"title": "Nested"}},
        {"op": "reset", "target": CB1, "keys": ["title"]},
        {"op": "restore_default", "target": CB1, "keys": ["textAlign"]},
        {"op": "set_fields", "target": {"kind": "global", "panel": "HeaderSection"}, "values": {"headerLayout": "split"}},
    ]
    t0 = time.monotonic()
    r = eng.run(base, ops)
    record_property("engine_boot_s", round(r.elapsed_s, 2))
    print(f"\n[engine] boot+apply {r.elapsed_s:.2f}s for {len(ops)} ops")
    assert r.ok and r.returncode == 0 and not r.cached
    assert r.elapsed_s > 0 and time.monotonic() - t0 >= r.elapsed_s * 0.9
    paths = {d["path"]: d for d in r.diff()}
    assert paths["homeSections[5].settings.tagline"]["after"] == "Fresh daily"
    assert paths["homeSections[5].settings.paddingY"]["after"] == 40
    assert paths["homeSections[0].settings.slides[0].headline"]["after"] == "<p>Yo</p>"
    assert paths["homeSections[0].settings.headline"]["after"] == "<p>Yo</p>"           # the mirror
    assert paths["homeSections[0].settings.slides[0].tagline"]["after"] == "Hi"
    assert paths["homeSections[0].settings.tagline"]["after"] == "Hi"                   # update slides[0] mirrors too
    assert paths["homeSections[0].settings.slides[0].bullets[1].title"]["after"] == "Nested"
    assert paths["globalSettings.headerLayout"]["after"] == "split"
    # gallery: add then remove index 0 → images[0..4] shift, images[5] becomes '' (the blank)
    gallery_before = base["homeSections"][7]["settings"]["images"]
    assert paths["homeSections[7].settings.images[0]"]["after"] == gallery_before[1]
    assert paths["homeSections[7].settings.images[5]"]["after"] == ""
    # contentBlock items swapped
    items_before = base["homeSections"][5]["settings"]["items"]
    assert paths["homeSections[5].settings.items[0].title"]["after"] == items_before[1]["title"]
    # reset writes unset_value (title has none → null), restore_default writes 'left'
    assert "homeSections[5].settings.title" in paths and paths["homeSections[5].settings.title"].get("after") is None
    assert paths.get("homeSections[5].settings.textAlign", {}).get("after", "left") == "left"
    assert all(res["ok"] for res in r.results)


@engine_test
def test_cache_hit_needs_no_boot(eng, base):
    ops = [{"op": "set_fields", "target": CB1, "values": {"tagline": "Fresh daily", "paddingY": 40}},
           {"op": "set_fields", "target": HERO, "values": {"headline": "<p>Yo</p>"}}]
    boots = eng.boots
    r1 = eng.run(base, ops)
    r2 = eng.run(base, ops)
    assert r2.cached and r2.diff() == r1.diff()
    assert eng.boots <= boots + 1
    assert eng.stats()["cache_hits"] >= 1
    # the disk cache survives a new Engine over the same dir
    other = E.Engine(cache_dir=eng.cache_dir)
    assert other.run(base, ops).cached


@engine_test
def test_exit_2_on_refused_ops_and_error_codes(eng, base):
    """E_READONLY (derived), E_FIELD, E_VALUE, W_HIDDEN warning, image companions — one boot."""
    pc1 = {"kind": "section", "page": "home", "sectionId": "productCarousel-1"}
    asset = S.load_assets_fixture()[0]
    ops = [
        {"op": "set_fields", "target": pc1, "values": {"tagId": "abc"}},                       # derived_from → E_READONLY
        {"op": "set_fields", "target": CB1, "values": {"noSuchKey": 1}},                       # E_FIELD
        {"op": "set_fields", "target": CB1, "values": {"textAlign": "diagonal"}},              # E_VALUE (enum)
        {"op": "set_fields", "target": CB1, "values": {"taglineImageSize": 20}},               # gated → W_HIDDEN, still applied
        {"op": "set_fields", "target": CB1, "values": {"taglineImage": asset["public_url"]},
         "assets": {"taglineImage": {"alt": asset["alt"], "srcset": asset["srcset"]}}},        # companions
    ]
    r = eng.run(base, ops)
    assert not r.ok and r.returncode == 2
    assert r.error_codes == ["E_READONLY", "E_FIELD", "E_VALUE"]
    assert r.results[3]["ok"] and any(w.startswith("W_HIDDEN:taglineImageSize") for w in r.results[3]["warnings"])
    paths = {d["path"]: d for d in r.diff()}
    assert paths["homeSections[5].settings.taglineImageSize"]["after"] == 20
    assert paths["homeSections[5].settings.taglineImage"]["after"].startswith("BREADSTACK_CDN/")  # compacted (gallery folder 'gal')
    srcset_leaves = [p for p in paths if p.startswith("homeSections[5].settings.taglineImageSrcSet")]
    print("[engine] companion leaves:", srcset_leaves)
    assert srcset_leaves, sorted(paths)                                                        # companions.srcset written (object → sub-leaves)


@engine_test
def test_w_hidden_is_emitted_whether_or_not_the_gate_is_open(eng, base, manifest):
    """Measured engine behaviour (validate.js:54): the warning marks a gated
    field, not a closed gate. Recorded so the verifier does not misread it."""
    import copy
    cb = manifest.section("contentBlock")
    opened = copy.deepcopy(base)
    S.apply_gate_writes(manifest, cb, opened, S.gate_writes(manifest, cb, cb.fields["taglineImageSize"], S.load_assets_fixture()), section_id="contentBlock-1")
    assert opened["homeSections"][5]["settings"]["taglineImage"]
    r = eng.run(opened, [{"op": "set_fields", "target": CB1, "values": {"taglineImageSize": 24}}])
    assert r.ok
    assert any(w.startswith("W_HIDDEN:taglineImageSize") for w in r.warnings)


@engine_test
def test_e_target_with_default_template_on_sections_add(eng, base):
    """R8: templateId 'default' is not a basic template → page sections add fails E_TARGET; editorial succeeds."""
    op = {"op": "edit_list", "target": {"kind": "page", "page": "home"}, "list": "sections", "action": "add", "item": {"id": "promotion"}}
    bad = eng.run(base, [op], {**E.GOLDEN_CTX, "templateId": "default"})
    assert not bad.ok and bad.error_codes == ["E_TARGET"]
    good = eng.run(base, [op])
    assert good.ok and good.results[0].get("sectionId", "").startswith("promotion")
    ids_after = [d for d in good.diff() if d["path"].endswith(".id")]
    assert any(d["after"].startswith("promotion") for d in ids_after)


@engine_test
def test_bare_section_target_is_filled_from_ctx_current_page(eng, base):
    """The tool description lets the model omit `target.page` ("the host fills
    it in"); the browser host does it with previewApi.withTargetDefaults(ops,
    ctx) and, since 2026-10-03, so does headless.mjs — the collector passes
    `currentPage` in ctx (collect.ctx_for) and int-run2 lost 16 rows to
    `E_TARGET "section target needs a page"` + a retry before it did. Without
    `currentPage` nothing is filled and the op still fails E_TARGET."""
    bare = {"kind": "section", "sectionId": "contentBlock-1"}  # CB1 minus page
    ops = [{"op": "set_fields", "target": bare, "values": {"tagline": "No page given"}},
           {"op": "reset", "target": bare, "keys": ["tagline"]}]
    filled = eng.run(base, ops, {**E.GOLDEN_CTX, "currentPage": "home"})
    assert filled.ok and filled.returncode == 0 and all(r["ok"] for r in filled.results)
    assert [r["op"]["target"] for r in filled.results] == [{**bare, "page": "home"}] * 2  # the engine saw filled targets
    assert set(filled.written) == {"homeSections[5].settings.tagline"}
    assert [d["path"] for d in filled.diff()] == ["homeSections[5].settings.tagline"]
    unfilled = eng.run(base, ops[:1])  # GOLDEN_CTX has no currentPage
    assert not unfilled.ok and unfilled.returncode == 2 and unfilled.error_codes == ["E_TARGET"]
    assert "needs a page" in unfilled.errors[0]["message"] and unfilled.results[0]["op"]["target"] == bare
    assert unfilled.diff() == []


@engine_test
def test_file_mode_and_out_file(eng, base, tmp_path):
    ops = [{"op": "set_page_layout", "page": "products", "layout": "filter-right"},
           {"op": "reset_pinned_section", "page": "products", "sectionId": "__pinned_product_list"}]
    st = json.loads(json.dumps(base))
    st["pageSectionsAll"]["products"] = [{"id": "__pinned_product_list", "enabled": False, "settings": {"stale": 1}}]
    s, o, c = E.write_request_files(tmp_path, st, ops, E.GOLDEN_CTX)
    r = eng.run_files(s, o, c, out_path=tmp_path / "out.json")
    assert r.ok and (tmp_path / "out.json").exists()
    paths = {d["path"]: d for d in r.diff()}
    assert paths["layouts.products"]["after"] == "filter-right"
    assert paths["pageSectionsAll.products[0].enabled"]["after"] is True
    assert paths["pageSectionsAll.products[0].settings.stale"].get("after") is None
    assert r.results[0].get("host") is True


@engine_test
def test_run_many_preserves_order(eng, base):
    reqs = [(base, [{"op": "set_fields", "target": CB1, "values": {"tagline": f"t{i}"}}], None) for i in range(2)]
    rs = eng.run_many(reqs, workers=2)
    assert [r.diff()[0]["after"] for r in rs] == ["t0", "t1"]
    assert all(r.ok for r in rs)


# ── engine identity (LAB.md §2: engine_fingerprint) ────────────────────────

def _ops_tree(root, files):
    ops = root / "launcher/client/src/sousChef/ops"
    ops.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (ops / name).write_text(text)
    return ops / "headless.mjs"


def test_engine_fingerprint_covers_sources_not_tests(tmp_path, monkeypatch):
    monkeypatch.delenv("SOUS_CHEF_ENGINE", raising=False)
    monkeypatch.delenv("COOKIE_CUTTER_DIR", raising=False)
    script = _ops_tree(tmp_path, {"headless.mjs": "// h", "engine.js": "// e", "engine.test.js": "// t", "README.md": "x"})
    a = E.engine_fingerprint(script)
    assert a and len(a["fingerprint"]) == 64 and a["commit"] is None and a["packages"] == []
    assert a["files"] == ["src/sousChef/ops/engine.js", "src/sousChef/ops/headless.mjs"]  # relative to the client root
    (script.parent / "engine.test.js").write_text("// changed test")
    assert E.engine_fingerprint(script)["fingerprint"] == a["fingerprint"]  # tests never move the fingerprint
    (script.parent / "engine.js").write_text("// changed engine")
    b = E.engine_fingerprint(script)
    assert b["fingerprint"] != a["fingerprint"]
    (script.parent / "validate.js").write_text("// new file")
    assert E.engine_fingerprint(script)["fingerprint"] not in (a["fingerprint"], b["fingerprint"])
    assert E.engine_fingerprint(tmp_path / "nowhere.mjs") in (None, E.engine_fingerprint())  # falls back to discovery


CLIENT_TREE = {
    "src/sousChef/ops/headless.mjs": "import fs from 'node:fs';\nconst { createServer } = await import('vite');\n"
                                     "const engine = await runner.import('/src/sousChef/ops/engine.js');\n",
    "src/sousChef/ops/engine.js": "import { compactAssetUrl } from '../../lib/cdnUrl.js';\n"
                                  "import {\n  getAvailableSectionsForPage,\n} from '../../data/pageSections.jsx';\n"
                                  "import { validate } from './validate.js';\nexport const x = 'from nowhere';\n",
    "src/sousChef/ops/validate.js": "import { PRODUCT_SORT_OPTIONS } from '../../components/basicEditor/constants.jsx';\n"
                                    "export * from './schema';\n",
    "src/sousChef/ops/schema.js": "export const ERRORS = {};\n",
    "src/sousChef/ops/validate.test.js": "import { validate } from './validate.js';\nimport './unrelated.js';\n",
    "src/sousChef/ops/unrelated.js": "// imported only by a test — reached through the ops/*.js union, not the closure\n",
    "src/data/pageSections.jsx": "import { Thing } from '../components/Thing';\nexport const getAvailableSectionsForPage = () => [];\n",
    "src/components/Thing/index.jsx": "export const Thing = 1;\n",
    "src/components/basicEditor/constants.jsx": "import { stripTags } from '@breadstack/ui';\nexport const PRODUCT_SORT_OPTIONS = [];\n",
    "src/lib/cdnUrl.js": "export const compactAssetUrl = (u) => u;\n",
    "node_modules/@breadstack/ui/package.json": '{"name": "@breadstack/ui", "version": "0.1.0"}',
}


def _client_tree(root, files=CLIENT_TREE):
    client = root / "launcher/client"
    for name, text in files.items():
        p = client / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return client / "src/sousChef/ops/headless.mjs"


def test_import_specifiers_and_resolution(tmp_path):
    specs = E.import_specifiers(CLIENT_TREE["src/sousChef/ops/engine.js"])
    assert specs == ["../../lib/cdnUrl.js", "../../data/pageSections.jsx", "./validate.js"]  # no 'from nowhere' string
    assert E.import_specifiers(CLIENT_TREE["src/sousChef/ops/headless.mjs"]) == ["node:fs", "vite", "/src/sousChef/ops/engine.js"]
    assert E.import_specifiers(CLIENT_TREE["src/sousChef/ops/validate.js"]) == ["../../components/basicEditor/constants.jsx", "./schema"]
    script = _client_tree(tmp_path)
    client = script.parents[3]
    assert E.resolve_import("./schema", script.parent / "validate.js", client) == (script.parent / "schema.js").resolve()
    assert E.resolve_import("../components/Thing", client / "src/data/pageSections.jsx", client) == \
        (client / "src/components/Thing/index.jsx").resolve()
    assert E.resolve_import("/src/sousChef/ops/engine.js", script, client) == (script.parent / "engine.js").resolve()
    assert E.resolve_import("vite", script, client) is None and E.resolve_import("./missing.js", script, client) is None
    assert E.workspace_package_version("@breadstack/ui", client / "src/components/basicEditor") == "@breadstack/ui@0.1.0"
    assert E.workspace_package_version("@breadstack/nothing", client) is None


def test_engine_fingerprint_hashes_the_import_closure_of_headless(tmp_path, monkeypatch):
    """Editing a scratch copy of src/data/pageSections.jsx (outside ops/) must
    move the fingerprint; so must a file two hops away and the workspace
    package version; a test file never does."""
    monkeypatch.delenv("SOUS_CHEF_ENGINE", raising=False)
    monkeypatch.delenv("COOKIE_CUTTER_DIR", raising=False)
    script = _client_tree(tmp_path)
    client = script.parents[3]
    files, packages = E.import_closure(script)
    rel = [p.relative_to(client.resolve()).as_posix() for p in files]
    assert rel == ["src/components/Thing/index.jsx", "src/components/basicEditor/constants.jsx", "src/data/pageSections.jsx",
                   "src/lib/cdnUrl.js", "src/sousChef/ops/engine.js", "src/sousChef/ops/headless.mjs", "src/sousChef/ops/schema.js",
                   "src/sousChef/ops/validate.js"]
    assert packages == ["@breadstack/ui@0.1.0"]
    a = E.engine_fingerprint(script)
    assert a["files"] == [*rel[:4], "src/sousChef/ops/engine.js", "src/sousChef/ops/headless.mjs", "src/sousChef/ops/schema.js",
                          "src/sousChef/ops/unrelated.js", "src/sousChef/ops/validate.js"]  # closure ∪ ops/*.js
    assert a["packages"] == ["@breadstack/ui@0.1.0"]
    seen = {a["fingerprint"]}

    def moved(path, text):
        (client / path).write_text(text)
        fp = E.engine_fingerprint(script)["fingerprint"]
        assert fp not in seen, path
        seen.add(fp)

    moved("src/data/pageSections.jsx", CLIENT_TREE["src/data/pageSections.jsx"] + "// behaviour change\n")
    moved("src/components/Thing/index.jsx", "export const Thing = 2;\n")
    moved("src/components/basicEditor/constants.jsx", CLIENT_TREE["src/components/basicEditor/constants.jsx"] + "//\n")
    moved("src/lib/cdnUrl.js", "export const compactAssetUrl = (u) => u + '';\n")
    moved("node_modules/@breadstack/ui/package.json", '{"name": "@breadstack/ui", "version": "0.2.0"}')
    (client / "src/sousChef/ops/validate.test.js").write_text("// changed test\n")
    assert E.engine_fingerprint(script)["fingerprint"] in seen  # tests never move the fingerprint
    (client / "src/components/Other.jsx").write_text("export const Other = 1;\n")  # not imported: not part of the engine
    assert E.engine_fingerprint(script)["fingerprint"] in seen


def test_engine_fingerprint_includes_git_head_when_available(tmp_path, monkeypatch):
    import shutil
    import subprocess
    if not shutil.which("git"):
        pytest.skip("git not on PATH")
    monkeypatch.delenv("SOUS_CHEF_ENGINE", raising=False)
    monkeypatch.delenv("COOKIE_CUTTER_DIR", raising=False)
    script = _ops_tree(tmp_path, {"headless.mjs": "// h"})
    before = E.engine_fingerprint(script)
    env = {"GIT_AUTHOR_NAME": "lab", "GIT_AUTHOR_EMAIL": "lab@example.invalid", "GIT_COMMITTER_NAME": "lab",
           "GIT_COMMITTER_EMAIL": "lab@example.invalid", "HOME": str(tmp_path), "PATH": os.environ.get("PATH", "")}
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, env=env)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "--allow-empty", "-m", "x"], check=True, env=env)
    head = E.git_head(tmp_path)
    after = E.engine_fingerprint(script)
    assert head and len(head) == 40 and after["commit"] == head and before["commit"] is None
    assert after["fingerprint"] != before["fingerprint"]  # the commit is part of the identity


def test_request_hash_is_engine_sensitive():
    base = E.request_hash({"s": 1}, [{"op": "reset"}], E.GOLDEN_CTX, "m")
    assert base == E.request_hash({"s": 1}, [{"op": "reset"}], E.GOLDEN_CTX, "m", "")
    assert base != E.request_hash({"s": 1}, [{"op": "reset"}], E.GOLDEN_CTX, "m", "fp-a")
    assert E.request_hash({"s": 1}, [{"op": "reset"}], E.GOLDEN_CTX, "m", "fp-a") != \
        E.request_hash({"s": 1}, [{"op": "reset"}], E.GOLDEN_CTX, "m", "fp-b")


@engine_test
def test_real_engine_carries_its_fingerprint_and_keys_the_cache_by_it(tmp_path):
    _engine_or_skip()
    eng = E.Engine(cache_dir=tmp_path)
    ident = E.engine_fingerprint(eng.path)
    assert eng.fingerprint == ident["fingerprint"] and eng.commit == ident["commit"]
    assert "src/sousChef/ops/headless.mjs" in ident["files"] and all(not f.endswith(".test.js") for f in ident["files"])
    # the engine imports behaviour from outside ops/ — the closure must reach it (refuter 2026-10-01: editing
    # src/data/pageSections.jsx left the old ops/*-only fingerprint unchanged)
    for f in ("src/data/pageSections.jsx", "src/components/basicEditor/constants.jsx", "src/utils/locationDetailSections.js",
              "src/lib/cdnUrl.js"):
        assert f in ident["files"], f
    assert any(p.startswith("@breadstack/ui@") for p in ident["packages"])
    assert eng.stats()["fingerprint"] == eng.fingerprint
    key = E.request_hash({}, [{"op": "x"}], E.GOLDEN_CTX, eng.manifest_hash, eng.fingerprint)
    eng._cache_put(key, {"ok": True, "results": [], "changes": []})
    assert (tmp_path / f"{key}.json").is_file()
    other = E.request_hash({}, [{"op": "x"}], E.GOLDEN_CTX, eng.manifest_hash, "another-engine")
    assert eng._cache_get(other) is None  # a diff cached under another engine is never reused
