"""retrain/carry.py: carrying an earlier round's accepted rows instead of re-asking the teacher (r6 plan §2b),
its use by collect_stage (the switch, the pass selection, the fallback) and r4dataset's re-render of a carried
row. A pure-Python engine, Fields and Prefix stand in for headless.mjs and the Go CLIs: no node, no Go, no model."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from souschef_lab import FIXTURES_DIR
from souschef_lab import fields as F
from souschef_lab import r4collect, r4dataset, render
from souschef_lab.engine import EngineResult
from souschef_lab.manifest import Manifest
from souschef_lab.retrain import carry as K
from souschef_lab.retrain import collect_stage as CS
from souschef_lab.retrain.common import open_store

MANIFEST = FIXTURES_DIR / "exam" / "manifest-real.json"
RULES = "You edit a cookie shop's site.\n"
INDEX_OLD = F.INDEX_HEAD + "s:ButtonsSection f=9\n"
TARGET = {"kind": "global", "panel": "ButtonsSection"}
RUN = "rt-20261020-0100-aaaaaa-bbbbbbbb"


# --- fakes ----------------------------------------------------------------------------------------------------------------

class FakeEngine:
    """headless.mjs in miniature: set_fields writes `globalSettings.<key>` (refused for `refuse` keys), every
    other op is accepted and writes nothing; the state keeps the writes under `w`."""

    fingerprint = "fp-test"

    def __init__(self, refuse=(), hidden=(), open_gate=()):
        self.refuse, self.hidden, self.open_gate = set(refuse), set(hidden), set(open_gate)

    def run(self, state, ops, ctx=None, *, print_state=False, use_cache=True):
        st = copy.deepcopy(state or {})
        results, diff = [], []
        for o in ops if isinstance(ops, list) else [ops]:
            if o.get("op") != "set_fields":
                results.append({"op": o.get("op"), "ok": True, "warnings": [], "written": []})
                continue
            vals = o.get("values") or {}
            bad = [k for k in vals if k in self.refuse]
            if bad:
                results.append({"op": "set_fields", "ok": False, "error": {"code": "E_FIELD", "message": f"unknown key {bad[0]}"}})
                continue
            written = []
            for k, v in vals.items():
                p = f"globalSettings.{k}"
                diff.append({"path": p, "before": (st.get("w") or {}).get(p), "after": v})
                st.setdefault("w", {})[p] = v
                written.append(p)
            warn = [f"W_HIDDEN:{k}" for k in vals if k in self.hidden | self.open_gate]
            # the engine warns on every gated write; `hidden` says whether the gate is open (byValue)
            hid = [{"key": k, "rule": "conditional_on", "by": "imagePosition",
                    "gate": {"key": "imagePosition", "value": "top"},
                    "byValue": "top" if k in self.open_gate else None} for k in vals if k in self.hidden | self.open_gate]
            results.append({"op": "set_fields", "ok": True, "warnings": warn, "written": written, "hidden": hid})
        return EngineResult(ok=all(r["ok"] for r in results), results=results,
                            changes=[{"slice": "globalSettings", "diff": diff}] if diff else [],
                            state=st if print_state else None)


class FakeFields:
    """fields.Fields: a block naming the turn's words (a follow-up adds the previous reply's first word)."""

    def __init__(self, tag="v1"):
        self.tag = tag
        self.primed = 0

    def prime(self, turns):
        self.primed += len(list(turns))

    def block(self, context, text, previous=None):
        extra = f" after:{previous[1].split()[0]}" if previous and previous[1] else ""
        return f"{F.OPEN_TAG}- {self.tag} borderRadius for '{render.go_trim_space(text)}'{extra}{F.CLOSE_TAG[:-1]}"

    def identity(self):
        return {"cli": "fake"}


class FakePrefix:
    def __init__(self, manifest, exe=None):
        self.manifest_hash = Manifest(json.loads(Path(manifest).read_text(encoding="utf-8"))).hash
        self.rules, self.index, self.tools = RULES, F.INDEX_HEAD + "s:ButtonsSection f=10 (new)\n", [
            {"name": "set_fields", "description": "new description", "input_schema": {"type": "object", "properties": {}}}]

    def identity(self):
        return {"cli": "fake-prefix", "manifest_hash": self.manifest_hash}


# --- material ---------------------------------------------------------------------------------------------------------------

def doc() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def manifest(d: dict | None = None) -> Manifest:
    return Manifest(d or doc())


def make_case(cid="r4-F1-0001", request="square-ish buttons, radius 4", value=4, family="F1", history=None, m=None):
    m = m or manifest()
    ops = [{"op": "set_fields", "target": TARGET, "values": {"borderRadius": value}}]
    st = {"template": {"id": "vintage"}}
    case = {"id": cid, "kind": "set_range", "target": TARGET, "fields": ["borderRadius"], "site_state": st,
            "context": {"page": "home"}, "request": request, "history": history or [], "holdout": False,
            "manifest_hash": m.hash, "engine_fingerprint": "fp-test", "template": "vintage",
            "expected": {"ops": ops, "diff": FakeEngine().run(st, ops, {}).diff()}, "score": {"mode": "exact"},
            "meta": {"family": family, "group": cid}}
    return case


def tool_msg(cid: str, name: str, item: dict) -> dict:
    return {"role": "tool", "tool_call_id": cid, "name": name, "content": render.tool_result_content(name, item)[0]}


def make_row(case: dict, fields=None, *, index=INDEX_OLD, rules=RULES, reply="Done: rounder corners at 4px.",
             lookup: dict | None = None, value=None) -> dict:
    """The built row an earlier round accepted for `case` (to_hf_messages' shape), its tool results the fake
    engine's, its blocks `fields`' on the source manifest."""
    fields = fields or FakeFields()
    msgs = [{"role": "system", "content": rules + index}]
    prev = None
    for k, h in enumerate(case.get("history") or []):
        msgs.append({"role": "user", "content": h["user"]})
        msgs.append({"role": "assistant", "content": h["assistant"]})
        prev = (h["user"], h["assistant"])
    ctx = case["context"]
    msgs.append({"role": "user", "content": render.context_block(ctx, case["request"], fields.block(ctx, case["request"], prev))})
    if lookup is not None:
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": "lk1", "type": "function", "function": {"name": "lookup_fields", "arguments": json.dumps({"query": "radius"})}}]})
        msgs.append({"role": "tool", "tool_call_id": "lk1", "name": "lookup_fields", "content": json.dumps(lookup)})
    v = case["expected"]["ops"][0]["values"]["borderRadius"] if value is None else value
    args = {"target": TARGET, "values": {"borderRadius": v}}
    msgs.append({"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "set_fields", "arguments": json.dumps(args)}}]})
    eng = FakeEngine().run(case["site_state"], [{"op": "set_fields", **args}], {}).to_dict()
    item = CS_tool_item("c1", eng)
    msgs.append(tool_msg("c1", "set_fields", item))
    msgs.append({"role": "assistant", "content": reply})
    return {"id": f"{case['id']}#pass1", "messages": msgs, "tools": [{"type": "function", "function": {"name": "set_fields"}}],
            "meta": {"case_id": case["id"], "family": case["meta"]["family"], "pass": "pass1", "model": "default",
                     "models": {"default": "claude-sonnet-5-5"}, "turn_id": "t-1", "at": "2026-10-11T00:00:00Z",
                     "history_assistant_turns": len(case.get("history") or [])}}


def CS_tool_item(cid: str, eng: dict) -> dict:
    from souschef_lab import collect as C
    return C.tool_results_for([{"id": cid, "name": "set_fields"}], eng)[0]


def source(cases_rows, name="model:r4", m=None, rejected=(), path: Path | None = None) -> dict:
    """load_source's shape for an in-memory source: [(case, row | None)]."""
    p = path or MANIFEST
    return {"name": name, "rows": {c["id"]: r for c, r in cases_rows if r is not None},
            "cases": {c["id"]: c for c, _ in cases_rows}, "manifest": p, "rejected": set(rejected)}


def write_manifest(tmp_path: Path, d: dict, name: str) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


def run_carry(tmp_path, cases, sources, *, m_new=None, engine=None, rules=RULES, fields_for=None, **kw):
    return K.carry(cases, sources, m_new=m_new or manifest(), engine=engine or FakeEngine(), extras={}, assets=[],
                   logs=tmp_path / "logs", rules=rules, fields_for=fields_for or (lambda p: FakeFields()),
                   progress=lambda m: None, workers=2, **kw)


# --- the switch -------------------------------------------------------------------------------------------------------------

def test_carry_is_on_by_default_in_full_mode_and_every_switch_turns_it_off():
    assert K.setting({}, "full", env={}) == (True, "default (full)")
    assert K.setting({}, "smoke", env={})[0] is False
    assert K.setting({}, "full", env={"COLLECT_CARRY": "off"}) == (False, "env COLLECT_CARRY")
    assert K.setting({"trigger": {"carry": "off"}}, "full", env={}) == (False, "trigger.carry")
    assert K.setting({"trigger": {"reask_all": True}}, "full", env={}) == (False, "trigger.reask_all")
    assert K.setting({"trigger": {"reask_all": True}}, "full", "on", env={}) == (True, "explicit")
    assert K.setting({}, "smoke", env={"COLLECT_CARRY": "on"})[0] is True
    assert K.excluded_ids({"trigger": {"carry_exclude": ["a", " b "]}}, env={"COLLECT_CARRY_EXCLUDE": "c,"}) == {"a", "b", "c"}
    assert K.hidden_strict({}, env={})  # on by default since the r6 review
    assert not K.hidden_strict({"trigger": {"carry_hidden": "keep"}}, env={})
    assert not K.hidden_strict({}, env={"COLLECT_CARRY_HIDDEN": "keep"})
    assert K.hidden_strict({"trigger": {"carry_hidden": "strict"}}, env={}) and K.hidden_strict({}, env={"COLLECT_CARRY_HIDDEN": "strict"})
    with pytest.raises(ValueError):
        K.hidden_strict({"trigger": {"carry_hidden": "stritc"}}, env={})


# --- sources ----------------------------------------------------------------------------------------------------------------

def test_sources_are_the_live_model_first_then_newer_parked_rounds_newest_first(tmp_path):
    s = open_store(tmp_path / "b")
    s.write_json("state/model.json", {"tag": "r4", "trained_at": "2026-10-11T03:00:00Z",
                                      "dataset": "gs://some-bucket/cases/pool/r4/dataset"})
    s.write_json("cases/pool/index.json", {"files": [{"path": "cases/pool/r4/cases-all.jsonl", "trained_in": "r4"},
                                                     {"path": "cases/pool/authored/x.jsonl", "trained_in": None}]})
    s.write_bytes("cases/pool/r4/dataset/dataset/train.jsonl", b"")
    for rid, at in (("rt-old", "2026-10-10T00:00:00Z"), ("rt-a", "2026-10-14T17:00:00Z"), ("rt-b", "2026-10-15T09:00:00Z"),
                    ("rt-nodata", "2026-10-16T00:00:00Z"), (RUN, "2026-10-17T00:00:00Z")):
        s.write_json(f"state/parked/c-{rid}.json", {"candidate_id": f"c-{rid}", "run_id": rid, "at": at})
        if rid != "rt-nodata":
            s.write_bytes(f"runs/{rid}/dataset/dataset/train.jsonl", b"")
            s.write_bytes(f"runs/{rid}/prepare/cases.jsonl", b"")
    src = K.find_sources(s, RUN)
    assert [x["name"] for x in src] == ["model:r4", "parked:c-rt-b", "parked:c-rt-a"]
    assert src[0]["dataset"] == "cases/pool/r4/dataset" and src[0]["cases"] == ["cases/pool/r4/cases-all.jsonl"]
    assert src[1]["dataset"] == "runs/rt-b/dataset" and src[1]["cases"] == ["runs/rt-b/prepare/cases.jsonl"]
    # an approved candidate names its own case file
    s.write_json("state/model.json", {"tag": "c-x", "trained_at": "2026-10-15T00:00:00Z", "dataset": "cases/trained/c-x/dataset",
                                      "training_cases": ["cases/trained/c-x/cases.jsonl"]})
    s.write_bytes("cases/trained/c-x/dataset/dataset/train.jsonl", b"")
    src = K.find_sources(s, RUN)
    assert [x["name"] for x in src] == ["model:c-x", "parked:c-rt-b"]
    assert src[0]["cases"] == ["cases/trained/c-x/cases.jsonl"]


def test_load_source_reads_only_the_rounds_cases_and_marks_rejects(tmp_path):
    s = open_store(tmp_path / "b")
    a, b, c = make_case("r4-F1-0001"), make_case("r4-F1-0002"), make_case("r4-F1-0003")
    s.write_jsonl("cases/pool/r4/cases-all.jsonl", [a, b, c])
    s.write_jsonl("ds/dataset/train.jsonl", [make_row(a)])
    s.write_jsonl(f"ds/dataset/{r4dataset.CAPPED_FILE}", [make_row(b)])
    s.write_jsonl("ds/dataset/rejects.jsonl", [{"case_id": "r4-F1-0003", "reasons": ["not_pass"]}])
    s.write_bytes("ds/manifest.json", MANIFEST.read_bytes())
    got = K.load_source(s, {"name": "model:r4", "dataset": "ds", "cases": ["cases/pool/r4/cases-all.jsonl"]}, tmp_path / "l",
                        {"r4-F1-0001", "r4-F1-0002", "r4-F1-0003"})
    assert set(got["rows"]) == {"r4-F1-0001", "r4-F1-0002"}  # a capped row is carriable too
    assert set(got["cases"]) == {"r4-F1-0001", "r4-F1-0002", "r4-F1-0003"} and got["rejected"] == {"r4-F1-0003"}
    assert not (tmp_path / "l" / "train.jsonl").exists()  # downloads deleted once read


# --- the checks ---------------------------------------------------------------------------------------------------------------

def test_an_unchanged_case_is_carried_and_logged_as_a_usable_collection_row(tmp_path):
    case = make_case()
    row = make_row(case)
    rep = run_carry(tmp_path, [case], [source([(case, row)])])
    assert rep["carried"] == 1 and rep["asked"] == 0 and rep["per_case"] == {case["id"]: "carried"}
    log = tmp_path / "logs" / K.CARRY_LOG
    logged = [json.loads(x) for x in log.read_text().splitlines()]
    assert logged[0]["carried"]["row"] == row and logged[0]["carried"]["source"] == "model:r4"
    assert logged[0]["score"]["usable"] and logged[0]["model"] == "default"
    assert r4collect.usable_cases([log]) == {case["id"]}  # what --prior / plan3 / status read
    assert r4collect.status(write_cases(tmp_path, [case]), [log])["total"]["usable"] == 1


def write_cases(tmp_path, cases) -> Path:
    p = tmp_path / "cases.jsonl"
    p.write_text("".join(json.dumps(c) + "\n" for c in cases), encoding="utf-8")
    return p


def test_a_changed_request_or_expectation_is_reasked(tmp_path):
    old = make_case()
    row = make_row(old)
    relabelled = dict(make_case(), request="square-ish buttons, corner radius 4")
    other_value = make_case(value=6)
    rep = run_carry(tmp_path, [relabelled], [source([(old, row)])])
    assert rep["reasons"] == {"case_changed": 1}
    rep = run_carry(tmp_path, [other_value], [source([(old, row)])])
    assert rep["reasons"] == {"case_changed": 1}


def test_a_written_fields_spec_change_is_reasked_but_provenance_is_not_a_change(tmp_path):
    case = make_case()
    row = make_row(case)
    moved = doc()
    moved["globalPanels"]["ButtonsSection"]["fields"]["borderRadius"]["source"]["line"] += 40
    relabelled = copy.deepcopy(moved)
    relabelled["globalPanels"]["ButtonsSection"]["fields"]["borderRadius"]["label"] = "Corner Radius"
    neighbour = doc()
    neighbour["globalPanels"]["ButtonsSection"]["fields"]["buttonColor"]["label"] = "Button Fill"
    old = write_manifest(tmp_path, doc(), "old.json")
    for d, want in ((moved, {"carried": 1}), (neighbour, {"carried": 1}), (relabelled, {"target_changed": 1})):
        src = source([(case, row)], path=old)
        rep = run_carry(tmp_path, [case], [src], m_new=Manifest(d))
        got = {"carried": 1} if rep["carried"] else rep["reasons"]
        assert got == want, (want, rep["reasons"], rep["examples"])


def test_a_lookup_result_on_a_changed_container_is_stale(tmp_path):
    case = make_case()
    old = write_manifest(tmp_path, doc(), "old.json")
    changed = doc()
    changed["globalPanels"]["ButtonsSection"]["fields"]["buttonColor"]["label"] = "Button Fill"
    lookup = {"container": "globalPanels.ButtonsSection", "kind": "global", "known": True,
              "fields": [{"key": "borderRadius", "type": "range"}], "total": 1}
    row = make_row(case, lookup=lookup)
    rep = run_carry(tmp_path, [case], [source([(case, row)], path=old)], m_new=Manifest(changed))
    assert rep["reasons"] == {"server_result_stale": 1}
    assert rep["examples"]["server_result_stale"][0]["containers"] == ["globalPanels.ButtonsSection"]
    rep = run_carry(tmp_path, [case], [source([(case, row)], path=old)], m_new=manifest())  # nothing changed: carried
    assert rep["carried"] == 1


def test_the_oracle_replay_rejects_a_row_the_engine_now_answers_otherwise(tmp_path):
    case = make_case()
    row = make_row(case)
    rep = run_carry(tmp_path, [case], [source([(case, row)])], engine=FakeEngine(refuse={"borderRadius"}))
    assert rep["reasons"] == {"tool_result_changed": 1}
    # a reply that does not do what the case expects is not usable on this round's case
    wrong = make_row(case, value=9)
    rep = run_carry(tmp_path, [case], [source([(case, wrong)])])
    assert rep["carried"] == 0 and list(rep["reasons"])[0].startswith("not_usable:")


def test_the_written_paths_order_is_not_a_tool_result_change():
    assert K.same_result('{"ok":true,"written":["a.y","a.x"]}', '{"ok":true,"written":["a.x","a.y"]}')
    assert not K.same_result('{"ok":true,"written":["a.x"]}', '{"code":"E_FIELD","message":"x"}')


def test_rules_and_block_parity_are_checked(tmp_path):
    case = make_case()
    row = make_row(case)
    rep = run_carry(tmp_path, [case], [source([(case, row)])], rules="Other rules.\n")
    assert rep["reasons"] == {"rules_changed": 1} and rep["sources"][0]["skipped"] == "rules_changed"
    rep = run_carry(tmp_path, [case], [source([(case, row)])], fields_for=lambda p: FakeFields("v2"))
    assert rep["reasons"] == {"block_parity": 1}


def test_hidden_writes_are_counted_and_refused_only_in_strict_mode(tmp_path):
    case = make_case()
    row = make_row(case)
    eng = FakeEngine(hidden={"borderRadius"})
    item = CS_tool_item("c1", eng.run(case["site_state"], case["expected"]["ops"], {}).to_dict())
    row["messages"][-2] = tool_msg("c1", "set_fields", item)  # collected on the same engine: the warning is in the result
    case["meta"]["measured"] = {"warnings": ["W_HIDDEN:borderRadius"]}  # r4score accepts what the case measured
    rep = run_carry(tmp_path, [case], [source([(case, row)])], engine=eng)
    assert rep["carried"] == 1 and rep["carried_hidden_write"] == 1
    rep = run_carry(tmp_path, [case], [source([(case, row)])], engine=eng, hidden_check=True)
    assert rep["reasons"] == {"hidden_write": 1}


def test_a_gated_write_with_its_gate_open_is_not_a_hidden_write(tmp_path):
    case = make_case()
    row = make_row(case)
    eng = FakeEngine(open_gate={"borderRadius"})
    item = CS_tool_item("c1", eng.run(case["site_state"], case["expected"]["ops"], {}).to_dict())
    row["messages"][-2] = tool_msg("c1", "set_fields", item)
    case["meta"]["measured"] = {"warnings": ["W_HIDDEN:borderRadius"]}
    rep = run_carry(tmp_path, [case], [source([(case, row)])], engine=eng, hidden_check=True)
    assert rep["carried"] == 1 and rep["carried_hidden_write"] == 0


def test_a_follow_up_is_replayed_turn_by_turn(tmp_path):
    from souschef_lab.state import context_of
    m = manifest()
    hist = [{"user": "the buttons look too round", "expect": "ask", "page": "home", "assistant": "How square do you want them?"}]
    case = make_case("r4-F2-0001", request="radius 4", family="F2", history=hist, m=m)
    case["context"] = context_of(case["site_state"], "home", {}, m, text="radius 4")
    row = make_row(case)
    rep = run_carry(tmp_path, [case], [source([(case, row)])], m_new=m)
    assert rep["carried"] == 1, rep["examples"]
    logged = json.loads((tmp_path / "logs" / K.CARRY_LOG).read_text().splitlines()[0])
    assert len(logged["history_rows"]) == 1 and logged["history_rows"][0]["text"] == hist[0]["user"]
    # the previous exchange is read back from the row exactly (block parity on a follow-up)
    t = r4dataset.hf_final_turn(row)
    assert t["previous"] == ("the buttons look too round", "How square do you want them?")
    # a history turn the teacher answered with a write where the case expects a question diverges
    bad = copy.deepcopy(row)
    bad["messages"][2] = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "h1", "type": "function", "function": {"name": "set_fields",
                                                       "arguments": json.dumps({"target": TARGET, "values": {"borderRadius": 2}})}}]}
    rep = run_carry(tmp_path, [case], [source([(case, bad)])], m_new=m)
    assert rep["carried"] == 0


def test_priority_fallback_exclusions_and_the_reask_reasons(tmp_path):
    a, b, c, d, e = (make_case(f"r4-F1-000{i}") for i in range(1, 6))
    new = make_case("aeX-F7-0001", family="F7")
    stale_a = dict(a, request="something else entirely")  # source 1 asked another question for a
    s1 = source([(stale_a, make_row(stale_a)), (b, make_row(b)), (c, None), (d, None)], rejected={c["id"]})
    s2 = source([(a, make_row(a)), (b, make_row(b, reply="Second source."))], name="parked:r5")
    e["meta"]["no_carry"] = True
    rep = run_carry(tmp_path, [a, b, c, d, e, new], [s1, s2], exclude={"r4-F1-0004"})
    pc = rep["per_case"]
    assert pc == {"r4-F1-0001": "carried", "r4-F1-0002": "carried", "r4-F1-0003": "no_row:rejected",
                  "r4-F1-0004": "excluded", "r4-F1-0005": "excluded", "aeX-F7-0001": "new"}
    rows = {json.loads(x)["case_id"]: json.loads(x) for x in (tmp_path / "logs" / K.CARRY_LOG).read_text().splitlines()}
    assert rows["r4-F1-0001"]["carried"]["source"] == "parked:r5"  # the live model's row no longer applied
    assert rows["r4-F1-0002"]["carried"]["source"] == "model:r4"  # the live model's row wins
    assert rep["reask_new"] == 1 and rep["reask_failing"] == 1
    assert rep["teacher_usd_saved_estimate"] == round(2 * K.USD_PER_CASE_DEFAULT, 2)


# --- r4dataset: the carried row re-rendered on the build's manifest -------------------------------------------------------------

def test_rerender_carried_replaces_the_prefix_and_the_block_and_keeps_every_answer():
    case = make_case()
    row = make_row(case)
    new, notes = r4dataset.rerender_carried(row, FakeFields("v2"), "RULES2" + F.INDEX_HEAD, [{"t": 1}])
    assert new["messages"][0]["content"] == "RULES2" + F.INDEX_HEAD and new["tools"] == [{"t": 1}]
    u = new["messages"][1]["content"]
    assert "v2 borderRadius" in u and u.endswith(case["request"]) and u.startswith("<context>\n")
    assert new["messages"][2:] == row["messages"][2:] and notes["block_changed"] and row["messages"][1]["content"] != u


def test_r4dataset_build_trains_a_carried_row_on_its_own_prefix_and_block(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "Prefix", FakePrefix)
    monkeypatch.setattr(F, "Fields", lambda manifest: FakeFields("v2"))
    cases = [make_case(f"r4-F1-00{i:02d}") for i in range(12)]
    rep = run_carry(tmp_path, cases, [source([(c, make_row(c)) for c in cases])])
    assert rep["carried"] == 12
    out = tmp_path / "data"
    st = r4dataset.build(write_cases(tmp_path, cases), [tmp_path / "logs" / K.CARRY_LOG], MANIFEST, out,
                         section_cap=0.0, family_max_share=0.0, progress=lambda m: None)
    assert st["rows"] == 12 and st["carried_rows"] == 12 and st["carried_by_source"] == {"model:r4": 12}
    assert st["block_drift_rows"] == 0 and st["render_sources"] == {"carried": 12}
    rows = [json.loads(x) for s in ("train", "val") for x in (out / "dataset" / f"{s}.jsonl").read_text().splitlines()]
    r = rows[0]
    assert r["messages"][0]["content"] == RULES + F.INDEX_HEAD + "s:ButtonsSection f=10 (new)\n"
    assert r["tools"][0]["function"]["description"] == "new description"
    assert "v2 borderRadius" in r["messages"][1]["content"] and r["meta"]["carried"]["from"] == "model:r4"
    assert r["meta"]["pass"] == "pass0-carry" and r["meta"]["model"] == "default" and r["meta"]["turn_id"] == "t-1"
    # the keep rules still apply to a carried row (here: over max_seq)
    st = r4dataset.build(write_cases(tmp_path, cases), [tmp_path / "logs" / K.CARRY_LOG], MANIFEST, tmp_path / "long",
                         section_cap=0.0, family_max_share=0.0, progress=lambda m: None, count=lambda hf: 10**6)
    assert st["rows"] == 0 and st["rejects_first_reason"] == {"over_length": 12}


def test_capped_rows_are_written(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "Prefix", FakePrefix)
    monkeypatch.setattr(F, "Fields", lambda manifest: FakeFields("v2"))
    cases = [make_case(f"r4-F1-00{i:02d}") for i in range(10)] + [make_case(f"r4-F3-00{i:02d}", family="F3") for i in range(2)]
    run_carry(tmp_path, cases, [source([(c, make_row(c)) for c in cases])])
    st = r4dataset.build(write_cases(tmp_path, cases), [tmp_path / "logs" / K.CARRY_LOG], MANIFEST, tmp_path / "d",
                         section_cap=0.0, family_max_share=0.40, progress=lambda m: None)
    capped = (tmp_path / "d" / "dataset" / r4dataset.CAPPED_FILE).read_text().splitlines()
    assert st["capped_family"] and len(capped) == sum(st["capped_family"].values()) and st["rows"] + len(capped) == 12


# --- collect_stage: the step and the passes -----------------------------------------------------------------------------------

class Runner:
    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, argv):
        self.calls.append(argv)
        return 0


def seeded(tmp_path, n=3):
    s = open_store(tmp_path / "bucket")
    cases = [make_case(f"r4-F1-000{i}") for i in range(1, n + 1)]
    s.write_jsonl(f"runs/{RUN}/prepare/cases.jsonl", cases)
    s.write_bytes(f"runs/{RUN}/prepare/manifest.json", MANIFEST.read_bytes())
    s.write_json("state/model.json", {"tag": "r4", "trained_at": "2026-10-11T03:00:00Z", "dataset": "cases/pool/r4/dataset",
                                      "training_manifest": cases[0]["manifest_hash"]})
    s.write_json("cases/pool/index.json", {"files": [{"path": "cases/pool/r4/cases-all.jsonl", "trained_in": "r4"}]})
    s.write_jsonl("cases/pool/r4/cases-all.jsonl", cases)
    s.write_jsonl("cases/pool/r4/dataset/dataset/train.jsonl", [make_row(c) for c in cases[:2]])  # the third: no row
    s.write_bytes("cases/pool/r4/dataset/manifest.json", MANIFEST.read_bytes())
    return s, cases


@pytest.fixture
def ok_render(monkeypatch):
    def fake(cases, logs, manifest):
        ids = [x for x in (logs[0].parent / "smoke.ids").read_text().split() if x]
        return {"rows": len(ids), "ok_rows": len(ids), "rendered": len(ids), "render_sources": {"conversation": len(ids)},
                "block_drift_rows": 0, "errors": {}, "by_case": [{"case_id": i} for i in ids]}
    monkeypatch.setattr(r4dataset, "render_check", fake)


def version(url):
    return {"build_id": "b1", "route_info": {"default": {"fields_context": True}, "opus": {"fields_context": True}}}


TOOLS = {"prefix_rules": RULES, "fields_for": lambda p: FakeFields(), "engine": FakeEngine()}


def test_collect_asks_only_what_was_not_carried(tmp_path, monkeypatch, ok_render):
    s, cases = seeded(tmp_path)
    monkeypatch.setenv("SOUS_CHEF_KEY", "k")
    monkeypatch.delenv("COLLECT_CARRY", raising=False)
    plan3 = {}
    monkeypatch.setattr("souschef_lab.r4collect.main", lambda argv: plan3.setdefault("argv", argv) and 0)
    runner = Runner()
    res = CS.run(s, RUN, mode="full", attempt=1, url="https://int.example", workdir=tmp_path / "w", expect_build="b1",
                 runner=runner, version=version, progress=lambda m: None, sync_s=3600, carry_tools=TOOLS)
    m = res["metrics"]
    # the third case is in r4's pool but r4 kept no row for it (capped before capped.jsonl, say): asked
    assert m["carry"]["on"] and m["carry"]["carried"] == 2 and m["carry"]["reasons"] == {"no_row:not_kept": 1}
    assert m["carried"] == 2 and m["reasked"] == 1 and m["new_cases"] == 0
    assert "pass0-carry.jsonl" in res["logs"]  # the dataset stage builds from it like any pass
    by = {a[a.index("--label") + 1]: a for a in runner.calls}
    p1 = by["pass1"]
    assert p1[p1.index("--select") + 1] == "unusable" and p1[p1.index("--prior") + 1].endswith("a1/pass0-carry.jsonl")
    p2 = by["pass2"]
    priors = [p2[i + 1] for i, x in enumerate(p2) if x == "--prior"]
    assert priors[0].endswith("pass1.jsonl") and priors[1].endswith("pass0-carry.jsonl")
    assert any(x.endswith("pass0-carry.jsonl") for x in plan3["argv"])
    assert s.exists(f"runs/{RUN}/collect/a1/pass0-carry.jsonl") and s.exists(f"runs/{RUN}/collect/a1/{K.REPORT}")
    # what r4collect run would select for pass 1: the case the carry left
    sel = {c["id"] for c in r4collect.case_index(tmp_path / "w" / "cases.jsonl")} - r4collect.usable_cases(
        [tmp_path / "w" / "a1" / "pass0-carry.jsonl"])
    assert sel == {"r4-F1-0003"}


def test_reask_all_and_smoke_skip_the_carry(tmp_path, monkeypatch, ok_render):
    s, _ = seeded(tmp_path)
    monkeypatch.setenv("SOUS_CHEF_KEY", "k")
    monkeypatch.setattr("souschef_lab.r4collect.main", lambda argv: 0)
    run = s.read_json(f"runs/{RUN}/run.json") or {}
    s.write_json(f"runs/{RUN}/run.json", {**run, "trigger": {"reason": "manual", "reask_all": True}})
    runner = Runner()
    res = CS.run(s, RUN, mode="full", attempt=1, url="https://int.example", workdir=tmp_path / "w", expect_build="b1",
                 runner=runner, version=version, progress=lambda m: None, sync_s=3600, carry_tools=TOOLS)
    assert res["metrics"]["carry"] == {"on": False, "why": "trigger.reask_all"} and res["metrics"]["carried"] == 0
    p1 = next(a for a in runner.calls if a[a.index("--label") + 1] == "pass1")
    assert "--prior" not in p1 and "pass0-carry.jsonl" not in res["logs"]
    runner = Runner()
    res = CS.run(s, RUN, mode="smoke", attempt=2, url="https://int.example", workdir=tmp_path / "w2", expect_build="b1",
                 runner=runner, version=version, progress=lambda m: None, sync_s=3600, carry_tools=TOOLS, carry=None)
    assert res["metrics"]["carry"]["on"] is False


def test_a_carry_that_fails_falls_back_to_asking_everything(tmp_path, monkeypatch, ok_render):
    s, _ = seeded(tmp_path)
    monkeypatch.setenv("SOUS_CHEF_KEY", "k")
    monkeypatch.setattr("souschef_lab.r4collect.main", lambda argv: 0)

    def boom(*a, **k):
        (tmp_path / "w" / "a1" / K.CARRY_LOG).write_text("{}\n")
        raise RuntimeError("engine died")
    runner = Runner()
    res = CS.run(s, RUN, mode="full", attempt=1, url="https://int.example", workdir=tmp_path / "w", expect_build="b1",
                 runner=runner, version=version, progress=lambda m: None, sync_s=3600, carry_tools=TOOLS, carry_fn=boom)
    assert res["status"] == "ok" and res["metrics"]["carry"]["on"] is False
    assert "engine died" in res["metrics"]["carry"]["why"]
    assert not (tmp_path / "w" / "a1" / K.CARRY_LOG).exists()
    p1 = next(a for a in runner.calls if a[a.index("--label") + 1] == "pass1")
    assert "--prior" not in p1


def test_a_required_carry_that_fails_stops_the_stage_before_any_teacher_call(tmp_path, monkeypatch, ok_render):
    from souschef_lab.retrain.common import ConfigError, StageError
    s, _ = seeded(tmp_path)
    monkeypatch.setenv("SOUS_CHEF_KEY", "k")
    monkeypatch.setattr("souschef_lab.r4collect.main", lambda argv: 0)
    assert K.setting({"trigger": {"carry": "required"}}, "full", env={}) == (True, "trigger.carry")
    assert K.required({"trigger": {"carry": "required"}}, env={}) and not K.required({}, env={})
    assert K.required({}, env={"COLLECT_CARRY": "required"})

    def boom(*a, **k):
        raise RuntimeError("engine died")
    run = s.read_json(f"runs/{RUN}/run.json") or {}
    s.write_json(f"runs/{RUN}/run.json", {**run, "trigger": {"reason": "manual", "carry": "required"}})
    runner = Runner()
    with pytest.raises(StageError, match="engine died"):
        CS.run(s, RUN, mode="full", attempt=1, url="https://int.example", workdir=tmp_path / "w", expect_build="b1",
               runner=runner, version=version, progress=lambda m: None, sync_s=3600, carry_tools=TOOLS, carry_fn=boom)
    assert not [a for a in runner.calls if "--label" in a]   # no pass ran
    # a typo in carry_hidden stops the stage too (it would otherwise ask everything silently)
    s.write_json(f"runs/{RUN}/run.json", {**run, "trigger": {"reason": "manual", "carry_hidden": "stritc"}})
    with pytest.raises(ConfigError, match="carry_hidden"):
        CS.run(s, RUN, mode="full", attempt=2, url="https://int.example", workdir=tmp_path / "w2", expect_build="b1",
               runner=Runner(), version=version, progress=lambda m: None, sync_s=3600, carry_tools=TOOLS, carry_fn=boom)


def test_a_retried_collect_reuses_its_carry(tmp_path, monkeypatch, ok_render):
    s, _ = seeded(tmp_path)
    monkeypatch.setenv("SOUS_CHEF_KEY", "k")
    monkeypatch.setattr("souschef_lab.r4collect.main", lambda argv: 0)
    CS.run(s, RUN, mode="full", attempt=1, url="https://int.example", workdir=tmp_path / "w", expect_build="b1",
           runner=Runner(), version=version, progress=lambda m: None, sync_s=3600, carry_tools=TOOLS)
    calls = []
    res = CS.run(s, RUN, mode="full", attempt=1, url="https://int.example", workdir=tmp_path / "w-retry", expect_build="b1",
                 runner=Runner(), version=version, progress=lambda m: None, sync_s=3600, carry_tools=TOOLS,
                 carry_fn=lambda *a, **k: calls.append(1))
    assert not calls and res["metrics"]["carry"]["reused"] and res["metrics"]["carry"]["carried"] == 2
