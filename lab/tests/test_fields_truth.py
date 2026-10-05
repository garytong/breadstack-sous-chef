"""The candidate-fields block follows the SERVICE, and the rows are checked
(docs/LAB.md §5):

* collect / eval read the route's `fields_context` from `GET /version`
  (`route_info: {name: {kind, fields_context, fallback}}`; a list of
  `{name, …}` is read too) and every `turn.done` (`route.fields_context`,
  or a top-level `fields_context`): `auto` follows it, `on`/`off` are refused
  when it says otherwise, a contradicting turn stops the run;
* pairs refuse samples collected partly with the block and partly without,
  dpo checks the pairs against the SFT adapter it continues;
* `fields rerender` refuses a manifest that is not the rows';
* the user turn's text is trimmed with Go's whitespace set.

All without go: the block is a stub (test_fields.StubFields).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
for _p in (LAB, LAB / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fake_service as FS  # noqa: E402
from souschef_lab import collect as C  # noqa: E402
from souschef_lab import dpo  # noqa: E402
from souschef_lab import eval as E  # noqa: E402
from souschef_lab import fields as F  # noqa: E402
from souschef_lab import pairs as P  # noqa: E402
from souschef_lab import render as R  # noqa: E402
from souschef_lab import serving  # noqa: E402
from souschef_lab import verify as V  # noqa: E402
from test_collect import TINY, TINY_HASH, USAGE, RecordingVerifier, done, fake_engine, make_case  # noqa: E402
from test_fields import BLOCK, StubFields  # noqa: E402
from test_pairs import FIX, PASS, RECORDED, WRONG, relabel, sample  # noqa: E402
from test_train_dry import a_pair, write_jsonl  # noqa: E402

STUDENT = {"name": "stu", "kind": "messages-compatible", "fields_context": True, "fallback": "default",
           "report_as": "blueberry-muffin"}  # the route's name is free; report_as is what its calls are reported as


@pytest.fixture
def svc():
    with FS.FakeService(manifest=TINY, key="k") as s:
        s.routes.append(dict(STUDENT))
        yield s


@pytest.fixture
def client(svc):
    return C.Client(svc.url, "k", timeout_s=10)


@pytest.fixture
def stub(monkeypatch):
    """collect.fields_for without go: `on` gives the stub."""
    made = []

    def fields_for(flag, manifest):
        if flag != "on":
            return None
        made.append(StubFields())
        return made[-1]

    monkeypatch.setattr(C, "fields_for", fields_for)
    return made


# --- what the service says ---------------------------------------------------------------------------------------------------

def test_version_states_each_route(svc, client):
    v = client.version()
    assert v["routes"] == ["default", "fake", "stu"] and v["default_route"] == "default"  # the name list stays
    assert v["route_info"]["stu"] == {k: x for k, x in STUDENT.items() if k != "name"}
    assert C.route_details(v)["stu"] == STUDENT
    assert C.route_info(v, "stu")["fields_context"] is True and C.route_info(v, "fake")["fields_context"] is False
    assert C.route_info(v, None)["name"] == "default" and C.route_info(v, "ghost") is None


def test_route_details_reads_the_shapes_a_service_may_answer_with():
    assert C.route_details({"routes": ["a", "b"]}) == {} and C.route_details(None) == {} and C.route_details({}) == {}
    d = [{"name": "a", "fields_context": "on"}]
    for key in C.ROUTE_DETAIL_KEYS + ("routes",):
        assert C.route_details({key: d}) == {"a": d[0]}  # the list form: [{name, kind, fields_context, fallback}]
    assert C.route_details({"route_info": {"a": {"kind": "anthropic", "fields_context": False, "fallback": None}}}) == \
        {"a": {"name": "a", "kind": "anthropic", "fields_context": False, "fallback": None}}  # the service's map
    assert C.route_details({"routes": ["a"], "routes_detail": {"a": {"kind": "anthropic"}}}) == \
        {"a": {"name": "a", "kind": "anthropic"}}
    two = {"routes_detail": [{"name": "x"}, {"name": "y"}]}
    assert C.route_info(two, None) is None  # two routes, neither called default: not guessed
    assert C.route_info({**two, "default_route": "y"}, None) == {"name": "y"}
    assert C.route_info({"routes_detail": [{"name": "only"}]}, None) == {"name": "only"}


def test_on_off_and_turn_done():
    assert [C.on_off(v) for v in (True, False, "on", "off", " ON ", None, "", 1, "maybe")] == \
        [True, False, True, False, True, None, None, None, None]
    assert C.done_fields_context({}) is None and C.done_fields_context({"fields_context": True}) is True
    assert C.done_fields_context({"fields_context": "off"}) is False
    assert C.done_fields_context({"route": {"name": "stu", "fields_context": True, "fallback": "default"}}) is True
    assert C.done_fields_context({"route": {"name": "d", "fields_context": False, "fallback": None}}) is False
    assert C.done_fields_context({"route": {"name": "d"}}) is None and C.done_fields_context({"route": "d"}) is None
    per_route = {"fields_context": {"stu": True, "default": False}, "routes": ["stu", "default"]}
    assert C.done_fields_context(per_route, "stu") is True  # the route the turn STARTED on rendered the user turn
    assert C.done_fields_context(per_route) is True and C.done_fields_context({"fields_context": {"x": False}}) is False


# --- the collector follows it ---------------------------------------------------------------------------------------------------

def test_auto_follows_version(svc, client, stub, tmp_path):
    fields, rec = C.resolve_fields("auto", tmp_path / "m.json", client, "stu")
    assert fields is stub[0] and rec == {"fields_context": True, "fields_context_by": "service", "route_info": STUDENT}
    fields, rec = C.resolve_fields("auto", tmp_path / "m.json", client, "fake")
    assert fields is None and rec["fields_context"] is False and rec["fields_context_by"] == "service"
    fields, rec = C.resolve_fields("auto", tmp_path / "m.json", client, None)  # the service's default route
    assert fields is None and rec["route_info"]["name"] == "default"


def test_a_flag_that_contradicts_the_service_is_refused(svc, client, stub, tmp_path):
    with pytest.raises(C.FieldsContextMismatch, match="says fields_context is on for stu"):
        C.resolve_fields("off", tmp_path / "m.json", client, "stu")  # the refuter's case: the student route, flag off
    with pytest.raises(C.FieldsContextMismatch, match="says fields_context is off for fake"):
        C.resolve_fields("on", tmp_path / "m.json", client, "fake")
    with pytest.raises(C.FieldsContextMismatch, match="ghost is not one of them"):
        C.resolve_fields("auto", tmp_path / "m.json", client, "ghost")
    assert C.resolve_fields("on", tmp_path / "m.json", client, "stu")[0] is stub[-1]  # agreeing is fine
    assert C.resolve_fields("off", tmp_path / "m.json", client, "fake")[0] is None
    assert not issubclass(C.FieldsContextMismatch, C.ServiceError) and issubclass(C.FieldsContextMismatch, RuntimeError)


def test_a_service_that_states_nothing_cannot_answer_auto(svc, client, stub, tmp_path):
    svc.routes = None  # a build before the statement: /version lists names only
    assert client.version() == {"build_id": "fake", "go": "none", "routes": ["default", "fake"]}
    with pytest.raises(C.FieldsContextMismatch, match="does not state fields_context for stu"):
        C.resolve_fields("auto", tmp_path / "m.json", client, "stu")
    fields, rec = C.resolve_fields("on", tmp_path / "m.json", client, "stu")  # said by hand, taken on trust
    assert fields is stub[0] and rec == {"fields_context": True, "fields_context_by": "flag", "route_info": None}
    assert C.resolve_fields("off", tmp_path / "m.json", client, "stu")[1]["fields_context_by"] == "flag"
    with pytest.raises(ValueError):
        C.resolve_fields("maybe", tmp_path / "m.json", client, "stu")


def test_turn_done_has_the_last_word(svc, client):
    svc.scripts["c1"] = [done()]
    row = C.run_case(make_case(), client, fake_engine, route="stu", fields=StubFields())
    assert row["fields_context"] is True and row["fields_context_by"] == "service" and row["candidate_fields"] == BLOCK
    assert row["events"][-1]["data"]["route"] == {"name": "stu", "fields_context": True, "fallback": "default",
                                                  "model": "blueberry-muffin"}  # as the service
    assert row["models"] == {"stu": "blueberry-muffin"} and row["model"] == "stu"  # the reported id, beside the route asked
    plain = C.run_case(make_case(), client, fake_engine, route="fake")
    assert "models" not in plain  # a build (or route entry) that states no reported model
    assert plain["fields_context"] is False and plain["fields_context_by"] == "service" and "candidate_fields" not in plain
    # the mismatch the review found: the student's route sends the block, the collector was not rendering it
    with pytest.raises(C.FieldsContextMismatch, match="rendered the user turn without the candidate-fields block"):
        C.run_case(make_case(), client, fake_engine, route="stu")
    with pytest.raises(C.FieldsContextMismatch, match="rendered the user turn with the candidate-fields block"):
        C.run_case(make_case(), client, fake_engine, route="fake", fields=StubFields())
    # a route the service states nothing about (or an older build): the flag is all there is
    unstated = C.run_case(make_case(), client, fake_engine, route="blueberry-muffin", fields=StubFields())
    assert unstated["fields_context"] is True and unstated["fields_context_by"] == "flag"
    # a per-route statement in turn.done (hybrid): the requested route's
    svc.scripts["c1"] = [[("turn.done", {"usage": USAGE, "stop": "end_turn", "routes": ["stu", "default"],
                                         "fields_context": {"stu": True, "default": False}})]]
    assert C.run_case(make_case(), client, fake_engine, route="stu", fields=StubFields())["fields_context_by"] == "service"
    svc.scripts["c1"] = [[("turn.done", {"usage": USAGE, "stop": "end_turn", "fields_context": True})]]  # the bare bool
    assert C.run_case(make_case(), client, fake_engine, route="stu", fields=StubFields())["fields_context_by"] == "service"
    with pytest.raises(C.FieldsContextMismatch):
        C.run_case(make_case(), client, fake_engine, route="stu")
    # an errored turn has no turn.done: nothing to check, the row says who decided
    svc.scripts["c1"] = [[("turn.error", {"code": "model_error", "message": "x"})]]
    err = C.run_case(make_case(), client, fake_engine, route="stu")
    assert err["status"] == "error" and err["fields_context_by"] == "flag"


def test_collect_stops_before_logging_a_row_the_service_contradicts(svc, client, tmp_path):
    log = tmp_path / "log.jsonl"
    with pytest.raises(RuntimeError, match="FieldsContextMismatch"):
        C.collect([make_case("a"), make_case("b")], client, fake_engine, log, route="stu", verifier=RecordingVerifier())
    assert C.read_jsonl(log) == []


def _files(tmp_path, *cases):
    cases_file = tmp_path / "cases.jsonl"
    cases_file.write_text("".join(json.dumps(c) + "\n" for c in cases), encoding="utf-8")
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps(TINY), encoding="utf-8")
    return cases_file, manifest


def test_collect_main_follows_the_service(svc, stub, tmp_path, monkeypatch, capsys):
    """`make collect ROUTE=<student>` with no flag: the rows carry the block."""
    cases, manifest = _files(tmp_path, make_case("a"))
    monkeypatch.setattr(C, "default_engine", lambda *a, **k: fake_engine)
    monkeypatch.setattr(C, "default_verifier", lambda *a, **k: RecordingVerifier("pass"))
    log = tmp_path / "log.jsonl"
    base = ["--cases", str(cases), "--log", str(log), "--url", svc.url, "--key", "k", "--manifest", str(manifest),
            "--assets", str(tmp_path / "none.json"), "--route", "stu"]
    assert C.build_parser().parse_args(base).fields_context == "auto"
    assert C.main(base + ["--fields-context", "off"]) == 1  # the flag the Makefile used to pass
    err = capsys.readouterr().err
    assert "cannot start" in err and "/version says fields_context is on for stu" in err and not log.exists()
    assert C.main(base) == 0
    assert "fields_context on for stu (says the service)" in capsys.readouterr().out
    (row,) = C.read_jsonl(log)
    assert row["candidate_fields"] == BLOCK and row["fields_context"] is True and row["fields_context_by"] == "service"
    assert row["messages"][0]["content"][0]["text"] == R.context_block(row["context"], row["text"], BLOCK)
    (run,) = C.read_jsonl(C.runs_log(log))
    assert run["fields_context"] is True and run["fields_context_by"] == "service" and run["route_info"] == STUDENT
    svc.routes = None
    assert C.main(base[:3] + [str(tmp_path / "old.jsonl")] + base[4:]) == 1
    assert "does not state fields_context for stu" in capsys.readouterr().err


def test_eval_main_follows_the_service(svc, stub, tmp_path, monkeypatch, capsys):
    cases, manifest = _files(tmp_path, make_case("h", holdout=True, kind="clarify"))
    monkeypatch.setattr(C, "default_engine", lambda *a, **k: fake_engine)
    monkeypatch.setattr(C, "default_verifier", lambda *a, **k: RecordingVerifier("clarify_ok"))
    base = ["--cases", str(cases), "--url", svc.url, "--key", "k", "--manifest", str(manifest), "--out-dir",
            str(tmp_path / "runs"), "--assets", str(tmp_path / "none.json"), "--route", "stu"]
    assert E.main(base + ["--fields-context", "off"]) == 1 and not (tmp_path / "runs").exists()
    assert "cannot start" in capsys.readouterr().err
    assert E.main(base) == 0
    report = E.load_run(next((tmp_path / "runs").glob("eval-stu-*.json")))
    assert report["fields_context"] is True and report["fields_context_by"] == "service" and report["route_info"] == STUDENT
    assert report["results"][0]["candidate_fields"] == BLOCK
    # a rescore reads what the rows say
    assert E.build_report([{"fields_context": True}], label="l", route="r", manifest_hash="m", started="t")["fields_context"]
    assert not E.build_report([{"fields_context": False}], label="l", route="r", manifest_hash="m", started="t")["fields_context"]


def test_eval_stops_on_a_contradicting_turn(svc, stub, tmp_path, monkeypatch, capsys):
    """/version and turn.done disagree (the route table changed under the run): nothing is saved."""
    cases, manifest = _files(tmp_path, make_case("h", holdout=True, kind="clarify"))
    svc.scripts["h"] = [[("turn.done", {"usage": USAGE, "stop": "end_turn", "fields_context": False})]]
    monkeypatch.setattr(C, "default_engine", lambda *a, **k: fake_engine)
    monkeypatch.setattr(C, "default_verifier", lambda *a, **k: RecordingVerifier("clarify_ok"))
    rc = E.main(["--cases", str(cases), "--url", svc.url, "--key", "k", "--manifest", str(manifest), "--out-dir",
                 str(tmp_path / "runs"), "--assets", str(tmp_path / "none.json"), "--route", "stu"])
    assert rc == 1 and "nothing saved" in capsys.readouterr().err and not list((tmp_path / "runs").glob("eval-*.json"))


def test_latest_rows_counts_a_retried_error_once():
    """collect's hybrid line reads the whole log: an errored sample that a
    later start asked again is one sample, the retry."""
    esc = {"from": "stu", "to": "default", "reason": "model_error"}
    rows = [{"id": "a#1", "status": "error", "escalated": esc}, {"id": "b#1", "status": "ok", "verdict": "pass"},
            {"id": "a#1", "status": "ok", "verdict": "pass", "routes": ["stu"]}, {"note": "no id"}]
    kept = C.latest_rows(rows)
    assert kept == [rows[3], rows[1], rows[2]]
    assert C.hybrid_summary(rows)["escalated"] == 1 and C.hybrid_summary(kept)["escalated"] == 0


# --- pairs and dpo check the rows -------------------------------------------------------------------------------------------------

def _pair_inputs(tmp_path, rows):
    cases = []
    for p in FIX.glob("*.json"):
        f = V.load_fixture(p)
        f["case"]["holdout"] = False
        cases.append(f["case"])
    cases_file = tmp_path / "cases.jsonl"
    cases_file.write_text("".join(json.dumps(c) + "\n" for c in cases), encoding="utf-8")
    log = tmp_path / "log.jsonl"
    log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return ["--log", str(log), "--cases", str(cases_file), "--out", str(tmp_path / "pairs"), "--recorded", str(RECORDED)]


def test_logged_fields_context_counts_the_collector_s_record():
    rows = [{"fields_context": True}, {"fields_context": False, "candidate_fields": ""}, {"candidate_fields": ""}, {}]
    assert P.logged_fields_context(rows) == {"with": 2, "without": 2}  # an old row: the key's presence is the record


def test_pairs_refuse_samples_collected_two_ways(tmp_path, capsys):
    good, bad = sample(PASS, 1), relabel(sample(WRONG), PASS, 2)
    args = _pair_inputs(tmp_path, [dict(good, candidate_fields=BLOCK, fields_context=True), dict(bad, fields_context=False)])
    assert P.main(args) == 2
    assert "collected with the candidate-fields" in capsys.readouterr().err and not (tmp_path / "pairs").exists()
    assert P.main(args + ["--fields-context", "off"]) in (0, 1)  # said by hand: every prompt without the block
    man = json.loads((tmp_path / "pairs" / "pairs-manifest.json").read_text())
    assert man["fields_context"]["mode"] == "off" and man["fields_context"]["pairs_with_block"] == 0


def test_pairs_record_how_many_prompts_carry_the_block(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(P, "length_guard", lambda pairs, share=0.5: (pairs, 0))  # one pair: the guard would drop it
    good, bad = sample(PASS, 1), relabel(sample(WRONG), PASS, 2)
    rows = [dict(good, candidate_fields=BLOCK, fields_context=True), dict(bad, candidate_fields=BLOCK, fields_context=True)]
    assert P.main(_pair_inputs(tmp_path, rows) + ["--max-per-prompt", "2"]) == 0
    out = tmp_path / "pairs"
    built = [json.loads(x) for f in ("train.jsonl", "val.jsonl") for x in (out / f).read_text().splitlines()]
    assert built and all(BLOCK in r["prompt"][-1]["content"] for r in built)
    fc = json.loads((out / "pairs-manifest.json").read_text())["fields_context"]
    assert fc == {"mode": "as-logged", "enabled": True, "pairs": len(built), "pairs_with_block": len(built)}
    assert json.loads((out / "stats.json").read_text())["fields_context"] == fc
    assert f"{len(built)} of {len(built)} pairs carry" in capsys.readouterr().out
    assert P.pair_block_stats(built) == {"rows": len(built), "with_block": len(built)}


def block_pair(pid: str) -> dict:
    row = a_pair(pid)
    user = next(m for m in row["prompt"] if m["role"] == "user")
    user["content"] = R.context_block({"page": "home"}, "Change the hero heading", BLOCK)
    return row


def _sft(tmp_path, fields_context=None) -> Path:
    adapter = tmp_path / "adapters" / "sft" / "adapter"
    adapter.mkdir(parents=True)
    run = {"kind": "sft", "name": "sft", "base": "b", "rank": 16, "alpha": 16, "max_seq_length": 32768}
    if fields_context is not None:
        run["fields_context"] = {"enabled": fields_context, "rows": 5, "rows_with_block": 5 if fields_context else 0}
    (adapter.parent / "run.json").write_text(json.dumps(run), encoding="utf-8")
    return adapter


def _pairs(tmp_path, name, make) -> Path:
    d = tmp_path / name
    write_jsonl(d / "train.jsonl", [make(f"c{i}#1-2") for i in range(3)])
    write_jsonl(d / "val.jsonl", [make("v0#1-2")])
    return d


def test_dpo_checks_the_pairs_against_the_sft_adapter(tmp_path, capsys):
    """The review's hazard: DPO pairs without the block for a student that is
    served (and was SFT-trained) with it."""
    plain, blocks = _pairs(tmp_path, "plain", a_pair), _pairs(tmp_path, "blocks", block_pair)
    with_fc, old = _sft(tmp_path / "a", True), _sft(tmp_path / "b")

    def run(pairs, sft, *extra):
        return dpo.main(["--pairs", str(pairs), "--sft-adapter", str(sft), "--name", "d", "--dry-run", *extra])

    def refused(pairs, sft, *extra) -> str:
        capsys.readouterr()
        with pytest.raises(SystemExit):
            run(pairs, sft, *extra)
        return capsys.readouterr().err

    assert dpo.parse_args(["--sft-adapter", "a", "--name", "n"]).fields_context == "auto"
    err = refused(plain, with_fc)
    assert "no pair in" in err and "trained with the <candidate_fields> block" in err
    assert run(blocks, with_fc) == 0
    assert "pairs_with_block" in capsys.readouterr().out  # the plan (and so run.json) says it
    assert run(plain, old) == 0  # an adapter from before the block, block-less pairs: the old form, consistently
    err = refused(blocks, old)
    assert "4 of 4 pairs" in err and "--fields-context on" in err
    assert run(blocks, old, "--fields-context", "on") == 0  # its run.json predates the record: said by hand
    assert "drop the flag" in refused(plain, with_fc, "--fields-context", "off")  # a flag cannot overrule the adapter
    assert "no pair in" in refused(plain, old, "--fields-context", "on")
    args = dpo.parse_args(["--pairs", str(blocks), "--sft-adapter", str(with_fc), "--name", "d"])
    p = dpo.plan(args, dpo.sft_run(with_fc), dpo.read_jsonl(blocks / "train.jsonl"), dpo.read_jsonl(blocks / "val.jsonl"))
    assert p["fields_context"] == {"enabled": True, "by": "sft_run", "sft_enabled": True, "pairs": 4, "pairs_with_block": 4,
                                   "pairs_manifest": None}
    (blocks / "pairs-manifest.json").write_text(json.dumps({"fields_context": {"mode": "as-logged"}}), encoding="utf-8")
    assert dpo.plan(args, dpo.sft_run(with_fc), [], [])["fields_context"]["pairs_manifest"] == {"mode": "as-logged"}


# --- `fields rerender` refuses another manifest -----------------------------------------------------------------------------------

class ManifestStub(StubFields):
    def __init__(self, manifest: Path):
        super().__init__()
        self.manifest = manifest


def _dataset(tmp_path, hashes=(TINY_HASH, TINY_HASH), declared=None) -> Path:
    data = tmp_path / "data"
    rows = [{"id": f"r{i}", "messages": [{"role": "user", "content": R.context_block({"page": "home"}, f"ask {i}")},
                                          {"role": "assistant", "content": "ok"}],
             "meta": {"manifest_hash": h} if h else {}} for i, h in enumerate(hashes)]
    write_jsonl(data / "train.jsonl", rows[:1])
    write_jsonl(data / "val.jsonl", rows[1:])
    if declared is not None:
        (data / "dataset-manifest.json").write_text(json.dumps({"manifest_hashes": declared}), encoding="utf-8")
    return data


def test_rerender_refuses_a_manifest_that_is_not_the_rows(tmp_path):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps(TINY), encoding="utf-8")
    assert F.manifest_file_hash(manifest) == TINY_HASH
    other = "ab" * 32
    for i, kw in enumerate(({"hashes": (TINY_HASH, other)}, {"hashes": (other, other)}, {"declared": [other]},
                            {"hashes": (None, None), "declared": [TINY_HASH, other]})):
        data, out = _dataset(tmp_path / f"bad{i}", **kw), tmp_path / f"bad{i}" / "out"
        with pytest.raises(F.FieldsError, match="another manifest's keys"):
            F.rerender(data, out, ManifestStub(manifest))
        assert not out.exists()  # refused before anything is written
    data = _dataset(tmp_path / "ok", declared=[TINY_HASH])
    rep = F.rerender(data, tmp_path / "ok" / "out", ManifestStub(manifest))
    assert rep["fields_context"]["manifest_hash"] == TINY_HASH and rep["fields_context"]["rows_checked"] == 2
    assert rep["fields_context"]["rows_with_block"] == 2
    man = json.loads((tmp_path / "ok" / "out" / "dataset-manifest.json").read_text())
    assert man["fields_context"]["manifest_hash"] == TINY_HASH and man["manifest_hashes"] == [TINY_HASH]
    # rows that carry no hash cannot be checked: counted, not refused
    bare = F.rerender(_dataset(tmp_path / "bare", hashes=(None, None)), tmp_path / "bare" / "out", ManifestStub(manifest))
    assert bare["fields_context"]["rows_checked"] == 0
    # stripping the block needs no manifest and checks none
    assert F.rerender(_dataset(tmp_path / "off", hashes=(other, other)), tmp_path / "off" / "out", None)["fields_context"] == \
        {"enabled": False, "rows": 2, "rows_with_block": 0}
    with pytest.raises(F.FieldsError, match="cannot be read"):
        F.manifest_file_hash(tmp_path / "nowhere.json")


def test_the_rerender_cli_exits_1_on_the_wrong_manifest(tmp_path, monkeypatch, capsys):
    """Direct CLI use with another manifest than the dataset's own (the
    review ran it with the fixture snapshot): exit 1, nothing written. The
    check runs before the Go CLI is ever started."""
    exe = tmp_path / "never-run"
    exe.write_text("#!/bin/sh\nexit 97\n", encoding="utf-8")
    exe.chmod(0o755)
    monkeypatch.setenv(F.BIN_ENV, str(exe))
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps(TINY), encoding="utf-8")
    data = _dataset(tmp_path, hashes=("cd" * 32, "cd" * 32))
    assert F.main(["rerender", "--data", str(data), "--manifest", str(manifest), "--out", str(tmp_path / "out")]) == 1
    err = capsys.readouterr().err
    assert "cdcdcdcdcdcd… (2 rows)" in err and TINY_HASH[:12] in err and "nothing was written" in err
    assert not (tmp_path / "out").exists()


# --- the text trim is Go's ---------------------------------------------------------------------------------------------------------

GO_SPACE = "\t\n\v\f\r \u0085                 　"
NOT_GO_SPACE = "\x1c\x1d\x1e\x1f"  # str.strip() cuts these; strings.TrimSpace (unicode.IsSpace) does not


def test_go_trim_space_is_unicode_is_space():
    assert set(R._GO_SPACE) == set(GO_SPACE) and serving.go_trim_space is R.go_trim_space
    assert R.go_trim_space(GO_SPACE + "a b" + GO_SPACE) == "a b" and R.go_trim_space(GO_SPACE) == "" == R.go_trim_space("")
    for ch in NOT_GO_SPACE + "﻿​":
        assert R.go_trim_space(f" {ch}x{ch}\n") == f"{ch}x{ch}" and ch not in R._GO_SPACE
    assert (NOT_GO_SPACE + "x").strip() == "x"  # why str.strip() was wrong here


def test_the_user_turn_keeps_what_go_keeps():
    text = "\x1c Make it bold \x1f"
    assert R.context_block(None, f" \n{text}  ") == text
    assert R.context_block({"page": "home"}, f" {text}\n") == '<context>\n{"page":"home"}\n</context>\n' + text
    assert R.context_block(None, f" {text} ", f"\n{BLOCK}\n ") == BLOCK + "\n" + text
    assert R.user_message(None, text)["content"][0]["text"] == text
    head, ctx, cand, rest = F.split_user_text(R.context_block({"a": 1}, text, BLOCK))
    assert (ctx, cand, rest) == ({"a": 1}, BLOCK, text) and F.join_user_text(head, f" {BLOCK}\n", rest) == head + BLOCK + "\n" + text


def test_fields_block_sends_the_cli_go_s_trim_of_the_text(tmp_path, monkeypatch):
    """loop.candidateFields trims with strings.TrimSpace before the
    retrieval: the lab hands the CLI the same text."""
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps(TINY), encoding="utf-8")
    f = F.Fields(manifest, exe=tmp_path / "unused")
    sent = []

    def run(turns):
        sent.extend(turns)
        return [f"<{t}>" for _, t in turns]

    monkeypatch.setattr(f, "_run", run)
    assert f.block({"p": 1}, " \x1chero title\x1f \n") == "<\x1chero title\x1f>"
    assert f.block({"p": 1}, "\x1chero title\x1f") == "<\x1chero title\x1f>" and len(sent) == 1  # one turn, cached
    assert f.block({"p": 1}, "hero title") == "<hero title>"  # another text than the one with the control characters
    f.prime([({"p": 1}, "\thero title "), (None, " \x1ex ")])
    assert sent == [({"p": 1}, "\x1chero title\x1f"), ({"p": 1}, "hero title"), (None, "\x1ex")]
