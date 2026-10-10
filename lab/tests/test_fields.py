"""The candidate-fields block and hybrid rows in the lab (docs/LAB.md §5
"Candidate fields", §6 "Hybrid").

Three layers:

* no go needed — where the block sits in the user turn (render), taking a
  rendered turn apart again, what the collector records of an escalated
  turn, the student-only / hybrid / escalation figures, the dataset's flags
  on escalated rows and train's check of what the rows carry;
* the real capture `fixtures/prompt/real/fields-context.*` (the real service
  binary on a route with `fields_context` on): the stored row, the rebuild
  and the wire are one and the same for the user turn — also no go, the row
  logged its block;
* go on PATH (skipped otherwise; SOUS_CHEF_LAB_REQUIRE_GO=1 fails instead) —
  the block the lab computes through the service's CLI IS the captured one,
  and a block-less built row re-rendered with it equals the wire.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
for _p in (LAB, LAB / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fake_service as FS  # noqa: E402
from souschef_lab import collect as C  # noqa: E402
from souschef_lab import dataset as D  # noqa: E402
from souschef_lab import eval as E  # noqa: E402
from souschef_lab import fields as F  # noqa: E402
from souschef_lab import render as R  # noqa: E402
from souschef_lab import train  # noqa: E402
from souschef_lab import verify as V  # noqa: E402
from test_collect import ASSETS, SET_TITLE, TINY, USAGE, RecordingVerifier, call, done, fake_engine, make_case, waiting  # noqa: E402

REAL = LAB / "fixtures" / "prompt" / "real"
WITH = REAL / "fields-context.set_field.contentBlock-1.title"  # route lab-fake-fields: fields_context on
WITHOUT = REAL / "ok.set_field.contentBlock-1.title"  # the same case and steps on the block-less route
MANIFEST = LAB / "fixtures" / "manifest.snapshot.json"
TRACES = LAB / "fixtures" / "traces"
RECORDED = TRACES / "engine" / "recorded.json"
BLOCK = "<candidate_fields>\nheader\ntarget {\"kind\":\"branding\"}\n  logo — Logo (image)\n</candidate_fields>"


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def capture(where: Path) -> tuple[dict, dict, list[dict]]:
    return (json.loads((where / "case.json").read_text(encoding="utf-8")), jsonl(where / "collect-real.jsonl")[0],
            [r["body"] for r in jsonl(where / "model-requests.jsonl")])


def wire_user(reqs: list[dict]) -> str:
    return reqs[0]["messages"][0]["content"][0]["text"]


def need_go() -> None:
    if not F.available():
        if os.environ.get("SOUS_CHEF_LAB_REQUIRE_GO"):
            pytest.fail("SOUS_CHEF_LAB_REQUIRE_GO is set but the sous-chef-fields CLI cannot be built (no go on PATH)")
        pytest.skip("go not on PATH (and SOUS_CHEF_FIELDS_BIN unset): the candidate-fields CLI is not built")


class StubFields:
    """fields.Fields without go: one fixed block for every turn."""

    def __init__(self, block: str = BLOCK):
        self._block, self.asked = block, []

    def prime(self, turns) -> None:
        list(turns)

    def block(self, context, text) -> str:
        self.asked.append((context, text))
        return self._block

    def identity(self) -> dict:
        return {"cli": "stub", "source_sha256": "0" * 64}


# --- where the block sits (prompt.UserText) ---------------------------------------------------------------------------------

def test_context_block_places_the_candidates_between_context_and_text():
    assert R.context_block({"page": "home"}, " hi ", BLOCK + "\n") == '<context>\n{"page":"home"}\n</context>\n' + BLOCK + "\nhi"
    assert R.context_block(None, "hi", BLOCK) == BLOCK + "\nhi"  # no context: the block opens the message
    assert R.context_block({"page": "home"}, "hi", " \n") == R.context_block({"page": "home"}, "hi")  # blank = none
    assert R.user_message({}, "hi", BLOCK)["content"][0]["text"] == "<context>\n{}\n</context>\n" + BLOCK + "\nhi"
    assert C.user_text({"a": 1}, "hi", BLOCK) == R.context_block({"a": 1}, "hi", BLOCK)
    assert R.turn_candidates({"candidate_fields": BLOCK}) == BLOCK and R.turn_candidates({}) == "" and R.turn_candidates(None) == ""


def test_a_rendered_user_turn_comes_apart_and_goes_back_together():
    for ctx, block, text in (({"page": "home", "n": 1.50, "s": "é<&>"}, BLOCK, "make it red\nplease"), (None, BLOCK, "hi"),
                             ({}, "", "hi"), (None, "", "<context> is a word here")):
        whole = R.context_block(ctx, text, block)
        head, got_ctx, got_block, got_text = F.split_user_text(whole)
        assert (got_ctx, got_block, got_text) == (ctx, block, text)
        assert F.join_user_text(head, got_block, got_text) == whole and F.has_block(whole) is bool(block)
        assert F.join_user_text(head, "", got_text) == R.context_block(ctx, text)
    # the head is kept as bytes: a context that Python would re-encode differently (1e3 → 1000.0) survives a re-render
    odd = '<context>\n{"n":1e3}\n</context>\nhi'
    assert F.join_user_text(*[F.split_user_text(odd)[i] for i in (0, 2, 3)]) == odd
    with pytest.raises(ValueError):
        F.split_user_text("<context>\n{}\nhi")
    assert F.has_block("<context>\n{}\nhi") is False
    assert F.block_stats([{"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": R.context_block({}, "a", BLOCK)}]},
                          {"messages": [{"role": "user", "content": "a"}]}]) == {"rows": 2, "with_block": 1}


# --- the real capture: stored == rebuilt == wire --------------------------------------------------------------------------------

def test_real_capture_with_fields_context_on_stored_rebuilt_and_wire_agree():
    case, row, reqs = capture(WITH)
    block = row["candidate_fields"]
    assert block.startswith("<candidate_fields>\n") and block.endswith("\n</candidate_fields>") and "\n  title — " in block
    wire = wire_user(reqs)
    stored = row["messages"][0]["content"][0]["text"]
    rebuilt = D.rebuild_messages(row, case)
    assert stored == rebuilt[0]["content"][0]["text"] == wire == R.context_block(row["context"], row["text"], block)
    assert rebuilt == row["messages"] and D.stored_messages_match(row, case)
    head, ctx, got, text = F.split_user_text(wire)
    assert ctx == row["context"] and got == block and text == row["text"].strip()
    # every later request of the turn replays the same user turn (the block is part of the stored message)
    assert all(wire_user([r]) == wire for r in reqs)
    # the same turn on the block-less route differs by exactly the block
    _, plain_row, plain_reqs = capture(WITHOUT)
    assert "candidate_fields" not in plain_row and F.join_user_text(head, "", text) == wire_user(plain_reqs)
    assert [b["text"] for b in reqs[0]["system"]] == [b["text"] for b in plain_reqs[0]["system"]] and reqs[0]["tools"] == plain_reqs[0]["tools"]
    # the training render: as logged by default, none when asked, another block when given
    assert D.to_hf(row, case, tools=[])["messages"][0]["content"] == wire
    assert D.to_hf(row, case, tools=[], candidates="")["messages"][0]["content"] == wire_user(plain_reqs)
    hf = D.to_hf(plain_row, case, tools=[], candidates=block + "\n")
    assert hf["messages"][0]["content"] == wire and hf["meta"]["fields_context"] is True
    assert hf["meta"]["fields_block_chars"] == len(block) and D.to_hf(plain_row, case, tools=[])["meta"]["fields_context"] is False
    assert json.loads((WITH / "plan.json").read_text()) == {"route": "lab-fake-fields", "fields_context": True}


# --- the service's own code (needs go) ----------------------------------------------------------------------------------------------

def test_the_lab_computes_the_block_the_service_sent():
    need_go()
    case, row, reqs = capture(WITH)
    f = F.Fields(MANIFEST)
    assert f.block(row["context"], "  " + row["text"] + "\n") == row["candidate_fields"]  # the service trims the text first
    assert f.block(None, "zzz qqq") == ""  # nothing matches → no block
    assert f.block({"sections": "all of them"}, row["text"]) == ""  # a context the retrieval refuses → no block, as the service
    f.prime([(row["context"], row["text"]), ({"sections": 3}, "x"), (None, row["text"])])  # one bad turn does not sink the batch
    assert f.block({"sections": 3}, "x") == "" and f.block(row["context"], row["text"]) == row["candidate_fields"]
    ident = f.identity()
    assert ident["source_sha256"] == F.source_sha() and len(ident["source_sha256"]) == 64 and ident["manifest_file"] == str(MANIFEST)
    with pytest.raises(F.FieldsError):
        F.Fields(LAB / "fixtures" / "nowhere.json")


def test_rerender_gives_a_block_less_row_the_wire_s_user_turn(tmp_path):
    """`fields rerender`: the built row of the block-less capture, re-rendered
    with the block, carries byte for byte the user turn the service sent on
    the route with fields_context on — and stripping it again gives the
    block-less wire. Nothing else in the row changes."""
    need_go()
    case, plain_row, plain_reqs = capture(WITHOUT)
    _, _, reqs = capture(WITH)
    hf = D.to_hf(plain_row, case, system=("RULES\n",), tools=[])
    data, out, back = tmp_path / "data", tmp_path / "out", tmp_path / "back"
    data.mkdir()
    (data / "train.jsonl").write_text(json.dumps(hf, ensure_ascii=False) + "\n", encoding="utf-8")
    (data / "exam.jsonl").write_text('{"id":"x"}\n', encoding="utf-8")
    assert F.main(["rerender", "--data", str(data), "--manifest", str(MANIFEST), "--out", str(out)]) == 0
    new = jsonl(out / "train.jsonl")[0]
    assert new["messages"][1]["content"] == wire_user(reqs) and new["messages"][0] == hf["messages"][0]
    assert new["messages"][2:] == hf["messages"][2:] and new["tools"] == hf["tools"]
    assert new["meta"]["fields_context"] is True and new["meta"]["fields_block_chars"] > 0
    assert {k: v for k, v in new["meta"].items() if not k.startswith("fields_")} == \
        {k: v for k, v in hf["meta"].items() if not k.startswith("fields_")}
    man = json.loads((out / "dataset-manifest.json").read_text())
    assert man["fields_context"]["enabled"] is True and man["fields_context"]["rows_with_block"] == 1
    assert man["fields_context"]["source_sha256"] == F.source_sha() and man["rerendered_from"] == str(data)
    assert (out / "exam.jsonl").read_text() == '{"id":"x"}\n'
    assert F.main(["rerender", "--data", str(out), "--out", str(back), "--fields-context", "off"]) == 0
    assert jsonl(back / "train.jsonl")[0]["messages"][1]["content"] == wire_user(plain_reqs)
    assert json.loads((back / "dataset-manifest.json").read_text())["fields_context"]["enabled"] is False
    assert F.main(["rerender", "--data", str(data), "--manifest", str(MANIFEST), "--out", str(data)]) == 1  # never in place
    # train.py reads which is which
    assert train.main(["--data", str(out), "--name", "t", "--output", str(tmp_path / "o"), "--dry-run"]) == 0
    with pytest.raises(SystemExit):
        train.main(["--data", str(back), "--name", "t", "--output", str(tmp_path / "o"), "--dry-run"])
    assert train.main(["--data", str(back), "--name", "t", "--output", str(tmp_path / "o"), "--fields-context", "off", "--dry-run"]) == 0


def test_the_prefix_cli_is_the_captured_prefix():
    """service/cmd/sous-chef-prefix prints system[0], system[1] and the tools
    every captured model call carried (fixtures/prompt/, re-captured
    2026-10-10) — so `rerender --system` writes the served bytes."""
    need_go()
    p = F.Prefix(MANIFEST)
    fx = LAB / "fixtures" / "prompt"
    assert p.rules == (fx / "system0.rules.txt").read_text(encoding="utf-8")
    assert p.index == (fx / "system1.index.txt").read_text(encoding="utf-8")
    assert p.tools == json.loads((fx / "tools.json").read_text(encoding="utf-8"))
    assert p.manifest_hash == F.manifest_file_hash(MANIFEST) and p.index.startswith(F.INDEX_HEAD)
    _, _, reqs = capture(WITHOUT)
    assert all([b["text"] for b in r["system"]] == [p.rules, p.index] and r["tools"] == p.tools for r in reqs)


def test_rerender_system_replaces_the_index_or_the_whole_prefix(tmp_path, capsys):
    """`fields rerender --system index` swaps system[1] only (the rules the
    row was collected under stay); `--system prefix` swaps system[0],
    system[1] and the tools. The user turn, the answer and the rest of meta
    are untouched either way, and the dataset manifest records the prefix."""
    need_go()
    case, plain_row, _ = capture(WITHOUT)
    hf = D.to_hf(plain_row, case, system=("OLD RULES\n", F.INDEX_HEAD + "old index\n"), tools=[])
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.jsonl").write_text(json.dumps(hf, ensure_ascii=False) + "\n", encoding="utf-8")
    p = F.Prefix(MANIFEST)
    for mode, rules in (("index", "OLD RULES\n"), ("prefix", p.rules)):
        out = tmp_path / mode
        assert F.main(["rerender", "--data", str(data), "--manifest", str(MANIFEST), "--out", str(out),
                       "--fields-context", "off", "--system", mode]) == 0
        new = jsonl(out / "train.jsonl")[0]
        assert new["messages"][0]["content"] == rules + p.index
        assert new["messages"][1:] == hf["messages"][1:]
        assert new["tools"] == (D.hf_tools(p.tools) if mode == "prefix" else hf["tools"])
        assert new["meta"]["system_rerendered"] == mode
        man = json.loads((out / "dataset-manifest.json").read_text())["system_prompt"]
        assert man["rerendered"] == mode and man["system1_sha256"] == p.identity()["system1_sha256"]
        assert man["rows_changed"] == {"rules": int(mode == "prefix"), "index": 1, "tools": int(mode == "prefix"), "any": 1}
        assert f"system {mode} · rows changed" in capsys.readouterr().out
    with pytest.raises(ValueError):
        F.split_system("no index here")
    with pytest.raises(SystemExit):  # --system needs the manifest
        F.main(["rerender", "--data", str(data), "--out", str(tmp_path / "x"), "--fields-context", "off", "--system", "index"])


def test_the_dataset_cli_renders_every_row_with_the_block_by_default(tmp_path, capsys):
    need_go()
    fx = V.load_fixture(TRACES / "set_field.contentBlock-1.title.json")
    row, case = V.trace_from_fixture(fx), dict(fx["case"], holdout=False)
    (tmp_path / "log.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "cases.jsonl").write_text(json.dumps(case) + "\n")
    out = tmp_path / "out"
    args = ["--log", str(tmp_path / "log.jsonl"), "--cases", str(tmp_path / "cases.jsonl"), "--recorded", str(RECORDED),
            "--val-pct", "0", "--manifest", str(MANIFEST)]
    assert D.main([*args, "--out", str(out)]) == 0
    hf = jsonl(out / "train.jsonl")[0]
    context, text = R.turn_inputs(row, case)
    block = F.Fields(MANIFEST).block(context, text)
    assert block and hf["messages"][0]["content"] == R.context_block(context, text, block)
    assert hf["meta"]["fields_context"] is True and hf["meta"]["fields_block_chars"] == len(block)
    man = json.loads((out / "dataset-manifest.json").read_text())
    assert man["fields_context"]["enabled"] is True and man["fields_context"]["mode"] == "on"
    assert man["fields_context"]["source_sha256"] == F.source_sha() and man["fields_context"]["rows_with_block"] == 1
    assert "candidate fields on: 1 of 1 kept rows carry the block" in capsys.readouterr().out
    assert D.main([*args, "--out", str(tmp_path / "off"), "--fields-context", "off"]) == 0
    off = jsonl(tmp_path / "off" / "train.jsonl")[0]
    assert off["messages"][0]["content"] == R.context_block(context, text) and off["meta"]["fields_context"] is False
    assert json.loads((tmp_path / "off" / "dataset-manifest.json").read_text())["fields_context"] == \
        {"enabled": False, "mode": "off", "rows": 1, "rows_with_block": 0}
    # a manifest that is not the cases' is refused: its keys would be another manifest's
    (tmp_path / "other.json").write_text(json.dumps(TINY))
    bad = dict(case, manifest_hash="f" * 64)
    (tmp_path / "cases.jsonl").write_text(json.dumps(bad) + "\n")
    assert D.main([*args[:-1], str(tmp_path / "other.json"), "--out", str(tmp_path / "bad")]) == 2


# --- the collector: the block it logs, the escalation it records ----------------------------------------------------------------------

@pytest.fixture
def svc():
    with FS.FakeService(manifest=TINY, key="k") as s:
        yield s


@pytest.fixture
def client(svc):
    return C.Client(svc.url, "k", timeout_s=10)


def escalated(reason="tool_error", frm="blueberry-muffin", to="default"):
    return ("turn.escalated", {"from": frm, "to": to, "reason": reason})


def done_routes(routes, text="Done."):
    return [("text.delta", {"text": text}), ("turn.done", {"usage": USAGE, "stop": "end_turn", "conversationId": "c", "routes": routes})]


BAD = {"target": SET_TITLE["target"], "values": {"itemDescriptions": "x"}}


def test_run_case_logs_the_block_it_was_told_the_route_sends(svc, client):
    svc.scripts["c1"] = [done()]
    f = StubFields()
    row = C.run_case(make_case(), client, fake_engine, route="blueberry-muffin", fields=f)
    assert row["candidate_fields"] == BLOCK and f.asked == [(make_case()["context"], "Set the hero title to Hi")]
    assert row["messages"][0]["content"][0]["text"] == R.context_block(row["context"], row["text"], BLOCK)
    assert D.rebuild_messages(row, make_case()) == row["messages"] and D.stored_messages_match(row, make_case())
    assert "candidate_fields" not in svc.turn_requests[0]  # the service computes its own; nothing extra is sent
    svc.scripts["c1"] = [done()]
    plain = C.run_case(make_case(), client, fake_engine, route="default")
    assert "candidate_fields" not in plain and plain["routes"] is None and plain["escalated"] is None
    assert plain["messages"][0]["content"][0]["text"] == R.context_block(plain["context"], plain["text"])
    nothing = C.run_case(make_case(), client, fake_engine, route="blueberry-muffin", fields=StubFields(""))
    assert nothing["candidate_fields"] == "" and nothing["messages"][0] == plain["messages"][0]  # on, but nothing matched


def test_run_case_records_the_escalation_and_the_routes(svc, client):
    """The student's op is refused; the next stream opens with turn.escalated
    and the fallback's corrected op; turn.done names both routes."""
    svc.scripts["c1"] = [[call("t1", "bogus_op", BAD), waiting("t1")],
                         [escalated(), call("t2", "set_fields", SET_TITLE), waiting("t2")],
                         done_routes(["blueberry-muffin", "default"])]
    row = C.run_case(make_case(), client, fake_engine, route="blueberry-muffin",
                     verifier=RecordingVerifier("invalid_op", ["invalid_op"]))
    assert row["status"] == "ok" and row["escalated"] == {"from": "blueberry-muffin", "to": "default", "reason": "tool_error"}
    assert row["routes"] == ["blueberry-muffin", "default"] and row["model"] == "blueberry-muffin"
    assert [e["event"] for e in row["events"]] == ["tool.call", "turn.waiting", "tool.result", "turn.escalated", "tool.call",
                                                   "turn.waiting", "tool.result", "text.delta", "turn.done"]
    assert D.rebuild_messages(row, make_case()) == row["messages"]  # the event is not a message; the rebuild is unchanged
    assert C.escalation_of(row) == row["escalated"] == C.escalation_of({"events": row["events"]})
    assert C.routes_of(row) == ["blueberry-muffin", "default"] == C.routes_of({"model": "blueberry-muffin", "events": row["events"]})
    assert C.completed_by(row) == "default" and C.refused_after_escalation(row) == 0
    # strict: the verifier's invalid_op stands; the result is right — the student's refused op applied nothing
    assert C.passed(row) is False and C.outcome_passed(row) is True
    svc.scripts["c1"] = [done_routes(["blueberry-muffin"])]
    alone = C.run_case(make_case(), client, fake_engine, route="blueberry-muffin", verifier=RecordingVerifier())
    assert alone["routes"] == ["blueberry-muffin"] and alone["escalated"] is None and C.completed_by(alone) == "blueberry-muffin"
    assert C.outcome_passed(alone) is True


def hrow(cid, verdict="pass", reasons=(), esc=None, refused_after=False, status="ok", routes=("blueberry-muffin",)):
    events = []
    if esc:
        events = [{"event": "tool.result", "data": {"ok": False}}, {"event": "turn.escalated", "data": {"from": "blueberry-muffin", "to": "default", "reason": esc}}]
        if refused_after:
            events.append({"event": "tool.result", "data": {"ok": False}})
        routes = ("blueberry-muffin", "default")
    return {"id": f"{cid}#1", "case_id": cid, "sample": 1, "model": "blueberry-muffin", "status": status,
            "verdict": verdict if status == "ok" else None, "reasons": list(reasons), "events": events,
            "escalated": {"from": "blueberry-muffin", "to": "default", "reason": esc} if esc else None, "routes": list(routes),
            "ops": [], "latency_ms": 10, "meta": {"kind": "set_field"}}


def test_hybrid_summary_student_only_hybrid_and_escalation_rate():
    rows = [hrow("a"), hrow("b"), hrow("c", "wrong_value", ["wrong_value"]),  # the student alone: 2 of 3
            hrow("d", "invalid_op", ["invalid_op"], esc="tool_error"),  # refused, corrected by the fallback → hybrid pass
            hrow("e", "invalid_op", ["invalid_op", "wrong_value"], esc="tool_error"),  # corrected to the wrong value
            hrow("f", "invalid_op", ["invalid_op"], esc="tool_error", refused_after=True),  # the fallback's own refusal
            hrow("g", "pass", esc="model_error"), hrow("h", "unsupported_ok", esc="unsupported"),
            hrow("i", "no_answer", ["no_answer", "turn_error:model_error"], esc="max_tokens"),
            hrow("j", status="error")]
    h = C.hybrid_summary(rows)
    assert h["rows"] == 10 and h["scored"] == 9 and h["fallback_seen"] is True
    assert h["escalated"] == 6 and h["escalation_rate"] == 60.0
    assert h["reasons"] == {"max_tokens": 1, "model_error": 1, "tool_error": 3, "unsupported": 1}
    assert h["student_only"] == {"n": 3, "passed": 2, "pass_rate": 22.2, "pass_rate_unescalated": 66.7}
    assert h["escalated_rows"] == {"scored": 6, "passed": 3, "pass_rate": 50.0}
    assert h["hybrid"] == {"passed": 5, "pass_rate": 55.6, "strict_passed": 4, "strict_pass_rate": 44.4}
    line = C.format_hybrid(h)
    # the student is labelled by what the rows say — here the route that was asked (no reported model stated)
    assert h["student"] == {"route": "blueberry-muffin", "model": None}
    assert "blueberry-muffin-only pass 2/9 (22.2 pts" in line and "hybrid pass 5/9 (55.6 pts; strict 44.4)" in line
    assert "escalated 6/10 (60.0%" in line
    plain = C.hybrid_summary([dict(hrow("a"), routes=None), dict(hrow("b", "wrong_value", ["wrong_value"]), routes=None)])
    assert plain["fallback_seen"] is False and plain["escalated"] == 0 and plain["escalation_rate"] == 0.0
    assert plain["student_only"]["pass_rate"] == plain["hybrid"]["pass_rate"] == plain["hybrid"]["strict_pass_rate"] == 50.0
    assert C.hybrid_summary([])["hybrid"]["pass_rate"] is None
    # the model id the service reported for the route (turn.done route.model / models) wins over the route's name;
    # nothing assumes a route name
    named = [dict(hrow("a"), model="gpu-a", routes=["gpu-a"], models={"gpu-a": "blueberry-muffin"}),
             dict(hrow("b", "invalid_op", ["invalid_op"], esc="tool_error"), model="gpu-a",
                  escalated={"from": "gpu-a", "to": "default", "reason": "tool_error"}, routes=["gpu-a", "default"],
                  models={"gpu-a": "blueberry-muffin", "default": "claude-sonnet-5-5"})]
    hn = C.hybrid_summary(named)
    assert hn["student"] == {"route": "gpu-a", "model": "blueberry-muffin"}
    assert C.format_hybrid(hn).startswith("blueberry-muffin-only pass 1/2")
    assert C.student_of([{"routes": ["r1"], "models": {"r1": "m1"}}]) == {"route": "r1", "model": "m1"}  # default route
    assert C.student_of([]) == {"route": None, "model": None} and C.student_label({}) == "student"
    done = {"route": {"name": "gpu-a", "model": "blueberry-muffin"}, "models": {"gpu-a": "blueberry-muffin", "default": "claude-sonnet-5-5"}}
    assert C.done_models(done) == {"gpu-a": "blueberry-muffin", "default": "claude-sonnet-5-5"}
    assert C.done_models({"route": {"name": "x"}}) == {} == C.done_models({})


def test_eval_reports_the_hybrid_figures(svc, client, tmp_path):
    cases = [make_case("x1", holdout=True), make_case("x2", holdout=True)]
    svc.scripts["x1"] = [done_routes(["blueberry-muffin"])]
    svc.scripts["x2"] = [[call("t1", "bogus_op", BAD), waiting("t1")],
                         [escalated(), call("t2", "set_fields", SET_TITLE), waiting("t2")], done_routes(["blueberry-muffin", "default"])]
    verdicts = iter([{"verdict": "pass", "reasons": []}, {"verdict": "invalid_op", "reasons": ["invalid_op"]}])
    lines = []
    path, report = E.run_exam(cases, client, fake_engine, label="hyb", route="blueberry-muffin", verifier=lambda c, r: next(verdicts),
                              assets=ASSETS, out_dir=tmp_path, progress=lines.append, fields=StubFields())
    h = report["summary"]["hybrid"]
    assert h["escalated"] == 1 and h["escalation_rate"] == 50.0 and h["student_only"]["pass_rate"] == 50.0
    assert h["hybrid"]["pass_rate"] == 100.0 and h["hybrid"]["strict_pass_rate"] == 50.0
    assert report["summary"]["overall"]["pass_rate"] == 50.0  # the verifier's strict figure is untouched
    assert report["fields_context"] is True and all(r["candidate_fields"] == BLOCK for r in report["results"])
    assert any("escalated→default (tool_error)" in x for x in lines)
    text = E.format_summary(report)
    assert "hybrid   blueberry-muffin-only pass 1/2 (50.0 pts" in text and "hybrid pass 2/2 (100.0 pts; strict 50.0)" in text
    saved = E.load_run(path)
    assert saved["summary"]["hybrid"] == h and saved["results"][1]["routes"] == ["blueberry-muffin", "default"]
    # a route without a fallback: the line is not printed, the figures are the plain pass
    assert "hybrid" not in E.format_summary(E.build_report([dict(hrow("a"), routes=None)], label="p", route="default",
                                                           manifest_hash=None, started="t"))
    assert "hybrid" in E.summarize([hrow("a")])


def test_collect_logs_escalations_and_the_block(svc, tmp_path, capsys):
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps(make_case("m1")) + "\n" + json.dumps(make_case("m2")) + "\n")
    svc.scripts["m1"] = [done_routes(["blueberry-muffin"])]
    svc.scripts["m2"] = [[escalated("model_error"), *done_routes(["blueberry-muffin", "default"])]]
    counts = C.collect(C.load_cases(cases), C.Client(svc.url, "k"), fake_engine, tmp_path / "log.jsonl", route="blueberry-muffin",
                       verifier=RecordingVerifier(), progress=print, fields=StubFields())
    assert counts["escalated"] == 1 and counts["asked"] == 2
    out = capsys.readouterr().out
    assert "m2#1 pass · escalated→default (model_error)" in out
    rows = C.read_jsonl(tmp_path / "log.jsonl")
    assert [bool(r["escalated"]) for r in rows] == [False, True] and all(r["candidate_fields"] == BLOCK for r in rows)
    assert C.read_jsonl(C.runs_log(tmp_path / "log.jsonl"))[0]["fields_context"] is True
    h = C.hybrid_summary(rows)
    assert h["student_only"]["pass_rate"] == 50.0 and h["hybrid"]["pass_rate"] == 100.0 and h["escalation_rate"] == 50.0


# --- the dataset: escalated rows are the fallback's, flagged ----------------------------------------------------------------------------

def fixture_row(name="set_field.contentBlock-1.title", **over):
    fx = V.load_fixture(TRACES / f"{name}.json")
    row = V.trace_from_fixture(fx, model="blueberry-muffin")
    row.update(over)
    return row, dict(fx["case"], holdout=False)


def test_an_escalated_row_is_kept_as_a_flagged_teacher_row():
    engine = V.RecordedEngine(RECORDED)
    esc = {"from": "blueberry-muffin", "to": "default", "reason": "model_error"}
    row, case = fixture_row(escalated=esc, routes=["blueberry-muffin", "default"])
    hf = D.to_hf(row, case)
    assert hf["meta"]["model"] == "default" and hf["meta"]["requested_model"] == "blueberry-muffin"
    assert hf["meta"]["escalated"] is True and hf["meta"]["escalation_reason"] == "model_error"
    plain, _ = fixture_row()
    meta = D.to_hf(plain, case)["meta"]
    assert meta["model"] == meta["requested_model"] == "blueberry-muffin" and meta["escalated"] is False and meta["escalation_reason"] is None
    # from the event alone (a trace row without the collector's field)
    ev_row, _ = fixture_row()
    ev_row["events"] = [{"event": "turn.escalated", "seq": 0, "data": esc}, *ev_row["events"]]
    assert D.to_hf(ev_row, case)["meta"]["model"] == "default"
    b = D.build([row], {case["id"]: case}, engine, D.BuildConfig(val_pct=0))
    assert len(b.train) == 1 and b.train[0]["meta"]["escalated"] is True and b.train[0]["meta"]["model"] == "default"
    assert b.stats["escalated"] == {"rows": 1, "kept": 1, "reasons": {"model_error": 1}, "kept_by_route": {"default": 1}}
    assert b.manifest["escalated_rows"]["kept"] == 1 and "flagged" in b.manifest["escalated_rows"]["policy"]
    dropped = D.build([row], {case["id"]: case}, engine, D.BuildConfig(val_pct=0, drop_escalated=True))
    assert not dropped.train and dropped.rejects[0]["reasons"] == ["escalated"] and dropped.manifest["optional_rules"] == ["escalated"]


def test_build_modes_for_the_block_without_go():
    engine = V.RecordedEngine(RECORDED)
    row, case = fixture_row(candidate_fields=BLOCK)
    row["messages"] = D.rebuild_messages(row, case)  # the cache a collector with --fields-context on would have logged
    cases = {case["id"]: case}
    logged = D.build([row], cases, engine, D.BuildConfig(val_pct=0))  # the library default: as logged
    assert F.has_block(logged.train[0]["messages"][0]["content"]) and logged.manifest["fields_context"]["enabled"] is True
    assert logged.stats["fields_context"]["drift"]["rows"] == 0 and logged.stats["fields_context"]["logged_with_block"] == 1
    off = D.build([row], cases, engine, D.BuildConfig(val_pct=0, fields_context="off"))
    assert not F.has_block(off.train[0]["messages"][0]["content"]) and off.manifest["fields_context"]["enabled"] is False
    assert off.stats["fields_context"]["drift"]["rows"] == 1  # logged with a block, rendered without: said so
    other = BLOCK.replace("logo — Logo", "favicon — Favicon")
    on = D.build([row], cases, engine, D.BuildConfig(val_pct=0, fields_context="on", fields=StubFields(other)))
    assert other in on.train[0]["messages"][0]["content"] and on.manifest["fields_context"]["source_sha256"] == "0" * 64
    assert on.stats["fields_context"]["drift"] == {"rows": 1, "sample": [row["id"]]} and on.stats["fields_context"]["rows_with_block"] == 1
    with pytest.raises(ValueError):
        D.build([row], cases, engine, D.BuildConfig(fields_context="on"))


# --- train: what the rows carry is checked and recorded ------------------------------------------------------------------------------------

def test_train_checks_the_rows_against_the_option():
    with_block = {"enabled": True, "rows": 3, "rows_with_block": 2, "dataset": None}
    assert train.check_fields_context(with_block, Path("d")) is None
    assert "no row" in train.check_fields_context({**with_block, "rows_with_block": 0}, Path("d"))
    assert "fields rerender" in train.check_fields_context({**with_block, "rows_with_block": 0}, Path("d"))
    assert "2 of 3 rows" in train.check_fields_context({**with_block, "enabled": False}, Path("d"))
    assert train.check_fields_context({**with_block, "enabled": False, "rows_with_block": 0}, Path("d")) is None
