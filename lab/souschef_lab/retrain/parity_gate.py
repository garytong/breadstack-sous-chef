"""parity_gate — the render-parity gate of the dataset stage (design §3.4 gate 1;
the tests/r4_parity.py pattern, on the ROUND's manifest).

Three conversations run through the REAL service binary of the deployed build
(the lab image's /opt/sous-chef/bin/sous-chef) on a messages-compatible route
`student` whose model is the lab's scripted fake (tests/fake_model.py), are
collected by r4collect (the browser emulator + the engine) and rendered by
r4dataset.student_source. Every model call of every scored turn must equal the
request the service sent, byte for byte after the r4_parity normalisation:
system (rules + index), tools, stable history (earlier user turns stripped,
thinking dropped), the follow-up-aware candidate-fields block, the server-tool
round trips (lookup_fields, report_unsupported) and `thinking: disabled`.

  1. a 2-turn follow-up whose answer looks the field up before writing it
     (a global text field: the announcement bar message when the manifest has it)
  2. an out-of-editor turn (report_unsupported, where: domain)
  3. a single write (a global number field: the buttons' border radius when there)

Postgres: $PIPELINE_DB_URL when set, else a throwaway cluster started with the
PG_BIN tools (initdb / pg_ctl) in the work folder on 127.0.0.1.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path
from typing import Iterator

from .. import LAB_DIR
from .common import ConfigError

DEFAULT_BIN = "/opt/sous-chef/bin/sous-chef"


def service_binary(explicit: str | None = None) -> Path | None:
    for cand in (explicit, os.environ.get("SOUS_CHEF_BIN"), DEFAULT_BIN):
        if cand and Path(cand).is_file():
            return Path(cand)
    return None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _pg_tool(name: str) -> str:
    pg_bin = os.environ.get("PG_BIN")
    if pg_bin and (Path(pg_bin) / name).is_file():
        return str(Path(pg_bin) / name)
    found = shutil.which(name)
    if not found:
        raise ConfigError(f"{name} not found (set PG_BIN to the Postgres 16 bin folder)")
    return found


@contextlib.contextmanager
def local_postgres(workdir: Path | None = None) -> Iterator[str]:
    """A throwaway Postgres on 127.0.0.1 (trust auth, the job's user); yields its URL."""
    if os.environ.get("PIPELINE_DB_URL"):
        yield os.environ["PIPELINE_DB_URL"]
        return
    base = Path(tempfile.mkdtemp(prefix="pg-", dir=str(workdir) if workdir else None))
    data = base / "data"
    sock = Path(tempfile.mkdtemp(prefix="pgs-", dir="/tmp" if Path("/tmp").is_dir() else None))  # a unix socket path is ≤ 107 bytes
    port = _free_port()
    run = lambda *a: subprocess.run(list(a), capture_output=True, text=True, timeout=120, check=False)  # noqa: E731
    r = run(_pg_tool("initdb"), "-D", str(data), "-U", "souschef", "--auth=trust", "-E", "UTF8", "--no-instructions")
    if r.returncode != 0:
        raise ConfigError(f"initdb failed: {r.stderr[-500:]}")
    r = run(_pg_tool("pg_ctl"), "-D", str(data), "-w", "-l", str(base / "pg.log"), "-o",
            f"-p {port} -k {sock} -c listen_addresses=127.0.0.1 -c fsync=off -c logging_collector=off", "start")
    if r.returncode != 0:
        raise ConfigError(f"pg_ctl start failed: {(base / 'pg.log').read_text(errors='replace')[-800:] if (base / 'pg.log').exists() else r.stderr}")
    try:
        r = run(_pg_tool("createdb"), "-h", "127.0.0.1", "-p", str(port), "-U", "souschef", "souschef_lab")
        if r.returncode != 0:
            raise ConfigError(f"createdb failed: {r.stderr[-500:]}")
        yield f"postgres://souschef@127.0.0.1:{port}/souschef_lab?sslmode=disable"
    finally:
        run(_pg_tool("pg_ctl"), "-D", str(data), "-m", "fast", "-w", "stop")
        shutil.rmtree(base, ignore_errors=True)
        shutil.rmtree(sock, ignore_errors=True)


def _pick(m, prefer: list[tuple[str, str]], types: tuple[str, ...]) -> tuple[str, str] | None:
    for panel, key in prefer:
        c = m.panel(panel)
        if c is not None and key in c.fields and c.fields[key].type in types:
            return panel, key
    for c in sorted(m.containers("global"), key=lambda c: c.name):
        for k, f in sorted(c.fields.items()):
            if f.type in types and not f.gated and not f.read_only and not f.keyed_by:
                return c.name, k
    return None


def plan(m) -> list[tuple[dict, list, int]]:
    """(case, scripted model steps, history steps) for the three conversations, on manifest m."""
    from .. import state as S
    from tests.r4_parity import msg

    st = S.build_state(m, template="vintage")
    extras = S.load_context_fixture()
    page = "home"
    text = _pick(m, [("HeaderSection", "announcementBar.message")], ("richtext", "text"))
    num = _pick(m, [("ButtonsSection", "borderRadius")], ("range", "number"))
    if text is None or num is None:
        raise ConfigError("parity gate: the manifest has no global text and number field to script")
    tpanel, tkey = text
    npanel, nkey = num
    nf = m.panel(npanel).fields[nkey]
    nval = int(nf.min if nf.min is not None else 0) + 1
    hb = {"target": {"kind": "global", "panel": tpanel}, "values": {tkey: "Free delivery over $60"}}
    follow = {"id": "pg-PAR-0001", "kind": "followup", "site_state": st, "template": "vintage", "manifest_hash": m.hash,
              "context": S.context_of(st, page, extras, m, text="Free delivery over $60"), "request": "Free delivery over $60",
              "history": [{"user": "change what the promo bar says", "page": page, "expect": "ask", "assistant": "",
                           "ops": [], "calls": []}],
              "expected": {"ops": [{"op": "set_fields", **hb}], "diff": [],
                           "keys": [{"target": hb["target"], "key": tkey, "list": ""}], "alt": [], "alt_exact": [],
                           "out_of_scope": None},
              "score": {"mode": "keys"}, "holdout": False, "meta": {"family": "F2", "page": page, "view": None}}
    steps_follow = [
        msg({"type": "thinking", "thinking": "the merchant wants the bar changed"}, {"type": "text", "text": "Sure — what should it say?"}),
        msg({"type": "text", "text": "Let me find it."}, {"type": "tool_use", "id": "toolu_l1", "name": "lookup_fields",
                                                          "input": {"query": "announcement bar message"}}),
        msg({"type": "tool_use", "id": "toolu_s1", "name": "set_fields", "input": hb}),
        msg({"type": "text", "text": "Done — it now says \"Free delivery over $60\". Save when you're happy."}),
    ]
    oos = {"id": "pg-PAR-0002", "kind": "unsupported", "site_state": st, "template": "vintage", "manifest_hash": m.hash,
           "context": S.context_of(st, page, extras, m, text="connect my domain prairieleaf.ca"),
           "request": "connect my domain prairieleaf.ca", "history": [],
           "expected": {"ops": [], "diff": [], "keys": [], "alt": [], "alt_exact": [],
                        "out_of_scope": {"reason": "domain", "expected_where": ["domain"]}},
           "score": {"mode": "where"}, "holdout": False, "meta": {"family": "F3", "page": page, "view": None}}
    steps_oos = [
        msg({"type": "tool_use", "id": "toolu_u1", "name": "report_unsupported",
             "input": {"reason": "Domains are connected in Site Settings.", "where": "domain"}}),
        msg({"type": "text", "text": "Connecting a domain is done in your domain settings."}),
    ]
    bt = {"target": {"kind": "global", "panel": npanel}, "values": {nkey: nval}}
    single = {"id": "pg-PAR-0003", "kind": "set_range", "site_state": st, "template": "vintage", "manifest_hash": m.hash,
              "context": S.context_of(st, page, extras, m, text=f"set it to {nval}"), "request": f"set it to {nval}",
              "history": [], "expected": {"ops": [{"op": "set_fields", **bt}], "diff": [],
                                          "keys": [{"target": bt["target"], "key": nkey, "list": ""}], "alt": [],
                                          "alt_exact": [], "out_of_scope": None},
              "score": {"mode": "keys"}, "holdout": False, "meta": {"family": "F1", "page": page, "view": None}}
    steps_single = [msg({"type": "tool_use", "id": "toolu_b1", "name": "set_fields", "input": bt}),
                    msg({"type": "text", "text": f"Done — it's {nval} now."})]
    return [(follow, steps_follow, 1), (oos, steps_oos, 0), (single, steps_single, 0)]


def run(manifest_path: Path, workdir: Path, *, binary: Path, engine_path: str | None = None) -> dict:
    """{ok, calls_checked, failures, service_version}. Needs node + the engine, the binary, Postgres tools."""
    import sys
    if str(LAB_DIR) not in sys.path:
        sys.path.insert(0, str(LAB_DIR))
    from tests import capture_real as CR
    from tests import fake_model as FM
    from tests.r4_parity import norm_messages

    from .. import collect as C
    from .. import fields as F
    from .. import r4collect as RC
    from .. import r4dataset as D
    from .. import state as S
    from ..engine import PersistentEngine
    from ..manifest import load

    m = load(str(manifest_path))
    fields = F.Fields(manifest_path)
    prefix = F.Prefix(manifest_path)
    out = Path(workdir) / "parity"
    out.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    checked = 0
    version: dict = {}
    eng = PersistentEngine(engine_path, manifest_path=manifest_path, size=1)
    try:
        with local_postgres(Path(workdir)) as db_url, FM.FakeModel([]) as fake:
            svc = CR.Service(binary, port=CR.free_port(), model_url=fake.url, db_url=db_url, log=out / "service.log")
            svc.env["MODEL_ROUTES"] = json.dumps({"student": {"kind": "messages-compatible", "url": fake.url, "auth": "none",
                                                              "model": "fake", "max_tokens": 4096, "stream": False}})
            svc.env["DEFAULT_ROUTE"] = "student"
            svc.env["MODEL_ROUTE_ALLOWLIST"] = "student"
            svc.env["BUILD_ID"] = "pipeline-parity"
            for k in ("ANTHROPIC_API_KEY", "SOUS_CHEF_KEY"):
                svc.env.pop(k, None)
            with svc:
                version = C.Client(svc.url, CR.KEY).version()
                for case, steps, hist_steps in plan(m):
                    start = len(fake.requests)
                    fake.append(*steps)
                    client, _ = C.connect(svc.url, CR.KEY, manifest_path, [case], sse=False, timeout_s=60,
                                          build_id="pipeline-parity")
                    row = RC.run_r4_case(case, client, eng, route="student", fields=fields, manifest=m,
                                         extras=S.load_context_fixture(), assets=C.load_assets(C.DEFAULT_ASSETS),
                                         site_prefix="lab-parity-")
                    reqs = [r["body"] for r in fake.requests[start:]]
                    (out / f"{case['id']}.row.json").write_text(json.dumps(row, ensure_ascii=False, indent=1), encoding="utf-8")
                    if len(reqs) != len(steps):
                        failures.append(f"{case['id']}: the fake served {len(reqs)} of {len(steps)} steps")
                    try:
                        src, notes = D.student_source(case, row, fields)
                    except D.RenderError as e:
                        failures.append(f"{case['id']}: render error {e}")
                        continue
                    src_n = norm_messages(src)
                    for k, body in enumerate(reqs[hist_steps:]):
                        checked += 1
                        wire = norm_messages(body["messages"])
                        if wire != src_n[:len(wire)]:
                            i = next((j for j, (x, y) in enumerate(zip(wire, src_n)) if x != y), min(len(wire), len(src_n)))
                            failures.append(f"{case['id']} call {k + 1}: messages differ at {i}")
                        if "".join(b.get("text", "") for b in body.get("system") or []) != prefix.rules + prefix.index:
                            failures.append(f"{case['id']} call {k + 1}: system differs")
                        if [t.get("name") for t in body.get("tools") or []] != [t.get("name") for t in prefix.tools]:
                            failures.append(f"{case['id']} call {k + 1}: tool list differs")
                        if body.get("thinking") != {"type": "disabled"}:
                            failures.append(f"{case['id']} call {k + 1}: thinking {body.get('thinking')}")
                    if notes.get("block_drift"):
                        failures.append(f"{case['id']}: block drift")
    finally:
        eng.close()
    return {"ok": not failures and checked > 0, "calls_checked": checked, "failures": failures,
            "service_build": version.get("build_id")}
