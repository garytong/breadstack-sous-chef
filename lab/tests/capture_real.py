"""Capture the REAL service binary's model requests for the prompt-parity
fixtures (docs/LAB.md §4 "Prompt parity", §9; fixtures/prompt/real/README.md).

For every capture in a plan this runs one scripted turn through the real
`sous-chef` binary with `tests/fake_model.py` as the model (no API spend,
no network beyond 127.0.0.1) and the real browser engine (headless.mjs),
driven by the lab's own collector (collect.run_case), and stores under
`<out>/<name>/`:

    case.json             the case the turn was asked (LAB.md §2 row)
    steps.json            the fake model's scripted replies (fake_model.py steps)
    collect-real.jsonl    the collector row (events incl. tool.result, messages, context, text)
    model-requests.jsonl  every POST /v1/messages body the service sent, as captured
    plan.json             only for a capture off the default route: {route, fields_context}

Two routes, both the fake model: `lab-fake` with `fields_context: off` (the
bytes an Anthropic route sends — the four tool_result shapes) and
`lab-fake-fields` with it on (the student's prompt: the candidate-fields
block in the user turn, which the collector computes too — `run_case(…,
fields=…)`, needs go — and logs as `row.candidate_fields`).

`tests/test_render.py` rebuilds each turn from `collect-real.jsonl` and
asserts the bytes equal `model-requests.jsonl`. Nothing here is a model
pass rate: the replies are scripted.

Prerequisites: go (the binary is built from service/cmd/sous-chef unless
--binary is given), node >= 20 + the cookie-cutter checkout (the engine),
Postgres reachable at --db-url (`make db-up` in the repo root starts the
docker one on 5433; the lab uses its own database name).

    .venv/bin/python tests/capture_real.py --default-plan data/e2e/cases.jsonl --out fixtures/prompt/real
    .venv/bin/python tests/capture_real.py --from fixtures/prompt/real --out /tmp/recapture   # re-run the stored plan
    … --only fields-context.set_field.contentBlock-1.title                                    # one capture of the plan
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
REPO = LAB.parent
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fake_model as FM  # noqa: E402
from souschef_lab import collect as C  # noqa: E402

KEY = "test-key"
ROUTE = "lab-fake"
FIELDS_ROUTE = "lab-fake-fields"  # the same fake model, with the candidate-fields block in the user turn
BUILD_ID = "lab-e2e"
DEFAULT_DB = "postgres://souschef:souschef@localhost:5433/souschef_lab?sslmode=disable"
TRACES = LAB / "fixtures" / "traces"


# --- the default plan: one capture per tool_result shape loop.browserResult can render -----------------------------------

def _case(cases: dict[str, dict], cid: str) -> dict:
    if cid in cases:
        return cases[cid]
    fx = TRACES / f"{cid}.json"
    if fx.is_file():
        return json.loads(fx.read_text(encoding="utf-8"))["case"]
    raise SystemExit(f"case {cid} is neither in the cases file nor in fixtures/traces/")


def _msg(text: str, *calls: dict, usage=(120, 30)) -> dict:
    content: list[dict] = [{"type": "text", "text": text}] if text else []
    content += [{"type": "tool_use", **c} for c in calls]
    return {"message": {"content": content, "usage": {"input_tokens": usage[0], "output_tokens": usage[1]}}}


def _op(op: dict) -> dict:
    return {k: v for k, v in op.items() if k != "op"}


def default_plan(cases_path: Path | None) -> list[dict]:
    """Five captures: the ok/written shape (the original 2026-10-01 case), an
    error result, a warnings + sectionId batch, a pick_asset asset — and the
    first again on the route with `fields_context` on (the candidate-fields
    block in the user turn)."""
    cases: dict[str, dict] = {}
    if cases_path:
        for line in Path(cases_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                cases[str(row["id"])] = row
    plan: list[dict] = []

    # 1. ok/written — set_field.contentBlock-1.title, the steps of the first real-binary run
    title = _case(cases, "set_field.contentBlock-1.title")
    t_op = title["expected"]["ops"][0]
    plan.append({"name": "ok.set_field.contentBlock-1.title", "shape": "ok/written", "case": title, "steps": [
        _msg("Updating the title.", {"id": "toolu_lab_1", "name": t_op["op"], "input": _op(t_op)}),
        _msg("Done — the title now reads as asked.", usage=(160, 12)),
    ]})

    # 2. error — set_field.contentBlock-1.tagline: an unknown key first (E_FIELD, is_error), then the right op
    tagline = _case(cases, "set_field.contentBlock-1.tagline")
    g_op = tagline["expected"]["ops"][0]
    plan.append({"name": "error.set_field.contentBlock-1.tagline", "shape": "error (api.ToolError, is_error) then ok", "case": tagline, "steps": [
        _msg("Turning the button on.", {"id": "toolu_lab_1", "name": "set_fields",
                                         "input": {"target": g_op["target"], "values": {"showButton": True}}}),
        _msg("That field does not exist on this block; updating the tagline instead.",
             {"id": "toolu_lab_2", "name": g_op["op"], "input": _op(g_op)}, usage=(170, 40)),
        _msg("Done — the tagline is updated.", usage=(210, 12)),
    ]})

    # 3. warnings + sectionId — one batch: edit_list add sections (sectionId) and a gated field (W_HIDDEN)
    add = _case(cases, "list_add/page/home/sections/loyaltyForm")
    a_op = add["expected"]["ops"][0]
    gated = {"target": {"kind": "section", "page": "home", "sectionId": "featuredCategories"}, "values": {"tileBorder": "all"}}
    plan.append({"name": "warnings-sectionid.list_add.home.sections.loyaltyForm", "shape": "ok + sectionId; ok + warnings", "case": add, "steps": [
        _msg("Adding a loyalty form and giving the category tiles a border.",
             {"id": "toolu_lab_1", "name": a_op["op"], "input": _op(a_op)},
             {"id": "toolu_lab_2", "name": "set_fields", "input": gated}),
        _msg("Done — the loyalty form is on the home page and the tiles have a border.", usage=(190, 14)),
    ]})

    # 4. pick_asset — the asset the collector will answer with is deterministic (collect.pick_asset_for →
    #    state.fixture_asset_for, keyed by the case id ALONE — the same function the generator used for the case's
    #    expected ops), so the scripted set_fields carries its URL + {alt, srcset} as the model would after reading the
    #    asset result, reproduces expected.diff, and the capture is a `pass`. The call id plays no part in the pick
    #    (before 377cefa it did, and this plan searched for the id whose pick matched the case).
    pick = _case(cases, "image/gallery/gallery/bgImage/pick")
    p_op, s_op = pick["expected"]["ops"][0], pick["expected"]["ops"][1]
    assets = C.load_assets(C.DEFAULT_ASSETS)
    key = next(iter(s_op["values"]))
    pick_id = "toolu_lab_1"
    chosen = C.pick_asset_for(assets, {"toolCallId": pick_id, "assetType": p_op.get("asset_type") or "image"}, str(pick["id"]))
    if chosen["public_url"] != s_op["values"][key]:
        raise SystemExit(f"{pick['id']}: the collector's pick ({chosen['public_url']}) is not the asset the case expects "
                         f"({s_op['values'][key]}) — the case predates the shared fixture_asset_for; regenerate it "
                         f"(souschef_lab.cases --kinds image, then `cases remeasure`) before capturing")
    with_asset = {"target": s_op["target"], "values": {key: chosen["public_url"]},
                  "assets": {key: {k: v for k, v in chosen.items() if k in ("alt", "srcset") and v is not None}}}
    plan.append({"name": "pick_asset.image.gallery.bgImage", "shape": "pick_asset → api.Asset; then ok/written", "case": pick, "steps": [
        _msg("Let's pick an image for the gallery background.",
             {"id": pick_id, "name": "pick_asset", "input": {"purpose": p_op["purpose"], "asset_type": p_op.get("asset_type") or "image"}}),
        _msg("Applying it to the gallery.", {"id": "toolu_lab_2", "name": "set_fields", "input": with_asset}, usage=(180, 60)),
        _msg("Done — the gallery has its new background image.", usage=(230, 12)),
    ]})

    # 5. fields_context on — the title case again, on the route that puts the candidate-fields block in the user turn
    plan.append({"name": "fields-context.set_field.contentBlock-1.title", "shape": "ok/written; <candidate_fields> in the user turn",
                 "case": title, "route": FIELDS_ROUTE, "fields_context": True, "steps": [
        _msg("Updating the title.", {"id": "toolu_lab_1", "name": t_op["op"], "input": _op(t_op)}),
        _msg("Done — the title now reads as asked.", usage=(160, 12)),
    ]})
    return plan


def plan_from_dir(root: Path) -> list[dict]:
    plan = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "case.json").is_file()):
        extra = json.loads((d / "plan.json").read_text(encoding="utf-8")) if (d / "plan.json").is_file() else {}
        plan.append({"name": d.name, "shape": "", "case": json.loads((d / "case.json").read_text(encoding="utf-8")),
                     "steps": FM.load_steps(d / "steps.json"), **{k: extra[k] for k in ("route", "fields_context") if k in extra}})
    return plan


# --- the service process -------------------------------------------------------------------------------------------------

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def build_binary(out: Path) -> Path:
    exe = out / "sous-chef"
    subprocess.run(["go", "build", "-o", str(exe), "./cmd/sous-chef"], cwd=str(REPO / "service"), check=True)
    return exe


def wait_http(url: str, timeout_s: float = 30.0) -> None:
    t0 = time.monotonic()
    last = ""
    while time.monotonic() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status < 500:
                    return
        except Exception as e:  # noqa: BLE001
            last = str(e)
        time.sleep(0.2)
    raise SystemExit(f"{url} did not come up in {timeout_s}s: {last}")


class Service:
    def __init__(self, binary: Path, *, port: int, model_url: str, db_url: str, log: Path):
        self.binary, self.port, self.log = binary, port, log
        self.env = {**os.environ, "PORT": str(port), "DATABASE_URL": db_url, "MIGRATE_AT_BOOT": "on", "SERVICE_KEYS": KEY,
                    # fields_context is spelled out on both: a messages-compatible route defaults to on (README §5),
                    # and the four tool_result captures are the block-less bytes.
                    "MODEL_ROUTES": json.dumps({name: {"kind": "messages-compatible", "url": model_url, "auth": "none",
                                                       "model": "fake", "max_tokens": 4096, "stream": False,
                                                       "fields_context": fc}
                                                for name, fc in ((ROUTE, "off"), (FIELDS_ROUTE, "on"))}),
                    "DEFAULT_ROUTE": ROUTE, "MODEL_ROUTE_ALLOWLIST": f"{ROUTE},{FIELDS_ROUTE}", "BUILD_ID": BUILD_ID}
        self.proc: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> "Service":
        self._log = self.log.open("w", encoding="utf-8")
        self.proc = subprocess.Popen([str(self.binary)], env=self.env, stdout=self._log, stderr=subprocess.STDOUT)
        wait_http(self.url + "/healthz")
        return self

    def __exit__(self, *exc) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self._log.close()


# --- the run ---------------------------------------------------------------------------------------------------------------

def capture(plan: list[dict], out: Path, *, binary: Path, db_url: str, engine_path: str | None, manifest: Path,
            model_port: int, port: int) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    engine = C.default_engine(engine_path, manifest)
    verifier = C.default_verifier(engine)
    assets = C.load_assets(C.DEFAULT_ASSETS)
    report = []
    fields = C.fields_for("on", manifest) if any(item.get("fields_context") for item in plan) else None
    with FM.FakeModel([], port=model_port) as fake, \
            Service(binary, port=port, model_url=fake.url, db_url=db_url, log=out / "service.log") as svc:
        version = json.loads(urllib.request.urlopen(svc.url + "/version", timeout=5).read())
        for item in plan:
            name, case, steps = item["name"], item["case"], item["steps"]
            d = out / name
            d.mkdir(exist_ok=True)
            start = len(fake.requests)
            fake.append(*steps)
            client, _ = C.connect(svc.url, KEY, manifest, [case], sse=False, timeout_s=120.0, build_id=BUILD_ID)
            t0 = time.monotonic()
            route = item.get("route") or ROUTE
            row = C.run_case(case, client, engine, route=route, assets=assets, verifier=verifier, build_id=BUILD_ID,
                             fields=fields if item.get("fields_context") else None)
            wall = time.monotonic() - t0
            reqs = fake.requests[start:]
            if fake.next != len(fake.steps):
                raise SystemExit(f"{name}: the fake model served {fake.next - start} of {len(steps)} steps")
            (d / "case.json").write_text(json.dumps(case, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            (d / "steps.json").write_text(json.dumps(steps, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            if route != ROUTE or item.get("fields_context"):
                (d / "plan.json").write_text(json.dumps({"route": route, "fields_context": bool(item.get("fields_context"))},
                                                        indent=1) + "\n", encoding="utf-8")
            (d / "collect-real.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            with (d / "model-requests.jsonl").open("w", encoding="utf-8") as f:
                for r in reqs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            summary = {"name": name, "shape": item.get("shape", ""), "route": route, "case_id": case["id"], "status": row["status"],
                       "candidate_fields_chars": len(row.get("candidate_fields") or ""),
                       "verdict": row.get("verdict"), "batches": row["batches"], "model_calls": len(reqs),
                       "tool_results": [e["data"].get("ok") for e in row["events"] if e["event"] == "tool.result"],
                       "wall_s": round(wall, 1), "service_version": version}
            report.append(summary)
            print(json.dumps(summary), file=sys.stderr)
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--default-plan", metavar="CASES.jsonl", help="build the four-shape plan from this cases file (+ fixtures/traces)")
    src.add_argument("--from", dest="from_dir", metavar="DIR", help="re-run the plan stored under DIR/*/{case,steps}.json")
    p.add_argument("--out", required=True, type=Path, help="where <name>/… lands (fixtures/prompt/real for the committed fixtures)")
    p.add_argument("--binary", type=Path, help="the sous-chef binary (default: go build into a temp dir)")
    p.add_argument("--db-url", default=os.environ.get("SOUS_CHEF_LAB_DB", DEFAULT_DB))
    p.add_argument("--engine", default=os.environ.get("ENGINE") or None, help="headless.mjs (default: $ENGINE / discovery)")
    p.add_argument("--manifest", type=Path, default=LAB / "fixtures" / "manifest.snapshot.json")
    p.add_argument("--model-port", type=int, default=0, help="fake model port (default: free)")
    p.add_argument("--port", type=int, default=0, help="service port (default: free)")
    p.add_argument("--only", action="append", metavar="NAME", help="run only this capture of the plan (repeatable)")
    a = p.parse_args(argv)
    plan = default_plan(Path(a.default_plan) if a.default_plan else None) if a.default_plan is not None else plan_from_dir(Path(a.from_dir))
    if a.only:
        plan = [item for item in plan if item["name"] in set(a.only)]
        if len(plan) != len(set(a.only)):
            raise SystemExit(f"--only: the plan has {[i['name'] for i in plan]} of {a.only}")
    tmp = Path(tempfile.mkdtemp(prefix="souschef-capture-"))
    try:
        binary = a.binary or build_binary(tmp)
        report = capture(plan, a.out, binary=binary, db_url=a.db_url, engine_path=a.engine, manifest=a.manifest,
                         model_port=a.model_port or free_port(), port=a.port or free_port())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
