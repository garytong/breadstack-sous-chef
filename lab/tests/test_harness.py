"""harness: the deployed service alone inside a job — Postgres initdb'd as
the user, the service binary on one messages-compatible route per arm with no
fallback, /version asserted. The route shape is checked here always; the real
binary runs when SOUS_CHEF_BIN names one (go build ./cmd/sous-chef) and the
Postgres server binaries are on PATH or in PG_BIN: a whole turn through the
lab's scripted fake model (tests/fake_model.py), no network, no spend."""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from souschef_lab import harness as H  # noqa: E402

REPO = LAB.parent


def test_route_is_the_deployed_inhouse_route_without_fallback():
    r = H.route_spec("https://m.run.app", "current")
    assert r == {"kind": "messages-compatible", "url": "https://m.run.app", "auth": "id-token", "model": "sous-chef",
                 "max_tokens": 1024, "stream": False, "temperature": 0.3, "prime": "on", "fast_finish": "on",
                 "report_as": "current"}
    assert "fallback" not in H.route_spec("u", "c", overrides={"fallback": "default"})


@pytest.mark.skipif(not (REPO / "infra" / "envs" / "int" / "main.tf").is_file(),
                    reason="the repository's infra/ is not here (the lab runtime image carries lab/ only)")
def test_the_route_matches_infra_int():
    """infra/envs/int/main.tf inhouse_route: the same keys, minus fallback."""
    tf = (REPO / "infra" / "envs" / "int" / "main.tf").read_text(encoding="utf-8")
    block = tf[tf.index("inhouse_route = {"): tf.index("}", tf.index("inhouse_route = {"))]
    for key in ("kind", "url", "auth", "model", "max_tokens", "stream", "temperature", "prime", "fast_finish", "report_as"):
        assert f"\n    {key}" in block, key
    assert '"id-token"' in block and 'prime       = "on"' in block and 'fast_finish = "on"' in block
    variables = (REPO / "infra" / "envs" / "int" / "variables.tf").read_text(encoding="utf-8")
    assert 'model        = "sous-chef"' in variables and "max_tokens   = 1024" in variables


def _tf_block(text: str, opener: str) -> dict:
    """`key = value` lines of an HCL object block (one level), literal values decoded, anything else kept as text."""
    start = text.index(opener)
    body = text[start + len(opener): text.index("\n  }", start)]
    out = {}
    for line in body.splitlines():
        m = re.match(r"\s*([a-z_]+)\s*=\s*(.+?)\s*(#.*)?$", line)
        if not m:
            continue
        raw = m.group(2)
        try:
            out[m.group(1)] = json.loads(raw)
        except ValueError:
            out[m.group(1)] = raw
    return out


@pytest.mark.skipif(not (REPO / "infra" / "envs" / "int" / "main.tf").is_file(),
                    reason="the repository's infra/ is not here (the lab runtime image carries lab/ only)")
def test_the_mistral_route_matches_infra_int_and_the_service_test():
    """chocolate-muffin's route (infra/envs/int/main.tf `mistral_route`, docs/CHOCOLATE-MUFFIN.md §2) is the mistral
    family's harness route — ROUTE_DEFAULTS with FAMILY_ROUTES["mistral"] on top — plus fallback and fields_context
    (on: the service's default, which the harness asserts in /version), so drift and the retrain eval ask a live
    chocolate-muffin as int's service does. service/internal/config/chocolate_test.go loads the same JSON."""
    tf = (REPO / "infra" / "envs" / "int" / "main.tf").read_text(encoding="utf-8")
    variables = (REPO / "infra" / "envs" / "int" / "variables.tf").read_text(encoding="utf-8")
    block = _tf_block(tf, "mistral_route = {")
    temp = re.search(r"temperature = optional\(number, ([0-9.]+)\)", variables[variables.index('variable "gpu_model_mistral_route"'):])
    name = re.search(r'name\s+= optional\(string, "([a-z-]+)"\)', variables[variables.index('variable "gpu_model_mistral_route"'):])
    fallback = re.search(r'fallback\s+= optional\(string, "([a-z-]+)"\)', variables[variables.index('variable "gpu_model_mistral_route"'):])
    resolved = {**block, "model": "sous-chef", "max_tokens": 1024, "temperature": float(temp.group(1)),
                "report_as": name.group(1), "fallback": fallback.group(1)}
    assert block["model"] == "var.gpu_model_route.model" and block["max_tokens"] == "var.gpu_model_route.max_tokens"
    assert block["temperature"] == "var.gpu_model_mistral_route.temperature" and name.group(1) == "chocolate-muffin"
    spec = H.route_spec("u", "x", family="mistral")
    for key in ("kind", "auth", "model", "max_tokens", "stream", "fast_finish", *H.FAMILY_ROUTES["mistral"]):
        assert resolved[key] == spec[key], key
    assert H.FAMILY_ROUTES["mistral"] == {"prime": "off", "temperature": 0.15, "parser_retries": "off"}
    assert resolved["fields_context"] == "on" and resolved["fallback"] == "default"
    go = REPO / "service" / "internal" / "config" / "chocolate_test.go"
    if go.is_file():
        const = re.search(r"const chocolateMuffin = `(.+?)`", go.read_text(encoding="utf-8")).group(1)
        route = json.loads("{" + const + "}")["chocolate-muffin"]
        assert {k: v for k, v in route.items() if k != "url"} == {k: v for k, v in resolved.items() if k != "url"}
    # the drift poll's route → family map (chatfmt.ROUTE_FAMILIES) names int's two in-house routes as Terraform names them
    from souschef_lab import chatfmt
    qwen_name = re.search(r'\n\s+name\s+= "([a-z-]+)"', variables[variables.index('variable "gpu_model_route"'):])
    assert chatfmt.ROUTE_FAMILIES == {qwen_name.group(1): "qwen", name.group(1): "mistral"}


def test_a_mistral_arm_needs_a_binary_that_honours_parser_retries(monkeypatch, tmp_path):
    info = {"current": {"kind": "messages-compatible", "fields_context": True, "fallback": None, "report_as": "current"},
            "candidate": {"kind": "messages-compatible", "fields_context": True, "fallback": None, "report_as": "candidate"}}
    h = H.Harness({"current": "https://eval", "candidate": "https://cand"}, families={"candidate": "mistral"},
                  binary=sys.executable, workdir=tmp_path / "h", port=1)
    monkeypatch.setattr(H, "_get", lambda url, headers=None, timeout=5.0: (
        200, json.dumps({"build_id": h.build_id, "route_info": info}).encode()))
    with pytest.raises(H.HarnessError, match="parser_retries None, want \"off\""):
        h.check_version()  # a binary from before the option: the candidate would redraw parser 500s unseen
    info["candidate"]["parser_retries"] = "off"
    assert h.check_version()["route_info"]["candidate"]["parser_retries"] == "off"
    q = H.Harness({"current": "https://eval"}, binary=sys.executable, workdir=tmp_path / "q", port=1)  # qwen: as before
    monkeypatch.setattr(H, "_get", lambda url, headers=None, timeout=5.0: (
        200, json.dumps({"build_id": q.build_id, "route_info": {"current": info["current"]}}).encode()))
    assert q.check_version()["route_info"]["current"]["fields_context"] is True


def test_no_binary_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.setenv("SOUS_CHEF_BIN", str(tmp_path / "nope"))
    monkeypatch.setattr(H, "DEFAULT_BIN", tmp_path / "nope-either", raising=False)
    with pytest.raises(H.HarnessError, match="SOUS_CHEF_BIN"):
        H.Harness({"current": "http://127.0.0.1:1"})


def _pg_ok() -> bool:
    try:
        H.pg_bin_dir()
        return True
    except H.HarnessError:
        return False


REAL = bool(os.environ.get("SOUS_CHEF_BIN")) and _pg_ok() and (os.getuid() != 0)


@pytest.mark.skipif(not REAL, reason="needs SOUS_CHEF_BIN (a built service) and Postgres server binaries (PG_BIN)")
def test_a_turn_through_the_real_binary(tmp_path):
    from souschef_lab import collect as C
    from souschef_lab.exam import MANIFEST_FILE
    from tests.fake_model import FakeModel

    steps = [{"message": {"content": [{"type": "text", "text": "Done — nothing to change."}],
                          "usage": {"input_tokens": 10, "output_tokens": 3}}}]
    with FakeModel(steps) as fm:
        with H.Harness({"current": fm.url, "candidate": fm.url}, run_id="t", workdir=tmp_path, auth="none",
                       route_overrides={"prime": "off"}) as h:
            v = h.version
            assert set(v["route_info"]) == {"current", "candidate"} and v["build_id"].endswith("-harness-t")
            assert h.key not in json.dumps(h.describe())
            client, hsh = C.connect(h.url, h.key, MANIFEST_FILE, [], sse=False, timeout_s=60, build_id="lab-drift")
            case = {"id": "harness-1", "request": "hello", "context": {"page": "home", "sections": []}, "site_state": {},
                    "expected": None, "template": "vintage", "manifest_hash": hsh}
            fields, rec = C.resolve_fields("auto", MANIFEST_FILE, client, "current")  # the service says: on
            assert rec["fields_context"] is True and rec["route_info"]["fallback"] is None
            row = C.run_case(case, client, lambda s, o, c: {"ok": True, "results": [], "changes": [], "state": s},
                             route="current", site_prefix="lab-harness-", fields=fields)
            assert row["status"] == "ok", row
            assert fm.requests and fm.requests[0]["body"]["model"] == "sous-chef"
            assert fm.requests[0]["body"].get("thinking") == {"type": "disabled"}
        assert (tmp_path / "service.log").is_file()
    assert not shutil.which("nonexistent-binary")
