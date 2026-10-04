"""Verifier (LAB.md §3): a trace's browser ops, replayed through the engine on
the case's fresh state, compared leaf by leaf with the case's measured
expected diff.

Input: a trace — a collector row (LAB.md §4), a Postgres ``traces.events``
array, or a fixture under ``lab/fixtures/traces/`` — and its case (LAB.md §2).

1. ``tool.call`` events in ``seq`` order, minus ``report_unsupported``,
   ``pick_asset`` and the server tools (``lookup_fields``,
   ``describe_target``), are the ops: ``name`` → ``op``, ``input`` → fields.
2. Replay through ``engine.run(case.site_state, ops, ctx)`` on a fresh copy
   of the case state with the golden ctx (``GOLDEN_CTX``).
3. Normalise both diffs to sets of ``(path, after)``: indexes kept, a
   removed leaf compares as ``after=None``; the hero mirror collapses in
   BOTH directions (a flat ``settings.<k…>`` and ``settings.slides[0].<k…>``
   are one write, nested companions such as ``backgroundImageSrcSet.800``
   included — ``align_mirror``); image companions (``<k>Alt``, ``<k>SrcSet``,
   ``<k>SrcSet.<w>``) of a primary both sides wrote are set aside and
   listed in ``companions`` — tolerated, never a failure — so the verdict
   rests on the image URL itself (``set_aside_companions``). Sets, so
   independent ops pass in any order; op SHAPE is never compared (a
   ``set_fields`` and an ``edit_list update slides[0]`` of a hero produce
   the same leaves and pass alike).
   A ``pick_asset`` case is compared against the asset the picker actually
   returned (the trace's ``tool.result`` asset): when it is not the asset
   the case was generated with, the expected ops are rewritten with it and
   re-measured through the engine (``substitute_picked``; info
   ``pick_substituted``) — exact by construction, no diff rewriting.
4. Verdict = the first of ``VERDICTS`` that applies; every reason that
   applies is kept in ``reasons``; ``info`` carries the non-failure notes
   (``companions``, ``pick_substituted``, ``pick_remeasure_failed``).

Exact by construction: no figure extraction or grounding heuristics
(thermometer's ``lab/verify.py`` is replaced, not copied).

Engine interface (LAB.md §1 ``engine.py``): an object with
``run(state, ops, ctx) -> dict`` returning the headless runner's output —
``{ok, results:[{op, ok, error?, warnings, written}], changes:[{slice,
written, diff:[{path, before?, after?}]}]}`` — plus, when the process
failed, ``exit_code`` (1 = usage/load error) and ``error``. ``RecordedEngine``
replays saved headless output for tests that must run without node.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

RULE_VERSION = "verify-2"  # bump when a verdict rule changes; passk refuses runs that mix versions
# verify-2 (2026-10-04): two-way hero mirror collapse incl. nested companions, image companion tolerance,
# pick_asset substitution + re-measure. verify-1 failed every correct hero image / pick write (LAB.md §3).

# Golden ctx (cookie-cutter ops/golden.test.js:97; LAB.md §3 step 2). `now`
# makes minted ids deterministic, `cdnUrl`/`galleryFolder` make image
# compaction (`BREADSTACK_CDN/…`) deterministic.
GOLDEN_CTX = {"cdnUrl": "https://cdn.int.breadstack.io", "galleryFolder": "gal", "now": 1700000000000}
DEFAULT_TEMPLATE = "editorial"  # a real BASIC_TEMPLATE_IDS member (headless.mjs header, R8)

SERVER_TOOLS = ("lookup_fields", "describe_target")
NON_OPS = ("report_unsupported", "pick_asset") + SERVER_TOOLS
BROWSER_OPS = ("set_fields", "edit_list", "reset", "restore_default", "set_page_layout", "reset_pinned_section")

VERDICTS = ("invalid_op", "unsupported_ok", "clarify_ok", "no_answer", "wrong_value", "missing_writes",
            "extra_writes", "pass")
PASSING = ("pass", "clarify_ok", "unsupported_ok")
FAILING = tuple(v for v in VERDICTS if v not in PASSING)

# The hero mirror: `settings.<k>` and `settings.slides[0].<k>` are written
# together (ops README §2 rule 6) for every key in the hero's `mirror.keys`,
# companions included (`backgroundImageSrcSet.800`, hence `rest`, not a bare
# key); the verifier keeps the slide path.
MIRROR_RE = re.compile(r"^(?P<prefix>.*\.settings)\.slides\[0\]\.(?P<rest>.+)$")
# An image companion leaf (transforms.js applyFieldWrites): `<primary>Alt`,
# `<primary>SrcSet`, `<primary>SrcSet.<width>`; parallel list companions carry
# the index after the suffix (`imagesAlt[0]` beside `images[0]`).
COMPANION_RE = re.compile(r"^(?P<base>.+?)(?P<kind>Alt|SrcSet)(?P<idx>\[\d+\])?(?P<width>\.[^.\[]+)?$")
INFO_NOTES = ("companions", "pick_substituted", "pick_remeasure_failed")


class Engine(Protocol):
    def run(self, state: dict, ops: list[dict], ctx: dict) -> dict: ...


class EngineUnavailable(RuntimeError):
    pass


def run_engine(engine: Engine, state: dict, ops: list[dict], ctx: dict) -> dict:
    """``engine.run`` as a plain dict: the headless output plus ``exit_code``.
    Accepts souschef_lab.engine's ``EngineResult`` (``to_dict()`` +
    ``returncode``) and turns its ``EngineError`` (exit 1 / no JSON) into
    ``{exit_code: 1, error}`` so the verdict is ``invalid_op``, never a crash
    of the whole build."""
    try:
        result = engine.run(state, ops, ctx)
    except Exception as e:  # noqa: BLE001 — only the engine's own error class is swallowed
        if type(e).__name__ != "EngineError":
            raise
        return {"ok": False, "exit_code": 1, "error": str(e), "results": [], "changes": []}
    if isinstance(result, dict):
        return result
    out = result.to_dict() if hasattr(result, "to_dict") else dict(vars(result))
    out.setdefault("exit_code", getattr(result, "returncode", 0 if out.get("ok") else 2))
    return out


def default_engine(manifest: str | Path | None = None) -> Engine:
    """The subprocess engine (``souschef_lab.engine``, LAB.md §1) — on
    ``manifest`` when given (the manifest the cases were generated from; the
    fixture snapshot otherwise). Imported lazily; callers may pass an
    ``Engine`` explicitly (``RecordedEngine`` in tests)."""
    try:
        from . import engine as mod  # type: ignore[attr-defined]
    except ImportError as e:  # TODO(integrator): remove the fallback message once engine.py exists
        raise EngineUnavailable("souschef_lab.engine is not available: pass engine= (see LAB.md §1)") from e
    factory = getattr(mod, "Engine", None)
    if not callable(factory):
        return mod  # type: ignore[return-value]
    try:
        return factory(manifest_path=manifest) if manifest else factory()
    except Exception as e:  # noqa: BLE001 — EngineError: no node / no headless.mjs found
        if type(e).__name__ != "EngineError":
            raise
        raise EngineUnavailable(str(e)) from e


def canonical(v: Any) -> str:
    """Canonical JSON (sorted keys, no whitespace): two values are the same
    write when these are equal. Copied from thermometer lab/bestofn.py."""
    return json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


# --- events -----------------------------------------------------------------------------------------------------------

def events_of(trace: dict | list) -> list[dict]:
    """The events of a trace in ``seq`` order (stable where ``seq`` is absent:
    heartbeats carry none)."""
    evs = trace if isinstance(trace, list) else (trace.get("events") or [])
    return sorted((e for e in evs if isinstance(e, dict)), key=lambda e: (e.get("seq") is None, e.get("seq") or 0))


def _name(e: dict) -> str:
    return str(e.get("event") or e.get("type") or "")


def tool_calls(events: Iterable[dict]) -> list[dict]:
    """Every ``tool.call``: ``{id, name, input, seq}``."""
    out = []
    for e in events_of(list(events)):
        if _name(e) == "tool.call":
            d = e.get("data") or {}
            inp = d.get("input") if isinstance(d.get("input"), dict) else {}
            out.append({"id": str(d.get("id") or ""), "name": str(d.get("name") or ""), "input": inp,
                        "seq": e.get("seq")})
    return out


def ops_of(events: Iterable[dict]) -> list[dict]:
    """The browser ops a trace asked for: ``tool.call`` minus NON_OPS,
    ``{op: name, **input}`` (README §1.2: the mapping is the identity)."""
    return [{"op": c["name"], **c["input"]} for c in tool_calls(events) if c["name"] not in NON_OPS]


def server_calls(events: Iterable[dict]) -> int:
    return sum(1 for c in tool_calls(events) if c["name"] in SERVER_TOOLS)


def saw_tool(events: Iterable[dict], name: str) -> bool:
    return any(c["name"] == name for c in tool_calls(events))


def final_text(events: Iterable[dict]) -> str:
    """The reply after the last batch: the ``text.delta`` events after the
    last ``turn.waiting``/``turn.suspended`` (all of them when there was no
    batch)."""
    parts: list[str] = []
    for e in events_of(list(events)):
        n = _name(e)
        if n in ("turn.waiting", "turn.suspended"):
            parts = []
        elif n == "text.delta":
            parts.append(str((e.get("data") or {}).get("text") or ""))
    return "".join(parts)


def all_text(events: Iterable[dict]) -> str:
    return "".join(str((e.get("data") or {}).get("text") or "") for e in events_of(list(events)) if _name(e) == "text.delta")


def outcome_of(events: Iterable[dict]) -> dict:
    """``{done, errored, stop, error_code}`` from the turn's terminal events."""
    out = {"done": False, "errored": False, "stop": None, "error_code": None}
    for e in events_of(list(events)):
        n = _name(e)
        if n == "turn.done":
            out["done"] = True
            out["stop"] = (e.get("data") or {}).get("stop")
        elif n == "turn.error":
            out["errored"] = True
            out["error_code"] = (e.get("data") or {}).get("code")
    return out


def warnings_of(engine_output: dict) -> list[str]:
    return [str(w) for r in engine_output.get("results") or [] for w in (r.get("warnings") or [])]


def engine_errors(engine_output: dict) -> list[dict]:
    errs = [r.get("error") or {"code": "E_UNKNOWN"} for r in engine_output.get("results") or [] if not r.get("ok")]
    if engine_output.get("exit_code") == 1 or engine_output.get("error"):
        errs.append({"code": "E_ENGINE", "message": str(engine_output.get("error") or "engine exited 1")})
    return errs


# --- diff normalisation -----------------------------------------------------------------------------------------------

def leaves_of(engine_output: dict) -> list[dict]:
    """Every ``diff`` leaf of every changed slice."""
    return [d for c in engine_output.get("changes") or [] for d in (c.get("diff") or [])]


def normalise(leaves: Iterable[dict]) -> dict[str, str]:
    """``{path: canonical(after)}`` — a removed leaf (no ``after``) is
    ``"null"``; a hero flat leaf whose ``slides[0]`` twin carries the same
    value collapses onto the slide path (so a flat ``set_fields`` and an
    ``edit_list update slides[0]`` compare equal — the engine writes both
    leaves for either form, measured 2026-10-01), nested companions
    included (``backgroundImageSrcSet.800``; before 2026-10-04 only a bare
    key collapsed and the flat srcset leaves surfaced as extra writes).
    Companions and indexes are kept here: ``align_mirror`` and
    ``set_aside_companions`` work on the pair."""
    d: dict[str, str] = {}
    for leaf in leaves:
        d[str(leaf.get("path") or "")] = canonical(leaf.get("after"))
    for path in list(d):
        m = MIRROR_RE.match(path)
        if m:
            flat = f"{m.group('prefix')}.{m.group('rest')}"
            if flat in d and d[flat] == d[path]:
                del d[flat]
    return d


def diff_set(leaves: Iterable[dict]) -> set[tuple[str, str]]:
    return set(normalise(leaves).items())


def _first_key(rest: str) -> str:
    return re.split(r"[.\[]", rest, maxsplit=1)[0]


def align_mirror(expected: dict[str, str], actual: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """The hero mirror in both directions: a section is a mirrored hero when
    either side wrote under its ``settings.slides[0]``; on such a section a
    flat ``settings.<k…>`` whose first key was seen under ``slides[0]`` on
    either side IS the ``slides[0].<k…>`` write — it moves onto the slide
    path when that path is absent or carries the same value (a slide path
    with a different value keeps both, so a real disagreement still shows).
    ``normalise`` only collapsed a flat leaf beside its own equal twin; a
    flat write that one side alone produced stayed an extra/missing path."""
    mirrored: dict[str, set[str]] = {}
    for d in (expected, actual):
        for p in d:
            m = MIRROR_RE.match(p)
            if m:
                mirrored.setdefault(m.group("prefix"), set()).add(_first_key(m.group("rest")))
    out = []
    for d in (expected, actual):
        d = dict(d)
        for path in list(d):
            for prefix, keys in mirrored.items():
                if not path.startswith(prefix + ".") or path.startswith(prefix + ".slides["):
                    continue
                rest = path[len(prefix) + 1:]
                if _first_key(rest) in keys:
                    slide = f"{prefix}.slides[0].{rest}"
                    if slide not in d or d[slide] == d[path]:
                        d[slide] = d.pop(path)
                break
        out.append(d)
    return out[0], out[1]


def companion_of(path: str) -> str | None:
    """The image primary a companion leaf belongs to (``…bgImageAlt`` /
    ``…bgImageSrcSet.480w`` → ``…bgImage``; ``…imagesAlt[0]`` →
    ``…images[0]``), or None for any other leaf."""
    m = COMPANION_RE.match(path)
    if not m:
        return None
    return f"{m.group('base')}{m.group('idx') or ''}"


def set_aside_companions(expected: dict[str, str], actual: dict[str, str]) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """Companion tolerance: every alt/srcset companion leaf whose image
    primary BOTH sides wrote is removed from the comparison and returned
    (sorted) as the third member. Matching primaries then pass whatever the
    companions say (present, absent, different — the case's asset metadata
    and the model's may legitimately differ, LAB.md §3); differing
    primaries are ``wrong_value`` on the URL itself rather than
    missing/extra srcset widths. A companion without its primary on both
    sides is kept and compared as any leaf."""
    exp, act = dict(expected), dict(actual)
    aside = [p for p in sorted(set(exp) | set(act))
             if (primary := companion_of(p)) is not None and primary in exp and primary in act]
    for p in aside:
        exp.pop(p, None)
        act.pop(p, None)
    return exp, act, aside


def compare(expected: dict[str, str], actual: dict[str, str]) -> list[str]:
    """The diff reasons, in verdict order: wrong_value (same paths, a value
    differs), missing_writes, extra_writes."""
    reasons = []
    ep, ap = set(expected), set(actual)
    if ep == ap and expected != actual:
        reasons.append("wrong_value")
    if ep - ap:
        reasons.append("missing_writes")
    if ap - ep:
        reasons.append("extra_writes")
    return reasons


# --- pick_asset -------------------------------------------------------------------------------------------------------

def picked_assets_of(trace: dict | list) -> list[dict]:
    """The assets the picker answered this turn's ``pick_asset`` calls with:
    the ``tool.result`` events carrying an ``asset`` (the collector's and the
    service trace's shape, LAB.md §4), else a collector row's
    ``picked_assets``."""
    out = [a for e in events_of(trace) if _name(e) == "tool.result"
           and isinstance(a := (e.get("data") or {}).get("asset"), dict) and a.get("public_url")]
    if not out and isinstance(trace, dict):
        out = [a for a in trace.get("picked_assets") or [] if isinstance(a, dict) and a.get("public_url")]
    return out


def substitute_picked(ops: list[dict], asset: dict) -> tuple[list[dict], bool]:
    """The expected browser ops with ``asset`` in place of the asset the
    case was generated with: for every image key an op names in ``assets``,
    ``values[key]`` / ``item[key]`` becomes the asset's ``public_url`` and
    ``assets[key]`` its ``{alt, srcset}`` (alt '' when the picker sent none,
    as ImageSelector does). Second member: whether any URL changed."""
    url = asset.get("public_url")
    companions = {"alt": asset.get("alt") or "", "srcset": asset.get("srcset")}
    out, changed = [], False
    for op in ops:
        op = copy.deepcopy(op)
        if isinstance(op.get("assets"), dict):
            for key in list(op["assets"]):
                for member in ("values", "item"):
                    if isinstance(op.get(member), dict) and key in op[member] and op[member][key] != url:
                        op[member][key] = url
                        changed = True
                op["assets"][key] = dict(companions)
        out.append(op)
    return out, changed


# --- clarify ----------------------------------------------------------------------------------------------------------

def candidates_of(case: dict) -> list[tuple[str, ...]]:
    """What a clarifying reply may name: per section of the case context,
    its id and title (either counts the section once); ``meta.candidates``
    when the generator spelled them."""
    meta = case.get("meta") or {}
    if meta.get("candidates"):
        return [tuple(str(x) for x in (c if isinstance(c, (list, tuple)) else [c])) for c in meta["candidates"]]
    out = []
    for s in (case.get("context") or {}).get("sections") or []:
        names = tuple(str(s[k]) for k in ("id", "title") if s.get(k))
        if names:
            out.append(names)
    return out


def is_clarifying(text: str, candidates: list[tuple[str, ...]], minimum: int = 2) -> bool:
    """A question mark and at least ``minimum`` candidates named."""
    if "?" not in text:
        return False
    t = text.casefold()
    named = sum(1 for names in candidates if any(n.casefold() in t for n in names))
    return named >= minimum


# --- verdict ----------------------------------------------------------------------------------------------------------

def ctx_of(case: dict) -> dict:
    """The replay ctx: the case template (``site_state.template.id``, else
    ``case.template``, else editorial) plus GOLDEN_CTX, the context's
    locations and — as ``collect.ctx_for`` sends it live — ``currentPage`` =
    the case's ``context.page``. The engine fills a section/page target that
    omits ``page`` from ``currentPage`` (headless.mjs, like the browser host,
    since 2026-10-03); without it here a bare target the collector's engine
    accepted would be refused E_TARGET on replay and a correct turn would
    read ``invalid_op``."""
    state = case.get("site_state") or {}
    tid = ((state.get("template") or {}).get("id")) or case.get("template") or DEFAULT_TEMPLATE
    ctx = {"templateId": tid, **GOLDEN_CTX}
    context = case.get("context") or {}
    locs = context.get("locations")
    if locs:
        ctx["locations"] = [{"id": str(l.get("id"))} for l in locs if isinstance(l, dict) and l.get("id")]
    if context.get("page"):
        ctx["currentPage"] = context["page"]
    return ctx


@dataclass
class Verdict:
    verdict: str
    reasons: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)  # non-failure notes (INFO_NOTES)
    companions: list[str] = field(default_factory=list)  # companion leaves set aside (set_aside_companions)
    ops: list[dict] = field(default_factory=list)
    expected_diff: dict[str, str] = field(default_factory=dict)
    actual_diff: dict[str, str] = field(default_factory=dict)
    engine_errors: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    server_calls: int = 0
    final_text_chars: int = 0
    stop: str | None = None
    rule_version: str = RULE_VERSION
    # The engine that replayed the ops, and the one that measured the case's expected diff (LAB.md §2); when they
    # differ the expected diff may be stale — `cases remeasure` refreshes it.
    engine_fingerprint: str | None = None
    case_engine_fingerprint: str | None = None

    @property
    def passed(self) -> bool:
        return self.verdict in PASSING

    @property
    def engine_mismatch(self) -> bool:
        return bool(self.engine_fingerprint and self.case_engine_fingerprint
                    and self.engine_fingerprint != self.case_engine_fingerprint)

    def to_dict(self) -> dict:
        """Counts, paths and values only — never the reply's text."""
        return {"verdict": self.verdict, "reasons": list(self.reasons), "passed": self.passed,
                "info": list(self.info), "companions": list(self.companions),
                "ops": [o.get("op") for o in self.ops],
                "expected_diff": [list(x) for x in sorted(self.expected_diff.items())],
                "actual_diff": [list(x) for x in sorted(self.actual_diff.items())],
                "engine_errors": list(self.engine_errors), "warnings": list(self.warnings),
                "server_calls": self.server_calls, "final_text_chars": self.final_text_chars, "stop": self.stop,
                "rule_version": self.rule_version, "engine_fingerprint": self.engine_fingerprint,
                "case_engine_fingerprint": self.case_engine_fingerprint, "engine_mismatch": self.engine_mismatch}


def verify(trace: dict | list, case: dict, engine: Engine | None = None) -> Verdict:
    """The verdict of one trace on its case (module docstring)."""
    events = events_of(trace)
    kind = str(case.get("kind") or "")
    expected_ops = (case.get("expected") or {}).get("ops") or []
    expected = normalise((case.get("expected") or {}).get("diff") or [])
    ops = ops_of(events)
    out = outcome_of(events)
    completed = out["done"] and not out["errored"] and out["stop"] == "end_turn"
    text = final_text(events)
    reasons: list[str] = []
    info: list[str] = []

    picked = picked_assets_of(trace) if any(o.get("op") == "pick_asset" for o in expected_ops) else []
    if picked:
        # The case expects the asset its generator chose; the model wrote the one the picker returned. When they
        # differ, the expectation follows the picker: same ops, picked asset, measured again by the same engine.
        substituted, changed = substitute_picked([o for o in expected_ops if o.get("op") != "pick_asset"], picked[0])
        if changed:
            engine = engine or default_engine()
            again = run_engine(engine, copy.deepcopy(case.get("site_state") or {}), substituted, ctx_of(case))
            if engine_errors(again):
                info.append("pick_remeasure_failed")
            else:
                expected = normalise(leaves_of(again))
                info.append("pick_substituted")

    if ops:
        engine = engine or default_engine()
        run = run_engine(engine, copy.deepcopy(case.get("site_state") or {}), copy.deepcopy(ops), ctx_of(case))
    else:
        run = {"ok": True, "results": [], "changes": []}
    errors = engine_errors(run)
    if errors:
        reasons.append("invalid_op")
    unsupported_seen = saw_tool(events, "report_unsupported")
    if kind == "unsupported" and not ops and unsupported_seen and completed:
        reasons.append("unsupported_ok")
    if kind == "clarify" and not ops and completed and is_clarifying(text, candidates_of(case)):
        reasons.append("clarify_ok")
    if not completed:
        reasons.append("no_answer")
        if out["errored"]:
            reasons.append(f"turn_error:{out['error_code'] or ''}")
        elif not out["done"]:
            reasons.append("no_turn_done")
        else:
            reasons.append(f"stop:{out['stop']}")
    if not ops and expected_ops:
        reasons.append("no_answer")
    # Tightenings beyond LAB.md §3's list: a clarify case that neither asked
    # nor acted, and an unsupported case that neither reported nor acted,
    # are no_answer (never a vacuous pass of two empty sets).
    if kind == "clarify" and not ops and "clarify_ok" not in reasons:
        reasons += ["no_answer", "not_clarifying"]
    if kind == "unsupported" and not ops and "unsupported_ok" not in reasons:
        reasons += ["no_answer", "not_reported"]
    expected, actual = align_mirror(expected, normalise(leaves_of(run)))
    exp_cmp, act_cmp, companions = set_aside_companions(expected, actual)
    if companions:
        info.append("companions")
    reasons += compare(exp_cmp, act_cmp)
    seen: list[str] = []
    for r in reasons:
        if r not in seen:
            seen.append(r)
    verdict = next((v for v in VERDICTS if v in seen), "pass")
    fp = getattr(engine, "fingerprint", None) if engine is not None else None
    return Verdict(verdict=verdict, reasons=seen, info=info, companions=companions, ops=ops,
                   expected_diff=expected, actual_diff=actual,
                   engine_errors=errors, warnings=warnings_of(run), server_calls=server_calls(events),
                   final_text_chars=len(text), stop=out["stop"], engine_fingerprint=str(fp) if fp else None,
                   case_engine_fingerprint=case.get("engine_fingerprint") or None)


# --- recorded engine --------------------------------------------------------------------------------------------------

class RecordedEngine:
    """Replays saved headless output (``lab/fixtures/traces/engine/recorded.json``:
    ``{state, ctx, runs:{key:{ops, output}}}``) keyed by the canonical ops.
    For tests that must run without node; an unrecorded batch is a
    ``KeyError`` naming the ops, never a guess."""

    def __init__(self, recorded: dict | str | Path):
        if not isinstance(recorded, dict):
            recorded = json.loads(Path(recorded).read_text(encoding="utf-8"))
        self.state = recorded.get("state") or {}
        self.ctx = recorded.get("ctx") or {}
        self.runs = {canonical(r["ops"]): r["output"] for r in (recorded.get("runs") or {}).values()}
        self.calls: list[dict] = []

    def run(self, state: dict, ops: list[dict], ctx: dict) -> dict:
        self.calls.append({"state": state, "ops": ops, "ctx": ctx})
        key = canonical(ops)
        if key not in self.runs:
            raise KeyError(f"no recorded engine output for ops {key[:200]}")
        return copy.deepcopy(self.runs[key])


# --- fixtures ---------------------------------------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "traces"


def load_fixture(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fixture_events(fx: dict) -> list[dict]:
    """The scripted streams flattened, with one ``tool.result`` event per
    posted result item after the stream it answers (``data`` = the item plus
    ``for_seq`` of the ``tool.call`` it answers — traces.events' shape,
    README §4), numbered from that stream's last seq: the service continues
    the next stream's seq after them, so the fixtures leave the gap."""
    streams = fx.get("streams") or []
    bodies = fx.get("tool_results") or []
    events: list[dict] = []
    turn_id = next((e.get("turnId") for s in streams for e in s if e.get("turnId")), None)
    for k, stream in enumerate(streams):
        events += [dict(e) for e in stream]
        if k >= len(bodies):
            continue
        by_id = {c["id"]: c["seq"] for c in tool_calls(events)}
        seq = max((e.get("seq") or 0 for e in stream), default=0)
        for item in bodies[k].get("results") or []:
            seq += 1
            data = dict(item, for_seq=by_id.get(str(item.get("toolCallId"))))
            events.append({"event": "tool.result", "data": data, "seq": seq, **({"turnId": turn_id} if turn_id else {})})
    return events_of(events)


def trace_from_fixture(fx: dict, sample: int = 1, model: str = "default") -> dict:
    """A collector row (LAB.md §4) built from a fixture: what collect.py would
    have logged had the fake service replayed these streams."""
    case = fx["case"]
    events = fixture_events(fx)
    out = outcome_of(events)
    done = next((e for e in events if _name(e) == "turn.done"), None)
    row = {
        "id": f"{case['id']}#{sample}", "case_id": case["id"], "sample": sample, "model": model,
        "status": "ok" if out["done"] or out["errored"] else "error",
        "http_status": 200, "code": out["error_code"], "error": out["error_code"] if out["errored"] else None,
        "latency_ms": None, "context": case.get("context"), "text": case.get("request"),
        "events": events, "ops": ops_of(events), "tools": fx.get("tools") or [],
        "usage": ((done or {}).get("data") or {}).get("usage"), "stop": out["stop"],
        "turn_id": next((e.get("turnId") for e in events if e.get("turnId")), None), "conversation_id": None,
        "at": fx.get("at"),
        "meta": {"kind": case.get("kind"), "section": (case.get("target") or {}).get("sectionId"),
                 "template": case.get("template"), "field_type": (case.get("meta") or {}).get("field_type"),
                 "holdout": bool(case.get("holdout")), "manifest_hash": case.get("manifest_hash"),
                 "buildId": fx.get("buildId")},
    }
    return row


def holdout_of(field_key: str, pct: int = 20) -> bool:
    """LAB.md §2: ``sha1(field_key) % 100 < 20``; field_key =
    ``<container>.<field or list path>``. Independent of wording."""
    return int(hashlib.sha1(field_key.encode("utf-8")).hexdigest(), 16) % 100 < pct


# --- CLI --------------------------------------------------------------------------------------------------------------

def read_jsonl(path: str | Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="re-verify collector rows against their cases (LAB.md §3)")
    p.add_argument("--log", action="append", required=True, help="collector log (jsonl), repeatable")
    p.add_argument("--cases", required=True, help="cases.jsonl")
    p.add_argument("--out", help="write one verdict per row here (jsonl)")
    p.add_argument("--recorded", help="replay recorded engine output (tests) instead of the subprocess engine")
    p.add_argument("--manifest", help="the manifest the cases were generated from (default: fixtures/manifest.snapshot.json)")
    args = p.parse_args(argv)
    cases = {c["id"]: c for c in read_jsonl(args.cases)}
    engine: Engine = RecordedEngine(args.recorded) if args.recorded else default_engine(args.manifest)
    rows = [r for path in args.log for r in read_jsonl(path)]
    hist: dict[str, int] = {}
    out_rows = []
    mismatched: dict[str, str] = {}
    for r in rows:
        case = cases.get(str(r.get("case_id")))
        if case is None:
            hist["no_case"] = hist.get("no_case", 0) + 1
            continue
        v = verify(r, case, engine)
        hist[v.verdict] = hist.get(v.verdict, 0) + 1
        if v.engine_mismatch:
            mismatched[case["id"]] = str(v.case_engine_fingerprint)
        out_rows.append({"id": r.get("id"), "case_id": case["id"], **v.to_dict()})
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for row in out_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    for k in [*VERDICTS, "no_case"]:
        if hist.get(k):
            print(f"{k:16} {hist[k]}")
    fp = getattr(engine, "fingerprint", None)
    print(f"rows {len(rows)} · rule {RULE_VERSION}" + (f" · engine {str(fp)[:12]}" if fp else ""))
    if mismatched:
        olds = sorted({v[:12] for v in mismatched.values()})
        print(f"warning: {len(mismatched)} case(s) were measured with another engine ({', '.join(olds)} vs current "
              f"{str(fp)[:12]}); their expected diffs may be stale — run `souschef-lab cases remeasure --cases {args.cases}`",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
