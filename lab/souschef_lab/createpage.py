"""create_page in the lab: the browser's slug rule and its same-batch stop.

Mirrors, read-only:

* cookie-cutter ``launcher/client/src/lib/pageSlug.js`` ``slugify`` — the one
  slug rule the Pages tab and the engine's ``create_page`` share (the
  service's Go copy is ``toolgen.Slugify``);
* ``launcher/client/src/sousChef/client/turns.js`` ``pageOf``,
  ``createPageSlugs`` and ``applyOps``: when a ``create_page`` of a batch is
  refused, every LATER call of the same batch aimed (``target.page``,
  ``page`` or ``for.target.page``) at the slug it passed or at its
  canonical slug (``slugify(slug || title)``) is answered ``E_TARGET``
  "page "x" was not created (create_page was refused earlier in this
  reply), so this op was not applied" and never reaches the engine. The
  service applies the same rule to a create IT refuses (``loop/turn.go``
  ``dispatch`` + ``toolgen.CreatePageSlugs``); those calls are never
  emitted, so the lab's browser emulator only meets the engine-refused
  case.

``run_batch`` applies a batch the way the browser does with that stop: the
headless engine takes the whole batch at once, so a batch whose create was
refused is run again without the stopped calls and their results are put
back in call order. Only ``create_page`` writes ``customPages`` and none of
the stopped calls is a ``create_page`` (it names no page), so taking them
out cannot change any create's outcome: two runs at most, and one — the
same single call as before — for a batch without a refused create.

Standard library only.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

CREATE_PAGE = "create_page"
E_TARGET = "E_TARGET"

# JavaScript's `\s` (and what String.prototype.trim strips): WhiteSpace + LineTerminator.
JS_WS = "\t\n\v\f\r    -     　﻿"
_NOT_SLUG_RE = re.compile(f"[^a-z0-9{JS_WS}-]")
_WS_RUN_RE = re.compile(f"[{JS_WS}]+")
_DASH_RUN_RE = re.compile("-+")
_TRIM_RE = re.compile(f"^[{JS_WS}]+|[{JS_WS}]+$")


def js_trim(s: str) -> str:
    """String.prototype.trim."""
    return _TRIM_RE.sub("", s)


def slugify(s: Any) -> str:
    """pageSlug.slugify: ``String(s || '')`` lowercased, everything but a-z,
    0-9, whitespace and ``-`` removed, trimmed, whitespace runs → ``-``,
    hyphen runs → one ``-`` (leading / trailing hyphens are kept, as there)."""
    if not isinstance(s, str) or not s:
        return ""
    out = _NOT_SLUG_RE.sub("", s.lower())
    return _DASH_RUN_RE.sub("-", _WS_RUN_RE.sub("-", js_trim(out)))


def page_of(op: Any) -> str | None:
    """turns.pageOf: ``op.target.page ?? op.page ?? op.for.target.page`` when
    it is a non-empty string, else None (no page named: the viewed page)."""
    if not isinstance(op, dict):
        return None

    def get(obj: Any, *keys: str) -> Any:
        for k in keys:
            if not isinstance(obj, dict):
                return None
            obj = obj.get(k)
        return obj

    for p in (get(op, "target", "page"), op.get("page"), get(op, "for", "target", "page")):
        if p is not None:  # `??`: the first non-nullish value decides, even an empty string
            return p if isinstance(p, str) and p else None
    return None


def create_page_slugs(op: Any) -> list[str]:
    """turns.createPageSlugs: the slug as passed (when a non-blank string) and
    ``slugify(slug || title)``, each once, in that order."""
    if not isinstance(op, dict):
        return []
    slug = op.get("slug")
    given = slug if isinstance(slug, str) and js_trim(slug) else None
    title = op.get("title")
    out: list[str] = []
    for s in (given, slugify(given or (title if isinstance(title, str) else ""))):
        if s and s not in out:
            out.append(s)
    return out


def not_created_message(slug: str) -> str:
    """The stop's message, byte for byte (turns.js applyOps; loop/turn.go dispatch)."""
    return (f"page {json.dumps(slug, ensure_ascii=False)} was not created "
            "(create_page was refused earlier in this reply), so this op was not applied")


def stopped_result(op: dict, slug: str) -> dict:
    """A stopped call's result in headless.mjs's result shape."""
    return {"op": op, "ok": False, "error": {"code": E_TARGET, "message": not_created_message(slug)},
            "warnings": [], "written": []}


def stopped_calls(ops: list[dict], results: dict[int, dict]) -> dict[int, str]:
    """Index → the slug it aimed at, for every call the browser stops: walk
    the batch in order; a call aimed at a slug of an earlier refused
    create_page is stopped; a create_page whose result (``results[i]``) is a
    refusal adds its slugs."""
    not_created: set[str] = set()
    out: dict[int, str] = {}
    for i, op in enumerate(ops):
        aimed = page_of(op)
        if aimed and aimed in not_created:
            out[i] = aimed
            continue
        r = results.get(i)
        if isinstance(op, dict) and op.get("op") == CREATE_PAGE and isinstance(r, dict) and not r.get("ok"):
            not_created.update(create_page_slugs(op))
    return out


def run_batch(run: Callable[[list[dict]], dict], ops: list[dict]) -> dict:
    """``run(ops)`` (headless.mjs's output shape: ``{ok, results, changes,
    claims?, state?, exit_code?}``) with the browser's same-batch stop after
    a refused create_page (module docstring). The returned ``results`` are
    one per op of ``ops``, in order; ``ok`` is false (and an ``exit_code`` of
    0 becomes 2, headless's "an op failed") when a call was stopped. A run
    without per-op results (the engine failed to load) is returned as is."""
    out = run(ops)
    if not any(isinstance(o, dict) and o.get("op") == CREATE_PAGE for o in ops):
        return out
    stopped: dict[int, str] = {}
    for _ in range(len(ops) + 1):
        kept = [i for i in range(len(ops)) if i not in stopped]
        results = list(out.get("results") or [])
        if len(results) != len(kept):
            return out
        by_index = dict(zip(kept, results))
        again = stopped_calls(ops, by_index)
        if again == stopped:
            merged = [by_index[i] if i in by_index else stopped_result(ops[i], stopped[i]) for i in range(len(ops))]
            final = dict(out)
            final["results"] = merged
            if stopped:
                final["ok"] = False
                if final.get("exit_code") == 0:
                    final["exit_code"] = 2
            return final
        stopped = again
        out = run([ops[i] for i in range(len(ops)) if i not in stopped])
    raise RuntimeError("create_page stop: the batch did not settle")  # pragma: no cover — bounded by the argument above


def run_engine_result(engine: Any, state: dict, ops: list[dict], ctx: dict | None = None, **kw: Any) -> Any:
    """``engine.run(state, ops, ctx, **kw)`` with the same-batch stop (``run_batch``), for the replay paths that
    read the engine's own result object (``souschef_lab.engine.EngineResult``: ``ok``, ``results``, ``diff()``,
    ``state``…) rather than verify.run_engine's dict: r4cases (rebase, remeasure, measure_page), r4collect (the
    exact re-measure on a live state), contrast (the hidden-write check) and retrain/author (the oracle replay).
    A batch without a ``create_page`` is the one call it always was; one whose create was refused is run again
    without the stopped calls (each run on its own copy of ``state``) and the LAST run's result comes back with
    the stopped calls' ``E_TARGET`` results put back in call order, ``ok`` false. The engine's errors propagate,
    as from ``engine.run``. A dict-returning engine (the tests' fakes) gets a dict back."""
    import copy
    import dataclasses

    if not any(isinstance(o, dict) and o.get("op") == CREATE_PAGE for o in ops or []):
        return engine.run(state, ops, ctx, **kw)
    last: dict[str, Any] = {}

    def once(batch: list[dict]) -> dict:
        r = engine.run(copy.deepcopy(state), batch, ctx, **kw)
        last["r"] = r
        if isinstance(r, dict):
            return r
        out = r.to_dict()
        out["exit_code"] = getattr(r, "returncode", 0 if out.get("ok") else 2)
        return out

    out = run_batch(once, ops)
    r = last["r"]
    if isinstance(r, dict):
        return out
    res = list(out.get("results") or [])
    if len(res) == len(r.results) and all(a is b for a, b in zip(res, r.results)):
        return r  # nothing was stopped: the engine's own result, untouched
    return dataclasses.replace(r, ok=bool(out.get("ok")), results=res,
                               returncode=int(out.get("exit_code", r.returncode)))
