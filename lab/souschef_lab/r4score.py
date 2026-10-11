"""r4score — verify-3: the verdict of one r4 turn on its case (docs/LAB.md §5b).

verify-2 (verify.py) compares exact diffs; the r4 cases add four other ways a
turn can be right (`case.score.mode`, set by r4cases):

  exact    verify.verify on expected.{ops, diff}; any of expected.alt_exact ({ops, diff}) passing also passes
  keys     exam.score_turn on expected.keys / alt (every expected key written, nothing else) and, when
           expected.direction is set, the written value moved the asked way (up / down: numbers;
           darker / lighter: colour luminance) — else `wrong_direction`; when expected.list_ops is set
           ([{list, action, index}]), an accepted edit_list made each of them — else `wrong_list_op`
  where    exam.score_turn's out-of-scope rule: no accepted write, report_unsupported.where ∈ expected_where
  clarify  verify.verify's clarify_ok (no write, a question naming >= 2 candidates)
  answer   no accepted write and the reply names one of expected.reply_any (`answer_ok`)

A turn that did not finish (`stop` ≠ end_turn, turn.error, transport error) is
`no_answer` / `error` whatever the mode.

`flags` are the dataset's keep rules that a passing turn can still break:
`preview_claim` (the reply says the change shows in the preview — the
deployed rules forbid it), `where_missing` (a report_unsupported call without
a `where` the browser can link), `empty_reply`, `lookup_storm`,
`unexpected_hidden` (a W_HIDDEN the case's own measured writes do not raise).
`usable` = passing verdict and no flag: what the collector's later passes
retry and what the dataset keeps.
"""
from __future__ import annotations

import copy
import re
from typing import Any

RULE_VERSION = "verify-3"
PASSING = ("pass", "unsupported_ok", "clarify_ok", "answer_ok")
# ONE preview rule for every keep check (the scorer's flag, the dataset's final reject and stat, the VM's row
# check): any form of the word — "preview", "previews", "previewing", "previewed", "live preview".
PREVIEW = re.compile(r"\bpreview(?:s|ed|ing)?\b", re.IGNORECASE)


def mentions_preview(text: Any) -> bool:
    return bool(PREVIEW.search(str(text or "")))
LOOKUP_STORM = 3
SERVER_TOOLS = ("lookup_fields", "describe_target")


def assistant_texts(row: dict) -> list[str]:
    """Every assistant text the turn produced (its text.delta runs)."""
    out, cur = [], []
    for e in row.get("events") or []:
        n = e.get("event")
        if n == "text.delta":
            cur.append(str((e.get("data") or {}).get("text") or ""))
        elif n in ("tool.call", "turn.waiting", "turn.suspended", "turn.done", "turn.error"):
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return [t for t in out if t.strip()]


def flags_of(case: dict, row: dict, history_rows: list[dict] | None = None) -> list[str]:
    from .exam import WHERE_VALUES

    flags = []
    rows = list(history_rows or []) + [row]
    if any(mentions_preview(t) for r in rows for t in assistant_texts(r)):
        flags.append("preview_claim")
    for r in rows:
        for c in r.get("tool_calls") or []:
            if c.get("name") == "report_unsupported":
                w = str(((c.get("input") or {}) if isinstance(c.get("input"), dict) else {}).get("where") or "").strip().lower()
                if w not in WHERE_VALUES:
                    flags.append("where_missing")
                    break
    from .verify import final_text
    if not final_text(row.get("events") or []).strip():  # dataset.py's rule: no reply after the last batch
        flags.append("empty_reply")
    if sum(1 for c in row.get("tool_calls") or [] if c.get("name") in SERVER_TOOLS) > LOOKUP_STORM:
        flags.append("lookup_storm")
    measured = set(((case.get("meta") or {}).get("measured") or {}).get("warnings") or [])
    measured |= {f"W_HIDDEN:{k}" for k in (case.get("expected") or {}).get("expect_hidden") or []}
    if (case.get("score") or {}).get("mode") in ("exact", "keys"):
        seen = {w for b in row.get("engine_results") or [] for r in b.get("results") or [] for w in r.get("warnings") or []
                if str(w).startswith("W_HIDDEN")}
        if seen - measured:
            flags.append("unexpected_hidden")
    return sorted(set(flags))


def _completed(row: dict) -> str | None:
    if row.get("status") != "ok":
        return "error"
    if row.get("stop") not in ("end_turn",):
        return "no_answer"
    return None


# --- direction --------------------------------------------------------------------------------------------------------

def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)", str(v or ""))
    return float(m.group(1)) if m else None


def _lum(v: Any) -> float | None:
    s = str(v or "").strip().lower()
    m = re.match(r"^#([0-9a-f]{3}|[0-9a-f]{6})([0-9a-f]{2})?$", s)
    if m:
        h = m.group(1)
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    else:
        m = re.match(r"^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*([\d.]+))?\s*\)$", s)
        if not m:
            return None
        r, g, b = (int(m.group(i)) for i in (1, 2, 3))
        if m.group(4) is not None:  # an overlay: more opaque black reads darker
            a = float(m.group(4))
            r, g, b = (x * a + 255 * (1 - a) for x in (r, g, b))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def check_direction(row: dict, direction: dict) -> str | None:
    """None = moved the asked way; else a reason (`wrong_direction:…`, or
    `direction_unchecked` when no before/after pair could be read)."""
    key = str(direction.get("key") or "")
    want = str(direction.get("dir") or "").lower()
    if not key or not want:
        return "direction_unchecked"
    leaf = key.split(".")[-1]
    before, after, seen = None, None, False
    for b in row.get("engine_results") or []:
        for ch in b.get("changes") or []:
            for d in ch.get("diff") or []:
                p = str(d.get("path") or "")
                if p == key or p.endswith("." + key) or p.endswith("." + leaf):
                    if not seen:
                        before = d.get("before")
                        seen = True
                    after = d.get("after")
    if not seen:
        return "direction_unchecked"
    if before is None:
        before = direction.get("from")
    if want in ("up", "down"):
        a, b = _num(after), _num(before)
        if a is None or b is None:
            return "direction_unchecked"
        ok = a > b if want == "up" else a < b
    elif want in ("darker", "lighter"):
        a, b = _lum(after), _lum(before)
        if a is None or b is None:
            return "direction_unchecked"
        ok = a < b if want == "darker" else a > b
    else:
        return "direction_unchecked"
    return None if ok else f"wrong_direction:{want}:{before}->{after}"


# --- list ops (keys mode) -----------------------------------------------------------------------------------------------

def _norm_index(v: Any) -> Any:
    if isinstance(v, list):
        return [_norm_index(x) for x in v]
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) or (isinstance(v, str) and v.strip().lstrip("-").isdigit()):
        return int(v)
    return v


def check_list_ops(row: dict, want: list[dict]) -> str | None:
    """None when every wanted list op {list, action, index?} was made by an
    ACCEPTED edit_list of the turn (the list by its name without `[]`, the
    index exactly when given — a bare int and a one-element path are the
    same), else `list_op_missing:<list> <action> <index>`. Keys mode checks
    WHICH rows a list write touched; the values (an FAQ answer's wording, a
    proposal the teacher made) are the teacher's."""
    from .exam import accepted_ops, norm_list

    accepted, _ = accepted_ops(row)
    made = [o for o in accepted if o.get("op") == "edit_list"]
    for w in want:
        def ok(o: dict) -> bool:
            if norm_list(o.get("list")) != norm_list(w.get("list")) or o.get("action") != w.get("action"):
                return False
            if "index" not in w or w.get("index") is None:
                return True
            a, b = _norm_index(o.get("index")), _norm_index(w.get("index"))
            a = a[0] if isinstance(a, list) and len(a) == 1 else a
            b = b[0] if isinstance(b, list) and len(b) == 1 else b
            return a == b
        if not any(ok(o) for o in made):
            return f"list_op_missing:{w.get('list')} {w.get('action')} {w.get('index')}"
    return None


# --- the verdict ------------------------------------------------------------------------------------------------------

def _verify(case: dict, row: dict, engine) -> dict:
    from .verify import verify
    v = verify(row, case, engine)
    return {"verdict": v.verdict, "reasons": list(v.reasons), "info": list(v.info), "detail": v.to_dict()}


def score(case: dict, row: dict, engine=None, history_rows: list[dict] | None = None) -> dict:
    """{verdict, reasons, info, mode, flags, usable, rule_version, …} of one turn."""
    from . import exam

    mode = (case.get("score") or {}).get("mode") or "exact"
    exp = case.get("expected") or {}
    out: dict[str, Any] = {"mode": mode, "rule_version": RULE_VERSION, "info": []}
    unfinished = _completed(row)
    if unfinished == "error":
        out.update(verdict="error", reasons=[f"error:{row.get('code')}"])
    elif mode == "exact":
        r = _verify(case, row, engine)
        if r["verdict"] not in PASSING:
            for i, alt in enumerate(exp.get("alt_exact") or []):
                c2 = copy.deepcopy(case)
                c2["expected"] = {**c2["expected"], "ops": alt.get("ops") or [], "diff": alt.get("diff") or []}
                r2 = _verify(c2, row, engine)
                if r2["verdict"] in PASSING:
                    r = r2
                    r["info"].append(f"alt_exact:{i}")
                    break
        out.update(verdict=r["verdict"], reasons=r["reasons"], info=r["info"],
                   detail={k: r["detail"][k] for k in ("expected_diff", "actual_diff", "engine_errors", "warnings")})
    elif mode == "clarify":
        r = _verify(case, row, engine)
        out.update(verdict=r["verdict"], reasons=r["reasons"], info=r["info"])
    elif unfinished:
        out.update(verdict="no_answer", reasons=[f"stop:{row.get('stop')}"])
    elif mode in ("keys", "where"):
        s = exam.score_turn(case, row)
        out.update(verdict=s.get("verdict") or "unlabelled", reasons=list(s.get("reasons") or []),
                   detail={k: s[k] for k in ("written", "stray", "where_sent", "expected_where", "keyword_verdict") if k in s})
        if out["verdict"] == "pass" and exp.get("list_ops"):
            why = check_list_ops(row, exp["list_ops"])
            if why:
                out.update(verdict="wrong_list_op", reasons=out["reasons"] + [why])
        if out["verdict"] == "pass" and exp.get("direction"):
            why = check_direction(row, exp["direction"])
            if why == "direction_unchecked":
                out["info"].append(why)
            elif why:
                out.update(verdict="wrong_direction", reasons=out["reasons"] + [why])
    elif mode == "answer":
        accepted, _ = exam.accepted_ops(row)
        reply = str(row.get("reply") or "").casefold()
        hits = [w for w in exp.get("reply_any") or [] if str(w).casefold() in reply]
        if accepted:
            out.update(verdict="wrote", reasons=["wrote_on_a_question"])
        elif hits:
            out.update(verdict="answer_ok", reasons=[], detail={"named": hits[:5]})
        else:
            out.update(verdict="no_answer", reasons=["named_none_of_reply_any"])
    else:
        out.update(verdict="unlabelled", reasons=[f"mode:{mode}"])
    out["passed"] = out["verdict"] in PASSING
    out["flags"] = flags_of(case, row, history_rows) if row.get("status") == "ok" else []
    out["usable"] = bool(out["passed"] and not out["flags"])
    return out
