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
  ask      a request too vague to act on: no accepted (or refused) write, no report_unsupported, and a question in
           the reply (`clarify_ok`; a write is `wrote`, no question `no_question`) — `score_ask`. The vague first turns
           of the follow-up family (history expect "ask" / "offer") as scored turns, and contrast.py's vague requests
  page     create_page (r6-4, the 2026-10-15 tool): exam.score_page on expected.page — scored on the OP (title without
           case or spacing, a canonical slug equal to slugify of the asked slug or title, in_nav and status as asked,
           a same-turn section add on the new page is the page's own write, nothing else written), or a decline
           (the page exists, the slug is reserved, the template can't show custom pages): nothing written and the
           reply says why. Verdict `pass` either way (the exam's pass sets know no other)
  list     listing a carousel source's choices (tags / brands / categories) in answer mode — `score_listing` on the
           context the turn was SENT: only names of that context list, a short list named in full, a longer one at
           most 15 names then "...and N more" with N = total − named (the *Truncated total when the list was cut, and
           then the editor's dropdown named), an empty list said to be empty, an unknown list (member missing) never
           said to be empty and the dropdown named; no write, no report_unsupported (`answer_ok`)

A turn that did not finish (`stop` ≠ end_turn, turn.error, transport error) is
`no_answer` / `error` whatever the mode.

`flags` are the dataset's keep rules that a passing turn can still break:
`preview_claim` (the reply says the change shows in the preview — the
deployed rules forbid it), `where_missing` (a report_unsupported call without
a `where` the browser can link), `empty_reply`, `lookup_storm`,
`unexpected_hidden` (a W_HIDDEN the case's own measured writes do not raise).
`usable` = passing verdict and no flag: what the collector's later passes
retry and what the dataset keeps.

Relaxations (r6, `POLICY_VERSION`; each one only turns a failing turn into a
passing one, so a verify-3 pass is still a pass and RULE_VERSION is unchanged;
the one that applied is named in `info` and in `relaxed`). Each must accept
only an answer that is RIGHT on the site, never a merely plausible one:

  refused_corrected  exact: an op the engine refused that a LATER batch of the same turn re-sent accepted (same
                     op, target, list and keys: the teacher read the error and fixed the value) is not `invalid_op`.
                     OFF by default: the row still holds the refused call and its error, so training it teaches
                     sending an invalid op first (the case is re-asked instead)
  mailto             exact: the answer wrote `mailto:x@y` where the case expected the bare `x@y`, on a field that takes
                     a link (a path segment says link / url / href). One way only: the site renders these fields as
                     the href itself (footer social e-mail: href={socialLinks.email}; a gallery link gets https://
                     prefixed), so a bare address is a broken link and never stands in for `mailto:`
  same_control       exact and keys: one editor control that writes ONE of several keys by the section's layout
                     (CONTROL_KEY; the content-block "Text Alignment" writes splitContentAlign on split,
                     contentTextAlign on feature columns, textAlign otherwise — launcher BasicEditor.jsx
                     contentAlignKey) accepts the sibling key the editor itself would write in the case's state,
                     where the case expected another one. Never a key the editor would NOT write there (it changes
                     nothing on the page), and never for a control without a CONTROL_KEY resolver
  acted_first_turn   a follow-up whose vague earlier turn the teacher acted on instead of asking (collect's
                     `history_diverged` did_not_ask) is excused when that turn's accepted writes are the case's own
                     keys ("keys"; "target": anything on the case's targets), wrote no field whose gate was closed,
                     wrote no text the user never gave (a text / richtext field or a list item's text: invented
                     content such as a made-up testimonial), and every one of its writes was written again by the
                     scored turn (nothing it chose on its own survives in the final state), and the scored turn
                     passed — `excuse_divergences` (r4collect wires it). OFF by default: an excused row still trains
                     "act on a vague request instead of asking", the direction r5 drifted
  font               not a relaxation but the answer key of "the font" (`apply_font_policy`, also on exam rows):
                     "all" (DEFAULT since r6-3, the user's 2026-10-15 decision: a font request that names no part of
                     the site — "i don't like the font", "use lora", "Roboto everywhere" — changes EVERY site font:
                     `site_font_fields` = each global-panel font family shown in the editor that is a theme font or
                     set on the site; the unset ones that inherit may be written too; a request that names one part
                     ("the heading font") still means that field alone) and `score_site_fonts` scores it, values
                     included; "as_authored" (the r4 / r5 answer key: the authored cases as written, the exam's
                     amended rows back to their old ids and keys — for re-scoring old runs), "body_only" and
                     "body_or_both" (r6-2 options, kept for re-scoring)

The policy is `DEFAULT_POLICY` (collect), overridden by $SOUSCHEF_SCORE_POLICY
(JSON) and $SOUSCHEF_FONT_POLICY, overridden by `score(policy=…)`. The eval
scores with `STRICT_POLICY` (every relaxation off: the r4 rules its stored
calibration replicates were scored with) unless the round's trigger says
otherwise (`adopt_trigger_policy(..., base=STRICT_POLICY)`). The font answer key
is the same "all" in collect, the eval and the noise arms (STRICT_POLICY["font"]).

Comparability (POLICY_VERSION r6-3, exam.ANSWER_KEY): the exam's answer key moved
with the 2026-10-15 decisions — exam.AMENDMENTS gives 19 rows new ids (`<id>.k2`):
the seven "the font" rows (p011, p013, p150 ×2, r02, r04, h059: every site font)
and twelve value-less requests that may now be answered with a question (ask_ok:
p031, p048, p109, p129, h024, h027, h033, h036, h037, h044, h048, h055). A drift
baseline or an earlier eval scored on the old key never pairs with them: they
leave every Δ until the live model is re-calibrated (drift calibrate
--fresh-baseline, or the next candidate eval's baseline on Approve). Re-scoring an
old run's rows under the old key: font policy "as_authored" (exam rows back to
their built ids and keys). r6-4 adds a 20th: p137.k2 ("create a new page for
events", report_unsupported pages before) expects create_page — or, on a site
that already has the page (the exam's vintage state does), the reply that it
exists (exam.score_page); "as_authored" restores p137 and its `pages` key.
"""
from __future__ import annotations

import copy
from collections import Counter
import functools
import json
import os
import re
from pathlib import Path
from typing import Any

RULE_VERSION = "verify-3"
POLICY_VERSION = "r6-4"  # r6-4: modes page (create_page) and list (dropdown choices), exam p137.k2
FONT_POLICIES = ("all", "as_authored", "body_only", "body_or_both")
ACTED_SCOPES = ("keys", "target", "off")
DEFAULT_POLICY: dict[str, Any] = {"refused_corrected": False, "mailto": True, "same_control": True,
                                  "acted_first_turn": "off", "font": "all"}
# verify-3 as r4 and r5 scored (and the eval's stored calibration replicates): every relaxation off. The font
# answer key is not a relaxation: "all" everywhere (the r4 / r5 key is "as_authored", for re-scoring old runs)
STRICT_POLICY: dict[str, Any] = {"refused_corrected": False, "mailto": False, "same_control": False,
                                 "acted_first_turn": "off", "font": "all"}
LINK_SEGMENT = re.compile(r"link|url|href", re.IGNORECASE)
EMAIL = re.compile(r"^[^@\s:/]+@[^@\s/]+\.[a-z]{2,}$", re.IGNORECASE)
TYPOGRAPHY_PANEL = "TypographyColorsSection"
NAMES_HEADINGS = re.compile(r"\b(head(?:ing|er|line)s?|titles?)\b", re.IGNORECASE)
# a site font setting (site_font_fields): a global select whose key ends in Font / FontFamily / fontFamily
FONT_FAMILY_KEY = re.compile(r"(?:Font|FontFamily|fontFamily)$")
THEME_FONTS = ("headingFont", "bodyFont", "navFont", "buttonFont", "productTitleFont", "productBodyFont")
# "every font" words: they win over a named part ("everything on the site in Montserrat, headings and paragraphs").
# Not a bare "all" / "across" ("set the body and headings all to Lora" names its two parts)
EVERYTHING = re.compile(r"\b(every(?:thing|where)|every\s+(?:single\s+)?(?:fonts?|typefaces?|text)|whole|entire|"
                        r"throughout|site[- ]?wide|all\s+(?:of\s+)?(?:the\s+|my\s+|our\s+)?(?:site'?s?\s+|website'?s?\s+)?"
                        r"(?:fonts?|typefaces?|text|typography)|across\s+(?:the\s+|my\s+|our\s+)?(?:whole\s+)?"
                        r"(?:site|website|store|shop))\b", re.IGNORECASE)
# a part of the site a font request can name (then it means that part's font, not every font)
NAMES_FONT_AREA = re.compile(r"\b(head(?:ing|er|line)s?|titles?|body|paragraphs?|nav(?:igation)?|menus?|links?|"
                             r"buttons?|cart|products?|footer|announcement|banner|cards?|prices?|descriptions?)\b",
                             re.IGNORECASE)
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
    mode = (case.get("score") or {}).get("mode")
    if mode == "page":
        from .exam import accepted_ops, page_expectation
        spec = page_expectation(case, row) or {}
        ex = spec.get("existing") or {}
        _, refused = accepted_ops(row)
        if spec.get("why") == "exists" and ex.get("in_context") and any(o.get("op") == "create_page" for o in refused):
            flags.append("create_refused")  # the context showed the page: a training row must not try to create it
    if mode in ("exact", "keys", "page"):
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


# --- policy -----------------------------------------------------------------------------------------------------------

def policy_of(policy: dict | None = None, env=None) -> dict:
    """DEFAULT_POLICY ← $SOUSCHEF_SCORE_POLICY (a JSON object) ← $SOUSCHEF_FONT_POLICY ← `policy`. An unknown font
    policy or acted scope is an error (a typo must not silently score with another rule)."""
    env = os.environ if env is None else env
    out = dict(DEFAULT_POLICY)
    raw = env.get("SOUSCHEF_SCORE_POLICY")
    if raw:
        out.update(json.loads(raw))
    if env.get("SOUSCHEF_FONT_POLICY"):
        out["font"] = env["SOUSCHEF_FONT_POLICY"]
    out.update(policy or {})
    if out["font"] not in FONT_POLICIES:
        raise ValueError(f"font policy {out['font']!r} is not one of {FONT_POLICIES}")
    if out["acted_first_turn"] in (False, None):
        out["acted_first_turn"] = "off"
    if out["acted_first_turn"] is True:
        out["acted_first_turn"] = "keys"
    if out["acted_first_turn"] not in ACTED_SCOPES:
        raise ValueError(f"acted_first_turn {out['acted_first_turn']!r} is not one of {ACTED_SCOPES}")
    return out


def adopt_trigger_policy(trigger: dict | None, env=None, base: dict | None = None) -> dict:
    """A round's scoring switches from run.json trigger (score_policy: an object of DEFAULT_POLICY keys,
    font_policy: one of FONT_POLICIES) into the environment the scorer reads — an environment value already set
    wins. `base` (the eval passes STRICT_POLICY) is the policy the trigger's keys are laid over instead of
    DEFAULT_POLICY. Returns the resolved policy (ValueError on a bad one, before anything is scored)."""
    env = os.environ if env is None else env
    trig = trigger or {}
    sp, fp = trig.get("score_policy"), trig.get("font_policy")
    if sp not in (None, "", {}):
        if not isinstance(sp, dict):
            raise ValueError("trigger.score_policy must be an object of " + ", ".join(DEFAULT_POLICY))
        unknown = sorted(set(sp) - set(DEFAULT_POLICY))
        if unknown:
            raise ValueError(f"trigger.score_policy: unknown key(s) {', '.join(unknown)}")
    if base is not None:
        sp = {**{k: v for k, v in base.items() if k != "font"}, **(sp or {})}
    if sp not in (None, "", {}):
        env.setdefault("SOUSCHEF_SCORE_POLICY", json.dumps(sp))
    if fp not in (None, ""):
        env.setdefault("SOUSCHEF_FONT_POLICY", str(fp))
    return policy_of(env=env)


@functools.lru_cache(maxsize=8)
def _load_manifest(path: str, mtime: float):
    from .manifest import load
    return load(path)


def manifest_for(engine=None, manifest=None):
    """The manifest the scorer reads field metadata from: `manifest` (a Manifest or a path), else the one the engine
    runs on (engine.Engine.manifest_path, also through collect.DictEngine.inner), else None."""
    from .manifest import Manifest
    if isinstance(manifest, Manifest):
        return manifest
    path = manifest
    if path is None:
        for e in (engine, getattr(engine, "inner", None), getattr(engine, "engine", None)):
            if e is not None and getattr(e, "manifest_path", None):
                path = e.manifest_path
                break
    if path is None or not Path(path).is_file():
        return None
    return _load_manifest(str(path), Path(path).stat().st_mtime)


# --- same-control siblings ----------------------------------------------------------------------------------------------

def _norm_label(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def container_for(m, target: dict | None):
    t = target or {}
    k = t.get("kind")
    if k == "section":
        return m.section(str(t.get("sectionId") or ""))
    if k == "global":
        return m.panel(str(t.get("panel") or "")) if t.get("panel") else None
    if k == "branding":
        return m.branding
    if k == "page":
        return m.page(str(t.get("page") or ""))
    return None


def _global_container_of_key(m, key: str):
    for c in m.containers("global"):
        if key in c.fields:
            return c
    return None


# One editor control, several keys: which key the control writes in a given section state. The launcher's content-block
# editor (BasicEditor.jsx ContentBlockEditor, also mounted for perksCards / perksIcons with their style defaulted to
# feature-cards / feature-icons — resolveSectionEditor) has ONE "Text Alignment" control: contentAlignKey = split →
# splitContentAlign, feature columns → contentTextAlign, every other layout → textAlign; the renderer reads only that
# key for that layout (DefaultContentBlock.jsx). Milestones has no Content subsection (no control).
SECTION_DEFAULT_STYLE = {"perksCards": "feature-cards", "perksIcons": "feature-icons"}
FEATURE_COLUMNS = ("feature-columns", "feature-cards", "feature-icons")


def content_align_key(section_id: str, settings: dict | None) -> str | None:
    base = re.sub(r"-\d+$", "", str(section_id or ""))
    if base == "milestones":
        return None
    st = (settings or {}).get("style") or SECTION_DEFAULT_STYLE.get(base)
    if st == "split":
        return "splitContentAlign"
    if st in FEATURE_COLUMNS:
        return "contentTextAlign"
    return "textAlign"


CONTROL_KEY = {frozenset({"textAlign", "contentTextAlign", "splitContentAlign"}): content_align_key}


def editor_key_for(target: dict | None, keys: list[str], state: dict | None) -> str | None:
    """The one key of `keys` (a control's sibling group) the editor writes for `target` in `state`, or None when it
    cannot tell (no resolver for the group, not a section, the section not in the state)."""
    t = target or {}
    fn = CONTROL_KEY.get(frozenset(k.split(".")[-1] for k in keys))
    if fn is None or t.get("kind") != "section" or not isinstance(state, dict):
        return None
    from .state import find_section
    sid = str(t.get("sectionId") or "")
    sec = find_section(state, sid, str(t.get("page") or "home"))
    if sec is None:
        return None
    k = fn(sid, sec.get("settings") or {})
    return next((x for x in keys if x.split(".")[-1] == k), None) if k else None


def same_control_siblings(m, target: dict | None, key: str, state: dict | None = None, *,
                          editor_only: bool = True) -> list[str]:
    """The other keys of `key`'s editor control in its container (module docstring: same source file and line, type,
    options and a label of >= 2 words, a different leaf — a per-item loop such as pageVisibility.<page>.headerNavName
    or the chip colours shares a source line but not a control). With `editor_only` (the scorer's use) only the
    sibling the editor itself writes in `state` (editor_key_for) — [] when it cannot tell. [] without a manifest
    entry."""
    if m is None or not key:
        return []
    c = container_for(m, target)
    if c is None and (target or {}).get("kind") == "global":
        c = _global_container_of_key(m, key)
    f = c.fields.get(key) if c else None
    if f is None:
        return []
    src = f.spec.get("source")
    label = _norm_label(f.label)
    if not isinstance(src, dict) or src.get("line") is None or len(label.split()) < 2 or "{" in label:
        return []
    leaf = key.split(".")[-1]
    out = []
    for k, g in c.fields.items():
        s2 = g.spec.get("source")
        if k == key or k.split(".")[-1] == leaf or not isinstance(s2, dict):
            continue
        if (s2.get("file"), s2.get("line")) == (src.get("file"), src.get("line")) and g.type == f.type \
                and _norm_label(g.label) == label and list(g.enum) == list(f.enum):
            out.append(k)
    if editor_only and out:
        ek = editor_key_for(target, [key] + out, state)
        out = [k for k in out if k == ek]
    return sorted(out)


def _swap_key_in_op(op: dict, key: str, new: str) -> dict | None:
    o = copy.deepcopy(op)
    if o.get("op") == "set_fields" and isinstance(o.get("values"), dict) and key in o["values"]:
        o["values"] = {(new if k == key else k): v for k, v in o["values"].items()}
        return o
    if o.get("op") in ("reset", "restore_default") and key in (o.get("keys") or []):
        o["keys"] = [new if k == key else k for k in o["keys"]]
        return o
    return None


def same_control_exact_variants(case: dict, m, engine=None, limit: int = 6) -> list[dict]:
    """[{ops, diff, swap}] — the case's expected ops with one key replaced by a same-control sibling, measured on the
    case's state by the engine (an op the engine refuses there gives no variant)."""
    from .verify import ctx_of, default_engine, engine_errors, leaves_of, run_engine
    ops = [o for o in (case.get("expected") or {}).get("ops") or [] if o.get("op") != "pick_asset"]
    out: list[dict] = []
    for i, op in enumerate(ops):
        if op.get("op") not in ("set_fields", "reset", "restore_default"):
            continue
        keys = list((op.get("values") or {}).keys()) if op.get("op") == "set_fields" else list(op.get("keys") or [])
        for k in keys:
            for sib in same_control_siblings(m, op.get("target"), str(k), case.get("site_state")):
                new = _swap_key_in_op(op, str(k), sib)
                if new is None:
                    continue
                vops = ops[:i] + [new] + ops[i + 1:]
                engine = engine or default_engine()
                run = run_engine(engine, copy.deepcopy(case.get("site_state") or {}), copy.deepcopy(vops), ctx_of(case))
                if engine_errors(run) or not leaves_of(run):
                    continue
                out.append({"ops": vops, "diff": leaves_of(run), "swap": f"{k}->{sib}"})
                if len(out) >= limit:
                    return out
    return out


def same_control_key_sets(case: dict, m) -> list[tuple[list[dict], str]]:
    """Every expected key set (exam.expected_sets) with one key replaced by a same-control sibling."""
    from .exam import expected_sets
    out = []
    for s in expected_sets(case):
        for j, e in enumerate(s):
            if not isinstance(e, dict) or e.get("list"):
                continue
            for sib in same_control_siblings(m, e.get("target"), str(e.get("key") or ""), case.get("site_state")):
                out.append((s[:j] + [{**e, "key": sib}] + s[j + 1:], f"{e.get('key')}->{sib}"))
    return out


# --- font policy --------------------------------------------------------------------------------------------------------

def _is_typography(t: dict | None) -> bool:
    t = t or {}
    return t.get("kind") == "global" and t.get("panel") in (None, "", TYPOGRAPHY_PANEL)


def _user_texts(case: dict) -> str:
    return " \n ".join([str(case.get("request") or "")] + [str(h.get("user") or "") for h in case.get("history") or []])


def _default_manifest():
    """The exam's manifest (fixtures/exam/manifest-real.json): the font-family fields when neither the engine nor the
    caller names a manifest (they are the same 16 on every manifest since 2026-10-06)."""
    from .exam import MANIFEST_FILE
    return manifest_for(manifest=MANIFEST_FILE)


def site_font_fields(m=None, state: dict | None = None) -> dict[str, Any]:
    """The site's font settings, from the manifest (2026-10-15 product decision: a font request that names no part
    of the site changes EVERY one of them). A site font = a global-panel select field whose key ends in Font /
    FontFamily / fontFamily (one entry per field path: buttonFont sits in two panels). Per the case's state:

      required  shown in the editor (its gate open) and either a theme font (the manifest gives it a default: heading,
                body, navigation, buttons, product title / description) or set on this site (announcementBar.fontFamily
                is "Inter" on every template) — left alone, it keeps the old font on the page
      optional  shown but unset (the cart, footer and shop-menu fonts fall back to the body / nav / button font in
                the storefront CSS): writing the new font there too is right, leaving it is right
      hidden    its gate is closed (shopMenuFont outside the "modern" shop menu): a write there is a hidden write

    Without a state every theme font is required and the rest optional. `paths`: {key: state path}; `panels`:
    {key: the panel that holds it (buttonFont: TypographyColorsSection)}."""
    from .state import get_path
    m = m if m is not None else _default_manifest()
    out: dict[str, Any] = {"required": [], "optional": [], "hidden": [], "paths": {}, "panels": {}}
    if m is None:
        return {"required": list(THEME_FONTS), "optional": [], "hidden": [],
                "paths": {k: f"globalSettings.{k}" for k in THEME_FONTS}, "panels": {}}
    seen: set[str] = set()
    panels = [TYPOGRAPHY_PANEL] + [p for p in m.panels if p != TYPOGRAPHY_PANEL]
    for pn in panels:
        c = m.panel(pn)
        for key, f in ((c.fields if c is not None else {}) or {}).items():
            if f.type not in ("select", "enum") or not FONT_FAMILY_KEY.search(key) or f.read_only:
                continue
            path = f.path or key
            if path in seen:
                continue
            seen.add(path)
            out["paths"][key] = path
            out["panels"][key] = pn
            if isinstance(state, dict) and f.gated:
                shown = True
                for rule in ("conditional_on", "hidden_when"):
                    g = f.spec.get(rule) or {}
                    gk = g.get("key")
                    if not gk:
                        continue
                    gf = c.fields.get(gk)
                    by = get_path(state, gf.path) if gf is not None and gf.path else None
                    if by is None and gf is not None and gf.has_default:
                        by = gf.default  # engine.js hiddenFor: an unset gate reads as its default
                    if not gate_open(rule, g, by):
                        shown = False
                if not shown:
                    out["hidden"].append(key)
                    continue
            is_set = False
            if isinstance(state, dict):
                v = get_path(state, path)
                is_set = v not in (None, "") and not (f.has_unset and v == f.unset_value)
            (out["required"] if f.has_default or is_set else out["optional"]).append(key)
    rank = {k: i for i, k in enumerate(THEME_FONTS)}
    for k in ("required", "optional"):
        out[k] = sorted(out[k], key=lambda x: (rank.get(x, len(rank)), out[k].index(x)))
    return out


def generic_font_request(case: dict) -> bool:
    """The user's words name no part of the site ("the site's text", "use it on the site"), or say everything / all /
    whole site ("everything on the site in Montserrat, headings and paragraphs" is still everything). Asked of a
    case whose answer already writes a font (site_font_case)."""
    t = _user_texts(case)
    return bool(EVERYTHING.search(t)) or not NAMES_FONT_AREA.search(t)


def site_font_case(case: dict, any_words: bool = False) -> str | None:
    """"all" when the case already expects every site font (expected.site_fonts); "both" when its answer writes
    Typography bodyFont AND headingFont with ONE value for a request that names no heading — the r4-F6 "site's text
    → body + heading" shape (r4-F6-0051…0068); "body" when an exam-shaped row expects exactly Typography bodyFont for
    a request about "the font"; else None. `any_words` (the "all" policy, which asks generic_font_request instead):
    "both" whatever the words name ("everything in Montserrat, headings and paragraphs" is still everything)."""
    e = case.get("expected") or {}
    if isinstance(e.get("site_fonts"), dict):
        return "all"
    if NAMES_HEADINGS.search(_user_texts(case)) and not any_words:
        return None
    vals: dict[str, Any] = {}
    for o in e.get("ops") or []:
        if o.get("op") == "set_fields" and _is_typography(o.get("target")):
            vals.update({k: v for k, v in (o.get("values") or {}).items() if k in ("bodyFont", "headingFont")})
    keys = [{x.get("key") for x in s if isinstance(x, dict) and _is_typography(x.get("target"))}
            for s in [e.get("keys") or []]]
    if ("bodyFont" in vals and "headingFont" in vals and vals["bodyFont"] == vals["headingFont"]) or \
            (not e.get("ops") and any(k >= {"bodyFont", "headingFont"} for k in keys)):
        return "both"
    if NAMES_HEADINGS.search(_user_texts(case)):
        return None
    sets = [[x for x in s if isinstance(x, dict)] for s in [e.get("keys") or []] + list(e.get("alt") or []) if s]
    if sets and all(len(s) == 1 and s[0].get("key") == "bodyFont" and _is_typography(s[0].get("target")) for s in sets) \
            and re.search(r"\bfonts?\b|\btypeface\b", _user_texts(case), re.IGNORECASE):
        return "body"
    return None


def _drop_heading(case: dict) -> dict:
    """The case answered by bodyFont alone: headingFont out of its ops, diff, keys and alternatives."""
    c = copy.deepcopy(case)
    e = c.setdefault("expected", {})
    ops = []
    for o in e.get("ops") or []:
        if o.get("op") == "set_fields" and _is_typography(o.get("target")) and "headingFont" in (o.get("values") or {}):
            o = {**o, "values": {k: v for k, v in o["values"].items() if k != "headingFont"}}
            if not o["values"]:
                continue
        ops.append(o)
    e["ops"] = ops
    e["diff"] = [d for d in e.get("diff") or [] if not str(d.get("path") or "").endswith("headingFont")]
    e["keys"] = [k for k in e.get("keys") or [] if not (isinstance(k, dict) and k.get("key") == "headingFont")]
    e["alt"] = [s for s in e.get("alt") or [] if not any(isinstance(k, dict) and k.get("key") == "headingFont" for k in s)]
    e["alt_exact"] = []
    if c.get("meta", {}).get("oracle_ops"):
        c["meta"]["oracle_ops"] = copy.deepcopy(ops)
    return c


def _font_value(case: dict) -> Any:
    """The one font a site-font case's answer writes (None when the merchant left it open)."""
    vals = {str(v) for o in (case.get("expected") or {}).get("ops") or [] if o.get("op") == "set_fields"
            for k, v in (o.get("values") or {}).items() if FONT_FAMILY_KEY.search(str(k))}
    return next(iter(vals)) if len(vals) == 1 else None


def to_all_fonts(case: dict, m=None, value: Any = None) -> dict:
    """The case as the "all" font policy scores it: expected.site_fonts = {value, required, optional, must}
    (site_font_fields on the case's state; `must` = the required ones whose value is not already `value`), keys =
    the required fonts on their panels (the site-font scorer takes any global target), and — when the value is known — the set_fields (one per
    panel) that write it on every required font as its ops and oracle ops (the expected diff is re-measured by whoever needs
    it; the site-font scorer reads site_fonts)."""
    from .state import get_path
    c = copy.deepcopy(case)
    e = c.setdefault("expected", {})
    v = value if value is not None else _font_value(case)
    state = c.get("site_state") if isinstance(c.get("site_state"), dict) else None
    sf = site_font_fields(m, state)
    must = list(sf["required"])
    if v is not None and state is not None:
        must = [k for k in sf["required"] if str(get_path(state, sf["paths"].get(k) or f"globalSettings.{k}")) != str(v)]
    e["site_fonts"] = {"value": v, "required": sf["required"], "optional": sf["optional"], "must": must}
    def tgt(k: str) -> dict:  # the panel that holds it (write_ids' hold-out grain is <panel>.<key>)
        p = sf["panels"].get(k)
        return {"kind": "global", **({"panel": p} if p else {})}
    e["keys"] = [{"target": tgt(k), "key": k, "list": ""} for k in sf["required"]]
    e["alt"] = []
    if v is not None:  # one set_fields per panel (write_ids / the hold-out grain is <panel>.<key>)
        by_panel: dict[str, dict] = {}
        for k in sf["required"]:
            by_panel.setdefault(sf["panels"].get(k) or "", {})[k] = v
        e["ops"] = [{"op": "set_fields", "target": {"kind": "global", **({"panel": p} if p else {})}, "values": vals}
                    for p, vals in by_panel.items()]
        e["alt_exact"] = []
        e["diff"] = []
        c.setdefault("meta", {})["oracle_ops"] = copy.deepcopy(e["ops"])
    return c


def _restore_superseded(case: dict) -> dict:
    """An exam row as it was written before an answer-key amendment (exam.AMENDMENTS): its id, keys, alt."""
    sup = (case.get("meta") or {}).get("superseded")
    if not isinstance(sup, dict):
        return case
    c = copy.deepcopy(case)
    e = c.setdefault("expected", {})
    for k in ("keys", "alt"):
        e[k] = copy.deepcopy(sup.get(k) or [])
    for k in ("optional", "ask_ok", "site_fonts"):
        e.pop(k, None)
    if sup.get("id"):
        c["id"] = sup["id"]
    if "fields" in sup:
        c["fields"] = copy.deepcopy(sup["fields"])
    for k, v in (sup.get("row") or {}).items():  # create-page (exam.amend): kind back to out_of_scope
        c[k] = copy.deepcopy(v)
    if "out_of_scope" in sup:
        e["out_of_scope"] = copy.deepcopy(sup["out_of_scope"])
    e.pop("page", None)
    c["meta"].pop("superseded", None)
    c["meta"].pop("answer_key", None)
    for k, v in (sup.get("meta") or {}).items():
        c["meta"][k] = copy.deepcopy(v)
    return c


def apply_font_policy(case: dict, font: str = "all", m=None) -> tuple[dict, str | None]:
    """(case as the font policy scores it, a note or None).

      all           (default since r6-3, the 2026-10-15 decision) a generic font request (generic_font_request) whose
                    answer is the "both" shape → to_all_fonts; an exam row the exam amended already carries
                    site_fonts → unchanged, noted
      as_authored   the r4 / r5 answer: training cases as authored; an amended exam row back to its old id and keys
      body_only     (r6-2 option) "both" → bodyFont alone; amended exam rows as authored first
      body_or_both  (r6-2 option) either answer passes; amended exam rows as authored first
    """
    if font not in FONT_POLICIES:
        raise ValueError(f"font policy {font!r} is not one of {FONT_POLICIES}")
    if font == "all" and (case.get("meta") or {}).get("answer_key"):
        # an exam row the exam amended (exam.AMENDMENTS) is today's key as written: h059 keeps body + heading
        # required with every site font as the alternative, never re-derived from its words
        return case, ("font:all" if site_font_case(case, any_words=True) else None)
    shape = site_font_case(case, any_words=font == "all")
    if font == "all":
        if shape == "all":
            return case, "font:all"
        if shape == "both" and generic_font_request(case):
            return to_all_fonts(case, m if m is not None else _default_manifest()), "font:all"
        return case, None
    if (case.get("meta") or {}).get("superseded"):
        case = _restore_superseded(case)
        shape = site_font_case(case)
        if font == "as_authored":
            return case, "font:as_authored"
    if font == "as_authored" or shape is None:
        return case, None
    if shape == "both" and font == "body_only":
        return _drop_heading(case), "font:body_only"
    if shape == "both" and font == "body_or_both":
        c = copy.deepcopy(case)
        body = _drop_heading(case)["expected"]
        e = c["expected"]
        if e.get("ops"):
            e["alt_exact"] = list(e.get("alt_exact") or []) + [{"ops": body["ops"], "diff": body["diff"], "why": "font:body_only"}]
        if e.get("keys"):
            e["alt"] = list(e.get("alt") or []) + [body["keys"]]
        return c, "font:body_or_both"
    if shape == "body" and font == "body_or_both":
        c = copy.deepcopy(case)
        e = c["expected"]
        tgt = (e.get("keys") or [{}])[0].get("target") or {"kind": "global", "panel": TYPOGRAPHY_PANEL}
        e["alt"] = list(e.get("alt") or []) + [[{"target": tgt, "key": "bodyFont", "list": ""},
                                                {"target": tgt, "key": "headingFont", "list": ""}]]
        return c, "font:body_or_both"
    return case, None


def _leaves(values: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (values or {}).items() if isinstance(values, dict) else []:
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(_leaves(v, key + "."))
        else:
            out[key] = v
    return out


def exam_asks(text: Any) -> bool:
    from .exam import asks_a_question
    return asks_a_question(text)


def _reply_of(row: dict) -> str:
    from .verify import final_text
    return str(row.get("reply") or final_text(row.get("events") or []) or "")


def score_site_fonts(case: dict, row: dict) -> dict:
    """The verdict of a turn on a site-font case (expected.site_fonts): every `must` font written, every write a
    site font (required or optional) on a global target, each with the asked value when the case names one.
    Verdicts (verify's names): invalid_op (an op the engine refused), no_answer (nothing written), wrong_value,
    missing_writes, extra_writes, pass."""
    from .exam import accepted_ops
    sf = (case.get("expected") or {}).get("site_fonts") or {}
    value = sf.get("value")
    fonts = set(sf.get("required") or []) | set(sf.get("optional") or [])
    accepted, refused = accepted_ops(row)
    writes: dict[str, Any] = {}
    other: list[str] = []
    for o in accepted:
        t = o.get("target") if isinstance(o.get("target"), dict) else {}
        if o.get("op") == "set_fields" and t.get("kind") == "global":
            writes.update(_leaves(o.get("values")))
        else:
            other.append(f"{o.get('op')}:{json.dumps(t, sort_keys=True)}")
    if not accepted and not refused and (case.get("expected") or {}).get("ask_ok") and exam_asks(_reply_of(row)) \
            and "report_unsupported" not in {c.get("name") for c in row.get("tool_calls") or []}:
        return {"verdict": "clarify_ok", "reasons": [], "info": ["site_fonts", "asked"],
                "detail": {"written": [], "missing": [], "stray": [], "value": value}}
    stray = other + sorted(k for k in writes if k not in fonts)
    wrong = sorted(k for k, v in writes.items() if k in fonts and value is not None and str(v) != str(value))
    if value is None:  # the merchant left the font open: ONE font, every setting the same (2026-10-15 decision)
        chosen = Counter(str(v) for k, v in writes.items() if k in fonts)
        if len(chosen) > 1:
            top = chosen.most_common(1)[0][0]
            wrong = sorted(k for k, v in writes.items() if k in fonts and str(v) != top) or wrong
    missing = [k for k in sf.get("must") or sf.get("required") or [] if k not in writes]
    reasons = ([f"refused:{len(refused)}"] if refused else []) + [f"wrong_value:{k}" for k in wrong] + \
              [f"missing:{k}" for k in missing] + [f"stray:{k}" for k in stray]
    verdict = ("invalid_op" if refused else "no_answer" if not accepted else "wrong_value" if wrong else
               "missing_writes" if missing else "extra_writes" if stray else "pass")
    return {"verdict": verdict, "reasons": reasons, "info": ["site_fonts"],
            "detail": {"written": sorted(writes), "missing": missing, "stray": stray, "value": value}}


# --- asking before acting --------------------------------------------------------------------------------------------

def score_ask(row: dict) -> dict:
    """`ask` mode: a request too vague to act on (no value, no direction, or more than one place it could mean) is
    answered with a question and nothing written — collect's history rule for an `expect: ask` turn
    (r4collect.diverged) applied to the scored turn. A write, even one the engine refused, is a guess."""
    from .exam import accepted_ops
    from .verify import final_text
    accepted, refused = accepted_ops(row)
    reply = str(row.get("reply") or final_text(row.get("events") or []) or "")
    called = {c.get("name") for c in row.get("tool_calls") or []}
    if accepted or refused:
        return {"verdict": "wrote", "reasons": ["acted_on_a_vague_request"], "info": []}
    if "report_unsupported" in called:
        return {"verdict": "reported", "reasons": ["reported_unsupported"], "info": []}
    if not exam_asks(reply):  # a sign-off ("Anything else?") is not a question about the request
        return {"verdict": "no_question", "reasons": ["did_not_ask"], "info": []}
    return {"verdict": "clarify_ok", "reasons": [], "info": ["asked"]}


# --- listing a dropdown's choices (mode list) ----------------------------------------------------------------------------

LIST_MAX = 15  # rules.go "Listing choices": at most 15 names, then "...and N more"
LIST_KINDS = ("tags", "brands", "categories", "locations")
_BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+(.+?)\s*$")
_MORE = re.compile(r"(?:\.\.\.|\u2026)?\s*\band\s+(\d+)\s+(?:more|others?)\b", re.IGNORECASE)
DROPDOWN = re.compile(r"\bdrop-?\s?downs?\b|\bpicker\b|\b(?:tag|brand|category|location)\s+(?:setting|field|menu|list|selector)\b"
                      r"|\bsource settings\b", re.IGNORECASE)
SAYS_NONE = re.compile(r"\b(?:no|zero|0)\s+(?:\w+\s+){0,2}(?:tags?|brands?|categor(?:y|ies)|locations?)\b|\bnone\b"
                       r"|\b(?:don't|doesn't|do not|does not|haven't|hasn't|have not|has not)\s+(?:\w+\s+)?"
                       r"(?:have|set up|got|added|created|made|defined)\s+any\b"
                       r"|\b(?:isn't|aren't|is not|are not)\s+any\b|\bnot set up\b|\bempty\b|\bnothing\b", re.IGNORECASE)
# a sentence that says the list cannot be SEEN (an unknown list: "There's no tag list loaded, so I can't see them")
# is not a claim that the store has none
CANT_SEE = re.compile(r"\b(?:can't|cannot|can not|couldn't|could not|unable to|not able to)\s+(?:\w+\s+)?"
                      r"(?:see|view|load|access|read|tell|find)\b|\b(?:list|lists|data)\s+(?:isn't|aren't|is not|are not|"
                      r"wasn't|weren't)\b|\b(?:not|isn't|wasn't|n't been)\s+loaded\b|\bloaded\b|\bvisible to me\b"
                      r"|\bdon't have (?:access|visibility)\b", re.IGNORECASE)
_SENTENCE = re.compile(r"[^.!?\n\u2026]+[.!?\u2026]*")


def says_none(reply: str) -> bool:
    """The reply says the store HAS none (a sentence matching SAYS_NONE that is not about being unable to see the
    list — CANT_SEE)."""
    return any(SAYS_NONE.search(x) and not CANT_SEE.search(x) for x in _SENTENCE.findall(str(reply or "")))


# a name the reply runs on after a list name in the same sentence ("Staff Picks, Local Finds and Moon Rocks"): any
# Title-Case run after a separator, in a line that names one of the list's own entries
_INLINE_ITEM = re.compile(r"(?:,|;|\band\b|\bor\b|&)\s+((?:[A-Z0-9][\w'&.-]*)(?:\s+[A-Z0-9][\w'&.-]*){0,3})")
_STEP_VERB = re.compile(r"^(?:open|click|go|select|choose|pick|press|scroll|find|use|tap|save|then|under|in)\b", re.IGNORECASE)


def _clean_item(s: str) -> str:
    t = re.sub(r"[*_`]+", "", s).strip()
    t = re.split(r"\s+[(\u2014\u2013]|\s+-\s+|:\s", t, maxsplit=1)[0]
    return t.strip().strip(".,;").strip()


def listing_view(case: dict, row: dict | None = None) -> dict:
    """What the context the turn was SENT says about the case's list: {list, known, names, slugs, total, shown,
    truncated, situation: short | long | truncated | empty | unknown}."""
    lst = str(((case.get("expected") or {}).get("listing") or {}).get("list") or "tags")
    ctx = (row or {}).get("context") if isinstance((row or {}).get("context"), dict) else (case.get("context") or {})
    entries = ctx.get(lst) if isinstance(ctx, dict) else None
    out: dict[str, Any] = {"list": lst, "known": isinstance(entries, list)}
    if not isinstance(entries, list):
        return {**out, "names": [], "slugs": [], "total": None, "shown": 0, "truncated": False, "situation": "unknown"}
    names = [str(e.get("name") or "") for e in entries if isinstance(e, dict) and e.get("name")]
    slugs = [str(e.get("slug") or "") for e in entries if isinstance(e, dict) and e.get("slug")]
    cut = (ctx.get(f"{lst}Truncated") or {}) if isinstance(ctx.get(f"{lst}Truncated"), dict) else {}
    total = int(cut.get("total") or len(names))
    sit = ("empty" if not names else "truncated" if cut else "long" if total > LIST_MAX else "short")
    return {**out, "names": names, "slugs": slugs, "total": total, "shown": len(names), "truncated": bool(cut),
            "situation": sit}


def _inline_invented(line: str, names: list[str], known: set[str]) -> list[str]:
    """Names a prose line runs on with after one of the list's own names (_INLINE_ITEM) that the list lacks."""
    low = line.casefold()
    spans = [(m.start(), m.end()) for n in names for m in re.finditer(r"(?<![\w])" + re.escape(n.casefold()) + r"(?![\w])", low)]
    if not spans:
        return []
    first = min(a for a, _ in spans)
    out = []
    for m in _INLINE_ITEM.finditer(line):
        if m.start() < first:
            continue
        item = _clean_item(m.group(1))
        if not item or item.casefold() in known or any(a <= m.start(1) < b for a, b in spans):
            continue
        if _STEP_VERB.match(item) or item.isdigit() or item.casefold() in ("i", "i'll", "i'd", "you", "we", "the", "a", "an"):
            continue
        out.append(item)
    return out


def score_listing(case: dict, row: dict) -> dict:
    """Mode list (module docstring): the reply to "what tags / brands / categories can I use" against the context
    list the turn was sent (listing_view). Verdicts: answer_ok, wrote, reported, invented_name, missing_names,
    no_more_count, wrong_more_count, no_dropdown_pointer, said_none, did_not_say_none."""
    from .exam import accepted_ops
    accepted, refused = accepted_ops(row)
    v = listing_view(case, row)
    reply = _reply_of(row)
    called = {c.get("name") for c in row.get("tool_calls") or []}
    detail: dict[str, Any] = {"list": v["list"], "situation": v["situation"], "total": v["total"]}
    if accepted or refused:
        return {"verdict": "wrote", "reasons": ["wrote_on_a_question"], "info": [], "detail": detail}
    if "report_unsupported" in called:
        return {"verdict": "reported", "reasons": ["reported_unsupported"], "info": [], "detail": detail}
    known = {x.casefold() for x in v["names"]} | {x.casefold() for x in v["slugs"]}
    named = [n for n in v["names"] if re.search(r"(?<![\w])" + re.escape(n.casefold()) + r"(?![\w])", reply.casefold())]
    invented = []
    for line in reply.splitlines():
        mm = _BULLET.match(line)
        if not mm:
            invented += _inline_invented(line, v["names"], known)
            continue
        item = _clean_item(mm.group(1))
        if not item or item.casefold() in known:
            continue
        if len(item.split()) <= 5 and not _STEP_VERB.match(item) and not item.endswith("?"):
            invented.append(item)
    detail.update(named=len(named), invented=invented[:5])
    if invented:
        return {"verdict": "invented_name", "reasons": [f"invented:{x}" for x in invented][:5], "info": [], "detail": detail}
    sit = v["situation"]
    if sit == "unknown":
        if says_none(reply):
            return {"verdict": "said_none", "reasons": ["claimed_an_unknown_list_is_empty"], "info": [], "detail": detail}
        if not DROPDOWN.search(reply):
            return {"verdict": "no_dropdown_pointer", "reasons": ["unknown_list_without_the_dropdown"], "info": [],
                    "detail": detail}
        return {"verdict": "answer_ok", "reasons": [], "info": ["list:unknown"], "detail": detail}
    if sit == "empty":
        if not says_none(reply):
            return {"verdict": "did_not_say_none", "reasons": ["empty_list_not_said"], "info": [], "detail": detail}
        return {"verdict": "answer_ok", "reasons": [], "info": ["list:empty"], "detail": detail}
    if sit == "short":
        missing = [n for n in v["names"] if n not in named]
        if missing:
            return {"verdict": "missing_names", "reasons": [f"missing:{n}" for n in missing][:5], "info": [], "detail": detail}
        return {"verdict": "answer_ok", "reasons": [], "info": ["list:short"], "detail": detail}
    k = len(named)
    more = [int(x) for x in _MORE.findall(reply)]
    detail["more"] = more
    if not named or k > LIST_MAX:
        return {"verdict": "missing_names" if not named else "invented_name",
                "reasons": [f"named:{k} (1..{LIST_MAX})"], "info": [], "detail": detail}
    if not more:
        return {"verdict": "no_more_count", "reasons": ["no_and_n_more"], "info": [], "detail": detail}
    if (v["total"] or 0) - k not in more:
        return {"verdict": "wrong_more_count", "reasons": [f"and_{more[0]}_more_not_{(v['total'] or 0) - k}"], "info": [],
                "detail": detail}
    if sit == "truncated" and not DROPDOWN.search(reply):
        return {"verdict": "no_dropdown_pointer", "reasons": ["cut_list_without_the_dropdown"], "info": [], "detail": detail}
    return {"verdict": "answer_ok", "reasons": [], "info": [f"list:{sit}"], "detail": detail}


def oracle_turn(case: dict) -> dict | None:
    """{ops, text}: what a right answer sends on a `page` or `list` case — for dry runs' scripts (r4collect
    oracle_scripts); None for any other mode."""
    mode = (case.get("score") or {}).get("mode")
    if mode == "page":
        from .exam import page_oracle
        ops, text = page_oracle(case)
        return {"ops": ops, "text": text}
    if mode != "list":
        return None
    v = listing_view(case)
    label = {"tags": "Tag", "brands": "Brand", "categories": "Category"}.get(v["list"], v["list"])
    if v["situation"] == "unknown":
        return {"ops": [], "text": f"I can't see your {v['list']} right now. The {label} dropdown in the carousel's "
                                   "Source settings lists them all."}
    if v["situation"] == "empty":
        return {"ops": [], "text": f"Your store has no {v['list']} set up yet."}
    shown = v["names"][:LIST_MAX]
    text = "\n".join(f"- {n}" for n in shown)
    if v["situation"] in ("long", "truncated"):
        text += f"\n...and {(v['total'] or 0) - len(shown)} more."
    if v["situation"] == "truncated":
        text += f" The {label} dropdown in the carousel's Source settings shows them all."
    return {"ops": [], "text": text + "\nWant me to switch the carousel to one of these?"}


# --- relaxing an exact verdict ------------------------------------------------------------------------------------------

def _is_link_path(path: str) -> bool:
    segs = re.split(r"[.\[\]]+", str(path or ""))
    return any(LINK_SEGMENT.search(s) for s in segs if s)


def _unquote(v: str) -> Any:
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return v


def mailto_equal(path: str, expected: str, actual: str) -> bool:
    """The answer wrote `mailto:x@y` (`actual`, canonical JSON) where the case expected the bare `x@y` (`expected`),
    on a link field. One way only (module docstring): a bare address where `mailto:` was expected is a broken link."""
    if not _is_link_path(path):
        return False
    x, y = _unquote(expected), _unquote(actual)
    if not isinstance(x, str) or not isinstance(y, str):
        return False
    x, y = x.strip(), y.strip()
    if x.lower().startswith("mailto:") or not y.lower().startswith("mailto:"):
        return False
    return y[len("mailto:"):].lower() == x.lower() and bool(EMAIL.match(x))


def _batch_of(row: dict) -> list[int]:
    """The engine batch each of row.ops ran in (index-aligned with row.ops / exam.accepted_ops)."""
    out = []
    for b, batch in enumerate(row.get("engine_results") or []):
        out += [b] * len(batch.get("results") or [])
    return out


def _op_identity(op: dict) -> tuple:
    from .exam import norm_list, touches_of
    t = op.get("target") if isinstance(op.get("target"), dict) else {}
    tgt = json.dumps({k: t.get(k) for k in ("kind", "sectionId", "panel", "page")}, sort_keys=True)
    keys = tuple(sorted({x["key"] for x in touches_of(op)}))
    return op.get("op"), tgt, norm_list(op.get("list")), op.get("action"), str(op.get("index")), keys


def refused_corrected(row: dict) -> bool:
    """Every op the engine refused this turn was re-sent ACCEPTED in a later batch (same op, target, list, action,
    index and keys) — the teacher read the engine's error and fixed it. False when nothing was refused."""
    ops = list(row.get("ops") or [])
    results = [r for b in row.get("engine_results") or [] for r in b.get("results") or []]
    batch = _batch_of(row)
    if not ops or len(results) < len(ops):
        return False
    refused = [i for i, r in enumerate(results[:len(ops)]) if not r.get("ok")]
    if not refused:
        return False
    for i in refused:
        ident = _op_identity(ops[i])
        if not any(results[j].get("ok") and batch[j] > batch[i] and _op_identity(ops[j]) == ident
                   for j in range(i + 1, len(ops))):
            return False
    return True


def relax_exact(r: dict, row: dict, pol: dict) -> dict:
    """An exact verdict (`_verify`'s dict) with the policy's exact relaxations applied (module docstring); the
    relaxations applied are listed in r["relaxed"]."""
    from .verify import PASSING as V_PASSING, VERDICTS, compare, set_aside_companions
    if r["verdict"] in V_PASSING:
        return r
    reasons = list(r["reasons"])
    relaxed: list[str] = []
    det = r.get("detail") or {}
    exp = {p: v for p, v in det.get("expected_diff") or []}
    act = {p: v for p, v in det.get("actual_diff") or []}
    if pol.get("mailto") and "wrong_value" in reasons:
        hits = [p for p in exp if p in act and mailto_equal(p, exp[p], act[p])]
        if hits:
            for p in hits:
                act[p] = exp[p]
            e2, a2, _ = set_aside_companions(exp, act)
            diff_reasons = compare(e2, a2)
            reasons = [x for x in reasons if x not in ("wrong_value", "missing_writes", "extra_writes")] + diff_reasons
            relaxed.append("mailto")
    if pol.get("refused_corrected") and "invalid_op" in reasons:
        live_refused = sum(1 for b in row.get("engine_results") or [] for x in b.get("results") or [] if not x.get("ok"))
        replay_errors = len(det.get("engine_errors") or [])
        if refused_corrected(row) and replay_errors == live_refused:
            reasons = [x for x in reasons if x != "invalid_op"]
            relaxed.append("refused_corrected")
    if not relaxed:
        return r
    out = dict(r)
    out["reasons"] = reasons
    out["verdict"] = next((v for v in VERDICTS if v in reasons), "pass")
    out["relaxed"] = relaxed
    out["info"] = list(r.get("info") or []) + [f"relaxed:{x}" for x in relaxed]
    return out


# --- gates ------------------------------------------------------------------------------------------------------------
#
# The engine warns W_HIDDEN:<key> for EVERY write to a gated field (validate.js: a spec with conditional_on /
# hidden_when), whatever the state; whether the gate is closed is in the result's `hidden` member
# ({key, by, byValue, rule, gate}: the gating setting's post-op value).

def gate_open(rule: str, gate: dict | None, by_value: Any) -> bool:
    vals = (gate or {}).get("value")
    vals = vals if isinstance(vals, list) else [vals]

    def match(v: Any) -> bool:
        if v == "set":
            return by_value not in (None, "", [], {})
        return by_value == v or (by_value is not None and v is not None and str(by_value) == str(v))
    hit = any(match(v) for v in vals)
    return hit if rule == "conditional_on" else not hit


def closed_gate_writes(engine_results: list[dict]) -> list[str]:
    """The keys a turn wrote while their gate was closed (not shown on the site). `engine_results` is a row's batch
    list ([{results: […]}]) or a plain result list; a W_HIDDEN without the `hidden` detail counts as closed."""
    results = []
    for x in engine_results or []:
        results += list(x.get("results") or []) if isinstance(x, dict) and "results" in x else [x]
    out: list[str] = []
    for r in results:
        if not isinstance(r, dict) or not r.get("ok"):
            continue
        detail = [h for h in r.get("hidden") or [] if isinstance(h, dict)]
        warned = [str(w).split(":", 1)[1] for w in r.get("warnings") or [] if str(w).startswith("W_HIDDEN:")]
        for k in warned:
            rules = [h for h in detail if h.get("key") == k.split(".")[-1] or h.get("key") == k]
            if not rules or any(h.get("rule") and not gate_open(h["rule"], h.get("gate"), h.get("byValue")) for h in rules):
                out.append(k)
    return sorted(set(out))


# --- acting on a vague first turn ---------------------------------------------------------------------------------------

def _case_touch_keys(case: dict, m=None) -> list[dict]:
    """Every (target, list, key) the case's answer may write: expected ops, oracle ops, keys and alternatives, plus
    their same-control siblings."""
    from .exam import expected_sets, touches_of
    e = case.get("expected") or {}
    out = [dict(t) for o in list(e.get("ops") or []) + list((case.get("meta") or {}).get("oracle_ops") or [])
           for t in touches_of(o)]
    out += [dict(k) for s in expected_sets(case) for k in s if isinstance(k, dict)]
    for t in list(out):
        for sib in same_control_siblings(m, t.get("target"), str(t.get("key") or ""), case.get("site_state")):
            out.append({**t, "key": sib})
    return out


TEXT_TYPES = ("text", "richtext")
NOT_TEXT_ITEM_KEY = re.compile(r"url|link|href|image|img|icon|src|id$|color|colour|bg$|style|variant|position", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9#@.%$]+")


def _words_of(v: Any) -> set[str]:
    t = re.sub(r"<[^>]+>", " ", str(v or "")).casefold()
    return {w.strip(".") for w in _WORD.findall(t) if w.strip(".")}


def _given_by_user(v: Any, user_words: set[str]) -> bool:
    """A written text the user typed: every word of it is one of the user's (case, punctuation and HTML aside)."""
    w = _words_of(v)
    return not w or w <= user_words


def invented_texts(case: dict, hrow: dict, m=None) -> list[str]:
    """The keys a history turn filled with text the user never gave: a text / richtext field (manifest type) or a
    list item's text member (an `add` / `update` item string that is not a url, image, icon, id or style), whose
    words are not all words of the case's user turns. r4-F2-0102/0103: the first turn invented a 5-star review."""
    from .exam import accepted_ops
    user = _words_of(_user_texts(case))
    acc, _ = accepted_ops(hrow)
    out: list[str] = []
    for o in acc:
        if o.get("op") == "set_fields":
            c = container_for(m, o.get("target")) if m is not None else None
            for k, v in (o.get("values") or {}).items():
                ck = c if c is not None or m is None or (o.get("target") or {}).get("kind") != "global" \
                    else _global_container_of_key(m, str(k))
                f = ck.fields.get(k) if ck is not None else None
                is_text = (f.type in TEXT_TYPES) if f is not None else (isinstance(v, str) and len(_words_of(v)) >= 2
                                                                         and not NOT_TEXT_ITEM_KEY.search(str(k)))
                if is_text and isinstance(v, str) and not _given_by_user(v, user):
                    out.append(str(k))
        elif o.get("op") == "edit_list" and o.get("action") in ("add", "update", "insert"):
            item = o.get("item") if isinstance(o.get("item"), dict) else (o.get("values") if isinstance(o.get("values"), dict) else {})
            if not isinstance(item, dict) and isinstance(o.get("item"), str):
                item = {"value": o.get("item")}
            for k, v in (item or {}).items():
                if isinstance(v, str) and not NOT_TEXT_ITEM_KEY.search(str(k)) and not _given_by_user(v, user):
                    out.append(f"{o.get('list')}[].{k}")
    return sorted(set(out))


def _rewritten_by(hrow: dict, row: dict | None) -> bool:
    """Every (target, list, key) the history turn wrote is written again by the scored turn's accepted ops."""
    from .exam import accepted_ops, norm_list, touches_of
    if row is None:
        return False
    tkey = lambda t: (json.dumps({k: (t.get("target") or {}).get(k) for k in ("kind", "sectionId", "panel", "page")},  # noqa: E731
                                 sort_keys=True), norm_list(t.get("list")), str(t.get("key") or ""))
    acc_h, _ = accepted_ops(hrow)
    acc_s, _ = accepted_ops(row)
    later = {tkey(t) for o in acc_s for t in touches_of(o)}
    later_lists = {(a, b) for a, b, k in later if not k}
    for o in acc_h:
        if o.get("op") == "edit_list" and o.get("action") in ("add", "insert"):
            return False  # an added item stays in the final state whatever the scored turn writes
        for t in touches_of(o):
            k = tkey(t)
            if k not in later and (k[0], k[1]) not in later_lists:
                return False
    return True


def acted_on_topic(case: dict, hrow: dict, scope: str = "keys", m=None, row: dict | None = None) -> bool:
    """The history turn made accepted writes, all of them on the case's own keys (scope "keys") or targets ("target"),
    none wrote a field whose gate was closed (closed_gate_writes), none wrote text the user never gave
    (invented_texts), and — given the scored `row` — the scored turn wrote every one of them again."""
    from .exam import accepted_ops, norm_list, target_matches, touches_of
    if scope == "off":
        return False
    acc, _ = accepted_ops(hrow)
    touches = [t for o in acc for t in touches_of(o)]
    if not touches:
        return False
    if closed_gate_writes(hrow.get("engine_results") or []):
        return False
    if invented_texts(case, hrow, m):
        return False
    if row is not None and not _rewritten_by(hrow, row):
        return False
    allowed = _case_touch_keys(case, m)
    page = (case.get("context") or {}).get("page")
    for t in touches:
        ok = False
        for e in allowed:
            if not e.get("target") or not target_matches(t["target"], e["target"], page):
                continue
            if scope == "target" or (str(t.get("key") or "") == str(e.get("key") or "")
                                     and norm_list(t.get("list")) == norm_list(e.get("list"))):
                ok = True
                break
            if not e.get("key") and norm_list(t.get("list")) == norm_list(e.get("list")):  # a whole-list write
                ok = True
                break
        if not ok:
            return False
    return True


def excuse_divergences(case: dict, history_rows: list[dict], divergences: list[dict], sc: dict,
                       policy: dict | None = None, engine=None, manifest=None,
                       row: dict | None = None) -> tuple[list[dict], list[dict]]:
    """(divergences still standing, divergences excused) — r4collect's `history_diverged` list after the
    acted_first_turn relaxation: an `ask`/`offer` turn the teacher answered by acting on the case's own keys, when
    the scored turn passed (sc = this module's score). `row` (the scored turn) is required for an excuse: every
    write of the excused turn must be written again by it."""
    pol = policy_of(policy)
    scope = pol["acted_first_turn"]
    if scope == "off" or not divergences or not sc.get("passed", sc.get("verdict") in PASSING) or row is None:
        return list(divergences), []
    m = manifest_for(engine, manifest)
    keep, excused = [], []
    for d in divergences:
        i = d.get("turn")
        h = history_rows[i] if isinstance(i, int) and 0 <= i < len(history_rows) else None
        if h is not None and d.get("expect") in ("ask", "offer") and d.get("why") == "did_not_ask" \
                and acted_on_topic(case, h, scope, m, row):
            excused.append({**d, "excused": f"acted_first_turn:{scope}"})
        else:
            keep.append(d)
    return keep, excused


# --- the verdict ------------------------------------------------------------------------------------------------------

def _verify(case: dict, row: dict, engine) -> dict:
    from .verify import verify
    v = verify(row, case, engine)
    return {"verdict": v.verdict, "reasons": list(v.reasons), "info": list(v.info), "detail": v.to_dict()}


def _exact_variants(case: dict, pol: dict, m, engine) -> list[tuple[dict, str]]:
    """The other (case, info) an exact turn may pass on, in order: alt_exact, then same-control swaps."""
    exp = case.get("expected") or {}
    out = []
    for i, alt in enumerate(exp.get("alt_exact") or []):
        c2 = copy.deepcopy(case)
        c2["expected"] = {**c2["expected"], "ops": alt.get("ops") or [], "diff": alt.get("diff") or []}
        out.append((c2, f"alt_exact:{i}" + (f":{alt['why']}" if alt.get("why") else "")))
    if pol.get("same_control") and m is not None:
        for v in same_control_exact_variants(case, m, engine):
            c2 = copy.deepcopy(case)
            c2["expected"] = {**c2["expected"], "ops": v["ops"], "diff": v["diff"]}
            out.append((c2, f"same_control:{v['swap']}"))
    return out


def score(case: dict, row: dict, engine=None, history_rows: list[dict] | None = None, *,
          policy: dict | None = None, manifest=None) -> dict:
    """{verdict, reasons, info, mode, flags, usable, rule_version, policy_version, relaxed?, …} of one turn."""
    from . import exam

    pol = policy_of(policy)
    shape = site_font_case(case, any_words=True) if pol["font"] == "all" else None
    case, font_note = apply_font_policy(case, pol["font"], manifest_for(engine, manifest) if shape == "both" else None)
    mode = (case.get("score") or {}).get("mode") or "exact"
    exp = case.get("expected") or {}
    out: dict[str, Any] = {"mode": mode, "rule_version": RULE_VERSION, "policy_version": POLICY_VERSION, "info": []}
    if font_note:
        out["font_policy"] = font_note
    unfinished = _completed(row)
    if unfinished == "error":
        out.update(verdict="error", reasons=[f"error:{row.get('code')}"])
    elif pol["font"] == "all" and isinstance(exp.get("site_fonts"), dict) and mode in ("exact", "keys") and not unfinished:
        s = score_site_fonts(case, row)
        out.update(verdict=s["verdict"], reasons=s["reasons"], info=s["info"], detail=s["detail"])
    elif mode == "ask":
        s = score_ask(row) if not unfinished else {"verdict": "no_answer", "reasons": [f"stop:{row.get('stop')}"], "info": []}
        out.update(verdict=s["verdict"], reasons=s["reasons"], info=s["info"])
    elif mode in ("page", "list") and unfinished:
        out.update(verdict="no_answer", reasons=[f"stop:{row.get('stop')}"])
    elif mode == "page":
        s = exam.score_page(case, row, refused_ok=bool(pol.get("refused_corrected")) and refused_corrected(row))
        out.update(verdict=s["verdict"], reasons=list(s.get("reasons") or []), info=list(s.get("info") or []),
                   detail=s.get("detail") or {})
        if pol.get("refused_corrected") and s["verdict"] == "pass" and (s.get("detail") or {}).get("expect") == "create" \
                and exam.accepted_ops(row)[1]:
            out["relaxed"] = ["refused_corrected"]
    elif mode == "list":
        s = score_listing(case, row)
        out.update(verdict=s["verdict"], reasons=s["reasons"], info=s["info"], detail=s["detail"])
    elif mode == "exact":
        r = relax_exact(_verify(case, row, engine), row, pol)
        if r["verdict"] not in PASSING:
            m = manifest_for(engine, manifest) if pol.get("same_control") else None
            for c2, note in _exact_variants(case, pol, m, engine):
                r2 = relax_exact(_verify(c2, row, engine), row, pol)
                if r2["verdict"] in PASSING:
                    r = r2
                    r["info"].append(note)
                    if note.startswith("same_control"):
                        r["relaxed"] = list(r.get("relaxed") or []) + ["same_control"]
                    break
        out.update(verdict=r["verdict"], reasons=r["reasons"], info=r["info"],
                   detail={k: r["detail"][k] for k in ("expected_diff", "actual_diff", "engine_errors", "warnings")})
        if r.get("relaxed"):
            out["relaxed"] = r["relaxed"]
    elif mode == "clarify":
        r = _verify(case, row, engine)
        out.update(verdict=r["verdict"], reasons=r["reasons"], info=r["info"])
    elif unfinished:
        out.update(verdict="no_answer", reasons=[f"stop:{row.get('stop')}"])
    elif mode in ("keys", "where"):
        s = exam.score_turn(case, row)
        if mode == "keys" and s.get("verdict") not in PASSING and pol.get("same_control"):
            m = manifest_for(engine, manifest)
            for kset, swap in same_control_key_sets(case, m) if m is not None else []:
                c2 = copy.deepcopy(case)
                c2["expected"]["alt"] = list(c2["expected"].get("alt") or []) + [kset]
                s2 = exam.score_turn(c2, row)
                if s2.get("verdict") in PASSING:
                    s = s2
                    out["info"].append(f"same_control:{swap}")
                    out["relaxed"] = ["same_control"]
                    break
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


# --- re-scoring a stored collect row ------------------------------------------------------------------------------------

def _run_with_state(engine, state: dict, ops: list[dict], ctx: dict) -> dict | None:
    """The state after `ops` (refused ones skipped, as live), or None when the engine cannot print it."""
    try:
        res = engine.run(copy.deepcopy(state), copy.deepcopy(ops), ctx, print_state=True)
    except TypeError:  # a dict engine (collect.DictEngine) prints the state unasked
        res = engine.run(copy.deepcopy(state), copy.deepcopy(ops), ctx)
    st = res.get("state") if isinstance(res, dict) else getattr(res, "state", None)
    return st if isinstance(st, dict) else None


def final_case_of(case: dict, row: dict, engine) -> dict | None:
    """The case as r4collect scored its last turn: the state after the earlier turns' ops (replayed batch by batch
    from the stored history rows), the live context of the scored turn and the re-measured expectation the row
    stored (`final_expected`). The case itself for a single turn; None when a history replay fails."""
    from .verify import ctx_of
    hist = row.get("history_rows") or []
    if not hist:
        return case
    final = copy.deepcopy(case)
    state = copy.deepcopy(case.get("site_state") or {})
    for h in hist:
        ops = list(h.get("ops") or [])
        batch = _batch_of(h)
        if not ops:
            continue
        ctx = ctx_of({"site_state": state, "context": h.get("context") or {}, "template": case.get("template")})
        for b in sorted(set(batch[:len(ops)])) if len(batch) >= len(ops) else [0]:
            bops = [o for o, x in zip(ops, batch) if x == b] if len(batch) >= len(ops) else ops
            st = _run_with_state(engine, state, bops, ctx)
            if st is None:
                return None
            state = st
    final["site_state"] = state
    if isinstance(row.get("context"), dict):
        final["context"] = row["context"]
    fe = row.get("final_expected") or {}
    if fe.get("mode"):
        final["score"] = {"mode": fe["mode"]}
    if fe.get("diff") is not None:
        final["expected"] = {**final["expected"], "diff": fe["diff"]}
    return final


def rescore_row(case: dict, row: dict, engine, *, policy: dict | None = None, manifest=None) -> dict:
    """A stored r4collect row scored again under today's rules, without asking the teacher: the same `score` dict
    r4collect stores (history_diverged / history_excused included). For carrying a round's accepted rows forward
    and for measuring a scorer change on a collect log."""
    from .r4collect import diverged
    if row.get("status") != "ok":
        return {"verdict": "error", "reasons": [f"error:{row.get('code')}"], "mode": (case.get("score") or {}).get("mode"),
                "flags": [], "passed": False, "usable": False, "rule_version": RULE_VERSION,
                "policy_version": POLICY_VERSION, "info": []}
    final = final_case_of(case, row, engine)
    if final is None:
        return {"verdict": "error", "reasons": ["history_replay_failed"], "mode": (case.get("score") or {}).get("mode"),
                "flags": [], "passed": False, "usable": False, "rule_version": RULE_VERSION,
                "policy_version": POLICY_VERSION, "info": []}
    hist = row.get("history_rows") or []
    sc = score(final, row, engine, hist, policy=policy, manifest=manifest)
    divs = []
    for i, h in enumerate(hist):
        d = diverged(str(h.get("expect") or ""), h)
        if d:
            divs.append({"turn": i, "expect": h.get("expect"), "why": d})
    divs, excused = excuse_divergences(final, hist, divs, sc, policy, engine, manifest, row=row)
    if excused:
        sc["history_excused"] = excused
    if divs:
        sc["history_diverged"] = divs
        sc["flags"] = sorted(set(sc["flags"]) | {"history_diverged"})
        sc["usable"] = False
    return sc
