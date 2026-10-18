"""dataset_stage — the training set and its gates (design §3.4): job `souschef-dataset`.

r4dataset.build on the round's cases and every collection log, on the deployed
manifest, `--section-cap-scope F0,F1`. Gates (each fails the stage):

  1. render parity   parity_gate.py: three conversations through the deployed binary,
                     every model call byte-equal to r4dataset.student_source
  2. stats           block_drift_rows == 0, preview_claim_kept == 0, and no kept
                     `where` row without report_unsupported's `where`
  3. row checks      tools/run-r4.sh's: rows in both splits, follow-ups > 0 (full mode),
                     no reply mentioning the preview, no row with block drift
  4. train plan      train.py's dry plan on the split: rows, tokens, updates, 0 over 32,768
  5. new keys        every new / retyped key of the round (prepare report.json
                     targets.wanted_list, else target_list; holdout keys excepted) has at
                     least MIN_NEW_KEY_POSITIVES (6) train rows that WRITE it on a field
                     the site shows (author.hidden_writes) — r5 trained 0 for
                     contentBlock.stackedImage* and gallery.imagesLink[] and nothing said so.
                     SMOKE is exempt; `min_new_key_positives` / SOUSCHEF_MIN_NEW_KEY_POSITIVES
                     sets the floor (0 = off), SOUSCHEF_NEW_KEY_POSITIVES_WAIVE (comma list
                     of addresses) waives keys a person looked at

  6. ask rule      (r7 §1–3; stats.json `ask`) every kept `ask` row's rendered user text is VAGUE under
                     askrule (no value, no direction, no action — carried rows included: a violation moves
                     to rejects.jsonl as `ask_rule_violation`); every ask row's WRITE twin (case
                     meta.ask_pair) has a kept row (else `ask_unpaired`; clarify and no-title create_page
                     asks are exempt); ask + clarify rows are at most `ask_share_max` (3 %, trigger.ask_share_max
                     / SOUSCHEF_ASK_SHARE_MAX, 0 = off) of the rows, over-cap rows moved as `ask_cap` (their twins
                     stay), deterministic (sha1 of the case id inside strata, round-robin over fields) with
                     floors per ask family (ASK_FLOORS). Gate: violations_kept == 0 and unpaired_kept == 0.
                     trigger.positives_required (or SOUSCHEF_POSITIVES_REQUIRED, comma list) adds addresses
                     to gate 5 (r7 §5.3: the re-authored keys)

stats.json is rewritten from the final files after every drop (where_missing, an
emptied val), with the new-key counts and the font-pair count: r5's bucket copy still
read the pre-drop 2,465 / 2,679 rows.

Font pair (decided 2026-10-15: "the font" = EVERY site font, r4score font policy "all"):
rows whose case writes both TypographyColorsSection bodyFont and headingFont for one
request ("the site's font") are counted in stats.json `font_pair` and kept
(font_pair="keep", the default) — under "all" prepare's fix_site_fonts made those cases
expect every site font, so a kept row writes the pair AND the rest (`font_pair.site_fonts`
counts the rows whose case expects every site font); font_pair="drop"
(SOUSCHEF_FONT_PAIR=drop) moves them to rejects.jsonl (reason font_pair_policy), for a
round that must not teach site-wide font changes at all. Neither changes the exam.

Capabilities (r6-4: create_page and the dropdown option lists, 2026-10-15): stats.json `capabilities` counts the
kept rows (train / val) of the cases tagged meta.capability — contrast.py's create_page and list_options groups by
group (page:create, page:exists, list:tags …) and mode, and prepare's relabelled page-create rows — beside the
rejects of those cases (so a group that collected nothing usable shows as 0 kept, not missing).

Outputs: runs/<id>/dataset/{dataset/{train,val,rejects}.jsonl, stats.json,
dataset-manifest.json, manifest.json} (the layout train.py and pipeline-train.sh
read) and runs/<id>/model/context.tar.gz (the lab image's lab/model-image, the build
context Cloud Build uses — no repository access needed there; StorageSource.object
must be a .zip or a .tar.gz, https://docs.cloud.google.com/build/docs/api/reference/rest/v1/projects.builds).
"""
from __future__ import annotations

import io
import json
import os
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .. import LAB_DIR
from .common import ConfigError, Store, load_run, notification, read_jsonl, run_prefix, write_jsonl, write_result

TRAIN_FLAGS = ["--fields-context", "on", "--eval-steps", "150", "--val-max-rows", "150", "--checkpoint-steps", "50"]
MIN_NEW_KEY_POSITIVES = 6
NEW_KEY_REASONS = ("added_field", "retyped_field", "added_list", "added_container", "drift_new_key", "reauthor",
                   "positives_required")
ASK_SHARE_MAX = 0.03  # r7 §3: between r4 (0.25 %, rarely asked) and r6 (7.8 %, over-asked)
ASK_MODES = ("ask", "clarify")
# r7 §3's floors: None = every row of the family; n = at least min(available, n)
ASK_FLOORS: dict[str, int | None] = {"complaint": None, "which_one": None, "page_title_missing": None,
                                     "text_missing": 25, "choice_missing": 25}
FONT_PAIR = ("globalPanels.TypographyColorsSection.bodyFont", "globalPanels.TypographyColorsSection.headingFont")
FONT_PAIR_POLICIES = ("keep", "drop")


def _where_ok(r: dict) -> bool:
    """A `where` row teaches the deep link only when its answer calls report_unsupported with `where`."""
    if (r.get("meta") or {}).get("mode") != "where":
        return True
    for m in r.get("messages") or []:
        for c in m.get("tool_calls") or []:
            fn = c.get("function") or {}
            if fn.get("name") == "report_unsupported":
                try:
                    if json.loads(fn.get("arguments") or "{}").get("where"):
                        return True
                except ValueError:
                    pass
    return False


def where_missing(rows: list[dict]) -> int:
    """Kept `where` rows whose answer has no report_unsupported call carrying `where`."""
    return sum(1 for r in rows if not _where_ok(r))


def drop_where_missing(data: Path) -> dict[str, int]:
    """Move kept `where` rows without a `where` into rejects.jsonl (reason where_missing).

    r4score flags a report_unsupported whose `where` is invalid, but a teacher answer that
    passes without calling report_unsupported at all used to stay in the set, and the row
    gate (row_checks) then refused the whole dataset (2026-10-14, the first full round:
    10 of 2,465 train rows). Dropping them here keeps the gate strict for anything else.
    """
    dropped: dict[str, int] = {}
    rejects = read_jsonl(data / "rejects.jsonl")
    for split in ("train", "val"):
        rows = read_jsonl(data / f"{split}.jsonl")
        keep = [r for r in rows if _where_ok(r)]
        gone = [r for r in rows if not _where_ok(r)]
        dropped[split] = len(gone)
        if gone:
            write_jsonl(data / f"{split}.jsonl", keep)
            rejects += [{"id": r.get("id"), "case_id": (r.get("meta") or {}).get("case_id"),
                         "family": (r.get("meta") or {}).get("family"), "verdict": "pass",
                         "reasons": ["where_missing"]} for r in gone]
    if any(dropped.values()):
        write_jsonl(data / "rejects.jsonl", rejects)
    return dropped


def load_cases_by_id(cases_path: Path, ids: set[str]) -> dict[str, dict]:
    """The round's cases the rows name (cases.jsonl is ~200 MB: one streaming pass)."""
    out: dict[str, dict] = {}
    if not ids or not Path(cases_path).is_file():
        return out
    with Path(cases_path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if str(c.get("id")) in ids:
                out[str(c["id"])] = c
    return out


def _row_case_ids(rows: list[dict]) -> set[str]:
    return {str((r.get("meta") or {}).get("case_id") or str(r.get("id") or "").split("#")[0]) for r in rows}


def _case_of(r: dict, cases: dict[str, dict]) -> dict | None:
    return cases.get(str((r.get("meta") or {}).get("case_id") or str(r.get("id") or "").split("#")[0]))


def write_kind(case: dict) -> str:
    """set | reset | relative: how a positive case writes (keys-scored = a relative / chosen value)."""
    ops = (case.get("expected") or {}).get("ops") or []
    if ops and all(o.get("op") in ("reset", "restore_default") for o in ops if isinstance(o, dict)):
        return "reset"
    return "relative" if (case.get("score") or {}).get("mode") == "keys" else "set"


def new_keys_of(report: dict | None) -> list[dict]:
    """The round's new / retyped keys from prepare's report.json: targets.wanted_list (every
    wanted address, authored this round or already covered), else the authored target_list."""
    rep = report or {}
    lst = ((rep.get("targets") or {}).get("wanted_list")) or rep.get("target_list") or []
    return [t for t in lst if not t.get("holdout") and t.get("reason") in NEW_KEY_REASONS]


def _unwritable(m, addr: str) -> str | None:
    """Why no case can write an address (read-only / json), or None."""
    from .rebase import split_addr
    ctr, lst, key = split_addr(addr)
    grp, _, name = ctr.partition(".")
    c = m.doc.get("branding") if grp == "branding" else (m.doc.get(grp) or {}).get(name)
    if not isinstance(c, dict) or not key:
        return None
    f = ((c.get("lists") or {}).get(lst) or {}).get("item", {}).get(key) if lst else (c.get("fields") or {}).get(key)
    f = f or {}
    if f.get("derived_from"):
        return "read_only"
    if f.get("type") == "json":
        return "json"
    return None


def new_key_positives(rows: list[dict], cases: dict[str, dict], m, keys: list[dict], *, minimum: int,
                      waive: set[str] | frozenset = frozenset()) -> dict:
    """Per new / retyped key: the train rows that write it (positives, by kind), the rows
    whose write is hidden in their state (not counted), and the keys short of `minimum`."""
    from . import author as A
    want = {t["address"]: t for t in keys}
    per: dict[str, dict] = {a: {"reason": t.get("reason"), "positives": 0, "kinds": {}, "hidden": 0} for a, t in want.items()}
    for r in rows:
        c = _case_of(r, cases)
        if c is None or (c.get("score") or {}).get("mode") not in ("exact", "keys"):
            continue
        hit = A.case_addresses(c, m) & set(want)
        if not hit:
            continue
        hidden = bool(A.hidden_writes(c, m))
        kind = write_kind(c)
        for a in hit:
            if hidden:
                per[a]["hidden"] += 1
                continue
            per[a]["positives"] += 1
            per[a]["kinds"][kind] = per[a]["kinds"].get(kind, 0) + 1
    short: dict[str, int] = {}
    for a, rec in per.items():
        why = "waived" if a in waive else _unwritable(m, a)
        if why:
            rec["exempt"] = why
        elif rec["positives"] < minimum:
            short[a] = rec["positives"]
    return {"min": minimum, "keys": per, "short": short, "ok": not short}


def site_font_rows(rows: list[dict], cases: dict[str, dict]) -> list[str]:
    """Row ids whose case expects every site font (expected.site_fonts: r4score font policy "all")."""
    return [str(r.get("id")) for r in rows
            if isinstance(((_case_of(r, cases) or {}).get("expected") or {}).get("site_fonts"), dict)]


def font_pair_rows(rows: list[dict], cases: dict[str, dict], m) -> list[str]:
    """Row ids whose case writes both bodyFont and headingFont for one request ("the site's font")."""
    from . import author as A
    out = []
    for r in rows:
        c = _case_of(r, cases)
        if c is not None and set(FONT_PAIR) <= A.case_addresses(c, m):
            out.append(str(r.get("id")))
    return out


# --- the ask rule, the ask / write pairs and the ask cap (r7 §1.2, §2.3, §3) ---------------------------------------------

def _sha(s: str) -> str:
    import hashlib
    return hashlib.sha1(str(s).encode("utf-8")).hexdigest()


def _cid(r: dict) -> str:
    return str((r.get("meta") or {}).get("case_id") or str(r.get("id") or "").split("#")[0])


def row_user_text(r: dict, case: dict | None = None) -> str:
    """The scored turn's user text as the row trains it (the last user message, the context and candidate-fields
    blocks stripped), else the case's request."""
    from ..r4dataset import _block_text, strip_turn_blocks
    users = [m for m in r.get("messages") or [] if m.get("role") == "user"]
    if users:
        t = strip_turn_blocks(_block_text(users[-1].get("content")))
        if t.strip():
            return t.strip()
    return str((case or {}).get("request") or "")


def ask_exempt(case: dict | None) -> str | None:
    """Why an ask / clarify case is outside the ask rule and the pairing: a clarify (explicit candidates), a target
    ambiguity (meta.ask_reason which_one), a no-title create_page (the rules: ask for the name)."""
    c = case or {}
    meta = c.get("meta") or {}
    if (c.get("score") or {}).get("mode") == "clarify":
        return "clarify"
    if meta.get("ask_reason") == "which_one":
        return "which_one"
    if meta.get("capability") == "create_page":
        return "create_page"
    return None


def _is_ask(r: dict) -> bool:
    return (r.get("meta") or {}).get("mode") in ASK_MODES


def ask_rule_rows(rows: list[dict], cases: dict[str, dict], m) -> dict[str, str]:
    """{row id: why} of the kept `ask` rows whose user text is NOT vague (askrule.classify_case on the case's
    vocabulary) — a carried row whose case predates the rule included."""
    from ..askrule import classify_case
    out: dict[str, str] = {}
    for r in rows:
        if (r.get("meta") or {}).get("mode") != "ask":
            continue
        c = _case_of(r, cases)
        if c is None or ask_exempt(c):
            continue
        v = classify_case(c, m, text=row_user_text(r, c))
        if not v.vague:
            out[str(r.get("id"))] = f"{v.kind}:{v.why}"
    return out


def _twins(c: dict) -> list[str]:
    p = (c.get("meta") or {}).get("ask_pair") or {}
    if p.get("role") != "ask":
        return []
    return [str(t) for t in (p.get("twins") or [p.get("twin")]) if t]


def unpaired_ask_rows(rows: list[dict], cases: dict[str, dict]) -> list[str]:
    """Row ids of the kept `ask` rows whose case has no write twin with a kept row (r7 §2.3; ask_exempt aside)."""
    kept = {_cid(r) for r in rows}
    out = []
    for r in rows:
        if (r.get("meta") or {}).get("mode") != "ask":
            continue
        c = _case_of(r, cases)
        if c is None or ask_exempt(c):
            continue
        if not any(t in kept for t in _twins(c)):
            out.append(str(r.get("id")))
    return out


def _field_key(c: dict | None) -> str:
    """The field a case is about (the ask cap's round-robin unit and the pairs report): contrast's key, the authored
    target, the expected keys, else the case's fields."""
    c = c or {}
    meta = c.get("meta") or {}
    if (meta.get("contrast") or {}).get("key"):
        return str(meta["contrast"]["key"])
    if meta.get("target_address"):
        return str(meta["target_address"]).rsplit(".", 1)[-1]
    for k in (c.get("expected") or {}).get("keys") or []:
        if isinstance(k, dict) and (k.get("key") or k.get("list")):
            return str(k.get("key") or k.get("list"))
    fs = c.get("fields") or []
    return str(fs[0]) if fs else "?"


def _round_robin(rows: list[dict], cases: dict[str, dict], n: int, have: set[str] | frozenset = frozenset()) -> list[dict]:
    """Up to n rows, one per field in turn (fields without a kept ask in `have` first; sha1 of the field, then of the
    case id inside a field) — deterministic."""
    by: dict[str, list[dict]] = {}
    for r in sorted(rows, key=lambda x: _sha(_cid(x))):
        by.setdefault(_field_key(_case_of(r, cases)), []).append(r)
    order = sorted(by, key=lambda f: (f in have, _sha(f)))
    out: list[dict] = []
    while len(out) < n and any(by.values()):
        for f in order:
            if by[f] and len(out) < n:
                out.append(by[f].pop(0))
    return out


def cap_ask_rows(rows: list[dict], cases: dict[str, dict], m, share: float) -> tuple[set[str], dict]:
    """(row ids over the ask cap, report): ask + clarify rows at most `share` of `rows` (0 = off), the families'
    floors (ASK_FLOORS) kept first, the rest filled round-robin over fields (module docstring)."""
    from ..askrule import ask_family_of
    asks = [r for r in rows if _is_ask(r)]
    fam = {str(r.get("id")): ask_family_of(_case_of(r, cases) or {"score": {"mode": (r.get("meta") or {}).get("mode")}}, m)
           for r in asks}
    rep: dict[str, Any] = {"share_max": share, "rows": len(rows), "ask_rows_before": len(asks),
                           "by_family_before": dict(sorted(Counter(fam.values()).items()))}
    if share <= 0:
        rep.update(cap=None, capped=0, off=True)
        return set(), rep
    cap = int(share * len(rows))
    keep: list[dict] = []
    for f, floor in ASK_FLOORS.items():
        members = [r for r in asks if fam[str(r.get("id"))] == f]
        keep += members if floor is None else _round_robin(members, cases, floor)
    floors = len(keep)
    kept_ids = {str(r.get("id")) for r in keep}
    rest = [r for r in asks if str(r.get("id")) not in kept_ids]
    have = {_field_key(_case_of(r, cases)) for r in keep}
    keep += _round_robin(rest, cases, max(0, cap - floors), have)
    kept_ids = {str(r.get("id")) for r in keep}
    drop = {str(r.get("id")) for r in asks} - kept_ids
    rep.update(cap=cap, floors_kept=floors, floors_over_cap=max(0, floors - cap), capped=len(drop),
               ask_rows=len(kept_ids), by_family=dict(sorted(Counter(fam[i] for i in kept_ids).items())))
    return drop, rep


def ask_pairs_report(rows: list[dict], cases: dict[str, dict]) -> dict:
    """{field: {ask, twin_write, other_relative}}: the kept ask rows, their twins' write rows and the other relative
    (direction-scored) write rows per field."""
    out: dict[str, dict] = {}
    for r in rows:
        c = _case_of(r, cases)
        if c is None:
            continue
        meta = c.get("meta") or {}
        mode = (r.get("meta") or {}).get("mode")
        role = (meta.get("ask_pair") or {}).get("role")
        if mode == "ask":
            k = "ask"
        elif role == "write":
            k = "twin_write"
        elif mode == "keys" and (c.get("expected") or {}).get("direction"):
            k = "other_relative"
        else:
            continue
        rec = out.setdefault(_field_key(c), {"ask": 0, "twin_write": 0, "other_relative": 0})
        rec[k] += 1
    return {f: out[f] for f in sorted(out) if out[f]["ask"] or out[f]["twin_write"]}


def enforce_ask_rules(data: Path, cases: dict[str, dict], m, *, share: float = ASK_SHARE_MAX) -> dict:
    """§1.2's ask_rule_rows, §2.3's pairing and §3's cap on the split files, in that order (each moves rows to
    rejects.jsonl with its reason); stats.json `ask` — the counts after every drop re-measured (violations_kept,
    unpaired_kept: the gate's numbers)."""
    rows = read_jsonl(data / "train.jsonl") + read_jsonl(data / "val.jsonl")
    before = sum(1 for r in rows if _is_ask(r))
    bad = ask_rule_rows(rows, cases, m)
    d1 = drop_rows(data, set(bad), "ask_rule_violation") if bad else {"train": 0, "val": 0}
    rows = read_jsonl(data / "train.jsonl") + read_jsonl(data / "val.jsonl")
    unpaired = unpaired_ask_rows(rows, cases)
    d2 = drop_rows(data, set(unpaired), "ask_unpaired") if unpaired else {"train": 0, "val": 0}
    rows = read_jsonl(data / "train.jsonl") + read_jsonl(data / "val.jsonl")
    over, cap_rep = cap_ask_rows(rows, cases, m, share)
    d3 = drop_rows(data, over, "ask_cap") if over else {"train": 0, "val": 0}
    rows = read_jsonl(data / "train.jsonl") + read_jsonl(data / "val.jsonl")
    asks = [r for r in rows if _is_ask(r)]
    exempt = Counter(ask_exempt(_case_of(r, cases)) for r in asks)
    exempt.pop(None, None)
    return {"rows_before": before, "rows": len(asks), "share": round(len(asks) / max(len(rows), 1), 4),
            "share_max": share, "cap": cap_rep,
            "violations": {"dropped": d1, "ids": dict(list(bad.items())[:50])},
            "violations_kept": len(ask_rule_rows(rows, cases, m)),
            "unpaired": {"dropped": d2, "ids": unpaired[:50]},
            "unpaired_kept": len(unpaired_ask_rows(rows, cases)),
            "capped": d3, "exempt": dict(exempt), "pairs": ask_pairs_report(rows, cases)}


def ask_share_of(trig: dict, explicit: float | None = None) -> float:
    """The ask cap: `explicit`, else SOUSCHEF_ASK_SHARE_MAX, else trigger.ask_share_max, else ASK_SHARE_MAX."""
    v: Any = explicit
    if v is None:
        v = os.environ.get("SOUSCHEF_ASK_SHARE_MAX", "").strip() or trig.get("ask_share_max")
    if v in (None, ""):
        return ASK_SHARE_MAX
    try:
        f = float(v)
    except (TypeError, ValueError) as e:
        raise ConfigError(f"ask_share_max {v!r} is not a number") from e
    if not 0 <= f < 1:
        raise ConfigError(f"ask_share_max {f} is not a share in [0, 1)")
    return f


def positives_required_of(trig: dict) -> list[dict]:
    """trigger.positives_required (a list or a comma string), else SOUSCHEF_POSITIVES_REQUIRED: addresses gate 5 counts
    like new keys (reason positives_required)."""
    v = os.environ.get("SOUSCHEF_POSITIVES_REQUIRED") or trig.get("positives_required") or []
    items = v if isinstance(v, list) else str(v).split(",")
    return [{"address": a, "reason": "positives_required", "holdout": False}
            for a in sorted({str(x).strip() for x in items if str(x).strip()})]


def capability_rows(train: list[dict], val: list[dict], rejects: list[dict], cases: dict[str, dict]) -> dict:
    """{capability: {train, val, rejected, by_group: {group: {train, val, rejected}}, by_mode: {mode: kept}}} for the
    rows whose case carries meta.capability (module docstring)."""
    out: dict[str, dict] = {}

    def rec(cap: str, group: str) -> tuple[dict, dict]:
        r = out.setdefault(cap, {"train": 0, "val": 0, "rejected": 0, "by_group": {}, "by_mode": {}})
        return r, r["by_group"].setdefault(group, {"train": 0, "val": 0, "rejected": 0})
    for split, rows in (("train", train), ("val", val), ("rejected", rejects)):
        for r in rows:
            c = cases.get(str(r.get("case_id") or "")) if split == "rejected" else _case_of(r, cases)
            meta = (c or {}).get("meta") or {}
            cap = meta.get("capability")
            if not cap:
                continue
            total, grp = rec(str(cap), str(meta.get("cap_group") or meta.get("group") or "?"))
            total[split] += 1
            grp[split] += 1
            if split != "rejected":
                mode = str(((c or {}).get("score") or {}).get("mode"))
                total["by_mode"][mode] = total["by_mode"].get(mode, 0) + 1
    for r in out.values():
        r["by_group"] = dict(sorted(r["by_group"].items()))
    return dict(sorted(out.items()))


def hidden_write_rows(rows: list[dict], cases: dict[str, dict], m) -> list[str]:
    """Row ids whose case writes a field its state hides (author.hidden_writes) — reported in
    stats.json, not dropped: r5 trained 22 perks stackedImage* rows like that."""
    from . import author as A
    return [str(r.get("id")) for r in rows if (c := _case_of(r, cases)) is not None and A.hidden_writes(c, m)]


def drop_rows(data: Path, ids: set[str], reason: str) -> dict[str, int]:
    """Moves rows (by id) from train / val to rejects.jsonl with `reason`."""
    dropped: dict[str, int] = {}
    rejects = read_jsonl(data / "rejects.jsonl")
    for split in ("train", "val"):
        rows = read_jsonl(data / f"{split}.jsonl")
        gone = [r for r in rows if str(r.get("id")) in ids]
        dropped[split] = len(gone)
        if gone:
            write_jsonl(data / f"{split}.jsonl", [r for r in rows if str(r.get("id")) not in ids])
            rejects += [{"id": r.get("id"), "case_id": (r.get("meta") or {}).get("case_id"),
                         "family": (r.get("meta") or {}).get("family"), "verdict": "pass", "reasons": [reason]} for r in gone]
    if any(dropped.values()):
        write_jsonl(data / "rejects.jsonl", rejects)
    return dropped


def refresh_stats(data: Path, st: dict, **extra: Any) -> dict:
    """stats.json (and dataset-manifest.json's counts) recomputed from the files as they
    stand, plus `extra`; returns the new stats."""
    from ..r4dataset import estimate_tokens
    from ..r4score import mentions_preview
    from collections import Counter
    train, val = read_jsonl(data / "train.jsonl"), read_jsonl(data / "val.jsonl")
    rejects = read_jsonl(data / "rejects.jsonl")
    kept = train + val
    fam = Counter(str((r.get("meta") or {}).get("family")) for r in kept)
    toks = sorted(estimate_tokens(r) for r in train)
    st = dict(st)
    st.update({
        "rows": len(kept), "train": len(train), "val": len(val),
        "by_family": dict(sorted(fam.items())),
        "by_family_share": {k: round(v / max(len(kept), 1), 3) for k, v in sorted(fam.items())},
        "by_mode": dict(Counter(str((r.get("meta") or {}).get("mode")) for r in kept)),
        "multi_turn_rows": sum(1 for r in kept if (r.get("meta") or {}).get("history_turns")),
        "rows_with_block": sum(1 for r in kept if (r.get("meta") or {}).get("fields_context")),
        "preview_claim_kept": sum(1 for r in kept for x in r.get("messages") or []
                                  if x.get("role") == "assistant" and mentions_preview(x.get("content"))),
        "tokens_train": {**(st.get("tokens_train") or {}), "total": sum(toks),
                         "median": toks[len(toks) // 2] if toks else 0, "max": toks[-1] if toks else 0},
        "rejects": len(rejects), "stats_refreshed": "after the dataset stage's drops",
    })
    st.update(extra)
    (data / "stats.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
    mp = data / "dataset-manifest.json"
    if mp.is_file():
        man = json.loads(mp.read_text(encoding="utf-8"))
        man["counts"] = {"train": len(train), "val": len(val), "rejects": len(rejects)}
        if isinstance(man.get("fields_context"), dict):
            man["fields_context"].update(rows=len(kept), rows_with_block=st["rows_with_block"])
        mp.write_text(json.dumps(man, indent=1), encoding="utf-8")
    return st


def row_checks(data: Path, *, need_followups: bool) -> dict:
    """tools/run-r4.sh's row checks, as a dict of problems (empty = pass)."""
    from ..r4score import mentions_preview
    out: dict[str, Any] = {"splits": {}, "problems": []}
    for split in ("train", "val"):
        rows = read_jsonl(data / f"{split}.jsonl")
        block = sum(1 for r in rows if (r.get("meta") or {}).get("fields_context"))
        follow = sum(1 for r in rows if (r.get("meta") or {}).get("history_assistant_turns"))
        drift = sum(1 for r in rows if (r.get("meta") or {}).get("block_drift"))
        preview = sum(1 for r in rows for m in r.get("messages") or [] if m.get("role") == "assistant"
                      and mentions_preview(m.get("content")))
        out["splits"][split] = {"rows": len(rows), "with_block": block, "followups": follow, "drift": drift,
                                "preview": preview, "where_missing": where_missing(rows)}
        if not rows:
            out["problems"].append(f"{split}: no rows")
        if preview:
            out["problems"].append(f"{split}: {preview} replies mention the preview")
        if drift:
            out["problems"].append(f"{split}: {drift} rows with block drift")
        if out["splits"][split]["where_missing"]:
            out["problems"].append(f"{split}: {out['splits'][split]['where_missing']} where rows without `where`")
    if need_followups and not out["splits"]["train"]["followups"]:
        out["problems"].append("train: no follow-up rows")
    return out


def train_plan(data: Path, name: str) -> dict:
    """train.py's dry plan (no torch): rows, tokens, updates, over-length."""
    from .. import train as T
    args = T.parse_args(["--data", str(data), "--name", name, *TRAIN_FLAGS, "--dry-run"])
    tr, va = T.load_split(args.data, "train"), T.load_split(args.data, "val")
    problems = T.check_rows(tr, "train") + T.check_rows(va, "val")
    p = T.plan(args, tr, va)
    mismatch = T.check_fields_context(p["fields_context"], args.data)
    over = int((p.get("rows") or {}).get("over_length_estimate") or 0)
    return {"problems": problems + ([mismatch] if mismatch else []) + ([f"{over} rows over max_seq"] if over else []),
            "updates": p.get("updates"), "rows": p.get("rows"), "tokens_estimate": p.get("tokens_estimate")}


def ensure_val(data: Path) -> int:
    """SMOKE's handful of rows can leave val empty after the group split: move the train rows of one group there."""
    tr, va = read_jsonl(data / "train.jsonl"), read_jsonl(data / "val.jsonl")
    if va or len(tr) < 2:
        return 0
    group = sorted({(r.get("meta") or {}).get("group") or r["id"] for r in tr})[0]
    moved = [r for r in tr if ((r.get("meta") or {}).get("group") or r["id"]) == group]
    if len(moved) == len(tr):
        moved = tr[:1]
    keep = [r for r in tr if r not in moved]
    for name, rows in (("train.jsonl", keep), ("val.jsonl", moved)):
        (data / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return len(moved)


def model_context_tgz(dest: Path) -> Path:
    """A gzip tar of lab/model-image (Dockerfile, entrypoint, model-context.sh): the Cloud Build source."""
    src = LAB_DIR / "model-image"
    if not (src / "Dockerfile").is_file():
        raise ConfigError(f"{src}/Dockerfile is missing")
    dest.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in sorted(src.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                tar.add(p, arcname=f"model-image/{p.relative_to(src)}")
    dest.write_bytes(buf.getvalue())
    return dest


def _env_int(name: str) -> int | None:
    v = os.environ.get(name, "").strip()
    try:
        return int(v) if v else None
    except ValueError as e:
        raise ConfigError(f"{name}={v!r} is not a whole number") from e


TRAIN_CONFIG_KEYS = ("seed", "seeds", "arms", "init_adapter", "init_lr", "gen_check")  # train.PIPELINE_ENV's keys


def _pipeline_bucket(store: Store | None) -> str:
    """gs://<bucket> the trainer resolves paths against (PIPELINE_BUCKET), for checking trigger.train here."""
    uri = str(getattr(store, "uri", "") or "")
    b = uri if uri.startswith("gs://") else (os.environ.get("PIPELINE_BUCKET") or "gs://pipeline-bucket")
    b = b.rstrip("/")
    return b if b.startswith("gs://") else "gs://" + b.split("/")[-1]


def train_config_of(run_file: dict, store: Store | None = None) -> dict | None:
    """run.json trigger.train → runs/<id>/train-config.json (the trainer's settings: train.pipeline_config).
    None when the trigger sets none. Checked HERE, before any GPU (r6 review: a bad value used to fail on the H100,
    after the image pull): the keys, every value through the trainer's own resolver (train.pipeline_config: seeds,
    arms — base:<seed> / warm:<seed>, a warm arm needing init_adapter and an init_adapter a warm arm — learning rate,
    generation check, an adapter path inside the pipeline bucket's runs/ or base/), and, with the store, that the
    warm-start adapter exists (adapter_config.json under it). ConfigError otherwise."""
    cfg = ((run_file or {}).get("trigger") or {}).get("train")
    if cfg in (None, {}, ""):
        return None
    if not isinstance(cfg, dict):
        raise ConfigError(f"trigger.train must be an object of {', '.join(TRAIN_CONFIG_KEYS)}")
    unknown = sorted(set(cfg) - set(TRAIN_CONFIG_KEYS))
    if unknown:
        raise ConfigError(f"trigger.train: unknown key(s) {', '.join(unknown)} (known: {', '.join(TRAIN_CONFIG_KEYS)})")
    from .. import train as T
    b = _pipeline_bucket(store)
    try:
        c = T.pipeline_config({}, cfg, b)
    except ValueError as e:
        raise ConfigError(f"trigger.train: {e}") from e
    if c["init_adapter"] and store is not None:
        rel = c["init_adapter"][len(b) + 1:].rstrip("/")
        found = [p for p in ("adapter_config.json", "adapter/adapter_config.json") if store.exists(f"{rel}/{p}")]
        if not found:
            raise ConfigError(f"trigger.train.init_adapter: no adapter_config.json under {rel}/ in the pipeline bucket "
                              "(r4's adapter is not uploaded yet: docs/RETRAIN.md R4)")
        # its LoRA shape and base, as train.py check_init_adapter checks them on the GPU (r16/α16 on the recipe's base)
        acfg = store.read_json(f"{rel}/{found[0]}") or {}
        if isinstance(acfg, dict) and acfg:
            r, alpha = acfg.get("r"), acfg.get("lora_alpha")
            if (r, alpha) != (T.RECIPE["rank"], T.RECIPE["alpha"]):
                raise ConfigError(f"trigger.train.init_adapter: {rel} is a r{r}/α{alpha} LoRA; the recipe trains "
                                  f"r{T.RECIPE['rank']}/α{T.RECIPE['alpha']} (a warm start continues the same LoRA)")
            base = str(acfg.get("base_model_name_or_path") or "")
            if base and "/" in base and not base.startswith("/") and base != T.RECIPE["base"]:
                raise ConfigError(f"trigger.train.init_adapter: {rel} was trained on {base}, the recipe's base is "
                                  f"{T.RECIPE['base']}")
    return dict(cfg)


def run(store: Store, run_id: str, *, mode: str, workdir: Path, parity: str = "required", engine_fp: str | None = None,
        binary: str | None = None, caps: bool | None = None, min_new_key_positives: int | None = None,
        waive_new_keys: set[str] | None = None, font_pair: str | None = None, ask_share_max: float | None = None,
        progress: Callable[[str], None] = print) -> dict:
    """parity: required (GCP) | auto (run when the binary is there) | skip (DRY without a binary).
    caps: r4dataset's section and family caps (tuned for thousands of rows) — off for SMOKE by default.
    min_new_key_positives: gate 5's floor (default SOUSCHEF_MIN_NEW_KEY_POSITIVES, else 6; 0 = off).
    waive_new_keys: addresses gate 5 skips (default SOUSCHEF_NEW_KEY_POSITIVES_WAIVE, comma-separated).
    font_pair: keep | drop (default SOUSCHEF_FONT_PAIR, else keep) — module docstring.
    ask_share_max: the ask cap (gate 6; default SOUSCHEF_ASK_SHARE_MAX, trigger.ask_share_max, else 0.03; 0 = off).
    Each default also reads run.json's trigger (font_pair, min_new_key_positives, new_key_positives_waive) after
    the environment; trigger.train is written to runs/<id>/train-config.json for the trainer (an existing file wins)."""
    from .. import r4dataset
    from . import parity_gate
    pre = run_prefix(run_id)
    trig = load_run(store, run_id).get("trigger") or {}
    train_cfg = train_config_of({"trigger": trig}, store)
    smoke = mode == "smoke"
    local = Path(workdir)
    cases, manifest = local / "cases.jsonl", local / "manifest.json"
    if not store.download(f"{pre}/prepare/cases.jsonl", cases) or not store.download(f"{pre}/prepare/manifest.json", manifest):
        raise ConfigError("the prepare stage's cases.jsonl / manifest.json are missing")
    col = store.read_json(f"{pre}/result-collect.json") or {}
    if col.get("status") != "ok":
        raise ConfigError(f"the collect stage has not finished ok (status {col.get('status')})")
    logs_dir = local / "logs"
    store.sync_down(col["logs_prefix"], logs_dir)
    logs = [logs_dir / n for n in col.get("logs") or [] if (logs_dir / n).is_file()]
    if not logs:
        raise ConfigError(f"no collection logs under {col['logs_prefix']}")
    why = None
    if store.list("cases/why"):
        why = local / "why"
        store.sync_down("cases/why", why)
    if engine_fp is None and os.environ.get("ENGINE"):
        from ..engine import engine_fingerprint
        engine_fp = (engine_fingerprint(os.environ["ENGINE"]) or {}).get("fingerprint")
    out = local / "data"
    progress(f"dataset: building from {len(logs)} log(s)")
    caps = (not smoke) if caps is None else caps
    st = r4dataset.build(cases, logs, manifest, out, why=why, val_pct=20 if smoke else 8, section_cap_scope=("F0", "F1"),
                         section_cap=0.05 if caps else 0.0, family_max_share=0.40 if caps else 0.0,
                         engine_fp=engine_fp, progress=progress)
    if not caps:
        st["val_moved"] = ensure_val(out / "dataset")
    dropped = drop_where_missing(out / "dataset")
    if any(dropped.values()):
        n = sum(dropped.values())
        progress(f"dataset: dropped {n} where row(s) without `where` (train {dropped['train']}, val {dropped['val']})")
        st["where_missing_dropped"] = dropped
        first = dict(st.get("rejects_first_reason") or {})
        first["where_missing"] = first.get("where_missing", 0) + n
        st["rejects_first_reason"] = first
    # the new-key gate and the font-pair count read the rows' cases on the deployed manifest
    from ..manifest import Manifest
    m_new = Manifest(json.loads(manifest.read_text(encoding="utf-8")))
    data = out / "dataset"
    all_rows = read_jsonl(data / "train.jsonl") + read_jsonl(data / "val.jsonl")
    cases_by_id = load_cases_by_id(cases, _row_case_ids(all_rows))
    # 6. the ask rule, the pairs and the cap (r7)
    ask_st = enforce_ask_rules(data, cases_by_id, m_new, share=ask_share_of(trig, ask_share_max))
    gone = {k: sum(v.values()) for k, v in (("ask_rule_violation", ask_st["violations"]["dropped"]),
                                           ("ask_unpaired", ask_st["unpaired"]["dropped"]), ("ask_cap", ask_st["capped"]))}
    if any(gone.values()):
        first = dict(st.get("rejects_first_reason") or {})
        for k, n in gone.items():
            if n:
                first[k] = first.get(k, 0) + n
        st["rejects_first_reason"] = first
        progress(f"dataset: ask rows {ask_st['rows_before']} → {ask_st['rows']} ({ask_st['share']:.1%}; dropped {gone})")
    all_rows = read_jsonl(data / "train.jsonl") + read_jsonl(data / "val.jsonl")
    font_pair = str(font_pair or os.environ.get("SOUSCHEF_FONT_PAIR") or trig.get("font_pair") or "keep").strip().lower()
    if font_pair not in FONT_PAIR_POLICIES:
        raise ConfigError(f"font_pair {font_pair!r} is not one of {', '.join(FONT_PAIR_POLICIES)}")
    pair_ids = font_pair_rows(all_rows, cases_by_id, m_new)
    font = {"policy": font_pair, "rows": len(pair_ids), "ids": pair_ids[:50],
            "site_fonts": len(site_font_rows(all_rows, cases_by_id))}
    if font_pair == "drop" and pair_ids:
        fd = drop_rows(data, set(pair_ids), "font_pair_policy")
        font["dropped"] = fd
        first = dict(st.get("rejects_first_reason") or {})
        first["font_pair_policy"] = first.get("font_pair_policy", 0) + sum(fd.values())
        st["rejects_first_reason"] = first
        progress(f"dataset: dropped {sum(fd.values())} font-pair row(s) (font_pair=drop)")
    minimum = min_new_key_positives if min_new_key_positives is not None else _env_int("SOUSCHEF_MIN_NEW_KEY_POSITIVES")
    if minimum is None and trig.get("min_new_key_positives") not in (None, ""):
        try:
            minimum = int(trig["min_new_key_positives"])
        except (TypeError, ValueError) as e:
            raise ConfigError(f"trigger.min_new_key_positives {trig['min_new_key_positives']!r} is not a whole number") from e
    minimum = MIN_NEW_KEY_POSITIVES if minimum is None else minimum
    if waive_new_keys is None:
        w = os.environ.get("SOUSCHEF_NEW_KEY_POSITIVES_WAIVE") or trig.get("new_key_positives_waive") or ""
        waive_new_keys = {str(a).strip() for a in (w if isinstance(w, list) else str(w).split(",")) if str(a).strip()}
    waive = waive_new_keys
    keys = new_keys_of(store.read_json(f"{pre}/prepare/report.json"))
    keys += [t for t in positives_required_of(trig) if t["address"] not in {k["address"] for k in keys}]
    nk = new_key_positives(read_jsonl(data / "train.jsonl"), cases_by_id, m_new, keys, minimum=minimum, waive=waive)
    hidden_ids = hidden_write_rows(read_jsonl(data / "train.jsonl"), cases_by_id, m_new)
    rejects_now = read_jsonl(data / "rejects.jsonl")
    cap_cases = {**cases_by_id, **load_cases_by_id(cases, {str(r.get("case_id")) for r in rejects_now if r.get("case_id")}
                                                   - set(cases_by_id))}
    capabilities = capability_rows(read_jsonl(data / "train.jsonl"), read_jsonl(data / "val.jsonl"), rejects_now, cap_cases)
    st = refresh_stats(data, st, font_pair=font, hidden_write_rows={"train": len(hidden_ids), "ids": hidden_ids[:50]},
                       new_key_positives={a: {k: v for k, v in r.items() if k != "reason"} for a, r in nk["keys"].items()},
                       capabilities=capabilities, ask=ask_st)
    gates: dict[str, Any] = {}
    problems: list[str] = []
    # 1. render parity
    exe = parity_gate.service_binary(binary)
    if parity == "skip" or (parity == "auto" and exe is None):
        gates["parity"] = {"skipped": True}
    else:
        if exe is None:
            raise ConfigError("the parity gate needs the deployed service binary (SOUS_CHEF_BIN / /opt/sous-chef/bin/sous-chef)")
        gates["parity"] = parity_gate.run(manifest, local, binary=exe, engine_path=os.environ.get("ENGINE"))
        if not gates["parity"]["ok"]:
            problems.append("render parity: " + "; ".join(gates["parity"]["failures"][:5] or ["no call checked"]))
    # 2. stats
    if st.get("block_drift_rows"):
        problems.append(f"stats: block_drift_rows {st['block_drift_rows']}")
    if st.get("preview_claim_kept"):
        problems.append(f"stats: preview_claim_kept {st['preview_claim_kept']}")
    if ask_st["violations_kept"] or ask_st["unpaired_kept"]:
        problems.append(f"stats: ask violations_kept {ask_st['violations_kept']}, unpaired_kept {ask_st['unpaired_kept']}")
    # 3. row checks
    gates["rows"] = row_checks(out / "dataset", need_followups=not smoke)
    problems += gates["rows"]["problems"]
    # 4. the train plan
    gates["train_plan"] = train_plan(out / "dataset", f"{run_id}-sft")
    problems += gates["train_plan"]["problems"]
    # 5. new keys: enough positive-write train rows for every new / retyped key
    if smoke or minimum <= 0:
        gates["new_key_positives"] = {"skipped": "smoke" if smoke else "min 0", "keys": nk["keys"]}
    else:
        gates["new_key_positives"] = nk
        if nk["short"]:
            listed = ", ".join(f"{a} {n}" for a, n in sorted(nk["short"].items())[:8])
            problems.append(f"new_key_positives: {len(nk['short'])} new/retyped key(s) have fewer than {minimum} "
                            f"positive-write train rows ({listed}); author more for them (prepare/author-report.json "
                            f"short_of_positive) or waive a key a person checked (SOUSCHEF_NEW_KEY_POSITIVES_WAIVE)")
    (out / "gates.json").write_text(json.dumps(gates, indent=1), encoding="utf-8")
    store.sync_up(out, f"{pre}/dataset")
    ctx = model_context_tgz(local / "context.tar.gz")
    store.upload(ctx, f"{pre}/model/context.tar.gz")
    metrics = {"rows": st.get("rows"), "train": st.get("train"), "val": st.get("val"),
               "followups": st.get("multi_turn_rows"), "by_family": st.get("by_family"),
               "rejects_first_reason": st.get("rejects_first_reason"), "block_drift_rows": st.get("block_drift_rows"),
               "preview_claim_kept": st.get("preview_claim_kept"), "updates": gates["train_plan"].get("updates"),
               "tokens_train": st.get("tokens_train"), "parity": {k: v for k, v in gates["parity"].items() if k != "failures"},
               "new_key_short": nk["short"], "font_pair_rows": font["rows"], "hidden_write_rows": len(hidden_ids),
               "ask": {k: ask_st[k] for k in ("rows", "share", "share_max", "violations_kept", "unpaired_kept")},
               "capabilities": {k: {"train": v["train"], "val": v["val"], "rejected": v["rejected"]}
                                for k, v in capabilities.items()}}
    if problems:
        note = notification("stage_failed", "error", "Sous Chef pipeline failed: dataset (gate)",
                            "The training set failed its gates: " + "; ".join(problems[:6]) +
                            f". Nothing was trained. Artifacts: {pre}/dataset/ · Playbook: docs/RETRAIN.md#r3-dataset",
                            {"stage": "dataset", "reason": "gate", "problems": problems}, dedup_key=f"{run_id}:dataset")
        write_result(store, run_id, "dataset", "gate_failed", metrics=metrics, notification_=note)
        raise ConfigError("dataset gates: " + "; ".join(problems[:6]), reason="dataset_gate")
    if train_cfg is not None:
        if store.exists(f"{pre}/train-config.json"):
            metrics["train_config"] = {"kept_existing": True}
        else:
            store.write_json(f"{pre}/train-config.json", train_cfg)
            metrics["train_config"] = train_cfg
        # the arms against the 12 h Vertex window, on this dataset's updates (every arm trains on the same rows):
        # a planning figure; the trainer's own guard skips an arm that would not end in time and records it
        from .. import train as T
        arms = T.pipeline_config({}, train_cfg, _pipeline_bucket(store))["arms"]
        if len(arms) > 1:
            win = T.window_estimate(len(arms), metrics.get("updates"))
            metrics["train_window"] = {"arms": [a["arm"] for a in arms], **win}
            if not smoke and not win["fits"]:
                progress(f"dataset: {len(arms)} arms of {metrics.get('updates')} updates need about {win['total_h']} h "
                         f"of H100, over the {win['window_h']:.0f} h window: about {win['fit_arms']} fit; the trainer "
                         "trains the arms in order and skips the rest (recorded in selection.skipped)")
    return write_result(store, run_id, "dataset", "ok", metrics=metrics,
                        artifacts=[f"{pre}/dataset/dataset/train.jsonl", f"{pre}/dataset/dataset/stats.json",
                                   f"{pre}/model/context.tar.gz"])
