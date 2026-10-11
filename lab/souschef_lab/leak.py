"""leak — the hold-out filter for training cases (docs/LAB.md §5b, r4 PLAN §3.1 / L2).

The real-phrasing exam (fixtures/exam/*.jsonl) and the merchant's real turns
are the test; nothing a model trains on may copy or closely paraphrase them.

Normalising: NFKC, lower case, apostrophes dropped, British spellings folded to
US ones (colour → color, centre → center, grey → gray …), tokens = `#hex`
colours, numbers (with `$`, `.`, `:` and a unit kept attached: `600px`, `50%`,
`1.5rem`, `300ms` are one token each) and words; a text is its token SET.

  rule 1  drop a case when ANY of its user texts (the request and every
          earlier turn's) has token-set Jaccard >= 0.6 with ANY exam user text
          (every `request` and `history[].user` of the exam sets, the study's
          own case files and the user turns of the real conversations).
  rule 2  drop a case that writes the same key as an exam row AND states the
          same literal value (a quoted string, a #hex, a number, a "… to X"
          ending) while one of its texts has Jaccard >= 0.4 with that row's.
  rule 3  containment: drop a case when >= 80 % of an exam text's content
          tokens (stop words and literal values left out; >= 3 of them, or
          exactly 2 found as the same adjacent pair) appear in one
          of the case's texts AND the case writes what that exam row writes —
          an exam text embedded whole in a longer request ("make the search
          box in the header wider, 600px" ⊃ "make the search box wider"), or
          an exam conversation's turn with the value swapped. An exam
          REQUEST matches on the row's keys (container, list, key); an exam
          HISTORY turn (its writes are not recorded) on the row's container
          and list.

`check_case` returns {max_jaccard, nearest, nearest_text, rule1, rule2}; the
CLI filters a case file and writes a report:

    python -m souschef_lab.leak --cases r4/cases-all.jsonl [--why <study dir>] \\
        --report leak-report.json [--out kept.jsonl]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from . import FIXTURES_DIR

EXAM_DIR = FIXTURES_DIR / "exam"
RULE1 = 0.6
RULE2 = 0.4
RULE3 = 0.8
RULE3_MIN_TOKENS = 2           # an exam REQUEST: content tokens left (stop words and values out)
RULE3_MIN_TOKENS_HISTORY = 3   # an exam HISTORY turn (matched on container only, so a longer text is needed)
STOP = frozenset("""a an the on in of to for at by with from into onto and or but my our your their its it this that these
those there here is are be been was were should would can could will you please pls i we me us all any some about up
so just too also very really bit little lot im id ill weve were youre thats dont do does""".split())
CHECKER = "leak-2"  # leak-1 + units kept on numbers + rule 3 (containment)

UK_US = {
    "colour": "color", "colours": "colors", "coloured": "colored", "colouring": "coloring", "recolour": "recolor",
    "centre": "center", "centres": "centers", "centred": "centered", "centring": "centering",
    "grey": "gray", "greys": "grays", "greyish": "grayish", "greyed": "grayed",
    "favourite": "favorite", "favourites": "favorites", "flavour": "flavor", "flavours": "flavors",
    "neighbour": "neighbor", "neighbourhood": "neighborhood", "harbour": "harbor", "harbourfront": "harborfront",
    "organise": "organize", "organised": "organized", "customise": "customize", "customised": "customized",
    "optimise": "optimize", "visualise": "visualize", "realise": "realize", "prioritise": "prioritize",
    "capitalise": "capitalize", "capitalised": "capitalized", "emphasise": "emphasize", "minimise": "minimize",
    "maximise": "maximize", "personalise": "personalize", "standardise": "standardize",
    "behaviour": "behavior", "licence": "license", "catalogue": "catalog", "programme": "program",
    "jewellery": "jewelry", "theatre": "theater", "metre": "meter", "litre": "liter", "cheque": "check",
    "enquiry": "inquiry", "enquiries": "inquiries", "labelled": "labeled", "cancelled": "canceled",
}
UNITS = r"(?:%|px|rem|em|pt|vh|vw|ms|s|k)"
TOKEN = re.compile(r"#[0-9a-f]{3,8}\b|\$?\d+(?:[.,:/]\d+)*" + UNITS + r"?(?![a-z0-9])|\$?\d+(?:[.,:/]\d+)*|[a-z0-9]+")


def tokens(text: Any) -> frozenset[str]:
    t = unicodedata.normalize("NFKC", str(text or "")).lower().replace("'", "").replace("’", "")
    return frozenset(UK_US.get(w, w) for w in TOKEN.findall(t))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def literal_values(text: str) -> set[str]:
    """The literal values a text states: quoted strings, #hex colours, numbers
    and a trailing "to <value>" (the exam rows carry no values, so rule 2
    compares what the two TEXTS say)."""
    t = unicodedata.normalize("NFKC", str(text or "")).lower()
    vals = set(re.findall(r"#[0-9a-f]{3,8}\b", t)) | set(re.findall(r"\b\d+(?:\.\d+)?\b", t))
    vals |= {q.strip() for q in re.findall(r"[\"'“‘]([^\"'”’]{2,})[\"'”’]", t)}
    m = re.search(r"\bto ([a-z0-9][\w\s-]{1,30})$", t.strip(" .!?"))
    if m:
        vals.add(m.group(1).strip())
    return {UK_US.get(v, v) for v in vals if v}


def content_tokens(text: str) -> frozenset[str]:
    """A text's tokens without stop words and without the literal values it
    states (rule 3 compares the wording, so "change the hero headline to Fresh
    drops" and "… to Pre-roll Fridays" are the same sentence)."""
    toks = {t for t in tokens(text) if t not in STOP and not re.match(r"^[$#]?\d|^#[0-9a-f]{3,8}$", t)}  # numbers, sizes, #hex
    for v in literal_values(text):
        toks -= set(tokens(v))
    return frozenset(toks)


def key_of(target: dict | None, lst: Any, key: Any) -> tuple[str, str, str]:
    """A write's identity, loose on the target's shape: the section / panel /
    page it lives in, the list, the key's first segment."""
    t = target or {}
    where = str(t.get("sectionId") or t.get("panel") or t.get("page") or t.get("kind") or "")
    return where, str(lst or "").replace("[]", ""), str(key or "").split(".", 1)[0]


def case_keys(case: dict, full: bool = False) -> set[tuple[str, str, str]]:
    """What a case writes, as key_of triples (full=True: the whole key, not its
    first segment — rule 3's stricter match). Key-scored cases add their
    oracle ops (meta.oracle_ops)."""
    from .exam import touches_of

    kf = full_key_of if full else key_of
    exp = case.get("expected") or {}
    out: set[tuple[str, str, str]] = set()
    sets = [exp.get("keys") or []] + [s for s in exp.get("alt") or [] if isinstance(s, list)]
    for s in sets:
        for e in s:
            if isinstance(e, dict):
                out.add(kf(e.get("target"), e.get("list"), e.get("key")))
    ops = list(exp.get("ops") or []) + (list((case.get("meta") or {}).get("oracle_ops") or []) if full else [])
    for op in ops:
        for t in touches_of(op):
            out.add(kf(t["target"], t["list"], t["key"]))
    return out


def content_seq(text: Any) -> list[str]:
    """A text's tokens in order, stop words out (rule 3's adjacency test)."""
    t = unicodedata.normalize("NFKC", str(text or "")).lower().replace("'", "").replace("’", "")
    return [w for w in (UK_US.get(x, x) for x in TOKEN.findall(t)) if w not in STOP]


def _adjacent(pair: list[str], seq: list[str]) -> bool:
    return any(seq[i] == pair[0] and seq[i + 1] == pair[1] for i in range(len(seq) - 1))


def full_key_of(target: dict | None, lst: Any, key: Any) -> tuple[str, str, str]:
    where, l, _ = key_of(target, lst, key)
    return where, l, str(key or "")


def case_texts(case: dict) -> list[str]:
    """Every user text of a case: the scored request and every earlier turn."""
    texts = [str(case.get("request") or "")]
    for h in case.get("history") or []:
        texts.append(str(h.get("user") or h.get("request") or ""))
    for t in case.get("turns") or []:
        texts.append(str(t.get("request") or t.get("user") or ""))
    seen, out = set(), []
    for t in texts:
        if t.strip() and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _conv_user_text(m: dict) -> str:
    c = m.get("content")
    t = c if isinstance(c, str) else " ".join(x.get("text", "") for x in c or [] if isinstance(x, dict) and x.get("type") == "text")
    if "</candidate_fields>" in t:
        t = t.rsplit("</candidate_fields>", 1)[1]
    elif "</context>" in t:
        t = t.rsplit("</context>", 1)[1]
    return t.strip()


class ExamIndex:
    """Every exam user text, with the keys its row expects (rule 2)."""

    def __init__(self, exam_dir: Path = EXAM_DIR, why: Path | None = None, extra: Iterable[Path] = ()):
        entries: list[dict] = []
        files = sorted(Path(exam_dir).glob("*.jsonl"))
        if why:
            files += [p for p in (Path(why) / "paraphrase-cases.jsonl", Path(why) / "paraphrase-holdout.jsonl",
                                  Path(why) / "real-cases.jsonl") if p.is_file()]
        files += [Path(p) for p in extra]
        for f in files:
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                exp = r.get("expected") or {}
                keys, fkeys = set(), set()
                for s in [exp.get("keys") or []] + [s for s in exp.get("alt") or [] if isinstance(s, list)]:
                    for e in s:
                        if isinstance(e, dict):
                            keys.add(key_of(e.get("target"), e.get("list"), e.get("key")))
                            fkeys.add(full_key_of(e.get("target"), e.get("list"), e.get("key")))
                src = f"{f.stem}:{r.get('id')}"
                ctr = {(w, l) for w, l, _ in keys}
                entries.append({"id": src, "text": str(r.get("request") or ""), "keys": keys, "fkeys": fkeys, "hist_ctr": set()})
                for j, h in enumerate(r.get("history") or []):
                    if h.get("user"):
                        entries.append({"id": f"{src}/h{j}", "text": str(h["user"]), "keys": set(), "fkeys": set(), "hist_ctr": set(ctr)})
        self.conversations = 0
        if why:
            for f in sorted(glob.glob(str(Path(why) / "conv-*.json"))):
                d = json.loads(Path(f).read_text(encoding="utf-8"))
                msgs = d.get("messages") or []
                if msgs:
                    self.conversations += 1
                for j, m in enumerate(msgs):
                    if m.get("role") == "user":
                        t = _conv_user_text(m)
                        if t:
                            entries.append({"id": f"{Path(f).stem}#u{j}", "text": t, "keys": set(), "fkeys": set(), "hist_ctr": set()})
        by_text: dict[str, dict] = {}
        for e in entries:
            k = e["text"].strip().lower()
            if not k:
                continue
            if k in by_text:
                by_text[k]["keys"] |= e["keys"]
                by_text[k]["fkeys"] |= e["fkeys"]
                by_text[k]["hist_ctr"] |= e["hist_ctr"]
                continue
            e["tokens"] = tokens(e["text"])
            e["ctokens"] = content_tokens(e["text"])
            e["cseq"] = [w for w in content_seq(e["text"]) if w in e["ctokens"]]
            e["values"] = literal_values(e["text"])
            by_text[k] = e
        self.entries = list(by_text.values())
        self.files = [str(f) for f in files]

    def nearest(self, text: str) -> tuple[float, dict | None]:
        tk = tokens(text)
        best: tuple[float, dict | None] = (0.0, None)
        for e in self.entries:
            j = jaccard(tk, e["tokens"])
            if j > best[0]:
                best = (j, e)
        return best

    def check(self, case: dict, rule1: float = RULE1, rule2: float = RULE2, rule3: float = RULE3) -> dict:
        texts = case_texts(case)
        best = (0.0, None, "")
        r2 = r3 = None
        keys = case_keys(case)
        fkeys = case_keys(case, full=True)
        ctr = {(w, l) for w, l, _ in fkeys}
        values = set().union(*(literal_values(t) for t in texts)) if texts else set()
        for t in texts:
            tk = tokens(t)
            seq = content_seq(t)
            for e in self.entries:
                j = jaccard(tk, e["tokens"])
                if j > best[0]:
                    best = (j, e, t)
                if r2 is None and j >= rule2 and e["keys"] & keys and e["values"] & values:
                    r2 = {"exam": e["id"], "jaccard": round(j, 3), "text": t, "exam_text": e["text"],
                          "values": sorted(e["values"] & values)}
                ct = e.get("ctokens") or frozenset()
                if r3 is None and ct:
                    by = ("keys" if len(ct) >= RULE3_MIN_TOKENS and e.get("fkeys", set()) & fkeys else
                          "history_container" if len(ct) >= RULE3_MIN_TOKENS_HISTORY and e.get("hist_ctr", set()) & ctr
                          else None)
                    cont = len(ct & tk) / len(ct) if by else 0.0
                    if by and len(ct) == 2 and cont >= rule3:  # two words: only as the same adjacent pair
                        cs = e.get("cseq") or []
                        cont = cont if len(cs) == 2 and _adjacent(cs, seq) else 0.0
                    if by and cont >= rule3:
                        r3 = {"exam": e["id"], "containment": round(cont, 3), "text": t, "exam_text": e["text"], "by": by}
        j, e, t = best
        return {"max_jaccard": round(j, 3), "nearest": e["id"] if e else None, "nearest_text": e["text"] if e else None,
                "case_text": t, "rule1": j >= rule1, "rule2": r2, "rule3": r3}


def filter_cases(cases: list[dict], index: ExamIndex, family=lambda c: (c.get("meta") or {}).get("family") or "?",
                 rule1: float = RULE1, rule2: float = RULE2) -> tuple[list[dict], dict]:
    """(kept cases with meta.leak set, report)."""
    kept, dropped = [], []
    counts: dict[str, Counter] = defaultdict(Counter)
    near: list[tuple[float, str, str, str]] = []
    for c in cases:
        res = index.check(c, rule1, rule2)
        fam = family(c)
        counts[fam]["checked"] += 1
        c.setdefault("meta", {})["leak"] = {"max_jaccard": res["max_jaccard"], "nearest": res["nearest"],
                                            "rule2": res["rule2"]["exam"] if res["rule2"] else None, "checker": CHECKER}
        if res["rule1"]:
            counts[fam]["dropped_rule1"] += 1
            dropped.append({"id": c.get("id"), "family": fam, "rule": 1, **res})
        elif res["rule2"]:
            counts[fam]["dropped_rule2"] += 1
            dropped.append({"id": c.get("id"), "family": fam, "rule": 2, **res})
        elif res["rule3"]:
            counts[fam]["dropped_rule3"] += 1
            dropped.append({"id": c.get("id"), "family": fam, "rule": 3, **res})
        else:
            counts[fam]["kept"] += 1
            kept.append(c)
            near.append((res["max_jaccard"], str(c.get("id")), res["case_text"] or "", res["nearest_text"] or ""))
    near.sort(reverse=True)
    report = {"rule1_threshold": rule1, "rule2_threshold": rule2, "exam_texts": len(index.entries),
              "exam_files": index.files, "real_conversations": index.conversations,
              "by_family": {k: dict(v) for k, v in sorted(counts.items())},
              "total": {"checked": len(cases), "kept": len(kept), "dropped": len(dropped)},
              "dropped": dropped, "nearest_kept": [{"jaccard": j, "id": i, "text": t, "exam_text": et}
                                                   for j, i, t, et in near[:20]],
              "rule3_threshold": RULE3,
              "exam_ids_causing_drops": sorted(Counter(d["nearest"] if d["rule"] == 1 else d[f"rule{d['rule']}"]["exam"]
                                                       for d in dropped).items())}
    return kept, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="souschef_lab.leak", description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", required=True, type=Path)
    ap.add_argument("--why", type=Path, help="the paraphrase study folder (its case files + conv-*.json real turns)")
    ap.add_argument("--exam-dir", type=Path, default=EXAM_DIR)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--out", type=Path, help="write the kept cases here (meta.leak set)")
    a = ap.parse_args(argv)
    cases = [json.loads(x) for x in a.cases.read_text(encoding="utf-8").splitlines() if x.strip()]
    index = ExamIndex(a.exam_dir, a.why)
    kept, report = filter_cases(cases, index)
    if a.report:
        a.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.out:
        with a.out.open("w", encoding="utf-8") as f:
            for c in kept:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("exam_texts", "real_conversations", "by_family", "total")}, indent=1))
    return 0 if not report["dropped"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
