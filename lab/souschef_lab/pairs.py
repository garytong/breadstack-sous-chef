"""DPO pairs from on-policy samples.

Copied from breadstack-thermometer lab/in-house-model/lab/pairs.py
(FOCACCIA-PLAN 4.5) and adapted to the exact verifier (LAB.md §5):

A pair is one prompt and two completions of it by the model being trained:
CHOSEN, a turn of a sample that passes (bestofn.Judge), and REJECTED, the
same turn of a sample of the same case whose verdict is
``extra_writes | wrong_value | missing_writes | invalid_op`` (``no_answer``
is INCOMPLETE: a broken run is no rejected completion). Both samples were
asked the same request on the same context (a fresh conversation each,
LAB.md §4), so their transcripts start alike; the pair is cut where they
part:

  final       the same tool calls and the same results, only the final
              text differs
  divergence  the first assistant turn whose tool calls differ: the shared
              turns before it are the prompt, the passing turn is chosen,
              the failing one rejected. Every Sous Chef failure is a failure
              of the calls (the ops), so thermometer's CALL_CATEGORIES gate
              is always open here; calls that differ only in their order are
              no divergence (no pair)

Which samples and prompts: a never-material sample (holdout, no case) is
neither side; a prompt with no pass or no usable fail gives none; ties are
final before divergence, then the rejected side's category
(bestofn.CATEGORY_ORDER), then the chosen side's rank_key, then the
rejected sample's number; each rejected sample is used once; at most
max_per_prompt (2) pairs per prompt; a side over max_tokens (36,864) is
dropped; quotas follow the residual failures; the length guard keeps
chosen no longer than rejected in at least half the pairs.

Row ``{id, prompt, chosen, rejected, tools, meta}`` — TRL's conversational
preference format, tool-call arguments as JSON strings (as the SFT rows).
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable

from .bestofn import CATEGORY_ORDER, INCOMPLETE_REASONS, Judged, rank_key
from .dataset import estimate_tokens
from .verify import canonical

PAIR_TYPES = ("final", "divergence")
REJECTED_VERDICTS = CATEGORY_ORDER


def completion_chars(m: dict) -> int:
    """The length of one assistant turn: its text and tool calls."""
    n = len(m.get("content") or "")
    for c in m.get("tool_calls") or []:
        f = c.get("function") or {}
        n += len(f.get("name") or "") + len(f.get("arguments") or "")
    return n


def calls_of(m: dict) -> tuple:
    """An assistant turn's tool calls: (name, canonical input) in order."""
    out = []
    for c in m.get("tool_calls") or []:
        f = c.get("function") or {}
        try:
            args = json.loads(f.get("arguments") or "{}")
        except ValueError:
            args = f.get("arguments")
        out.append((str(f.get("name") or ""), canonical(args)))
    return tuple(out)


def turns_of(row: dict) -> tuple[list[dict], list[tuple[int, dict, list[dict]]]]:
    """(the prompt's messages, this answer's turns): each turn its message
    index, the assistant message and the tool results after it."""
    msgs = row["messages"]
    start = int(row["meta"]["answer_start"])
    turns: list[tuple[int, dict, list[dict]]] = []
    i = start
    while i < len(msgs):
        if msgs[i].get("role") != "assistant":
            i += 1
            continue
        j = i + 1
        while j < len(msgs) and msgs[j].get("role") != "assistant":
            j += 1
        turns.append((i, msgs[i], msgs[i + 1:j]))
        i = j
    return msgs[:start], turns


def _results(ms: list[dict]) -> tuple:
    return tuple((m.get("role"), m.get("name"), m.get("content")) for m in ms)


def split_point(a: dict, b: dict) -> tuple[str, int, int] | str:
    """Where two samples' rows part: ("final" | "divergence", a's message
    index, b's message index), or why no pair: "prompt_mismatch",
    "calls_reordered", "results_differ" or "same_answer". Thermometer's,
    verbatim."""
    pa, ta = turns_of(a)
    pb, tb = turns_of(b)
    if pa != pb or a.get("tools") != b.get("tools"):
        return "prompt_mismatch"
    for k in range(min(len(ta), len(tb))):
        (ia, ma, ra), (ib, mb, rb) = ta[k], tb[k]
        ca, cb = calls_of(ma), calls_of(mb)
        if ca != cb:
            if sorted(ca) == sorted(cb) or sorted(c for _, m, _ in ta[k:] for c in calls_of(m)) == \
                    sorted(c for _, m, _ in tb[k:] for c in calls_of(m)):
                return "calls_reordered"
            return "divergence", ia, ib
        if not ca:
            return ("final", ia, ib) if (ma.get("content") or "") != (mb.get("content") or "") else "same_answer"
        if _results(ra) != _results(rb):
            return "results_differ"
    return "same_answer"


@dataclass
class Pair:
    case_id: str
    kind: str
    ptype: str
    chosen: Judged
    rejected: Judged
    prompt: list[dict]
    chosen_msg: dict
    rejected_msg: dict
    tools: list[dict]
    meta: dict = field(default_factory=dict)

    @property
    def category(self) -> str:
        return self.rejected.category

    @property
    def chosen_chars(self) -> int:
        return completion_chars(self.chosen_msg)

    @property
    def rejected_chars(self) -> int:
        return completion_chars(self.rejected_msg)

    def to_row(self) -> dict:
        meta = {"case_id": self.case_id, "kind": self.kind, "type": self.ptype,
                "chosen_sample": self.chosen.sample_id, "rejected_sample": self.rejected.sample_id,
                "rejected_reasons": list(self.rejected.reasons),
                "rejected_verdict": self.rejected.verdict.verdict if self.rejected.verdict else None,
                "rejected_category": self.category, "chosen_chars": self.chosen_chars,
                "rejected_chars": self.rejected_chars, "model": self.chosen.model,
                "family": (self.chosen.hf.get("meta") or {}).get("family"),
                "holdout": bool((self.chosen.case or {}).get("holdout")), **self.meta}
        return {"id": f"{self.case_id}:{sample_tag(self.chosen)}-{sample_tag(self.rejected)}", "prompt": self.prompt,
                "chosen": [self.chosen_msg], "rejected": [self.rejected_msg], "tools": self.tools, "meta": meta}


@dataclass
class PairConfig:
    max_per_prompt: int = 2
    max_pairs: int | None = None
    max_tokens: int = 36864
    length_guard: float = 0.5


def sample_tag(j: Judged) -> str:
    return j.sample_id.split("#", 1)[1] if "#" in j.sample_id else str(j.sample)


def usable_rejected(j: Judged) -> bool:
    """A failing sample that finished as an answer and failed on its ops."""
    if j.passed or j.never or j.verdict is None:
        return False
    if any(r in INCOMPLETE_REASONS for r in j.reasons):
        return False
    return j.verdict.verdict in REJECTED_VERDICTS


def build_pairs(judged: list[Judged], cfg: PairConfig, count_tokens: Callable[[dict], int] = estimate_tokens) \
        -> tuple[list[Pair], dict]:
    """The pairs of judged samples (every sample of every case), and a report
    of counts. See the module's docstring for the rules."""
    rep: Counter = Counter()
    groups: dict[str, list[Judged]] = defaultdict(list)
    for j in judged:
        groups[j.case_id].append(j)
    fails_by_cat: Counter = Counter()
    candidates: list[Pair] = []
    for qid, js in sorted(groups.items()):
        if any(j.prompt_excluded for j in js):
            rep["prompts_held_out"] += 1
            continue
        passes = [j for j in js if j.passed and not j.never]
        fails = [j for j in js if usable_rejected(j)]
        fails_by_cat.update(j.category for j in fails)
        if not passes:
            rep["prompts_no_pass"] += 1
            continue
        if not fails:
            rep["prompts_no_fail"] += 1
            continue
        options = []
        for c in passes:
            for r in fails:
                sp = split_point(c.hf, r.hf)
                if isinstance(sp, str):
                    rep[f"no_pair_{sp}"] += 1
                    continue
                ptype, ia, ib = sp
                options.append((PAIR_TYPES.index(ptype), _cat_rank(r.category), rank_key(c), r.sample, c, r, sp))
        options.sort(key=lambda o: o[:4])
        used: set[int] = set()
        mine = 0
        for *_, c, r, (ptype, ia, ib) in options:
            if mine >= cfg.max_per_prompt or id(r) in used:
                continue
            ca, rb = c.hf, r.hf
            p = Pair(case_id=qid, kind=c.kind, ptype=ptype, chosen=c, rejected=r, prompt=ca["messages"][:ia],
                     chosen_msg=ca["messages"][ia], rejected_msg=rb["messages"][ib], tools=ca["tools"])
            sides = [count_tokens({"messages": p.prompt + [m], "tools": p.tools}) for m in (p.chosen_msg, p.rejected_msg)]
            p.meta["tokens"] = {"chosen_side": sides[0], "rejected_side": sides[1]}
            if max(sides) > cfg.max_tokens:
                rep["dropped_over_max_tokens"] += 1
                continue
            used.add(id(r))
            mine += 1
            candidates.append(p)
        if not mine:
            rep["prompts_no_pair"] += 1
    rep["candidates"] = len(candidates)
    chosen = quotas(candidates, fails_by_cat, cfg.max_pairs) if cfg.max_pairs is not None else list(candidates)
    rep["dropped_by_quota"] = len(candidates) - len(chosen)
    kept, guard = length_guard(chosen, cfg.length_guard)
    rep["dropped_by_length_guard"] = guard
    rep["pairs"] = len(kept)
    rep["by_type"] = dict(Counter(p.ptype for p in kept))
    rep["by_category"] = {c: n for c in CATEGORY_ORDER + ("other",) if (n := sum(1 for p in kept if p.category == c))}
    rep["failures_by_category"] = dict(fails_by_cat)
    rep["chosen_not_longer_share"] = round(sum(1 for p in kept if p.chosen_chars <= p.rejected_chars) / len(kept), 3) \
        if kept else None
    return kept, dict(rep)


def _cat_rank(c: str) -> int:
    return CATEGORY_ORDER.index(c) if c in CATEGORY_ORDER else len(CATEGORY_ORDER)


def quotas(pairs: list[Pair], failures: Counter, total: int) -> list[Pair]:
    """At most total pairs, shared among the failure categories as the
    failing samples are (thermometer quotas, verbatim)."""
    if total >= len(pairs):
        return list(pairs)
    by_cat: dict[str, list[Pair]] = defaultdict(list)
    for p in pairs:
        by_cat[p.category].append(p)
    order = sorted(by_cat, key=_cat_rank)
    n_fail = sum(failures[c] for c in order) or 1
    want = {c: math.floor(total * failures[c] / n_fail) for c in order}
    left = total - sum(want.values())
    for c in order:
        if left <= 0:
            break
        want[c] += 1
        left -= 1
    out: list[Pair] = []
    spare = 0
    for c in order:
        take = min(len(by_cat[c]), want[c] + spare)
        spare = want[c] + spare - take
        out += by_cat[c][:take]
    if spare:
        for c in order:
            extra = [p for p in by_cat[c] if p not in out][:spare]
            out += extra
            spare -= len(extra)
            if not spare:
                break
    return out


def length_guard(pairs: list[Pair], share: float = 0.5) -> tuple[list[Pair], int]:
    """(pairs, dropped): pairs whose chosen side is longer than its rejected
    one go, the largest excess first, until at least share of the pairs have
    a chosen side no longer than the rejected (thermometer, verbatim)."""
    kept = list(pairs)
    dropped = 0
    longer = sorted((p for p in kept if p.chosen_chars > p.rejected_chars),
                    key=lambda p: (-(p.chosen_chars - p.rejected_chars), p.case_id, p.rejected.sample))
    while kept and sum(1 for p in kept if p.chosen_chars <= p.rejected_chars) < share * len(kept) and longer:
        p = longer.pop(0)
        kept.remove(p)
        dropped += 1
    return kept, dropped


# --- CLI --------------------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    from .bestofn import Judge, is_sample_log, unique_samples
    from .dataset import MANIFEST_NAME, estimate_tokens as _est, split_of, utc_now, write_jsonl
    from .verify import RULE_VERSION, RecordedEngine, default_engine, read_jsonl

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", action="append", required=True, help="best-of-N collector log (jsonl), repeatable")
    p.add_argument("--cases", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--val-pct", type=int, default=10)
    p.add_argument("--max-pairs", type=int)
    p.add_argument("--max-per-prompt", type=int, default=PairConfig.max_per_prompt)
    p.add_argument("--max-tokens", type=int, default=PairConfig.max_tokens)
    p.add_argument("--model", help="pair only this route's samples (needed when the logs hold several)")
    p.add_argument("--system-rules")
    p.add_argument("--system-index")
    p.add_argument("--tools")
    p.add_argument("--recorded", help="replay recorded engine output (tests)")
    args = p.parse_args(argv)
    rows = [r for path in args.log for r in read_jsonl(path)]
    if is_sample_log(rows) is not True:
        p.error("--log must be best-of-N logs (every row carries `sample`)")
    rows, renamed = unique_samples(rows)
    models = {str(r.get("model") or "") for r in rows if r.get("status") == "ok"} - {""}
    model = args.model or (next(iter(models)) if len(models) == 1 else "")
    if not model:
        p.error(f"the logs hold samples of {len(models)} routes: say which with --model")
    cases = {c["id"]: c for c in read_jsonl(args.cases)}
    system = tuple(Path(f).read_text(encoding="utf-8") for f in (args.system_rules, args.system_index) if f)
    tools = json.loads(Path(args.tools).read_text(encoding="utf-8")) if args.tools else None
    engine = RecordedEngine(args.recorded) if args.recorded else default_engine()
    judge = Judge(cases, engine, system=system, tools=tools)
    judged = [judge(r) for r in rows if r.get("status") == "ok" and str(r.get("model") or "") == model]
    pairs, rep = build_pairs(judged, PairConfig(max_per_prompt=args.max_per_prompt, max_pairs=args.max_pairs,
                                                max_tokens=args.max_tokens), _est)
    train, val = [], []
    for pr in pairs:
        (val if split_of(pr.case_id, args.val_pct) == "val" else train).append(pr.to_row())
    out = Path(args.out)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    (out / "stats.json").write_text(json.dumps({"built_at": utc_now(), **rep, "train": len(train), "val": len(val),
                                                "sample_ids": renamed}, indent=2), encoding="utf-8")
    (out / "pairs-manifest.json").write_text(json.dumps({
        "built_at": utc_now(), "logs": list(args.log), "model": model, "rule_version": RULE_VERSION,
        "max_per_prompt": args.max_per_prompt, "max_tokens": args.max_tokens, "max_pairs": args.max_pairs,
        "length_guard": PairConfig.length_guard, "rejected_verdicts": list(REJECTED_VERDICTS),
        "incomplete_reasons": list(INCOMPLETE_REASONS), "dataset_manifest": MANIFEST_NAME}, indent=2), encoding="utf-8")
    print(f"samples {len(judged)} · pairs {rep['pairs']} (train {len(train)} / val {len(val)}) · by type {rep['by_type']} "
          f"· by category {rep['by_category']}")
    return 0 if pairs else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
