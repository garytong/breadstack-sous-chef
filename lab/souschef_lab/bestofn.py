"""Best-of-N: judging samples, stopping early, and choosing the rows.

Copied from breadstack-thermometer lab/in-house-model/lab/bestofn.py
(FOCACCIA-PLAN 4.3) and adapted to Sous Chef's exact verifier (LAB.md §4–§5):

  - a sample PASSES when the verifier passes it and every keep rule holds
    (dataset.reasons_of); each failure is named as it is;
  - rank_key: pass first, fewest browser ops, fewest server-tool calls,
    shortest reply, then the sample number — total and repeatable;
  - the adaptive stop (Adaptive, question_state) is verbatim;
  - select: the thinking guard becomes a reply-length p90 guard per kind
    (traces carry no thinking), the "tool path" is the op path (op names and
    list actions, first-seen order), the template family is the case's
    section type / panel (dataset.family_of), cap 0.03, floors kept as an
    optional dict and empty by default;
  - Claude is the teacher and the day-one model: thermometer's
    refuse_claude_model and is_claude are dropped (LAB.md fixed decisions).

Nothing here keeps, drops or ranks an individual row by hand.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .dataset import estimate_tokens, family_of, reasons_of, to_hf
from .verify import FAILING, NON_OPS, SERVER_TOOLS, Engine, Verdict, canonical, events_of, final_text, tool_calls, verify

# Reasons about the prompt, not the answer: every sample of the case has
# them, so the case is never sampled again and never trained or paired.
PROMPT_REASONS = ("holdout", "no_case")
# Reasons a sample is never material at all — not a row, not a pair side.
NEVER_REASONS = PROMPT_REASONS
# A failing sample whose answer did not finish as a whole answer: a broken
# run is no "rejected" completion to learn from (pairs). no_answer is
# INCOMPLETE (LAB.md §5), never a rejected side.
INCOMPLETE_REASONS = ("error", "stop_reason", "empty_reply", "no_answer")
# A failing sample's category is its verdict (verify.FAILING minus no_answer),
# most teachable first: a refused op, a wrong value, a missing write, an
# extra write.
CATEGORY_ORDER = ("invalid_op", "wrong_value", "missing_writes", "extra_writes")


def category_of(reasons: Iterable[str]) -> str:
    rs = set(reasons)
    return next((c for c in CATEGORY_ORDER if c in rs), "other")


@dataclass
class Judged:
    """One sample, judged."""

    sample_id: str  # "<case id>#<n>"
    case_id: str
    sample: int
    model: str
    meta: dict
    row: dict
    case: dict | None
    passed: bool
    reasons: list[str]
    verdict: Verdict | None = None
    hf: dict = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return str((self.case or {}).get("kind") or self.meta.get("kind") or "")

    @property
    def never(self) -> bool:
        return any(r in NEVER_REASONS for r in self.reasons)

    @property
    def prompt_excluded(self) -> bool:
        return any(r in PROMPT_REASONS for r in self.reasons)

    @property
    def category(self) -> str:
        if self.passed:
            return ""
        return category_of([self.verdict.verdict] if self.verdict else []) if self.verdict else category_of(self.reasons)


class Judge:
    """Judges a collector row with the current rules: the verifier on the
    row's case, then every keep rule (dataset.reasons_of). The hf row is
    built once and kept on the Judged (select counts its tokens, pairs cut
    it)."""

    def __init__(self, cases: dict[str, dict], engine: Engine, system: tuple[str, ...] | list[str] = (),
                 tools: list[dict] | None = None, max_seq: int | None = None,
                 tokens: Callable[[dict], int] = estimate_tokens):
        self.cases = cases
        self.engine = engine
        self.system = tuple(system)
        self.tools = tools
        self.max_seq = max_seq
        self.tokens = tokens

    def __call__(self, row: dict) -> Judged:
        case_id = str(row.get("case_id") or "")
        sample = int(row.get("sample") or 1)
        sid = str(row.get("id") or f"{case_id}#{sample}")
        model = str(row.get("model") or "")
        case = self.cases.get(case_id)
        if case is None:
            return Judged(sid, case_id, sample, model, row.get("meta") or {}, row, None, False, ["no_case"])
        v = verify(row, case, self.engine)
        hf = to_hf(row, case, self.system, self.tools)
        reasons = reasons_of(row, case, v, self.max_seq, self.tokens, hf)
        return Judged(sid, case_id, sample, model, row.get("meta") or {}, row, case, not reasons, reasons, v, hf)


# --- what a sample did ----------------------------------------------------------------------------------------------

def browser_ops(j: Judged) -> int:
    return sum(1 for c in tool_calls(events_of(j.row)) if c["name"] not in NON_OPS)


def server_calls(j: Judged) -> int:
    return sum(1 for c in tool_calls(events_of(j.row)) if c["name"] in SERVER_TOOLS)


def op_path(j: Judged) -> tuple[str, ...]:
    """The ops a sample used, in the order it first used each: ``op`` or
    ``edit_list:<action>``. Two samples take different op paths when these
    differ (a repeated op is no other path)."""
    out = []
    for c in tool_calls(events_of(j.row)):
        if c["name"] in NON_OPS:
            continue
        key = f"edit_list:{c['input'].get('action')}" if c["name"] == "edit_list" else c["name"]
        out.append(key)
    return tuple(dict.fromkeys(out))


def reply_chars(j: Judged) -> int:
    return len(final_text(events_of(j.row)))


def rank_key(j: Judged) -> tuple:
    """How good a sample is to teach from, best first (LAB.md §5): pass
    first, the fewest browser ops, the fewest server-tool calls, the
    shortest reply, then the earliest sample, so the order is total and
    repeatable."""
    return (0 if j.passed else 1, browser_ops(j), server_calls(j), reply_chars(j), j.sample, j.sample_id)


# --- the adaptive stop (collect.py --samples) -----------------------------------------------------------------------

@dataclass(frozen=True)
class Adaptive:
    """When collect.py stops sampling a case before its N samples (thermometer
    lab/bestofn.py Adaptive, verbatim):

      enough     it has `enough` passing samples (default 2: selection keeps
                 one per case, or two that take different op paths);
      hopeless   none of its first `hopeless_after` samples passed (default
                 half of N, rounded up) AND its family — every case of its
                 section type / panel — has passed less than `family_rate`
                 (1/8, the learnable band's floor) of at least `family_min`
                 (8) finished samples;
      excluded   a sample of it named a reason about the case itself
                 (holdout): no sample of it is ever training material.

    A sample that failed to answer (an HTTP error, a timeout) is not a
    finished sample: it is asked again on the next run."""

    enough: int = 2
    hopeless_after: int | None = None
    family_min: int = 8
    family_rate: float = 1 / 8

    def after(self, n: int) -> int:
        return self.hopeless_after if self.hopeless_after is not None else max(1, math.ceil(n / 2))

    def to_dict(self, n: int) -> dict:
        return {"enough": self.enough, "hopeless_after": self.after(n), "family_min": self.family_min,
                "family_rate": self.family_rate}


def question_state(outcomes: list[bool], n: int, family: tuple[int, int], rule: Adaptive | None,
                   excluded: bool = False) -> str:
    """"open" (sample it again), "done" (N samples), or — with rule — the
    adaptive stops: "enough", "hopeless", "excluded". outcomes are its
    finished samples' verdicts; family is (passes, finished samples) over
    its family. Thermometer's question_state, verbatim."""
    if len(outcomes) >= n:
        return "done"
    if rule is None:
        return "open"
    if excluded:
        return "excluded"
    if sum(outcomes) >= rule.enough:
        return "enough"
    passes, samples = family
    if not any(outcomes) and len(outcomes) >= rule.after(n) and samples >= rule.family_min \
            and passes < rule.family_rate * samples:
        return "hopeless"
    return "open"


# --- selection (dataset.build --best-of-n) --------------------------------------------------------------------------

def p90(values: list[int], pct: float = 0.9) -> int:
    """The nearest-rank percentile (pct) of values; 0 for none."""
    if not values:
        return 0
    v = sorted(values)
    return v[max(0, math.ceil(pct * len(v)) - 1)]


def stable_hash(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:12], 16)


def groups_of(j: Judged) -> list[tuple]:
    """The floor groups a row counts in: its kind."""
    return [("kind", j.kind)]


def conversation_key(j: Judged) -> str:
    """What makes two samples the same example across case ids: the context
    and the request (thermometer trace_conversation_key)."""
    case = j.case or {}
    return canonical([j.row.get("context") or case.get("context"), j.row.get("text") or case.get("request")])


@dataclass
class Kept:
    judged: Judged
    rank: int  # 1: the case's best sample; 2: a second one on another op path
    tokens: int = 0


@dataclass
class SelectConfig:
    max_seq: int = 36864
    family_share: float = 0.03
    floors: dict[tuple, int] = field(default_factory=dict)
    reply_pct: float = 0.9
    second_path: bool = True


def select(judged: list[Judged], tokens: Callable[[Judged], int], cfg: SelectConfig) -> tuple[list[Kept], dict]:
    """The rows best-of-N keeps (thermometer select, adapted), and a report:

      1. only passing samples;
      2. drop those whose reply is above the p90 (reply_pct) of the passing
         samples of their kind — but only where the case has a passing
         sample at or below it (kept_above_p90_only_passes);
      3. drop those longer than max_seq tokens — before the choice;
      4. per case, the best by rank_key, and a second when another passing
         sample takes a genuinely other op path (its ops are not a superset
         of the best's);
      5. one conversation (context + request) under two case ids is kept
         once (the lowest id's);
      6. no family above family_share of the rows before this step: its
         extra rows go, second rows first, then the worst-ranked, then by a
         hash of the case id — never one whose removal takes a floor group
         below its floor."""
    rep: dict = {"samples": len(judged), "passing": sum(1 for j in judged if j.passed)}
    passing = [j for j in judged if j.passed]
    by_kind: dict[str, list[int]] = defaultdict(list)
    for j in passing:
        by_kind[j.kind].append(reply_chars(j))
    limits = {k: p90(v, cfg.reply_pct) for k, v in by_kind.items()}
    rep["reply_p90_chars"] = dict(sorted(limits.items()))
    within = {j.case_id for j in passing if reply_chars(j) <= limits[j.kind]}
    after_len = [j for j in passing if reply_chars(j) <= limits[j.kind] or j.case_id not in within]
    rep["dropped_reply_above_p90"] = len(passing) - len(after_len)
    rep["kept_above_p90_only_passes"] = len({j.case_id for j in passing} - within)
    sized: list[tuple[Judged, int]] = []
    over = 0
    for j in after_len:
        n = tokens(j)
        if n > cfg.max_seq:
            over += 1
        else:
            sized.append((j, n))
    rep["dropped_over_max_seq"] = over
    per_q: dict[str, list[tuple[Judged, int]]] = defaultdict(list)
    for j, n in sized:
        per_q[j.case_id].append((j, n))
    chosen: list[Kept] = []
    for qid in sorted(per_q):
        cands = sorted(per_q[qid], key=lambda x: rank_key(x[0]))
        best, n = cands[0]
        chosen.append(Kept(best, 1, n))
        if cfg.second_path:
            path = set(op_path(best))
            other = next(((j, m) for j, m in cands[1:] if not set(op_path(j)) >= path), None)
            if other is not None:
                chosen.append(Kept(other[0], 2, other[1]))
    rep["cases_with_a_pass"] = len(per_q)
    owner: dict[str, str] = {}
    deduped: list[Kept] = []
    for k in chosen:
        key = conversation_key(k.judged)
        if owner.setdefault(key, k.judged.case_id) == k.judged.case_id:
            deduped.append(k)
    rep["dropped_duplicate_conversation"] = len(chosen) - len(deduped)
    total = len(deduped)
    cap = max(1, math.floor(cfg.family_share * total))
    counts: Counter = Counter(g for k in deduped for g in groups_of(k.judged))
    by_family: dict[str, list[Kept]] = defaultdict(list)
    for k in deduped:
        by_family[family_of(k.judged.case or {})].append(k)
    dropped: set[int] = set()
    capped: dict[str, int] = {}
    held: dict[str, int] = {}
    for fam in sorted(by_family):
        rows = by_family[fam]
        excess = len(rows) - cap
        if excess <= 0:
            continue
        order = sorted(rows, key=lambda k: (-k.rank, tuple(-x for x in rank_key(k.judged)[:4]),
                                            stable_hash(k.judged.case_id)))
        for k in order:
            if excess <= 0:
                break
            gs = groups_of(k.judged)
            if any(counts[g] - 1 < cfg.floors[g] for g in gs if g in cfg.floors):
                held[fam] = held.get(fam, 0) + 1
                continue
            dropped.add(id(k))
            for g in gs:
                counts[g] -= 1
            excess -= 1
            capped[fam] = capped.get(fam, 0) + 1
    kept = [k for k in deduped if id(k) not in dropped]
    rep["family_cap"] = {"share": cfg.family_share, "rows_before": total, "cap": cap,
                         "dropped_by_family": dict(sorted(capped.items())), "held_by_floors": dict(sorted(held.items()))}
    final: Counter = Counter(g for k in kept for g in groups_of(k.judged))
    rep["floors_short"] = {" ".join(g): {"have": final[g], "floor": f}
                           for g, f in sorted(cfg.floors.items()) if final[g] < f}
    rep["rows"] = len(kept)
    rep["second_rows"] = sum(1 for k in kept if k.rank == 2)
    return kept, rep


def selection_meta(k: Kept) -> dict:
    """What a kept row records about why it was chosen (counts, no text)."""
    j = k.judged
    return {"sample_id": j.sample_id, "case_id": j.case_id, "rank": k.rank, "op_path": list(op_path(j)),
            "browser_ops": browser_ops(j), "server_calls": server_calls(j), "reply_chars": reply_chars(j),
            "tokens": k.tokens, "rule_version": j.verdict.rule_version if j.verdict else None}


# --- logs -----------------------------------------------------------------------------------------------------------

def sample_rows(log_rows: list[dict]) -> list[dict]:
    """The best-of-N rows of a collect log (they carry "sample")."""
    return [r for r in log_rows if r.get("sample") is not None]


def is_sample_log(log_rows: list[dict]) -> bool | None:
    """True for a best-of-N log, False for a plain one, None for an empty or
    mixed one (thermometer, verbatim)."""
    s = sum(1 for r in log_rows if r.get("sample") is not None)
    if not log_rows:
        return None
    if s == len(log_rows):
        return True
    return False if s == 0 else None


def unique_samples(log_rows: list[dict]) -> tuple[list[dict], dict]:
    """One sample id per sample across logs (thermometer unique_samples, with
    ``turn_id`` for ``trace_id``): a row whose turn an earlier row already
    logged is that sample again and goes; rows that share an id but not a
    turn each get "<id>~<turn id>"."""
    def key(r: dict) -> tuple[str, str]:
        return str(r.get("model") or ""), str(r.get("id") or f"{r.get('case_id')}#{r.get('sample')}")

    turns_of: dict[tuple, set] = defaultdict(set)
    for r in log_rows:
        if r.get("status") == "ok" and r.get("sample") is not None:
            turns_of[key(r)].add(str(r.get("turn_id") or ""))
    out, seen, rep = [], set(), Counter()
    for r in log_rows:
        if r.get("status") != "ok" or r.get("sample") is None:
            out.append(r)
            continue
        k, tid = key(r), str(r.get("turn_id") or "")
        if (k, tid) in seen:
            rep["repeated_rows"] += 1
            continue
        seen.add((k, tid))
        if len(turns_of[k]) > 1:
            r = dict(r, id=f"{k[1]}~{tid}")
            rep["renamed"] += 1
        out.append(r)
    return out, dict(rep)


def family_outcomes(judged: Iterable[Judged]) -> dict[str, tuple[int, int]]:
    """(passes, finished samples) per family — what Adaptive's hopeless rule
    reads."""
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for j in judged:
        if j.case is None:
            continue
        fam = family_of(j.case)
        out[fam][1] += 1
        out[fam][0] += int(j.passed)
    return {k: (v[0], v[1]) for k, v in out.items()}
