#!/usr/bin/env python3
"""mx_latency — per-arm latency of a Stage 3 paired eval (docs/MISTRAL.md §Latency), standard library only.

Two sources, both already written by every eval attempt (mxarm.py eval):

  per CALL   ollama-<role>.log. Ollama 0.34.4 runs llama-server, which prints a timing block per task
             ("prompt eval time = X ms / N tokens", "eval time = X ms / N tokens", "total time = …"), and Ollama's
             access log a [GIN] line per HTTP request with its wall time. One server per arm, OLLAMA_NUM_PARALLEL=1:
             tasks run one at a time, so each request line closes the timing block printed just before it.
               service_ms  prompt eval + decode on the GPU (llama-server's own clock; no queueing)
               wall_ms     the HTTP request as Ollama saw it — service_ms plus the wait for the slot (the eval runs
                           2 workers per arm against one slot) plus template/parse overhead
               decode tok/s  eval tokens / eval ms (calls that decoded ≥ 8 tokens)
               prefill tok/s prompt tokens / prompt ms on cold prefixes (≥ 1024 tokens evaluated: a cache miss)
  per TURN   eval/<role>/rows.jsonl — latency_ms per turn (the service's turn: every model call, tool round and
             history replay of that turn, as the client saw it) and usage.calls / output_tokens; rows with a
             latency (the exam sets, rederived and recorded; the built F0H/F9H rows carry none) and status ok.

The serving smokes (mxarm.assert_served: a load-only /api/generate, two /v1/messages, one /api/chat) come before
any row and are left out: the first SMOKE_MESSAGES /v1/messages requests and every /api/* request.
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable

GIN_LINE = re.compile(r'^\[GIN\][^|]*\|\s*(\d{3})\s*\|\s*([^|]*?)\s*\|[^|]*\|\s*([A-Z]+)\s+"([^"?]*)(\?[^"]*)?"')
TIMING = re.compile(r"print_timing:.*\|\s*(prompt eval time|eval time|total time)\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens")
SMOKE_MESSAGES = 2  # assert_served: the prompt-size smoke and the tool smoke
MODEL_PATHS = ("/v1/messages", "/api/chat", "/api/generate")
_DUR = re.compile(r"([0-9.]+)(h|ms|m|s|µs|μs|us|ns)")
_UNIT_MS = {"h": 3_600_000.0, "m": 60_000.0, "s": 1000.0, "ms": 1.0, "µs": 1e-3, "μs": 1e-3, "us": 1e-3, "ns": 1e-6}


def go_duration_ms(s: str) -> float | None:
    """A Go time.Duration string ("7.861531973s", "95.714407ms", "88.828µs", "1m2.5s") in milliseconds."""
    parts = _DUR.findall(str(s or "").strip())
    if not parts or "".join(a + b for a, b in parts) != str(s).strip():
        return None
    return sum(float(v) * _UNIT_MS[u] for v, u in parts)


def ollama_calls(text: str, *, smoke_messages: int = SMOKE_MESSAGES) -> list[dict]:
    """The model calls the eval's rows made, in order, from one Ollama log: [{status, path, wall_ms, prompt_ms,
    prompt_tokens, eval_ms, eval_tokens, service_ms}] (timing fields None when no block preceded the request)."""
    out: list[dict] = []
    pending: dict[str, Any] = {}
    seen_messages = 0
    for ln in text.splitlines():
        m = TIMING.search(ln)
        if m:
            kind, ms, n = m.group(1), float(m.group(2)), int(m.group(3))
            if kind == "prompt eval time":
                pending = {"prompt_ms": ms, "prompt_tokens": n}
            elif kind == "eval time":
                pending.update(eval_ms=ms, eval_tokens=n)
            elif kind == "total time":
                pending["service_ms"] = ms
            continue
        g = GIN_LINE.search(ln)
        if not g:
            continue
        status, dur, method, path = g.group(1), g.group(2), g.group(3), g.group(4)
        if method != "POST" or path not in MODEL_PATHS:
            continue
        timing, pending = pending, {}
        if path != "/v1/messages":
            continue  # /api/generate (load) and /api/chat (decode smoke): the serving asserts
        seen_messages += 1
        if seen_messages <= smoke_messages:
            continue
        out.append({"status": int(status), "path": path, "wall_ms": go_duration_ms(dur),
                    "prompt_ms": timing.get("prompt_ms"), "prompt_tokens": timing.get("prompt_tokens"),
                    "eval_ms": timing.get("eval_ms"), "eval_tokens": timing.get("eval_tokens"),
                    "service_ms": timing.get("service_ms")})
    return out


def pct(values: Iterable[float], q: float) -> float | None:
    """The nearest-rank q-quantile (q in (0, 1]), rounded to 0.1; None when empty."""
    v = sorted(x for x in values if isinstance(x, (int, float)))
    if not v:
        return None
    return round(v[min(len(v) - 1, max(0, math.ceil(q * len(v)) - 1))], 1)


def _stats(values: list[float]) -> dict:
    v = [float(x) for x in values if isinstance(x, (int, float))]
    return {"n": len(v), "p50": pct(v, 0.5), "p90": pct(v, 0.9), "mean": round(sum(v) / len(v), 1) if v else None}


def summarize_calls(calls: list[dict]) -> dict:
    ok = [c for c in calls if c.get("status") == 200]
    timed = [c for c in ok if isinstance(c.get("service_ms"), (int, float))]
    dec = [c["eval_tokens"] / (c["eval_ms"] / 1000.0) for c in timed
           if (c.get("eval_tokens") or 0) >= 8 and (c.get("eval_ms") or 0) > 0]
    pre = [c["prompt_tokens"] / (c["prompt_ms"] / 1000.0) for c in timed
           if (c.get("prompt_tokens") or 0) >= 1024 and (c.get("prompt_ms") or 0) > 0]
    statuses: dict[str, int] = {}
    for c in calls:
        statuses[str(c.get("status"))] = statuses.get(str(c.get("status")), 0) + 1
    return {"calls": len(calls), "status": statuses, "timed": len(timed),
            "wall_ms": _stats([c.get("wall_ms") for c in ok]),
            "service_ms": _stats([c["service_ms"] for c in timed]),
            "prompt_ms": _stats([c.get("prompt_ms") for c in timed]),
            "decode_ms": _stats([c.get("eval_ms") for c in timed]),
            "queue_ms": _stats([max(0.0, c["wall_ms"] - c["service_ms"]) for c in timed
                                if isinstance(c.get("wall_ms"), (int, float))]),
            "output_tokens": _stats([c.get("eval_tokens") for c in timed]),
            "prompt_tokens_evaluated": _stats([c.get("prompt_tokens") for c in timed]),
            "decode_tok_s": _stats(dec), "prefill_tok_s": _stats(pre), "cold_prefills": len(pre)}


def turn_group(set_name: str) -> str:
    s = str(set_name or "")
    return "recorded" if s.endswith(":recorded") else "exam" if ":" not in s else "other"


def summarize_turns(rows: list[dict]) -> dict:
    """Turn latency from an arm's eval rows (status ok, latency_ms present), all and per group (exam = the six
    rederived sets the bar uses; recorded = their `:recorded` runs)."""
    good = [r for r in rows if r.get("status") == "ok" and isinstance(r.get("latency_ms"), (int, float))]

    def one(rs: list[dict]) -> dict:
        u = [r.get("usage") or {} for r in rs]
        out_tok = [x.get("output_tokens") for x in u]
        calls = [x.get("calls") for x in u]
        lat = [r["latency_ms"] for r in rs]
        tps = [(x.get("output_tokens") or 0) / (r["latency_ms"] / 1000.0) for x, r in zip(u, rs) if r["latency_ms"] > 0]
        return {"turns": len(rs), "latency_ms": _stats(lat), "calls_per_turn": _stats(calls),
                "output_tokens": _stats(out_tok), "output_tok_per_turn_s": _stats(tps)}
    groups: dict[str, list[dict]] = {}
    for r in good:
        groups.setdefault(turn_group(r.get("set")), []).append(r)
    return {"all": one(good), **{g: one(rs) for g, rs in sorted(groups.items())},
            "errors": sum(1 for r in rows if r.get("status") != "ok")}


def pair_latency(logs: dict[str, str | None], rows: dict[str, list[dict]], names: dict[str, str]) -> dict:
    """{role: {model, calls: summarize_calls, turns: summarize_turns}} for one pair (roles current / candidate)."""
    out = {}
    for role in ("current", "candidate"):
        text = logs.get(role)
        out[role] = {"model": names.get(role),
                     "calls": summarize_calls(ollama_calls(text)) if text else None,
                     "turns": summarize_turns(rows.get(role) or []) if rows.get(role) else None}
    return out


def by_model(pairs: dict[str, dict]) -> dict[str, dict]:
    """The same model measured in several pairs: {model: {pair: role summary}}."""
    out: dict[str, dict] = {}
    for p, lat in pairs.items():
        for role, d in (lat or {}).items():
            if isinstance(d, dict) and d.get("model"):
                out.setdefault(d["model"], {})[f"{p} ({role})"] = d
    return out


def _f(x: Any, nd: int = 0) -> str:
    if x is None:
        return "–"
    return f"{x:,.{nd}f}" if isinstance(x, (int, float)) else str(x)


def markdown(pairs: dict[str, dict], *, title: str = "## Latency (Stage 3, Vertex L4 + Ollama 0.34.4)") -> str:
    lines = [title, "",
             "Per call: one model request (Ollama log). `service` = llama-server prompt eval + decode; `wall` = the "
             "request as Ollama saw it, including the wait for the single slot the arm's 2 eval workers share. Per "
             "turn: the service's turn latency from rows.jsonl (every call of the turn). ms unless noted; p50 / p90. "
             "Token counts are each model's own tokenizer (on r6's transcripts Tekken ran ≈ 3.7 characters a token, "
             "Qwen3.5 ≈ 2.0–2.4, for about the same ~150 characters a call): compare output SIZE in characters, "
             "decode TIME per call across models. The 2 workers share 1 slot, so absolute times are not Cloud Run's.",
             "",
             "| pair (role) | model | calls | service p50 / p90 | wall p50 / p90 | decode tok/s p50 | prefill tok/s p50 (cold n) "
             "| out tok/call p50 | turns | turn p50 / p90 | calls/turn mean | 5xx |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p, lat in pairs.items():
        for role in ("current", "candidate"):
            d = (lat or {}).get(role) or {}
            c, t = d.get("calls") or {}, ((d.get("turns") or {}).get("all") or {})
            fivexx = sum(v for k, v in (c.get("status") or {}).items() if k.startswith("5"))
            lines.append(
                f"| {p} ({role}) | {d.get('model') or '–'} | {_f(c.get('calls'))} | "
                f"{_f((c.get('service_ms') or {}).get('p50'))} / {_f((c.get('service_ms') or {}).get('p90'))} | "
                f"{_f((c.get('wall_ms') or {}).get('p50'))} / {_f((c.get('wall_ms') or {}).get('p90'))} | "
                f"{_f((c.get('decode_tok_s') or {}).get('p50'), 1)} | "
                f"{_f((c.get('prefill_tok_s') or {}).get('p50'))} ({_f(c.get('cold_prefills'))}) | "
                f"{_f((c.get('output_tokens') or {}).get('p50'))} | {_f(t.get('turns'))} | "
                f"{_f((t.get('latency_ms') or {}).get('p50'))} / {_f((t.get('latency_ms') or {}).get('p90'))} | "
                f"{_f((t.get('calls_per_turn') or {}).get('mean'), 2)} | {fivexx} |")
    return "\n".join(lines) + "\n"
