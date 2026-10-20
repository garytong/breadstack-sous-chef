"""tools/mx_latency.py: the per-call parse of an Ollama 0.34.4 log (llama-server timing blocks + Ollama's [GIN]
access lines), the per-turn summary of eval rows, and the report section. Offline."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
for p in (str(HERE.parent.parent), str(HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import mx_latency as L  # noqa: E402


def timing(prompt_ms: float, prompt_n: int, eval_ms: float, eval_n: int, task: int = 1) -> str:
    total = prompt_ms + eval_ms
    return "\n".join([
        f"slot launch_slot_: id  0 | task {task} | processing task, is_child = 0",
        f"slot print_timing: id  0 | task {task} | prompt processing, n_tokens =   9216, progress = 0.62, t =   3.38 s / 2726.41 tokens per second",
        f"slot print_timing: id  0 | task {task} | prompt eval time =    {prompt_ms:.2f} ms / {prompt_n:5d} tokens (    0.46 ms per token,  2167.47 tokens per second)",
        f"slot print_timing: id  0 | task {task} |        eval time =     {eval_ms:.2f} ms /    {eval_n} tokens (   31.77 ms per token,    31.48 tokens per second)",
        f"slot print_timing: id  0 | task {task} |       total time =    {total:.2f} ms / {prompt_n + eval_n} tokens",
        f"slot print_timing: id  0 | task {task} |    graphs reused =          0"])


def gin(status: int, dur: str, path: str, method: str = "POST") -> str:
    return f'[GIN] 2026/10/17 - 22:50:04 | {status} | {dur:>14} |       127.0.0.1 | {method}     "{path}"'


SMOKES = "\n".join([
    gin(200, "88.828µs", "/api/version", "GET"),
    gin(200, "1.2s", "/api/generate"),
    timing(86.05, 61, 0.0, 1, task=0), gin(200, "95.714407ms", "/v1/messages"),  # the prompt-size smoke
    timing(30.0, 20, 180.0, 8, task=1), gin(200, "300ms", "/api/chat"),  # the decode smoke
    timing(35.82, 57, 186.54, 9, task=2), gin(200, "235.123939ms", "/v1/messages"),  # the tool smoke
])


def test_go_durations():
    assert L.go_duration_ms("7.861531973s") == pytest.approx(7861.531973)
    assert L.go_duration_ms("95.714407ms") == pytest.approx(95.714407)
    assert L.go_duration_ms("88.828µs") == pytest.approx(0.088828)
    assert L.go_duration_ms("1m2.5s") == pytest.approx(62500.0)
    assert L.go_duration_ms("garbage") is None and L.go_duration_ms("") is None


def test_the_rows_calls_follow_the_smokes_and_carry_their_timing():
    log = "\n".join([SMOKES,
                     timing(6853.14, 14854, 889.45, 29, task=15), gin(200, "7.861531973s", "/v1/messages?beta=true"),
                     timing(42.56, 10, 825.71, 27, task=59), gin(500, "1.5s", "/v1/messages?beta=true"),
                     gin(200, "40µs", "/api/version", "GET"),
                     timing(40.0, 12, 400.0, 13, task=60), gin(200, "2.0s", "/v1/messages?beta=true")])
    calls = L.ollama_calls(log)
    assert [c["status"] for c in calls] == [200, 500, 200]  # the two smokes and every /api/* request left out
    first = calls[0]
    assert first["wall_ms"] == pytest.approx(7861.53, abs=0.01) and first["prompt_tokens"] == 14854
    assert first["service_ms"] == pytest.approx(7742.59, abs=0.01) and first["eval_tokens"] == 29
    s = L.summarize_calls(calls)
    assert s["calls"] == 3 and s["status"] == {"200": 2, "500": 1} and s["timed"] == 2  # the 500 is not a timed answer
    assert s["cold_prefills"] == 1 and s["prefill_tok_s"]["p50"] == pytest.approx(14854 / 6.85314, abs=0.1)
    assert s["decode_tok_s"]["n"] == 2  # both answers decoded ≥ 8 tokens
    assert s["queue_ms"]["p50"] is not None and s["queue_ms"]["p50"] >= 0


def test_a_request_without_a_timing_block_has_no_service_time():
    calls = L.ollama_calls(SMOKES + "\n" + gin(200, "1s", "/v1/messages?beta=true"))
    assert len(calls) == 1 and calls[0]["service_ms"] is None and calls[0]["wall_ms"] == 1000.0
    assert L.summarize_calls(calls)["service_ms"]["n"] == 0


def row(set_, lat, *, calls=1, out=50, ok=True):
    return {"set": set_, "status": "ok" if ok else "error", "latency_ms": lat,
            "usage": {"calls": calls, "output_tokens": out}}


def test_turns_by_group():
    rows = [row("paraphrase", 1000), row("paraphrase", 3000, calls=3, out=150), row("holdout:recorded", 2000),
            row("real", 9000, ok=False), {"set": "F9H", "status": "ok"}]  # built rows carry no latency
    t = L.summarize_turns(rows)
    assert t["all"]["turns"] == 3 and t["errors"] == 1
    assert t["exam"]["latency_ms"] == {"n": 2, "p50": 1000.0, "p90": 3000.0, "mean": 2000.0}
    assert t["recorded"]["turns"] == 1 and t["exam"]["calls_per_turn"]["mean"] == 2.0
    assert t["exam"]["output_tok_per_turn_s"]["p50"] == 50.0


def test_pair_latency_and_markdown():
    log = SMOKES + "\n" + timing(100.0, 2000, 900.0, 30, task=9) + "\n" + gin(200, "2s", "/v1/messages?beta=true")
    lat = L.pair_latency({"current": log, "candidate": None}, {"current": [row("paraphrase", 2500)], "candidate": []},
                         {"current": "r4", "candidate": "mx"})
    assert lat["current"]["model"] == "r4" and lat["current"]["calls"]["calls"] == 1
    assert lat["candidate"]["calls"] is None and lat["candidate"]["turns"] is None
    md = L.markdown({"r4-mx": lat})
    assert "## Latency" in md and "| r4-mx (current) | r4 | 1 | 1,000 / 1,000 | 2,000 / 2,000 |" in md
    assert "| r4-mx (candidate) | mx | – |" in md
    bm = L.by_model({"r4-mx": lat, "r4-r7c": lat})
    assert set(bm["r4"]) == {"r4-mx (current)", "r4-r7c (current)"}


def test_pct_is_nearest_rank():
    assert L.pct([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.9) == 9.0 and L.pct([], 0.5) is None and L.pct([5], 0.5) == 5.0
