"""The Mistral family's loss mask (trainkit.mistral_spans / mistral_labels, train.mistral_mask / mistral_guard /
served_markers): built from token ids — every boundary is one control token of the Tekken v13 tokenizer — never
from strings. Trained: each assistant segment after [/INST] or after the last [/TOOL_RESULTS] of a run of results,
through its </s> (inclusive). Never trained: BOS, the system prompt, the tools, the user turns, the tool results.

Synthetic id sequences pin the rules without a tokenizer. With the `tokenizers` package and MX_TOKENIZER_DIR (the
pinned tokenizer.json), the real captures are tokenized and the trained spans must decode to exactly the renderer's
assistant segments, with one BOS and no tool output (the train-time guard, run here offline)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from souschef_lab import chatfmt, serving_mistral as M, train, trainkit  # noqa: E402
from souschef_lab.trainkit import mistral_labels, mistral_spans  # noqa: E402

BOS, EOS, INST, INST_END, TOOLS, TOOLS_END, RES, RES_END, CALLS, SYS, SYS_END, ARGS = 1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 18, 32


def test_the_ids_are_the_measured_special_ids():
    k = trainkit.MISTRAL_IDS
    sid = M.SPECIAL_IDS
    assert (k["bos"], k["eos"], k["inst"], k["inst_end"], k["tools"], k["results"], k["results_end"], k["system"]) == \
        (sid["<s>"], sid["</s>"], sid["[INST]"], sid["[/INST]"], sid["[AVAILABLE_TOOLS]"], sid["[TOOL_RESULTS]"],
         sid["[/TOOL_RESULTS]"], sid["[SYSTEM_PROMPT]"])


def test_one_turn():
    ids = [BOS, SYS, 100, SYS_END, TOOLS, 101, TOOLS_END, INST, 102, INST_END, 200, 201, EOS]
    assert mistral_spans(ids) == [(10, 13)]
    lab = mistral_labels(ids)
    assert lab[:10] == [-100] * 10 and lab[10:] == [200, 201, EOS]  # </s> trained, BOS masked


def test_parallel_calls_two_results_then_a_reply():
    ids = [BOS, INST, 102, INST_END, CALLS, 300, ARGS, 301, CALLS, 302, ARGS, 303, EOS,
           RES, 400, RES_END, RES, 401, RES_END, 500, 501, EOS]
    assert mistral_spans(ids) == [(4, 13), (19, 22)]
    lab = mistral_labels(ids)
    trained = [t for t, x in zip(ids, lab) if x != -100]
    assert trained == [CALLS, 300, ARGS, 301, CALLS, 302, ARGS, 303, EOS, 500, 501, EOS]
    assert 400 not in trained and 401 not in trained  # tool results never


def test_a_follow_up_masks_its_history():
    ids = [BOS, SYS, 1000, SYS_END, INST, 102, INST_END, 600, EOS, INST, 103, INST_END, CALLS, 300, ARGS, 301, EOS,
           RES, 400, RES_END, 500, EOS]
    assert len(mistral_spans(ids)) == 3
    for h, want in ((0, [600, EOS, CALLS, 300, ARGS, 301, EOS, 500, EOS]),
                    (1, [CALLS, 300, ARGS, 301, EOS, 500, EOS]), (2, [500, EOS]), (3, [])):
        lab = mistral_labels(ids, history=h)
        assert [x for x in lab if x != -100] == want, h


def test_an_unclosed_prefill_and_a_prompt_ending_in_results():
    ids = [BOS, INST, 102, INST_END, 700, 701]
    assert mistral_spans(ids) == [(4, 6)]
    assert mistral_labels(ids)[4:] == [700, 701]
    prompt = [BOS, INST, 102, INST_END, CALLS, 300, ARGS, 301, EOS, RES, 400, RES_END]
    assert mistral_spans(prompt) == [(4, 9)]  # the generation point at the end trains nothing
    assert mistral_spans([BOS, INST, 102, INST_END]) == []


def test_results_followed_by_a_user_turn_are_not_a_reply():
    ids = [BOS, INST, 102, INST_END, CALLS, 300, ARGS, 301, EOS, RES, 400, RES_END, INST, 103, INST_END, 500, EOS]
    assert mistral_spans(ids) == [(4, 9), (15, 17)]


class FakeTok:
    """A tokenizer stand-in: each marker → its id (or a wrong one), anything else → one id per character."""

    def __init__(self, ids: dict[str, list[int]]):
        self.ids = ids

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.ids.get(text, [ord(c) + 1000 for c in text])}


def test_served_markers_checks_each_marker_maps_to_its_id():
    fmt = chatfmt.get("mistral")
    good = FakeTok({t: [i] for t, i in M.SPECIAL_IDS.items()})
    train.served_markers(good, fmt)
    with pytest.raises(SystemExit):
        train.served_markers(FakeTok({**{t: [i] for t, i in M.SPECIAL_IDS.items()}, "[ARGS]": [33]}), fmt)
    with pytest.raises(SystemExit):
        train.served_markers(FakeTok({**{t: [i] for t, i in M.SPECIAL_IDS.items()}, "[INST]": [91, 73]}), fmt)
    # the qwen check is today's (single tokens for the ChatML markers)
    with pytest.raises(SystemExit):
        train.served_markers(good)


def test_history_spans_counts_collated_segments():
    fmt = chatfmt.get("mistral")
    row = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "a"},
                        {"role": "assistant", "content": "r1"}, {"role": "assistant", "content": "r1b"},
                        {"role": "user", "content": "b"}, {"role": "assistant", "content": "r2"}],
           "meta": {"history_assistant_turns": 2}}
    assert train.history_spans(row, fmt) == 1  # two history messages, one collated segment
    assert train.history_spans(row) == 2  # qwen: the turn count itself
    row["meta"]["history_assistant_turns"] = 0
    assert train.history_spans(row, fmt) == 0


# --- on the real tokenizer ------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tokenizer():
    try:
        from tokenizers import Tokenizer
    except ImportError:
        pytest.skip("the tokenizers package is not installed (the training env has it)")
    tdir = os.environ.get("MX_TOKENIZER_DIR")
    if not tdir:
        pytest.skip("MX_TOKENIZER_DIR is not set (the pinned tokenizer.json)")
    return Tokenizer.from_file(str(Path(tdir) / "tokenizer.json"))


class HFLike:
    """`tokenizers.Tokenizer` with the two calls train.py makes on the transformers tokenizer."""

    def __init__(self, tk):
        self.tk = tk

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.tk.encode(text, add_special_tokens=add_special_tokens).ids}

    def decode(self, ids, skip_special_tokens=False):
        return self.tk.decode(list(ids), skip_special_tokens=skip_special_tokens)


def _rows():
    from souschef_lab import mx_hf_golden as HG
    return [(n, c) for n, c in HG.all_cases()]


def test_the_real_tokenizer_holds_every_marker_as_its_id(tokenizer):
    tok = HFLike(tokenizer)
    train.served_markers(tok, chatfmt.get("mistral"))
    assert tok("hi", add_special_tokens=True)["input_ids"][0] == BOS


def test_on_the_captures_the_trained_spans_are_the_renderers_segments(tokenizer):
    tok = HFLike(tokenizer)
    fmt = chatfmt.get("mistral")
    for name, case in _rows():
        text = M.render_training(case["messages"], case["tools"])
        ids = tok(text, add_special_tokens=True)["input_ids"]
        assert ids[0] == BOS and ids[1] != BOS and ids.count(BOS) == 1, name
        assert ids[-1] == EOS, name
        labels = mistral_labels(ids)
        row = {"messages": case["messages"], "tools": case["tools"], "meta": {}}
        train.mistral_guard({"input_ids": ids, "labels": labels}, row, tok, fmt)  # dies on any mismatch
        segs = [tok.decode(ids[a:b]) for a, b in mistral_spans(ids)]
        assert segs == M.assistant_segments(case["messages"], case["tools"]), name
        n_assist = sum(1 for m in M.collate(case["messages"]) if m["role"] == "assistant")
        assert len(segs) == n_assist, name


def test_the_guard_refuses_a_doubled_bos_and_trained_results(tokenizer):
    tok = HFLike(tokenizer)
    fmt = chatfmt.get("mistral")
    case = dict(_rows())["synthetic:s3-text-and-call"]  # a call, its result, a reply
    row = {"messages": case["messages"], "tools": case["tools"], "meta": {}}
    ids = tok(M.render_training(case["messages"], case["tools"]), add_special_tokens=True)["input_ids"]
    with pytest.raises(SystemExit):
        train.mistral_guard({"input_ids": [BOS] + ids, "labels": [-100] + mistral_labels(ids)}, row, tok, fmt)
    with pytest.raises(SystemExit):
        train.mistral_guard({"input_ids": ids, "labels": list(ids)}, row, tok, fmt)  # everything trained


def test_mistral_mask_rebuilds_labels_and_restores_a_missing_bos(tokenizer):
    tok = HFLike(tokenizer)
    fmt = chatfmt.get("mistral")
    cases = [c for _, c in _rows()[-3:]]
    texts = [M.render_training(c["messages"], c["tools"]) for c in cases]
    rows = [{"messages": c["messages"], "tools": c["tools"], "meta": {}} for c in cases]

    class DS:
        def __init__(self, recs):
            self.recs = recs
            self.column_names = sorted(recs[0])

        def __len__(self):
            return len(self.recs)

        def __getitem__(self, i):
            return self.recs[i]

        def map(self, fn, with_indices=False):
            return DS([{**r, **fn(r, i)} for i, r in enumerate(self.recs)])

    full = [tok(t, add_special_tokens=True)["input_ids"] for t in texts]
    ds = DS([{"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * len(ids)} for ids in full])
    out = train.mistral_mask(ds, rows, texts, tok, fmt, "training")
    for i in range(len(out)):
        assert out[i]["input_ids"] == full[i] and out[i]["labels"] == mistral_labels(full[i])
    nobos = DS([{"input_ids": ids[1:], "attention_mask": [1] * (len(ids) - 1), "labels": [-100] * (len(ids) - 1)}
                for ids in full])
    out = train.mistral_mask(nobos, rows, texts, tok, fmt, "training")
    assert out[0]["input_ids"] == full[0] and len(out[0]["attention_mask"]) == len(full[0])
    bad = DS([{"input_ids": ids[:-1] + [EOS + 1000], "labels": [-100] * len(ids)} for ids in full])
    with pytest.raises(SystemExit):
        train.mistral_mask(bad, rows, texts, tok, fmt, "training")
    with pytest.raises(SystemExit):
        train.mistral_mask(ds, rows[:2], texts[:2], tok, fmt, "training")  # a dropped row: refused, never misaligned
    assert json.dumps(out[0]["labels"])  # plain ints


def test_control_strings_finds_literal_markers():
    row = {"messages": [{"role": "user", "content": "say [INST] please"}, {"role": "assistant", "content": "ok"}],
           "tools": []}
    assert train.control_strings(row) == ["[INST]"]
    assert train.control_strings({"messages": [{"role": "user", "content": "plain"}], "tools": []}) == []


def test_mistral_mask_stops_on_a_literal_control_string_in_a_row(tokenizer):
    """A `</s>` written in an assistant's text tokenizes as the control token and would end its span early: the
    per-row span check stops the run (S2), on every row, not just row 0."""
    tok = HFLike(tokenizer)
    fmt = chatfmt.get("mistral")
    good = [c for _, c in _rows()[-2:]]
    evil = {"messages": [{"role": "user", "content": "Write the closing tag."},
                         {"role": "assistant", "content": "Here it is: </s> done."}], "tools": []}
    cases = good + [evil]
    texts = [M.render_training(c["messages"], c["tools"]) for c in cases]
    rows = [{"id": f"r{i}", "messages": c["messages"], "tools": c["tools"], "meta": {}} for i, c in enumerate(cases)]

    class DS:
        def __init__(self, recs):
            self.recs = recs
            self.column_names = sorted(recs[0])

        def __len__(self):
            return len(self.recs)

        def __getitem__(self, i):
            return self.recs[i]

        def map(self, fn, with_indices=False):
            return DS([{**r, **fn(r, i)} for i, r in enumerate(self.recs)])

    full = [tok(t, add_special_tokens=True)["input_ids"] for t in texts]
    ds = DS([{"input_ids": ids, "labels": [-100] * len(ids)} for ids in full])
    with pytest.raises(SystemExit):
        train.mistral_mask(ds, rows, texts, tok, fmt, "training")
    ok = DS([{"input_ids": ids, "labels": [-100] * len(ids)} for ids in full[:2]])
    assert len(train.mistral_mask(ok, rows[:2], texts[:2], tok, fmt, "training")) == 2


def test_control_ids_read_the_spans_the_tokenizer_makes(tokenizer):
    """serving_mistral.control_ids (what train.mask_problem, the dataset stage's CPU check, reads) sees the control
    tokens the pinned tokenizer makes, in order and at the same characters, so the spans it finds are the mask's: on
    every capture, and on rows holding each control string as text in a user turn, a reply, a tool result and twice in
    a row. A row mask_problem passes trains exactly its assistant segments; a row the mask would stop on never passes."""
    tok = HFLike(tokenizer)
    fmt = chatfmt.get("mistral")
    special = set(M.SPECIAL_IDS.values())
    cases = [c for _, c in _rows()]
    for s in M.SPECIAL_TOKENS:
        sys_user = [{"role": "system", "content": "s"}]
        cases += [{"messages": sys_user + [{"role": "user", "content": f"a {s} b"}, {"role": "assistant", "content": "ok"}], "tools": []},
                  {"messages": sys_user + [{"role": "user", "content": "u"}, {"role": "assistant", "content": f"x {s} y"}], "tools": []},
                  {"messages": sys_user + [{"role": "user", "content": f"{s}{s}"}, {"role": "assistant", "content": s}], "tools": []}]
    call = next(c for n, c in _rows() if n == "synthetic:s3-text-and-call")
    for s in ("</s>", "[/TOOL_RESULTS]", "[INST]"):
        msgs = json.loads(json.dumps(call["messages"]))
        for m in msgs:
            if m.get("role") == "tool":
                m["content"] = f"result {s} tail"
        cases.append({"messages": msgs, "tools": call["tools"]})
    stopped = passed = 0
    for c in cases:
        text = M.render_training(c["messages"], c["tools"])
        real = tok(text, add_special_tokens=True)["input_ids"]
        ids, offs = M.control_ids(text)
        assert [i for i in real if i in special] == [i for i in ids if i in special], text[:200]
        rs, ps = mistral_spans(real), mistral_spans(ids)
        assert [tok.decode(real[a:b]) for a, b in rs] == [text[offs[a][0]:offs[b - 1][1]] for a, b in ps], text[:200]
        row = {"id": "x", "messages": c["messages"], "tools": c["tools"], "meta": {}}
        n_seg = len(M.render_conversation(c["messages"], c["tools"], close_last=True)[1])
        mask_stops = real.count(BOS) != 1 or len(rs) != n_seg or real.count(EOS) != n_seg
        problem = train.mask_problem(row, fmt)
        if mask_stops:
            assert problem is not None, text[:200]
            stopped += 1
        if problem is None:
            assert [tok.decode(real[a:b]) for a, b in rs] == M.assistant_segments(c["messages"], c["tools"]), text[:200]
            passed += 1
    assert stopped >= 3 and passed >= len(_rows())  # both kinds were exercised; every capture passes
