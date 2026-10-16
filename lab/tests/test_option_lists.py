"""The turn context's option lists (launcher 1d4193e2 optionLists.js +
turnContext.addOptionLists + carouselOptions.js) as state.context_of mirrors
them: the browser's own output for the lab's edge cases
(fixtures/parity/turnContext.options.json, tests/option_lists_golden.py),
byte for byte, plus the unit rules. The node-backed check regenerates the
golden from the checkout when it has optionLists.js and skips otherwise."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
for _p in (LAB, LAB / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import option_lists_golden as G  # noqa: E402
from souschef_lab import browser_defaults as B  # noqa: E402
from souschef_lab import engine as E  # noqa: E402
from souschef_lab import state as S  # noqa: E402


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(G.GOLDEN.read_text(encoding="utf-8"))


def test_context_of_equals_the_browsers_option_list_golden(golden):
    assert len(golden["cases"]) >= 30
    assert golden["source"]["snapshot_of"]  # which launcher commit produced it
    for c in golden["cases"]:
        got = G.lab_context(c, golden["manifest"])
        assert got == c["expected"], c["name"]
        assert json.dumps(got, ensure_ascii=False) == json.dumps(c["expected"], ensure_ascii=False), c["name"]  # key order


def test_the_golden_covers_cut_empty_unknown_and_excluded(golden):
    by = {c["name"]: c["expected"] for c in golden["cases"]}
    cut = by["a focused prefix-family carousel off home; over 100 the lists are cut and flagged"]
    assert cut["tagsTruncated"] == {"shown": 100, "total": 130} and cut["brandsTruncated"] == {"shown": 100, "total": 101}
    empty = by["an unknown list is left out; a list that loaded empty is sent empty"]
    assert empty["brands"] == [] and "tags" not in empty
    assert list(cut)[-4:] == ["tags", "tagsTruncated", "brands", "brandsTruncated"]  # after the outline, in this order
    excluded = [n for n, want in G.GATE_TEXTS if not want]
    assert all("tags" not in by[f"gate {i:02d}: {t}"] for i, (t, w) in enumerate(G.GATE_TEXTS) if not w) and excluded
    assert all("tags" in by[f"gate {i:02d}: {t}"] for i, (t, w) in enumerate(G.GATE_TEXTS) if w)
    cats = by["category ids are text; slug falls back to the id; blank ids are left out"]["categories"]
    assert cats[0] == {"slug": "flower", "name": "Flower", "id": "12"} and all(isinstance(c.get("id", ""), str) for c in cats)


def test_the_golden_is_what_the_checkout_gives_now(golden):
    """Regenerate the browser's side from COOKIE_CUTTER_DIR (one Vite boot) when it has optionLists.js."""
    root = B.cookie_cutter_dir()
    if not G.has_option_lists(root) or E.node_version() is None:
        pytest.skip("node and a launcher checkout with sousChef/optionLists.js (1d4193e2+) are needed")
    items = G.cases()
    expected = G.browser_expected(items, root)
    assert [c["name"] for c in items] == [c["name"] for c in golden["cases"]], "cases changed: tests/option_lists_golden.py --write"
    for c, want, have in zip(items, expected, golden["cases"]):
        assert json.dumps(want, ensure_ascii=False) == json.dumps(have["expected"], ensure_ascii=False), \
            f"{c['name']}: the checkout's turn context moved — regenerate with tests/option_lists_golden.py --write"


@pytest.mark.parametrize("text,want", G.GATE_TEXTS)
def test_the_gate_regex(text, want):
    assert bool(S.OPTION_TEXT_RE.search(text)) is want


def test_wants_option_lists_by_section_type():
    m = G.trimmed_manifest()
    assert S.option_ref_section_types(m) == {"productCarousel"}  # tagId / brandId are derived, categorySlug is not tag/brand
    assert S.option_ref_section_types(None) == {"productCarousel"}  # the fallback
    assert S.wants_option_lists(None, ["productCarousel-deals"], m)
    assert not S.wants_option_lists(None, ["contentBlock", "heroBanner"], m)
    assert not S.wants_option_lists("", [], None)


def test_option_list_sorts_like_javascript_and_counts_before_the_cut():
    rows, total = S.option_list("tags", [{"slug": "b", "name": "｡"}, {"slug": "a", "name": "\U0001F600"}], 1)
    assert rows == [{"slug": "a", "name": "\U0001F600"}] and total == 2  # UTF-16 order: U+D83D… sorts before U+FF61
    rows, total = S.option_list("brands", [{"slug": "x", "name": ""}, {"slug": "y", "name": "Y"}])
    assert rows == [{"slug": "y", "name": "Y"}] and total == 1
