"""askrule.py — when a turn may be answered with a question (r7 spec §1.1): VAGUE iff no value, no direction, no
action. The calibration rows are the exam's own texts (fixtures/exam), never copied into training."""
from __future__ import annotations

import json

import pytest

from souschef_lab import FIXTURES_DIR
from souschef_lab import askrule as AR
from souschef_lab.manifest import load

EXAM = FIXTURES_DIR / "exam"
EXAM_MANIFEST = EXAM / "manifest-real.json"

# the rows r6 lost by asking (flips.md "over-ask"): each gives a direction or an action
NOT_VAGUE = {
    ("holdout", "h004"): "direction", ("paraphrase", "p104"): "direction", ("paraphrase", "p112"): "direction",
    ("paraphrase", "p085"): "direction", ("paraphrase", "p111"): "direction", ("holdout", "h025"): "direction",
    ("holdout", "h043"): "direction", ("holdout", "h016"): "direction", ("holdout", "h051"): "direction",
    ("holdout", "h009"): "direction", ("holdout", "h020"): "direction", ("paraphrase", "p107"): "direction",
    ("paraphrase", "p113"): "direction", ("real", "r05"): "direction",
    ("holdout", "h018"): "action", ("paraphrase", "p084"): "action",
}
# the exam's ask_ok rows (exam.VAGUE_ASK_OK + p013) and r7's three new ones (h022, p030, p047): vague
VAGUE = [("paraphrase", "p031"), ("paraphrase", "p048"), ("paraphrase", "p109"), ("paraphrase", "p129"),
         ("paraphrase", "p013"), ("holdout", "h024"), ("holdout", "h027"), ("holdout", "h033"), ("holdout", "h036"),
         ("holdout", "h037"), ("holdout", "h044"), ("holdout", "h048"), ("holdout", "h055"),
         ("holdout", "h022"), ("paraphrase", "p030"), ("paraphrase", "p047")]


def _row(s: str, rid: str) -> dict:
    for line in (EXAM / f"{s}.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r["id"] == rid:
                return r
    raise KeyError(rid)


def _text(s: str, rid: str) -> str:
    return _row(s, rid)["request"]


@pytest.fixture(scope="module")
def m():
    return load(str(EXAM_MANIFEST))


@pytest.mark.parametrize("key,kind", sorted(NOT_VAGUE.items()))
def test_the_rows_r6_lost_by_asking_are_not_vague(key, kind):
    v = AR.classify(_text(*key))
    assert not v.vague and v.kind == kind, (key, v)


@pytest.mark.parametrize("key", VAGUE)
def test_the_ask_ok_rows_are_vague(key):
    v = AR.classify(_text(*key))
    assert v.vague, (key, v)


def test_the_three_values_the_vocabulary_catches(m):
    """h006 "Inter" (a font of the manifest's font lists), h010 "paper" (the texture asset), r09 "is HashiTong"
    (V5): the regex prototype missed them; classify_case reads the case's vocabulary."""
    h006, h010, r09 = _row("holdout", "h006"), _row("holdout", "h010"), _row("real", "r09")
    assert "Inter" in AR.font_names(m)
    for row in (h006, h010, r09):
        case = {"request": row["request"], "fields": row.get("fields"), "target": row.get("target"),
                "expected": {"keys": (row.get("expected") or {}).get("keys") or []}, "site_state": {}}
        v = AR.classify_case(case, m)
        assert not v.vague and v.kind == "value", (row["id"], v)
    assert AR.classify("add a paper texture behind everything", values=("paper texture",)).why.startswith("V6")
    assert AR.classify("the name of my site is HashiTong").why.startswith("V5")
    assert AR.classify("the body copy font should be Inter", values=("Inter",)).kind == "value"


def test_values_quotes_and_texts():
    # V3: a quote of what the site already shows names the thing, it gives no value
    assert AR.classify("the 'Shop now' button wording on the hero is meh", names=("Shop now",)).vague
    assert AR.classify("the 'Shop now' button wording on the hero is meh").kind == "value"
    # V5 negatives and positives
    assert AR.classify("should say something else").vague
    assert AR.classify("what the promo bar says").vague
    assert AR.classify("should read Hand-trimmed, always").kind == "value"
    assert AR.classify("rename the about link to Our Story").kind == "value"
    # V1 / V2 / V4
    assert AR.classify("make the content block headline #7a1f2b").why == "V1:colour_literal"
    assert AR.classify("black background please").why == "V1:colour:black"
    assert AR.classify("the background of the site is too white").kind == "direction"  # "too" + colour: a direction
    assert AR.classify("less padding on the content block, 24px").kind == "value"
    assert AR.classify("i want to change what the button on the second panel says").vague  # panel 2 / second: no value
    assert AR.classify("change the button on panel 2").vague
    assert AR.classify("link our leafly page https://www.leafly.com/x").kind == "value"
    # idioms: "doesn't feel right" is no `right` option
    assert AR.classify("the text alignment doesn't feel right", values=("left", "right", "center")).vague
    assert AR.classify("align the text right", values=("left", "right")).kind in ("value", "action")


def test_directions():
    assert AR.classify("too white").kind == "direction"
    v = AR.classify("show fewer products in the New Arrivals row")
    assert v.kind == "direction" and AR.direction_of("show fewer products in the New Arrivals row", "number") == "down"
    assert AR.direction_of("the footer text is too light", "color") == "darker"
    assert AR.direction_of("make it a bit lighter", "color") == "lighter"
    assert AR.direction_of("bump the paragraph text up a bit", "range") == "up"
    assert AR.direction_of("it's too big", "number") == "down"
    assert AR.direction_of("section titles are too subtle", "color") is None  # no clear way: no direction
    assert AR.direction_of("make it bigger", "text") is None
    assert AR.direction_of("bigger but a bit smaller too", "number") is None
    # not a direction: the readability words and the vague templates' own words
    for t in ("the footer text is hard to read", "the colours look washed out", "the header needs work",
              "the hero doesn't feel right", "can you update the announcement bar", "tweak the footer",
              "the read more color doesn't feel right", "the promo is no longer running, change the bar"):
        assert AR.classify(t).vague, t


def test_actions():
    v = AR.classify("change the big image at the top")
    assert v.kind == "action" and v.why == "action:pick_asset"
    assert AR.classify("change the picture that shows when someone shares my site on facebook").why == "action:pick_asset"
    # a picture named as a PLACE is no image request
    assert AR.classify("i want to change the gap between the button and the edge of the banner image").vague
    assert AR.classify("add our weedmaps page").vague
    assert AR.classify("add a newsletter signup", section_names=("newsletter",)).why == "action:add:newsletter"
    assert AR.classify("add a newsletter signup").vague
    assert AR.classify("add the search bar", bool_labels=("Show search bar",)).kind == "action"
    assert AR.classify("hide the FAQ link from the top menu").kind == "action"
    assert AR.classify("turn off the announcement bar").kind == "action"
    assert AR.classify("switch up the font on the footer headings").vague
    assert AR.classify("show the product code").kind == "action"
    assert AR.classify("the pack filter display needs work").vague  # display: a noun here
    assert AR.classify("change what shows up as the title on google").vague


def test_every_vague_template_stays_vague():
    from souschef_lab.contrast import VAGUE_TEMPLATES
    for t in VAGUE_TEMPLATES:
        for label in ("footer text color", "shop menu link font size", "location picker headline", "cards per row"):
            assert AR.classify(t.format(label=label)).vague, t


def test_case_vocabulary_and_family(m):
    from souschef_lab.state import build_state
    state = build_state(m, template="vintage")
    tgt = {"kind": "global", "panel": "TypographyColorsSection"}
    font = {"id": "x", "request": "i don't like the body font", "target": tgt, "fields": ["bodyFont"],
            "site_state": state, "expected": {"ops": []}, "score": {"mode": "ask"}, "meta": {}}
    assert "Lora" in AR.values_for(font, m)
    assert AR.classify_case(font, m).vague
    assert not AR.classify_case(font, m, text="body font in Lora please").vague
    assert AR.ask_family_of(font, m) == "choice_missing"
    # V7: the source oracle's literal in the turn
    op = {"op": "set_fields", "target": tgt, "values": {"bodyFont": "Fraunces"}}
    assert "fraunces" in AR.values_for(font, m, oracle_ops=[op])
    assert AR.ask_family_of({"score": {"mode": "clarify"}, "meta": {}}) == "which_one"
    assert AR.ask_family_of({"score": {"mode": "ask"}, "meta": {"capability": "create_page"}}) == "page_title_missing"
    assert AR.ask_family_of({"score": {"mode": "ask"}, "meta": {"ask_family": "complaint"}}) == "complaint"


def _single_turn_rows():
    from souschef_lab import exam as EX
    for s in ("holdout", "paraphrase", "real"):
        for line in (EXAM / f"{s}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("history"):
                continue
            yield s, r, bool((EX.amend(r, s).get("expected") or {}).get("ask_ok"))


def test_every_single_turn_exam_row_classifies_as_its_answer_key_says(m):
    """r7 review: the rule must call every single-turn exam row that is NOT ask-allowed clear (else an ask on it leaks
    past the dataset check and the report's clear-asks count undercounts) and every ask-allowed row vague — on the
    case's whole vocabulary (classify_case), as the report and the dataset use it."""
    wrong = []
    for s, r, ask_ok in _single_turn_rows():
        v = AR.classify_case(r, m)
        if v.vague != ask_ok:
            wrong.append((s, r["id"], ask_ok, v.why, r["request"]))
    assert not wrong, wrong


def test_review_cues_are_values_directions_or_actions():
    # the 15 rows the r7 review found called vague, by cue
    for t, kind in (("slow the image gallery down a little", "direction"), ("make the slideshow faster", "direction"),
                    ("speed up the carousel", "direction"), ("page titles should be centered", "action"),
                    ("don't show categories that are sold out in the menu", "action"),
                    ("square buttons please", "value"), ("make the product photos square", "value"),
                    ("pill shaped buttons", "value"), ("use a serif font for the titles", "value"),
                    ("sort by price low to high by default", "value"), ("sort the carousel by THC, highest first", "value"),
                    ("add our google analytics G-ABC123", "value"), ("the logo should link to the products page", "value"),
                    ("add a picture to the instagram grid", "action"), ("let shoppers close the promo bar", "action"),
                    ("start with the filters closed", "action"), ("add a question about delivery times", "action"),
                    ("add a new review from Sarah: best shop in town", "action")):
        v = AR.classify(t)
        assert not v.vague and v.kind == kind, (t, v)
    # …and their look-alikes that stay vague
    for t in ("change how round the buttons are", "the round logo looks off", "the headings look too plain",
              "the product cards look too boxy", "can we make the hero more modern", "the menu text is too wordy"):
        assert AR.classify(t).vague, t


def test_a_name_the_site_shows_hides_no_new_value():
    """r7 review: names matched by prefix, so "rename delivery to Home delivery" (the page name "Home") was vague. A
    describing word names a thing only by its exact name; "rename … to X" / "should be X" always give X."""
    names = ("Home", "Delivery", "Shop", "Staff Picks", "Our Story", "FEATURED CATEGORIES")
    for t in ("rename delivery to Home delivery", "rename the shop link to Shop All",
              "rename FEATURED CATEGORIES to Shop by category", "the product carousel title should be Staff Picks"):
        v = AR.classify(t, names=names)
        assert not v.vague and v.kind == "value", (t, v)
    assert AR.classify("the section called Our Story needs work", names=names).vague
    assert AR.classify("the bit that says Shop is off", names=names).vague
    assert AR.classify("the section called Our Story Today", names=names).kind == "value"  # not the exact name


def test_a_complaint_is_no_value_or_direction():
    # a colour word that describes what is there; reported speech; a style word after too / more
    for t in ("the grey text in the newsletter email input should change", "everything looks so white and sterile",
              "people say they can't search for products", "customers keep saying it looks off",
              "the headings look too plain", "more modern buttons please"):
        assert AR.classify(t).vague, t
    # the values next to them stay values
    assert AR.classify("make the grey text white").why == "V1:colour:white"
    assert AR.classify("the header should be navy").why == "V1:colour:navy"
    assert AR.classify("change the text to grey").why == "V1:colour:grey"
    assert AR.classify("there is not enough space between the sections").kind == "direction"


def test_direction_of_an_opacity_field_is_never_guessed():
    t = "fade the header background image a lot more"
    assert AR.direction_of(t, "range", "headerBgImageOpacity") is None
    assert AR.direction_of(t, "range") is None  # the words alone
    assert AR.direction_of("make the overlay more see-through", "number", "overlayStrength") is None
    assert AR.direction_of("make the logo bigger", "range", "logoMaxHeight") == "up"


def test_write_evidence():
    from types import SimpleNamespace as F
    colour, size = F(type="color", key="titleColor", label="Headline"), F(type="range", key="paddingY", label="Padding")
    toggle = F(type="bool", key="showAddToCart", label="Show add to cart")
    font = F(type="select", key="headingFont", label="Heading font")
    op = lambda **v: [{"op": "set_fields", "values": v}]  # noqa: E731
    assert AR.write_evidence("make the content block headline #7a1f2b", colour,
                             oracle_ops=op(titleColor="#7A1F2B")).startswith("oracle:")
    assert AR.write_evidence("the content section heading should read Local and proud", F(type="text", key="title", label="t"),
                             oracle_ops=op(title="Local and proud")).startswith("oracle:")
    assert AR.write_evidence("my newsletter section is too tall", size, oracle_ops=op(paddingY=40)) == "direction:down"
    assert AR.write_evidence("remove the cart buttons from the product carousel", toggle,
                             oracle_ops=op(showAddToCart=False)) == "toggle:cart"
    sort = F(type="select", key="sortBy", label="Sort by", enum=["newest", "price-asc", "thc-desc"], enum_labels={})
    assert AR.write_evidence("sort the carousel by THC, highest first", sort, oracle_ops=op(sortBy="thc-desc")) == "option:sort"
    assert AR.write_evidence("sort the product row by newest", sort) == "option:newest"
    # r6's arbitrary writes: no evidence
    assert AR.write_evidence("the headings look too plain", font, oracle_ops=op(headingFont="Oswald")) is None
    assert AR.write_evidence("show what payments we accept", F(type="bool", key="paymentMethods.etransfer",
                                                               label="E-transfer"), oracle_ops=op(etransfer=True)) is None
    assert AR.write_evidence("section titles are too subtle", F(type="color", key="headingColor", label="h"),
                             oracle_ops=op(headingColor="#111111")) is None
    # no oracle (a pool ask): a literal only
    assert AR.write_evidence("make it #222222", colour).startswith("literal:V1")
    assert AR.write_evidence("hide the button shadow", colour) is None
