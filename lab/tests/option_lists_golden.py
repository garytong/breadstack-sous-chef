"""The option-lists parity golden: the BROWSER's own turn context
(turnContext.buildTurnContext, launcher 1d4193e2+) for the lab's tag / brand
/ category cases, which tests/test_option_lists.py asserts
souschef_lab.state.context_of reproduces byte for byte.

The launcher's own golden (``__fixtures__/turnContext.parity.json``, asserted
by tests/test_state.py) has five option-list cases; these add the edges the
lab mirrors by hand: every gate word and exclusion of OPTION_TEXT_RE
(brand colour / font / logo …, "tag line", NBSP, hyphens, plurals, case), an
unknown list (left out) next to a list that loaded empty (``[]``), the sort
(case, ties by slug, UTF-16 order of astral characters), labels (markup,
entities, blank / numeric names, numeric slugs, the 60-character clip), the
filters (``visible: false`` tags, nameless brands, entries without a slug,
non-objects), the cap of 100 with its ``{shown, total}`` flags, a prefix-family
focus, a named page's carousel, a manifest-less turn (the productCarousel
fallback), a location-ref section (no lists) and category ids as text.

Regenerate — node + a launcher checkout or snapshot at or after 1d4193e2,
read-only (one Vite boot; never two cold boots at once):

    COOKIE_CUTTER_DIR=<snapshot> .venv/bin/python tests/option_lists_golden.py --write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from souschef_lab import FIXTURES_DIR  # noqa: E402
from souschef_lab import browser_defaults as B  # noqa: E402
from souschef_lab import state as S  # noqa: E402

GOLDEN = FIXTURES_DIR / "parity" / "turnContext.options.json"
SOURCES = ("launcher/client/src/sousChef/turnContext.js", "launcher/client/src/sousChef/optionLists.js",
           "launcher/client/src/components/basicEditor/editors/carouselOptions.js")
CLIENT_OPTION_LISTS = Path("launcher/client/src/sousChef/optionLists.js")


def trimmed_manifest() -> dict:
    """The lab manifest's section types with their `ref` fields only (enough for optionRefSectionTypes and
    lookupSectionTypeId; no lists, so the outline stays small): productCarousel (tag / brand / category refs,
    the derived tagId / brandId), contentBlock (a location ref), heroBanner (none)."""
    doc = json.loads((FIXTURES_DIR / "manifest.snapshot.json").read_text(encoding="utf-8"))
    out: dict = {"sections": {}}
    for typ in ("productCarousel", "contentBlock", "heroBanner"):
        fields = (doc["sections"][typ].get("fields") or {})
        out["sections"][typ] = {"fields": {k: {kk: f[kk] for kk in ("type", "ref", "derived_from") if kk in f}
                                           for k, f in fields.items() if isinstance(f, dict) and f.get("type") == "ref"}}
    return out


def _ctx(page: str = "home", template: str = "cozy", **focus) -> dict:
    return {"currentPage": page, "templateId": template, "editingStoreId": None, "tab": focus.get("tab", "editor"),
            "activeSectionId": focus.get("activeSectionId"), "globalSection": focus.get("globalSection"),
            "locations": [{"id": "loc-1", "name": "Main St"}]}


def _state(template: str = "cozy", home: list | None = None, **more) -> dict:
    return {"template": {"id": template}, "homeSections": home if home is not None else [], **more}


HERO = {"id": "heroBanner", "enabled": True, "settings": {"title": "Hi"}}
CAROUSEL = {"id": "productCarousel-1", "enabled": True, "settings": {"title": "Picks", "subType": "tag", "tagSlug": "new"}}
TAGS = [{"id": 7, "slug": "staff-picks", "name": "Staff Picks", "visible": True, "count": 12},
        {"id": 3, "slug": "hidden", "name": "Hidden", "visible": False}, {"id": 9, "slug": "sale", "name": "Sale"}]
BRANDS = [{"id": "b2", "slug": "house", "name": "House Brand"}, {"id": "b1", "slug": "acme", "name": "Acme"},
          {"id": "b3", "slug": "blank", "name": ""}]

# (text, lists expected?) — the gate words and their exclusions, each asked on a home page without a carousel.
GATE_TEXTS = [
    ("Which brands can I feature?", True),
    ("Change my brand color to red", False),
    ("Make the brand-logo bigger and use the Brand Fonts", False),
    ("Our brand voice and our brand story need work", False),
    ("Brands colours should match", False),
    ("Use the brand colour everywhere", False),  # NBSP is JavaScript whitespace
    ("Edit the tag line", False),
    ("Make the tag lines shorter", False),
    ("Change the tagline", False),                    # one word: no `tag` word at all
    ("Show the tag lineup", True),                     # `lines?\b` fails inside "lineup": not excluded
    ("Show the TAGS please", True),
    ("Pick a brand", True),
    ("A brand-new carousel for the home page", True),
    ("Our brand names are long", True),               # `name\b` fails on "names"
    ("Which brand name sells most", False),
    ("Add some hashtags and get tagged", False),
    ("Swap the CAROUSELS around", True),
]


def cases() -> list[dict]:
    m = trimmed_manifest()
    fx = S.load_context_fixture()
    out: list[dict] = []

    def add(name: str, state: dict, ctx: dict, extras: dict, text: str | None = None, manifest: bool = True) -> None:
        out.append({"name": name, "state": state, "ctx": ctx, "extras": extras, "text": text, "manifest": manifest})

    add("the lab fixture's lists with a carousel on home",
        _state("editorial", [HERO, CAROUSEL]), _ctx("home", "editorial"),
        {"categories": fx["categories"], "tags": fx["tags"], "brands": fx["brands"]})
    add("category ids are text; slug falls back to the id; blank ids are left out",
        _state("cozy", [HERO]), _ctx(),
        {"categories": [{"id": 12, "slug": "flower", "name": "Flower", "parent": 0},
                        {"id": "13", "slug": "pre-rolls", "name": "Pre-Rolls", "parent": 12},
                        {"id": 0, "slug": "", "name": "Zero"}, {"id": "", "slug": "no-id", "name": "No id"},
                        {"id": None, "slug": "null-id", "title": "Null id"}, {"id": 1.0, "slug": "float", "label": "Float"},
                        {"id": True, "slug": 5, "name": 7}, {"id": 14, "slug": "x", "name": "", "title": "", "label": "L"}]})
    for i, (text, _) in enumerate(GATE_TEXTS):
        add(f"gate {i:02d}: {text}", _state("cozy", [HERO]), _ctx(), {"tags": TAGS, "brands": BRANDS}, text)
    add("an unknown list is left out; a list that loaded empty is sent empty",
        _state("cozy", [HERO, CAROUSEL]), _ctx(), {"brands": []})
    add("a null list is unknown too", _state("cozy", [HERO, CAROUSEL]), _ctx(), {"tags": None, "brands": BRANDS})
    add("both lists unknown: nothing, even with the carousel and the word",
        _state("cozy", [HERO, CAROUSEL]), _ctx(), {"categories": []}, "Which tags can I use?")
    add("sort, labels and filters",
        _state("cozy", [CAROUSEL]), _ctx(),
        {"tags": [{"id": 1, "slug": "b-2", "name": "banana"}, {"id": 2, "slug": "b-1", "name": "Banana"},
                  {"id": 3, "slug": "apple", "name": "apple"}, {"id": 4, "slug": "smile", "name": "\U0001F600 Smile"},
                  {"id": 5, "slug": "half", "name": "｡ halfwidth"}, {"id": 6, "slug": "new-hot", "name": "<b>New</b> &amp; hot"},
                  {"id": 7, "slug": "null-name", "name": None}, {"id": 8, "slug": "blank-name", "name": ""},
                  {"id": 9, "slug": "num-name", "name": 42}, {"id": 10, "slug": 77, "name": "Numeric slug"},
                  {"id": 11, "slug": "zero-visible", "name": "Zero visible", "visible": 0},
                  {"id": 12, "slug": "gone", "name": "Gone", "visible": False}, "a string", None, 5,
                  {"id": 13, "slug": "", "name": "Blank slug"}, {"id": 14, "name": "No slug"},
                  {"id": 15, "slug": "long", "name": "A very long tag name that goes on and on past the sixty character clip"},
                  {"id": 16, "slug": "spaces", "name": "  \t "}],
         "brands": [{"id": "b1", "slug": "zero", "name": 0}, {"id": "b2", "slug": "blank", "name": ""},
                    {"id": "b3", "slug": "spaces", "name": "   "}, {"id": "b4", "slug": "ok", "name": "Okay & Co"},
                    {"id": "b5", "name": "No slug"}, {"id": "b6", "slug": None, "name": "Null slug"}]})
    add("a focused prefix-family carousel off home; over 100 the lists are cut and flagged",
        _state("bold", []), _ctx("products", "bold", activeSectionId="productCarousel-deals"),
        {"tags": [{"id": 1000 + i, "slug": f"tag-{i:03d}", "name": f"Tag {i:03d}"} for i in range(130, 0, -1)],
         "brands": [{"id": f"b{i}", "slug": f"brand-{i:03d}", "name": f"Brand {i:03d}"} for i in range(101)]})
    add("exactly 100 is not cut", _state("bold", [CAROUSEL]), _ctx("home", "bold"),
        {"tags": [{"id": i, "slug": f"t{i:03d}", "name": f"T {i:03d}"} for i in range(100)], "brands": BRANDS})
    add("a named custom page's carousel brings the lists",
        _state("sweet", [HERO], customPages=[{"slug": "deals", "title": "Deals", "status": "published"}],
               pageSectionsAll={"deals": [{"id": "productCarousel-1", "enabled": True, "settings": {"title": "On sale"}}]}),
        _ctx("home", "sweet"), {"tags": TAGS, "brands": BRANDS}, "Make the Deals page title bigger")
    add("a disabled carousel row still counts", _state("cozy", [HERO, {**CAROUSEL, "enabled": False}]), _ctx(),
        {"tags": TAGS, "brands": BRANDS})
    add("a location-ref section brings no lists", _state("cozy", [{"id": "contentBlock-1", "enabled": True, "settings": {}}]),
        _ctx(), {"tags": TAGS, "brands": BRANDS})
    add("without a manifest the productCarousel fallback decides",
        _state("cozy", [HERO, {**CAROUSEL, "id": "productCarousel-2"}]), _ctx(), {"tags": TAGS, "brands": BRANDS},
        manifest=False)
    add("without a manifest a request word still decides", _state("cozy", [HERO]), _ctx(),
        {"tags": TAGS, "brands": BRANDS}, "what brands are there", manifest=False)
    return [{**c, "manifest_doc": m if c["manifest"] else None} for c in out]


def browser_expected(items: list[dict], cc_dir: str | Path | None = None) -> list[dict]:
    """buildTurnContext for each case, in one node boot (browser_defaults.browser_contexts)."""
    reqs = []
    for c in items:
        extras = dict(c["extras"])
        if c["manifest_doc"] is not None:
            extras["manifest"] = c["manifest_doc"]
        if c["text"]:
            extras["text"] = c["text"]
        reqs.append({"state": c["state"], "ctx": c["ctx"], "extras": extras})
    return B.browser_contexts(reqs, cc_dir)


def lab_context(c: dict, manifest: dict | None) -> dict:
    """state.context_of as the lab calls it for a golden case (the extras' lists, the ctx's locations)."""
    ctx = c["ctx"]
    extras = {**c["extras"], "locations": ctx.get("locations")}
    return S.context_of(c["state"], ctx.get("currentPage") or "home", extras, manifest if c["manifest"] else None, ctx,
                        text=c.get("text"))


def has_option_lists(cc_dir: Path | None) -> bool:
    return cc_dir is not None and (cc_dir / CLIENT_OPTION_LISTS).is_file()


def generate(cc_dir: str | Path | None = None) -> dict:
    root = B.cookie_cutter_dir(cc_dir)
    if not has_option_lists(root):
        raise SystemExit("needs a launcher checkout or snapshot with sousChef/optionLists.js (1d4193e2 or later): COOKIE_CUTTER_DIR")
    items = cases()
    expected = browser_expected(items, root)
    snap = root / "SNAPSHOT_OF"
    return {
        "what": "turnContext.buildTurnContext's own output for the lab's option-list cases (tests/option_lists_golden.py); "
                "tests/test_option_lists.py asserts souschef_lab.state.context_of gives the same object, key order "
                "included. Regenerate: COOKIE_CUTTER_DIR=<launcher at or after 1d4193e2> .venv/bin/python "
                "tests/option_lists_golden.py --write",
        "source": {"snapshot_of": snap.read_text(encoding="utf-8").strip() if snap.is_file() else None,
                   "files": {rel: hashlib.sha256((root / rel).read_bytes()).hexdigest() for rel in SOURCES}},
        "manifest": items[0]["manifest_doc"],
        "cases": [{k: c[k] for k in ("name", "state", "ctx", "extras", "text", "manifest")} | {"expected": e}
                  for c, e in zip(items, expected)],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cookie-cutter", help="the launcher checkout or snapshot (default $COOKIE_CUTTER_DIR)")
    ap.add_argument("--write", action="store_true", help=f"write {GOLDEN.relative_to(LAB)}")
    a = ap.parse_args(argv)
    doc = generate(a.cookie_cutter)
    text = json.dumps(doc, ensure_ascii=False, indent=1) + "\n"
    if a.write:
        GOLDEN.write_text(text, encoding="utf-8")
        print(f"wrote {GOLDEN} ({len(doc['cases'])} cases)")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
