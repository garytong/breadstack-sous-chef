"""EditorState builder and turn-context derivation (docs/LAB.md §1 `state.py`).

`EditorState` is the plain object of PreviewPage slices the basic editor
writes (engine README §3). The empty shape is `golden.test.js:104`
(`emptyState`); the home rows are `DEFAULT_DEFAULT_HOME_SECTIONS`
(pageSections.jsx:4012); rich settings for the sampled sections and the
global / branding / pageVisibility slices come from the probe fixture
(`fixtures/states/probe-initials.json`, copied from
`ops/__fixtures__/probe-sample.json`), and every manifest field with a
`default` that the seed left unset is filled from the manifest.

`context_of` mirrors the browser's `turnContext.buildTurnContext`
(launcher/client/src/sousChef/turnContext.js): `page`, `template`, `layout`
(`layouts[page]` or `''`), `sections[{id,type,title,enabled}]`, `categories`,
`customPages`, `nav` (the storefront's header links), `locations`, `focus`
(tab / open section / open global panel), `shopMenu` and the manifest-derived
list `outline` (+ `outlineTruncated`); with `text` (the request) the pages it
names (`named_pages`, mirroring the browser's `pageNames.js`) join the
outline after the viewed page. The shared golden
`launcher/client/src/sousChef/__fixtures__/turnContext.parity.json` in the
cookie-cutter checkout holds the browser's output; tests/test_state.py
asserts this module reproduces it.

Path helpers resolve a manifest field to its state path for a section row,
a global panel, branding or a page, and `gate_writes` computes the values
that open a field's `conditional_on` / `hidden_when` gate (docs/LAB.md §2).

Standard library only.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

from . import FIXTURES_DIR, browser_defaults
from .manifest import SECTION_PREFIX_FAMILIES, Container, Field, Manifest, lookup_section_type_id, section_type_id

# pageSections.jsx:379-382 — the ten basic templates; `default` is NOT one
# (a section `add` on it fails E_TARGET, engine README §2 R8).
BASIC_TEMPLATE_IDS = ("bold", "boutique", "cozy", "editorial", "elegant", "playful", "pluto", "retro", "sweet", "vintage")
DEFAULT_TEMPLATE = "editorial"

# Copied from launcher/client/src/data/pageSections.jsx:4012
# DEFAULT_DEFAULT_HOME_SECTIONS (breadstack-cookie-cutter sous-chef-v2 0f9fd067).
DEFAULT_HOME_SECTIONS: list[dict] = [
    {"id": "heroBanner", "enabled": True},
    {"id": "featuredCategories", "enabled": True},
    {"id": "productCarousel-1", "enabled": True, "settings": {"title": "New Arrivals", "subType": "newest"}},
    {"id": "productCarousel-2", "enabled": True, "settings": {"title": "Customer Favorites", "subType": "popular"}},
    {"id": "testimonials", "enabled": True, "settings": {"layout": "carousel"}},
    {"id": "contentBlock-1", "enabled": True, "settings": {"title": "Crafted with Care", "imageLayout": "3-col"}},
    {"id": "contentBlock-2", "enabled": True, "settings": {"title": "Our Story", "imageLayout": "2-col", "reversed": True}},
    {"id": "gallery", "enabled": True},
    {"id": "contentBlock-3", "enabled": True, "settings": {"title": "Visit Us", "imageLayout": "1-col"}},
    {"id": "contentBlock-features", "enabled": True, "settings": {"title": "Why Choose Us?", "style": "feature-cards"}},
    {"id": "blogSection", "enabled": True},
    {"id": "contentBlock-icons", "enabled": True, "settings": {"title": "Why Choose Us?", "style": "feature-icons"}},
    {"id": "newsletter", "enabled": True},
]

# The sampled sections whose rich settings the probe recorded (golden §6).
PROBE_SECTIONS = ("heroBanner", "productCarousel", "gallery", "contentBlock")

# Copied from schema.js PINNED_BY_PAGE (ids only; `reorderable` rows persist a
# `{id, enabled}` placeholder, the rest are virtual).
PINNED_BY_PAGE: dict[str, dict] = {
    "faq": {"id": "__pinned_faq", "reorderable": False},
    "product-details": {"id": "__pinned_product_detail", "reorderable": False},
    "products": {"id": "__pinned_product_list", "reorderable": True},
    "blog": {"id": "__pinned_blog_list", "reorderable": False},
    "blog-detail": {"id": "__pinned_blog_detail", "reorderable": False},
    "privacy-policy": {"id": "__pinned_privacy", "reorderable": False},
    "terms-of-service": {"id": "__pinned_terms", "reorderable": False},
    "return-refund": {"id": "__pinned_return_refund", "reorderable": False},
    "age-gate": {"id": "__pinned_age_gate", "reorderable": False},
    "location-selection": {"id": "__pinned_location_selection", "reorderable": False},
    "cart": {"id": "__pinned_cart", "reorderable": False},
    "checkout": {"id": "__pinned_checkout", "reorderable": False},
    "account": {"id": "__pinned_account", "reorderable": False},
    "login": {"id": "__pinned_login", "reorderable": False},
    "order-confirmation": {"id": "__pinned_order_confirmation", "reorderable": False},
    "order-details": {"id": "__pinned_order_details", "reorderable": False},
}

# schema.js PATH_ROOT_TO_SLICE: manifest path roots that are not the slice name
PATH_ROOT_TO_SLICE = {
    "globalSettings": "globalSettings", "branding": "branding", "pageVisibility": "pageVisibility", "seo": "seo",
    "layouts": "layouts", "faqCategories": "faqCategories", "termsContent": "terms", "privacyContent": "privacy",
    "returnRefundContent": "returnRefund", "disclaimerContent": "disclaimer", "eulaContent": "eula",
    "cookiePolicyContent": "cookiePolicy",
}


def slice_path(path: str) -> str:
    """A manifest path (`termsContent[0].title`) → the EditorState path (`terms[0].title`)."""
    m = re.match(r"^([A-Za-z_]+)(.*)$", path or "")
    if not m:
        return path
    return PATH_ROOT_TO_SLICE.get(m.group(1), m.group(1)) + m.group(2)


# golden.test.js:104 emptyState()
EMPTY_STATE: dict = {
    "template": {"id": DEFAULT_TEMPLATE},
    "homeSections": [],
    "pageSectionsAll": {},
    "locationDetailsOverrides": {},
    "globalSettings": {},
    "branding": {},
    "pageVisibility": {},
    "layouts": {},
    "customPages": [],
    "seo": {},
    "faqCategories": [],
    "terms": [], "privacy": [], "returnRefund": [], "disclaimer": [], "eula": [], "cookiePolicy": [],
}

# probe `initial` member → EditorState slice (golden.test.js buildState)
_INITIAL_SLICES = {
    "global": "globalSettings", "branding": "branding", "pageVisibility": "pageVisibility",
    "faqCategories": "faqCategories", "termsContent": "terms", "privacyContent": "privacy",
    "returnRefundContent": "returnRefund",
}

# ── fixtures ────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_fixture(name: str, fixtures_dir: Path | None = None) -> Any:
    return _load_json((fixtures_dir or FIXTURES_DIR) / name)


def load_context_fixture(fixtures_dir: Path | None = None) -> dict:
    return load_fixture("context.json", fixtures_dir)


def load_layouts_fixture(fixtures_dir: Path | None = None) -> dict:
    return load_fixture("layouts.json", fixtures_dir)


def load_assets_fixture(fixtures_dir: Path | None = None) -> list[dict]:
    return list(load_fixture("assets.json", fixtures_dir)["assets"])


def stable_hash(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:12], 16)


def fixture_asset_for(assets: list[dict], case_id: str, asset_type: str = "image", asset_id: str | None = None) -> dict:
    """The fixture asset the collector's fake picker answers a case's
    `pick_asset` with (collect.pick_asset_for): the asset the case NAMES
    (`asset_id` — the generator's `meta.asset_id`) when it is one of the
    asked type (a merchant who asked for "the image 'Sunlit field at
    harvest'" picks that one), else one of the asked type chosen from the
    case id alone. Either way the answer is a function of the case, so the
    generator can put exactly this asset in the expected ops and every
    sample of the case gets the same one. The generator and the collector
    must share this function: when they disagreed (the picker also hashed
    the service's random tool call id, 2026-10-04) every pick case compared
    the model's correct write against another asset; and while the picker
    ignored the asset a `url`-variant request named (int-run3, 2026-10-04:
    the model owns no asset library, so it opens the picker for a request
    that names an image by its alt and writes what the picker returns) 15 of
    the run's 21 `url` rows failed on the picker's other asset."""
    pool = [a for a in assets if (a.get("asset_type") or a.get("type") or "image") == asset_type] or list(assets)
    if not pool:
        raise ValueError("no fixture assets to answer pick_asset with (fixtures/assets.json)")
    if asset_id:
        named = next((a for a in pool if a.get("id") == asset_id), None)
        if named is not None:
            return named
    return pool[stable_hash(case_id) % len(pool)]


def case_asset_id(case: dict) -> str | None:
    """The fixture asset a case names (`meta.asset_id`: the `url` variant's
    requested asset, the pick variant's hash pick); None for other cases."""
    meta = case.get("meta") if isinstance(case, dict) else None
    v = (meta or {}).get("asset_id") if isinstance(meta, dict) else None
    return str(v) if v else None


def load_probe_initials(fixtures_dir: Path | None = None) -> dict:
    return load_fixture("states/probe-initials.json", fixtures_dir)


# ── paths ───────────────────────────────────────────────────────────────────

_SEG_RE = re.compile(r"^([^\[\]]*)((?:\[\d+\])*)$")


def split_path(path: str) -> list[str | int]:
    """`a.b[0].c` → ['a','b',0,'c'] (turnContext.splitWrittenPath); `a.b.0.c` keeps '0' as a key."""
    out: list[str | int] = []
    for raw in str(path or "").split("."):
        if raw == "":
            continue
        m = _SEG_RE.match(raw)
        if not m:
            out.append(raw)
            continue
        if m.group(1) != "":
            out.append(m.group(1))
        for idx in re.findall(r"\d+", m.group(2)):
            out.append(int(idx))
    return out


def get_path(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for seg in split_path(path):
        if isinstance(seg, int):
            if not isinstance(cur, list) or seg >= len(cur):
                return default
            cur = cur[seg]
        else:
            if not isinstance(cur, dict) or seg not in cur:
                return default
            cur = cur[seg]
    return cur


def set_path(obj: Any, path: str, value: Any) -> Any:
    """Writes `value` at `path`, creating dicts / padding lists on the way. Returns obj."""
    segs = split_path(path)
    if not segs:
        raise ValueError("empty path")
    cur = obj
    for i, seg in enumerate(segs):
        last = i == len(segs) - 1
        nxt = segs[i + 1] if not last else None
        if isinstance(seg, int):
            if not isinstance(cur, list):
                raise TypeError(f"{path}: segment {i} expects a list")
            while len(cur) <= seg:
                cur.append(None)
            if last:
                cur[seg] = value
            else:
                if cur[seg] is None or not isinstance(cur[seg], (dict, list)):
                    cur[seg] = [] if isinstance(nxt, int) else {}
                cur = cur[seg]
        else:
            if not isinstance(cur, dict):
                raise TypeError(f"{path}: segment {i} expects a dict")
            if last:
                cur[seg] = value
            else:
                if not isinstance(cur.get(seg), (dict, list)):
                    cur[seg] = [] if isinstance(nxt, int) else {}
                cur = cur[seg]
    return obj


def fill_indexes(path: str, indexes: list[int] | tuple[int, ...]) -> str:
    """`settings.slides[].bullets[].title` + [0, 2] → `settings.slides[0].bullets[2].title`."""
    out = path
    for i in indexes:
        out = out.replace("[]", f"[{i}]", 1)
    return out


def field_key_with_id(field: Field, ident: str) -> str:
    """`locationOverrides.{locationId}.mode` + `loc-main` → `locationOverrides.loc-main.mode`."""
    return re.sub(r"\{[a-zA-Z]+\}", ident, field.key)


# ── sections ────────────────────────────────────────────────────────────────

def sections_of(state: dict, page: str = "home") -> list[dict]:
    if page == "home":
        return state.get("homeSections") or []
    return (state.get("pageSectionsAll") or {}).get(page) or []


def sections_slice_path(page: str = "home") -> str:
    return "homeSections" if page == "home" else f"pageSectionsAll.{page}"


def find_section_index(state: dict, section_id: str, page: str = "home") -> int:
    for i, s in enumerate(sections_of(state, page)):
        if isinstance(s, dict) and s.get("id") == section_id:
            return i
    return -1


def find_section(state: dict, section_id: str, page: str = "home") -> dict | None:
    i = find_section_index(state, section_id, page)
    return sections_of(state, page)[i] if i >= 0 else None


def section_settings(state: dict, section_id: str, page: str = "home") -> dict:
    s = find_section(state, section_id, page)
    if s is None:
        raise KeyError(f"no section {section_id} on {page}")
    s.setdefault("settings", {})
    return s["settings"]


def type_of_section(section_id: str, known: set[str] | None = None) -> str:
    """The manifest type a row edits with: `-<n>` stripped, prefix families collapsed
    (`contentBlock-features` → `contentBlock`). Unknown ids keep their stripped form."""
    return lookup_section_type_id(section_id, known | set(SECTION_PREFIX_FAMILIES) if known else set(SECTION_PREFIX_FAMILIES)) \
        or section_type_id(section_id)


def sections_of_type(state: dict, type_id: str, page: str = "home") -> list[dict]:
    return [s for s in sections_of(state, page) if type_of_section(s.get("id")) == type_id]


# ── field ⇄ state ───────────────────────────────────────────────────────────

def field_state_path(
    container: Container, field: Field, *, state: dict | None = None, section_id: str | None = None,
    page: str = "home", key: str | None = None, indexes: list[int] | tuple[int, ...] = (),
) -> str:
    """The absolute EditorState path a field write lands on.

    section fields: `<slice>[i].settings.<…>` with `[]` filled from `indexes`
    (hero mirror keys fill `[0]`); keyed fields take the id from `key`
    (`locationOverrides.loc-main.mode`); other containers use `field.path`
    with `{page}` substituted."""
    path = field.path
    if key and field.keyed_by:
        # `locationOverrides.loc-main.mode` / `footerLocationEnabled.loc-main`: the id is segment 1
        ident = key.split(".")[1] if "." in key else key
        path = re.sub(r"\{[a-zA-Z]+\}", ident, path)
    if "[]" in path:
        path = fill_indexes(path, indexes or [0] * path.count("[]"))
    path = path.replace("{page}", page)
    if container.kind == "section":
        if state is None or section_id is None:
            raise ValueError("section fields need state + section_id")
        i = find_section_index(state, section_id, page)
        if i < 0:
            raise KeyError(f"no section {section_id} on {page}")
        return f"{sections_slice_path(page)}[{i}].{path}"
    return slice_path(path)


def read_field(container: Container, field: Field, state: dict, *, section_id: str | None = None,
               page: str = "home", key: str | None = None, indexes=()) -> Any:
    return get_path(state, field_state_path(container, field, state=state, section_id=section_id, page=page, key=key, indexes=indexes))


def write_field(container: Container, field: Field, state: dict, value: Any, *, section_id: str | None = None,
                page: str = "home", key: str | None = None, indexes=()) -> dict:
    set_path(state, field_state_path(container, field, state=state, section_id=section_id, page=page, key=key, indexes=indexes), value)
    return state


def resolve_key(manifest: Manifest, container: Container, key: str) -> Field | None:
    """A top-level field, a hero mirror key, a page `_common` key or a keyed field
    (`locationOverrides.loc-main.mode` → `locationOverrides.{locationId}.mode`)."""
    f = manifest.resolve_field(container, key)
    if f:
        return f
    parts = key.split(".")
    if len(parts) < 2:
        return None
    for cand in container.fields.values():
        if not cand.keyed_by:
            continue
        cparts = cand.key.split(".")
        if len(cparts) != len(parts):
            continue
        # every literal segment must match; `{locationId}` / `{slug}` segments take any id
        if all(cp == p or (cp.startswith("{") and cp.endswith("}")) for cp, p in zip(cparts, parts)):
            return cand
    return None


def write_values(manifest: Manifest, container: Container, state: dict, values: dict, *,
                 section_id: str | None = None, page: str = "home") -> dict:
    """Seeds `values` (field key → value) into the state the way the editor
    would store them: mirror keys go to `slides[0].<key>` AND the flat key."""
    for key, value in values.items():
        f = resolve_key(manifest, container, key)
        if f is None:
            raise KeyError(f"{container.name} has no field {key}")
        if container.kind == "section" and container.mirror and key in container.mirror_keys and key not in container.fields:
            write_field(container, f, state, value, section_id=section_id, page=page, indexes=[0])
            flat = f"{sections_slice_path(page)}[{find_section_index(state, section_id, page)}].settings.{key}"
            set_path(state, flat, copy.deepcopy(value))
        else:
            write_field(container, f, state, value, section_id=section_id, page=page, key=key if f.keyed_by else None)
    return state


# ── gates ───────────────────────────────────────────────────────────────────

def _gate_field(manifest: Manifest, container: Container, gate_key: str) -> Field | None:
    """The field a gate names; `slides[].layout` resolves to the slides item field."""
    if "[]" in gate_key:
        list_key, _, item_key = gate_key.rpartition("[].")
        l = container.lists.get(list_key)
        if l and item_key in l.item_fields:
            return l.item_fields[item_key]
        return manifest.resolve_field(container, item_key)
    return manifest.resolve_field(container, gate_key)


def _truthy_for(gate: Field | None, assets: list[dict] | None) -> Any:
    if gate is None:
        return "x"
    if gate.type == "image":
        return (assets or [{}])[0].get("public_url") or "https://cdn.int.breadstack.io/gal/lab/gate.webp"
    if gate.type == "bool":
        return True
    if gate.enum:
        return next((v for v in gate.enum if v not in (None, "")), gate.enum[0])
    if gate.type in ("range", "number"):
        return gate.min if gate.min not in (None, 0) else (gate.max or 1)
    if gate.type == "color":
        return "#336699"
    return "x"


def _falsy_for(gate: Field | None) -> Any:
    if gate is None:
        return ""
    if gate.has_unset:
        return gate.unset_value
    if gate.type == "bool":
        return False
    return ""


def _differs(gate: Field | None, avoid: list) -> Any:
    if gate and gate.enum:
        for v in gate.enum:
            if v not in avoid:
                return v
    if gate and gate.type == "bool":
        return not avoid[0] if avoid and isinstance(avoid[0], bool) else True
    return "other"


def gate_writes(manifest: Manifest, container: Container, field: Field, assets: list[dict] | None = None) -> dict[str, Any]:
    """Field key → value that makes `field` live (docs/LAB.md §2: the gate is
    set in `site_state`). Keys are container-relative; `slides[].x` gates are
    returned as the item key `x` with a `slides[].` prefix kept so the caller
    writes them into slide 0."""
    out: dict[str, Any] = {}
    cond = field.conditional_on
    if cond and cond.get("key"):
        gate = _gate_field(manifest, container, cond["key"])
        v = cond.get("value")
        if v == "set":
            out[cond["key"]] = _truthy_for(gate, assets)
        elif v == "unset":
            out[cond["key"]] = _falsy_for(gate)
        elif isinstance(v, list):
            out[cond["key"]] = v[0] if v else _truthy_for(gate, assets)
        else:
            out[cond["key"]] = v
    hid = field.hidden_when
    if hid and hid.get("key"):
        gate = _gate_field(manifest, container, hid["key"])
        v = hid.get("value")
        if v == "set":
            out[hid["key"]] = _falsy_for(gate)
        elif v == "unset":
            out[hid["key"]] = _truthy_for(gate, assets)
        elif isinstance(v, bool):
            out[hid["key"]] = not v
        elif isinstance(v, list):
            out[hid["key"]] = _differs(gate, v)
        else:
            out[hid["key"]] = _differs(gate, [v])
    return out


def apply_gate_writes(manifest: Manifest, container: Container, state: dict, writes: dict, *,
                      section_id: str | None = None, page: str = "home") -> dict:
    """Writes gate values; `slides[].layout` lands on slide 0 (and the flat
    mirror key when the hero mirrors it)."""
    for key, value in writes.items():
        if "[]" in key:
            list_key, _, item_key = key.rpartition("[].")
            l = container.lists.get(list_key)
            f = l.item_fields.get(item_key) if l else None
            if f is None:
                continue
            idx = find_section_index(state, section_id, page)
            base = f"{sections_slice_path(page)}[{idx}].{l.path}[0].{item_key}"
            set_path(state, base, value)
            if container.mirror and item_key in container.mirror_keys:
                set_path(state, f"{sections_slice_path(page)}[{idx}].settings.{item_key}", copy.deepcopy(value))
        else:
            f = resolve_key(manifest, container, key)
            if f is None:
                continue
            write_values(manifest, container, state, {key: value}, section_id=section_id, page=page)
    return state


# ── builder ─────────────────────────────────────────────────────────────────

def _defaults_for(container: Container) -> dict[str, Any]:
    """Top-level fields with a `default` (no list items, no keyed/templated paths)."""
    out = {}
    for k, f in container.fields.items():
        if f.has_default and "[]" not in f.path and "{" not in f.path and "." not in k:
            out[k] = copy.deepcopy(f.default)
    return out


def _fill_defaults(target: dict, container: Container, *, prefix: str) -> None:
    """Fills manifest defaults the seed left unset, at the field's path below `prefix`."""
    for k, f in container.fields.items():
        if not f.has_default or "[]" in f.path or "{" in f.path:
            continue
        path = f.path
        if prefix and path.startswith(prefix + "."):
            path = path[len(prefix) + 1:]
        elif prefix:
            continue
        if get_path(target, path, _SENTINEL) is _SENTINEL:
            set_path(target, path, copy.deepcopy(f.default))


_SENTINEL = object()


# ── the probe fixture's placeholder copy ─────────────────────────────────────
# The probe fixture names its list rows "Zq Fixture Bullet One", "Zq Fixture
# Post Two" … (a marker its tests grep for). A real site never shows that: the
# outline the browser sends carries row titles, so a training context full of
# "Zq Fixture Column One" teaches the model a site nobody has (the FAQ fix of
# 2026-10-10 removed the premium FAQ rows for the same reason). `humanize`
# swaps every such title / name for plausible dispensary copy — strings only,
# no row added, removed or reshaped, ids (`zq-cat-1` …) untouched — the same
# copy the r4 F4/F5 author used, so their states are unchanged by it.
HUMAN_ROWS: dict[tuple[str, str], list[str]] = {
    ("heroBanner", "slides[0].bullets"): ["Lab-tested quality", "Same-day delivery"],
    ("heroBanner", "slides[1].bullets"): ["Members earn points", "Curbside pickup"],
    ("contentBlock-1", "items"): ["Locally grown", "Budtender picks", "Fair prices"],
    ("contentBlock-1", "bullets"): ["Small-batch growers", "Hand-trimmed buds"],
    ("contentBlock-2", "items"): ["Founded in 2019", "Family owned", "Community first"],
    ("contentBlock-2", "bullets"): ["Locally owned", "Award-winning staff"],
    ("contentBlock-3", "items"): ["Open 7 days", "Free parking", "Wheelchair accessible"],
    ("contentBlock-3", "bullets"): ["Curbside pickup", "ATM on site"],
    ("contentBlock-features", "items"): ["Huge selection", "Expert budtenders", "Loyalty rewards"],
    ("contentBlock-features", "bullets"): ["Price match", "Daily deals"],
    ("contentBlock-icons", "items"): ["Fast delivery", "Secure checkout", "Discreet packaging"],
    ("contentBlock-icons", "bullets"): ["Text order updates", "Tap to pay"],
}
HUMAN_FALLBACK = ["Quality you can trust", "Friendly budtenders", "Local favourites", "Fresh every week",
                  "Easy pickup", "Members save more"]
HUMAN_POSTS = ["Beginner's guide to edibles", "What are terpenes?", "Indica vs sativa: does it matter?",
               "How to store your flower", "Meet our budtenders", "New: local craft pre-rolls"]
HUMAN_LEGAL = ["Introduction", "Information we collect", "How we use your information", "Sharing your information",
               "Cookies", "Data retention", "Your rights", "Age requirements", "Security", "Third-party links",
               "Changes to this policy", "Contact us", "Governing law", "Limitation of liability", "Termination"]
HUMAN_CERTS = ["Health Canada Licensed", "AGCO Authorized Retailer"]
HUMAN_TERMS = ["Acceptance of terms", "Eligibility", "Your account", "Orders and payment", "Pricing", "Delivery", "Pickup",
               "Returns", "Prohibited use", "Intellectual property", "Disclaimers", "Limitation of liability",
               "Governing law", "Changes to these terms", "Contact us"]
HUMAN_RETURNS = ["Our return policy", "How refunds work"]
HUMAN_CATEGORIES = {"zq-cat-1": "Topicals", "zq-cat-2": "Beverages"}
PROBE_MARK = "Zq Fixture"


def _is_probe(v: Any) -> bool:
    return isinstance(v, str) and v.startswith("Zq")


def humanize(state: dict) -> dict:
    """`state` with the probe fixture's "Zq Fixture …" titles and names replaced
    by realistic copy (in place; returned). Idempotent."""
    def rows(lst: Any, names: list[str], member: str) -> None:
        for j, it in enumerate(lst or []):
            if isinstance(it, dict) and _is_probe(it.get(member)):
                it[member] = names[j % len(names)]

    for page_rows in [state.get("homeSections") or []] + [v for v in (state.get("pageSectionsAll") or {}).values()
                                                         if isinstance(v, list)]:
        for row in page_rows:
            if not isinstance(row, dict):
                continue
            sid, st = str(row.get("id") or ""), row.get("settings") or {}
            for i, slide in enumerate(st.get("slides") or []):
                if isinstance(slide, dict):
                    rows(slide.get("bullets"), HUMAN_ROWS.get((sid, f"slides[{i}].bullets"), HUMAN_FALLBACK), "title")
            for lst in ("items", "bullets"):
                rows(st.get(lst), HUMAN_ROWS.get((sid, lst), HUMAN_FALLBACK), "title")
    pv = state.get("pageVisibility") or {}
    rows((pv.get("blog") or {}).get("posts"), HUMAN_POSTS, "title")
    rows((pv.get("_meta") or {}).get("headerNavCustomLinks"), ["Deals"], "name")
    rows((state.get("branding") or {}).get("certifications"), HUMAN_CERTS, "name")
    for key, names in (("terms", HUMAN_TERMS), ("privacy", HUMAN_LEGAL), ("returnRefund", HUMAN_RETURNS)):
        rows(state.get(key), names, "title")
    for slug, v in ((state.get("globalSettings") or {}).get("categories") or {}).items():
        if isinstance(v, dict) and _is_probe(v.get("name")):
            v["name"] = HUMAN_CATEGORIES.get(slug, "Specials")
    for col in (state.get("globalSettings") or {}).get("footerCustomColumns") or []:
        if isinstance(col, dict):
            if _is_probe(col.get("title")):
                col["title"] = "Shop"
            rows(col.get("links"), ["Flower", "Pre-rolls", "Edibles"], "label")
    return state


def build_state(
    manifest: Manifest,
    *,
    template: str = DEFAULT_TEMPLATE,
    rich: bool = True,
    fixtures_dir: Path | None = None,
    home_sections: list[dict] | None = None,
) -> dict:
    """A basic-template EditorState: `DEFAULT_DEFAULT_HOME_SECTIONS` rows,
    rich settings from the probe fixture for the sampled section types and
    the HeaderSection / pinned:products global, branding and pageVisibility
    slices (its "Zq Fixture …" titles replaced by realistic copy: `humanize`),
    then manifest `default`s for whatever is still unset."""
    if template not in BASIC_TEMPLATE_IDS:
        raise ValueError(f"{template!r} is not a basic template ({', '.join(BASIC_TEMPLATE_IDS)})")
    state = copy.deepcopy(EMPTY_STATE)
    state["template"] = {"id": template}
    probe = load_probe_initials(fixtures_dir) if rich else {}
    ctxf = load_context_fixture(fixtures_dir)

    rows = copy.deepcopy(home_sections if home_sections is not None else DEFAULT_HOME_SECTIONS)
    for row in rows:
        c = manifest.section(row["id"])
        type_id = c.name if c else section_type_id(row["id"])
        settings: dict = {}
        init = (probe.get(f"section:{type_id}") or {}).get("rich") if rich else None
        if init and isinstance(init.get("section"), dict):
            settings.update(copy.deepcopy(init["section"].get("settings") or {}))
        settings.update(row.get("settings") or {})  # the row's own preset wins
        if c:
            for k, v in _defaults_for(c).items():
                settings.setdefault(k, v)
        row["settings"] = settings
        row.setdefault("enabled", True)
    state["homeSections"] = rows

    if rich:
        for key in ("globalPanel:HeaderSection", "pinned:products"):
            init = (probe.get(key) or {}).get("rich") or {}
            for member, slice_name in _INITIAL_SLICES.items():
                if member in init and not state.get(slice_name):
                    state[slice_name] = copy.deepcopy(init[member])
    for c in manifest.containers("global"):
        _fill_defaults(state, c, prefix="")
    _fill_defaults(state, manifest.branding, prefix="")
    for pg in manifest.pages:
        for k, f in manifest.page_fields(pg).items():
            if f.has_default and "[]" not in f.path and get_path(state, f.path, _SENTINEL) is _SENTINEL:
                set_path(state, f.path, copy.deepcopy(f.default))
    state["customPages"] = copy.deepcopy(ctxf.get("customPages") or [])
    # The FAQ categories a basic-template site starts with: PreviewPage.jsx
    # seeds `faqCategories` with getDefaultFaqForTemplate(template) when
    # nothing was saved — `defaultBasicFaqCategories` ({name, items:[{q, a}]},
    # the schema of the manifest's faqCategories[].items) for every basic id.
    # The probe's rows are the premium {name, icon, questions:[…]} schema a
    # basic site never holds, so they never stand in for it (r1-r3 states
    # carried them; docs/LAB.md §2 "Browser defaults").
    faq = browser_defaults.faq_categories(template)
    if faq is not None:
        state["faqCategories"] = faq
    # The products page persists its reorderable pinned placeholder (schema.js PINNED_BY_PAGE).
    state["pageSectionsAll"].setdefault("products", [{"id": PINNED_BY_PAGE["products"]["id"], "enabled": True}])
    # No "Zq Fixture …" placeholder copy: no real site has it (see `humanize`).
    return humanize(state)


# ── context (turnContext.js) ────────────────────────────────────────────────

def _s(v: Any) -> str:
    return "" if v is None else str(v)


def section_title(section: dict) -> str:
    s = section.get("settings") or {}
    for k in ("title", "heading", "headline", "label"):
        if isinstance(s.get(k), str):
            return s[k]
    for k in ("title", "label"):
        if isinstance(section.get(k), str):
            return section[k]
    return ""


def section_summaries(sections: list[dict]) -> list[dict]:
    return [
        {"id": _s(s["id"]), "type": section_type_id(s["id"]), "title": section_title(s), "enabled": s.get("enabled") is not False}
        for s in sections or [] if isinstance(s, dict) and s.get("id") is not None
    ]


# Copied from components/basicEditor/widgets/HeaderNavigationEditor.jsx via
# turnContext.js HEADER_NAV_ITEMS / HEADER_NAV_DEFAULTS (= DefaultNavbar
# BUILTIN_NAV_ITEMS / DEFAULT_HEADER_PAGES).
HEADER_NAV_ITEMS: tuple[tuple[str, str], ...] = (
    ("Shop", "products"), ("Locations", "locations"), ("Delivery", "delivery"), ("Mail Order", "mail-order"),
    ("About Us", "about"), ("FAQ", "faq"), ("Blog", "blog"), ("Contact", "contact"), ("Loyalty", "loyalty"),
    ("Careers", "careers"),
)
HEADER_NAV_DEFAULTS = ("products", "locations", "about", "contact", "faq")

# turnContext.js GLOBAL_PANEL_BY_SECTION: GlobalSettingsDrawer `data-section` id → manifest globalPanels name.
GLOBAL_PANEL_BY_SECTION = {
    "branding": "BrandingSection", "headerMenu": "HeaderSection", "shopMenu": "ShopMenuSection",
    "footerMenu": "FooterSection", "typography": "TypographyColorsSection", "buttons": "ButtonsSection",
    "productCard": "ProductCardSection", "social": "SocialMediaSection", "layout": "LayoutSection",
    "sidebarCart": "SidebarCartSection", "notification": "NotificationSection", "insiderBanner": "InsiderBannerSection",
}

# turnContext.js outline constants
LABEL_KEYS = ("q", "question", "name", "title", "label", "headline")
LABEL_MAX = 60
ROWS_MAX = 20
OUTLINE_MAX = 4096
LADDER = (ROWS_MAX, 10, 5, 3, 2, 1, 0)
_LABEL_TEXT_TYPES = ("text", "richtext")
_VALUE_LABEL_TYPES = ("text", "richtext", "enum", "select")
_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_BLOCK_TAG_RE = re.compile(r"</?(?:p|div|br|li|ul|ol|h[1-6]|tr|td|th|blockquote|section|article|hr)\b[^>]*>", re.I | re.A)
_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
_TAG_OPEN_RE = re.compile(r"<(?=[/!A-Za-z])")
_STRIP_PASSES = 5
_ENTITIES = (("&nbsp;", " "), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'"), ("&#x27;", "'"), ("&amp;", "&"))
_WS_RE = re.compile("[ \t\n\r\f\v\u00a0]+")
_CTRL_RE = re.compile("[\u0000-\u001f\u007f\ud800-\udfff]")
# Non-whitespace controls and lone surrogates, removed FIRST: removed after the tag strip they could
# reassemble a tag (`<\u0001/context>` → `</context>`).
_EARLY_CTRL_RE = re.compile("[\u0000-\u0008\u000e-\u001f\u007f\ud800-\udfff]")
_LIST_SEG_RE = re.compile(r"^([^\[\]]*)((?:\[\])*)$")


def _truthy(v: Any) -> bool:
    """JavaScript truthiness for JSON values (`[]` and `{}` are truthy)."""
    if isinstance(v, (list, dict)):
        return True
    return bool(v)


def nav_summaries(page_visibility: dict | None, custom_pages: list | None = None) -> list[dict]:
    """turnContext.navSummaries: the header links DefaultNavbar renders — built-in pages and published
    custom pages whose `inHeaderNav` is true or, when unset, in HEADER_NAV_DEFAULTS; never a
    `published:false` page; then `_meta.headerNavCustomLinks`; ordered by `_meta.headerNavOrder`."""
    pv = page_visibility if isinstance(page_visibility, dict) else {}
    builtin = {page for _, page in HEADER_NAV_ITEMS}
    items = list(HEADER_NAV_ITEMS)
    for p in custom_pages if isinstance(custom_pages, list) else []:
        if not isinstance(p, dict) or not _truthy(p.get("slug")) or not _truthy(p.get("title")):
            continue
        if p.get("status") == "draft" or p.get("slug") in builtin:
            continue
        items.append((_s(p.get("title")), _s(p.get("slug"))))
    rows = []
    for name, page in items:
        cfg = pv.get(page) if isinstance(pv.get(page), dict) else {}
        if cfg.get("published", True) is False:
            continue
        shown = _truthy(cfg["inHeaderNav"]) if "inHeaderNav" in cfg else page in HEADER_NAV_DEFAULTS
        if not shown:
            continue
        custom = cfg.get("headerNavName")
        label = custom if isinstance(custom, str) and custom.strip() else name
        rows.append({"key": page, "label": label, "href": f"/{page}"})
    meta = pv.get("_meta") if isinstance(pv.get("_meta"), dict) else {}
    links = meta.get("headerNavCustomLinks")
    for link in links if isinstance(links, list) else []:
        if not isinstance(link, dict):
            continue
        rows.append({"key": link.get("id"), "label": _s(link.get("label")) or "Link", "href": _s(link.get("url")) or "/"})
    order = meta.get("headerNavOrder") if isinstance(meta.get("headerNavOrder"), list) else []
    if order:
        picked = [r for r in (next((r for r in rows if r["key"] == k), None) for k in order) if r is not None]
        rows = picked + [r for r in rows if r["key"] not in order]
    return [{"label": r["label"], "href": r["href"]} for r in rows]


def focus_summary(view: dict | None) -> dict:
    """turnContext.focusSummary: `area` (Pantry tab; `settings` → `global`), the open section editor, and the
    open Global Settings panel as a manifest panel name (global area only)."""
    v = view if isinstance(view, dict) else {}
    tab = _s(v.get("tab"))
    area = "global" if tab == "settings" else (tab or "editor")
    gs = _s(v.get("globalSection"))
    return {
        "area": area,
        "sectionId": _s(v.get("activeSectionId")),
        "panel": (GLOBAL_PANEL_BY_SECTION.get(gs) or gs) if area == "global" and gs else "",
    }


def _js_number(v: Any) -> float:
    """JavaScript `Number(v)` for JSON values (NaN when not numeric)."""
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    if v is None:
        return 0.0
    if isinstance(v, str):
        t = v.strip()
        if t == "":
            return 0.0
        try:
            return float(t)
        except ValueError:
            return float("nan")
    return float("nan")


def _num_out(n: float) -> int | float:
    # JS prints whole numbers from 1e21 up in exponent form (`1e+21`), as Python's float repr does.
    return int(n) if float(n).is_integer() and abs(n) < 1e21 else n


def _js_num_str(n: float) -> str:
    """String(n) for a finite JS number: shortest round-trip digits, positional for 1e-6 <= |n| < 1e21,
    else `d.ddde±x` (no zero-padded exponent)."""
    if n == 0:
        return "0"
    if float(n).is_integer() and abs(n) < 1e21:
        return str(int(n))
    r = repr(float(n))
    if 1e-6 <= abs(n) < 1e21:
        return format(Decimal(r), "f")
    # Outside that range Python's repr is already in exponent form (`1e+21`, `1.5e-07`): drop exponent zero-padding.
    return re.sub(r"e([+-])0*(\d)", r"e\1\2", r)


def shop_menu_summary(global_settings: dict | None) -> dict:
    """turnContext.shopMenuSummary."""
    g = global_settings if isinstance(global_settings, dict) else {}
    mx = _js_number(g.get("shopMenuMaxItems")) if "shopMenuMaxItems" in g else float("nan")
    items = g.get("shopMenuItems") if isinstance(g.get("shopMenuItems"), list) else []
    return {
        "style": _s(g.get("shopMenuStyle")) or "classic",
        "maxItems": _num_out(mx) if math.isfinite(mx) and mx > 0 else 0,
        "hidden": sum(1 for it in items if isinstance(it, dict) and it.get("visible") is False),
        "inStockOnly": g.get("shopMenuShowOnlyInStockCategories") is True,
    }


def plain_label(value: Any, max_len: int = LABEL_MAX) -> str:
    """turnContext.plainLabel: non-whitespace control chars and lone surrogates removed first, then entities
    decoded once (`&amp;` last), then comments, block tags (→ space) and
    tag-like spans removed until stable (at most 5 passes), then any `<` still opening a tag (`<` + `/`, `!`
    or a letter) deleted; whitespace collapsed, control chars / lone surrogates removed, clipped to `max_len`
    code points with `…`. The result never carries markup; `a < b and c > d` survives."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        value = _js_num_str(float(value)) if isinstance(value, float) else str(value)
    if not isinstance(value, str):
        return ""
    s = _EARLY_CTRL_RE.sub("", value)
    for frm, to in _ENTITIES:
        s = s.replace(frm, to)
    for _ in range(_STRIP_PASSES):
        nxt = _TAG_RE.sub("", _BLOCK_TAG_RE.sub(" ", _COMMENT_RE.sub("", s)))
        if nxt == s:
            break
        s = nxt
    s = _CTRL_RE.sub("", _WS_RE.sub(" ", _TAG_OPEN_RE.sub("", s))).strip(" ")
    return s[: max_len - 1].rstrip(" ") + "…" if len(s) > max_len else s


def json_size(v: Any) -> int:
    """turnContext.jsonSize: code points of the compact JSON."""
    return len(json.dumps(v, separators=(",", ":"), ensure_ascii=False))


def _item_fields_of(spec: dict, doc: dict) -> dict:
    out = dict(spec["item"]) if isinstance(spec.get("item"), dict) else {}
    ref = re.sub(r"^details\.", "", spec["item_ref"]) if isinstance(spec.get("item_ref"), str) else ""
    details = doc.get("details") if isinstance(doc.get("details"), dict) else {}
    panel = details.get(ref) if ref and isinstance(details.get(ref), dict) else None
    if panel and isinstance(panel.get("fields"), dict):
        out.update(panel["fields"])
    return out


def _label_rule(fields: dict) -> tuple[str, str | None] | None:
    for k in LABEL_KEYS:
        if isinstance(fields.get(k), dict) and fields[k].get("type") in _LABEL_TEXT_TYPES:
            return ("key", k)
    if isinstance(fields.get("value"), dict) and fields["value"].get("type") in _VALUE_LABEL_TYPES:
        return ("value", None)
    return None


def _segments_of(path: str, indexes: list[int], absolute: bool) -> list[str | int]:
    idx = list(indexes)
    out: list[str | int] = []
    for n, part in enumerate(str(path).split(".")):
        m = _LIST_SEG_RE.match(part)
        name = m.group(1) if m else part
        if name != "":
            out.append(PATH_ROOT_TO_SLICE.get(name, name) if absolute and n == 0 else name)
        for _ in range(len(m.group(2)) // 2 if m else 0):
            out.append(idx.pop(0) if idx else 0)
    return out


def _read_at(root: Any, segs: list[str | int]) -> Any:
    cur = root
    for s in segs:
        if isinstance(cur, dict):
            cur = cur.get(s if isinstance(s, str) else str(s))
        elif isinstance(cur, list) and isinstance(s, int):
            cur = cur[s] if 0 <= s < len(cur) else None
        else:
            return None
    return cur


def _parent_tuples(root: Any, path: str, absolute: bool, flags: dict) -> list[list[int]]:
    """Every parent index tuple of a nested list path, capped at ROWS_MAX per level (a cap that hid parents
    sets `flags["capped"]`)."""
    cut = path.split("[]")
    tuples: list[list[int]] = [[]]
    for level in range(len(cut) - 1):
        parent_path = "[]".join(cut[: level + 1])
        nxt = []
        for t in tuples:
            arr = _read_at(root, _segments_of(parent_path, t, absolute))
            if isinstance(arr, list):
                if len(arr) > ROWS_MAX:
                    flags["capped"] = True
                nxt.extend(t + [i] for i in range(min(len(arr), ROWS_MAX)))
        tuples = nxt
    return tuples


def _list_entries(lists: Any, doc: dict, state: dict, section: dict | None, absolute_of, seen: set,
                  flags: dict) -> list[dict]:
    out = []
    for key, spec in (lists.items() if isinstance(lists, dict) else []):
        if not isinstance(spec, dict) or not isinstance(spec.get("path"), str) or not spec["path"]:
            continue
        absolute = absolute_of(spec["path"])
        base = state if absolute else section
        rule = _label_rule(_item_fields_of(spec, doc))
        for at in _parent_tuples(base, spec["path"], absolute, flags):
            sig = f"{spec['path']}@{','.join(str(i) for i in at)}" if absolute else None
            if sig and sig in seen:
                continue
            arr = _read_at(base, _segments_of(spec["path"], at, absolute))
            if not isinstance(arr, list) or not arr:
                continue
            if sig:
                seen.add(sig)
            labels = None
            if rule:
                labels = []
                for i, row in enumerate(arr):
                    v = (row.get(rule[1]) if isinstance(row, dict) else None) if rule[0] == "key" else row
                    lab = plain_label(v)
                    labels.append(f"[{i}] {lab}" if lab else f"[{i}]")
            out.append({"key": key, "at": at, "count": len(arr), "labels": labels})
    return out


def _page_list_owner(state: dict, doc: dict, page: str, owner: dict, seen: set, flags: dict) -> dict:
    pages = doc.get("pages") if isinstance(doc.get("pages"), dict) else {}
    page_spec = pages.get(page) if isinstance(pages.get(page), dict) else None
    lists = _list_entries(page_spec.get("lists"), doc, state, None, lambda p: True, seen, flags) if page_spec else []
    return {"owner": owner, "lists": lists}


def _section_owners(state: dict, doc: dict, sections: list, page: str, seen: set, flags: dict) -> list[dict]:
    """Every section row with a style or non-empty lists; `page` set → entries carry it (another page's)."""
    types = doc.get("sections") if isinstance(doc.get("sections"), dict) else {}
    known = {k for k, v in types.items() if _truthy(v)}
    out = []
    for row in sections if isinstance(sections, list) else []:
        if not isinstance(row, dict) or row.get("id") is None:
            continue
        sid = _s(row["id"])
        typ = lookup_section_type_id(sid, known)
        settings = row.get("settings") if isinstance(row.get("settings"), dict) else {}
        style = plain_label(settings["style"]) if isinstance(settings.get("style"), str) else ""
        lists = (_list_entries(types[typ].get("lists"), doc, state, row, lambda p: not p.startswith("settings."), seen, flags)
                 if typ and isinstance(types.get(typ), dict) else [])
        if not style and not lists:
            continue
        owner: dict = {"section": sid}
        if page:
            owner["page"] = page
        if style:
            owner["style"] = style
        out.append({"owner": owner, "lists": lists})
    return out


def _outline_owners(state: dict, doc: dict, page: str, sections: list, focus_id: str, seen: set,
                    flags: dict) -> list[dict]:
    page_owner = _page_list_owner(state, doc, page, {"page": page}, seen, flags)
    owners = [page_owner] if page_owner["lists"] else []
    sec_owners = _section_owners(state, doc, sections, "", seen, flags)
    fi = next((i for i, o in enumerate(sec_owners) if o["owner"]["section"] == focus_id), -1) if focus_id else -1
    if fi >= 0:
        return [sec_owners[fi], *owners, *(o for i, o in enumerate(sec_owners) if i != fi)]
    return owners + sec_owners


def _other_page_owners(state: dict, doc: dict, page: str, sections: list, seen: set, flags: dict) -> list[dict]:
    """A named other page: header `{page, other:true, sections, lists?}` then `{section, page, style?, lists?}`."""
    summaries = [{**row, "title": plain_label(row["title"])} for row in section_summaries(sections)]
    header = _page_list_owner(state, doc, page, {"page": page, "other": True, "sections": summaries}, seen, flags)
    return [header, *_section_owners(state, doc, sections, page, seen, flags)]


def _render_owners(owners: list[dict], rows_max: int) -> list[dict]:
    out = []
    for o in owners:
        d = dict(o["owner"])
        if o["lists"]:
            ls = []
            for e in o["lists"]:
                lst: dict = {"key": e["key"]}
                if e["at"]:
                    lst["at"] = e["at"]
                lst["count"] = e["count"]
                if e["labels"] is not None and rows_max > 0:
                    rows = e["labels"][:rows_max]
                    if len(e["labels"]) > rows_max:
                        rows = rows + [f"+{len(e['labels']) - rows_max} more"]
                    lst["rows"] = rows
                ls.append(lst)
            d["lists"] = ls
        out.append(d)
    return out


def build_outline(state: dict, manifest: Any, page: str, sections: list, focus_id: str = "",
                  max_size: int = OUTLINE_MAX, others: list[dict] | None = None) -> tuple[list[dict], bool]:
    """turnContext.buildOutline → (outline, truncated); ([], False) without a manifest. `others` =
    [{page, sections}] for the pages the request names, appended after the viewed page within the one
    budget (the viewed page is never shrunk for them)."""
    doc = manifest.doc if isinstance(manifest, Manifest) else manifest
    if not isinstance(doc, dict):
        return [], False
    st = state if isinstance(state, dict) else {}
    seen: set = set()
    flags = {"capped": False}
    owners = _outline_owners(st, doc, page, sections, focus_id, seen, flags)
    truncated = False
    outline = None
    for n in LADDER:
        r = _render_owners(owners, n)
        if json_size(r) <= max_size:
            outline = r
            truncated = truncated or n < ROWS_MAX
            break
    if outline is None:
        outline = _render_owners(owners, 0)
        while outline and json_size(outline) > max_size:
            outline = outline[:-1]
        truncated = True
    for other in others or []:
        block = _other_page_owners(st, doc, _s(other.get("page")), other.get("sections"), seen, flags)
        fit = None
        for n in LADDER:
            r = _render_owners(block, n)
            if json_size(outline + r) <= max_size:
                fit = r
                truncated = truncated or n < ROWS_MAX
                break
        if fit is None:
            truncated = True
            r = _render_owners(block, 0)
            while len(r) > 1 and json_size(outline + r) > max_size:
                r = r[:-1]
            if json_size(outline + r) > max_size:
                continue
            fit = r
        outline = outline + fit
    return outline, truncated or flags["capped"]


# ── named pages (pageNames.js) ──────────────────────────────────────────────

# pageNames.js BUILTIN_PAGE_NAMES: slug → names, the first is the display label (PreviewPage buildPageOptions).
BUILTIN_PAGE_NAMES: dict[str, tuple[str, ...]] = {
    "home": ("Home", "Home page", "Homepage"), "products": ("Products", "Shop"),
    "product-details": ("Product Details", "Product page"), "cart": ("Cart",), "checkout": ("Checkout",),
    "order-confirmation": ("Order Confirmation",), "locations": ("Locations",), "faq": ("FAQ", "FAQs"),
    "about": ("About Us", "About page"), "blog": ("Blog",), "blog-detail": ("Blog Post",), "delivery": ("Delivery",),
    "mail-order": ("Mail Order",), "contact": ("Contact Us", "Contact"), "careers": ("Careers",),
    "loyalty": ("Loyalty Program", "Loyalty"), "privacy-policy": ("Privacy Policy",),
    "terms-of-service": ("Terms of Service", "Terms"),
    "return-refund": ("Refund & Return Policy", "Return Policy", "Refund Policy"), "login": ("Login",),
    "account": ("Account",), "order-details": ("Track Order",), "location-selection": ("Location Selection",),
    "age-gate": ("Age Gate",),
}
NAMED_PAGES_MAX = 3
NAME_MIN = 3
GENERIC_NAMES = (" about ",)
_SKIPPED_SLUGS = ("location-details",)
_NAME_FOLD_RE = re.compile(r"[\W_]+")
_NAV_HREF_RE = re.compile(r"/([^/?#]+)")


def norm_name(s: Any) -> str:
    """pageNames.normName: `' ' + lowercase words joined by single spaces + ' '`."""
    return " " + _NAME_FOLD_RE.sub(" ", _s(s).lower()).strip() + " "


def _custom_title(p: dict) -> str:
    return _s(p.get("title") if _truthy(p.get("title")) else p.get("name"))


def page_label(slug: Any, custom_pages: list | None = None) -> str:
    """pageNames.pageLabel: a custom page's title, else the built-in label, else the humanised slug."""
    s = _s(slug)
    for p in custom_pages if isinstance(custom_pages, list) else []:
        if isinstance(p, dict) and _s(p.get("slug")) == s and _custom_title(p).strip():
            return _custom_title(p).strip()
    if s in BUILTIN_PAGE_NAMES:
        return BUILTIN_PAGE_NAMES[s][0]
    words = re.sub(r"[-_]+", " ", s).strip()
    return words[:1].upper() + words[1:]


def _name_candidates(custom_pages: list | None, nav: list | None) -> list[tuple[str, str]]:
    out = []
    for slug, names in BUILTIN_PAGE_NAMES.items():
        out.extend((slug, n) for n in names)
        out.append((slug, slug))
    custom = [p for p in (custom_pages if isinstance(custom_pages, list) else []) if isinstance(p, dict) and _s(p.get("slug"))]
    for p in custom:
        out.append((_s(p.get("slug")), _custom_title(p)))
        out.append((_s(p.get("slug")), _s(p.get("slug"))))
    known = set(BUILTIN_PAGE_NAMES) | {_s(p.get("slug")) for p in custom}
    for row in nav if isinstance(nav, list) else []:
        if not isinstance(row, dict):
            continue
        m = _NAV_HREF_RE.fullmatch(_s(row.get("href")))
        if m and m.group(1) in known:
            out.append((m.group(1), _s(row.get("label"))))
    return [(slug, name) for slug, name in out
            if slug not in _SKIPPED_SLUGS and len(norm_name(name)) >= NAME_MIN + 2 and norm_name(name) not in GENERIC_NAMES]


def named_pages(text: Any, custom_pages: list | None = None, nav: list | None = None, exclude: str = "") -> list[str]:
    """pageNames.namedPages: slugs of the pages `text` names (built-in names and slugs, custom page titles and
    slugs, header nav labels; whole words, case-insensitive), earliest mention first, `exclude` dropped, at
    most NAMED_PAGES_MAX."""
    if not isinstance(text, str) or not text:
        return []
    hay = norm_name(text)
    best: dict[str, tuple[int, int]] = {}
    for order, (slug, name) in enumerate(_name_candidates(custom_pages, nav)):
        at = hay.find(norm_name(name))
        if at < 0:
            continue
        if slug not in best or at < best[slug][0]:
            best[slug] = (at, order)
    ranked = sorted((v, slug) for slug, v in best.items() if slug != _s(exclude))
    return [slug for _, slug in ranked[:NAMED_PAGES_MAX]]


def _page_rows(state: dict, page: str, template: Any = None) -> list:
    """The page's section rows as targets.resolveSectionSlot gives them to
    the browser's turn context: home = `homeSections`; another page = its
    saved rows when there are any, else the template's defaults
    (`getDefaultPageSections(template, page) || []`, browser_defaults) —
    never a pinned row the browser does not list."""
    rows = sections_of(state, page)
    if page == "home" or (isinstance(rows, list) and rows):
        return rows if isinstance(rows, list) else []
    return browser_defaults.page_sections(template, page)


# ── option lists (optionLists.js, carouselOptions.js, previewApi.buildCtx) ──

OPTION_LIST_MAX = 100
OPTION_KINDS = ("tags", "brands")
OPTION_REF_KINDS = ("tag", "brand")
OPTION_SECTION_FALLBACK = ("productCarousel",)
# optionLists.OPTION_TEXT_RE: tag(s) / brand(s) / carousel(s), whole words, any case — except "tag line(s)" and the
# merchant's own brand ("brand colour / font / logo …"). JavaScript semantics: `\b` and the case folding are ASCII
# (re.ASCII; `/i` without `u` never folds a non-ASCII character onto an ASCII one), `\s` is JavaScript's set.
_JSWS = "\t\n\v\f\r \u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
OPTION_TEXT_RE = re.compile(
    "\\b(?:tags?(?![" + _JSWS + "]+lines?\\b)|brands?(?![" + _JSWS + "-]+(?:colou?rs?|fonts?|logos?|identity|kit|guidelines?"
    "|palette|styles?|voice|assets?|story|name|images?)\\b)|carousels?)\\b", re.IGNORECASE | re.ASCII)


def js_str(v: Any) -> str:
    """JavaScript ``String(v)`` for a JSON value (``null`` → ``"null"``; callers handle null themselves)."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        return _js_num_str(v)
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return ",".join("" if x is None else js_str(x) for x in v)
    return "[object Object]"


def _jstr(v: Any) -> str:
    """turnContext `str`: ``v == null ? '' : String(v)``."""
    return "" if v is None else js_str(v)


def carousel_tag_options(raw: Any) -> list:
    """carouselOptions.carouselTagOptions: visible tags only (``visible: false`` is not selectable)."""
    return [t for t in (raw if isinstance(raw, list) else []) if _truthy(t) and not (isinstance(t, dict) and t.get("visible") is False)]


def carousel_brand_options(raw: Any) -> list:
    """carouselOptions.carouselBrandOptions: named brands only."""
    return [b for b in (raw if isinstance(raw, list) else []) if _truthy(b) and isinstance(b, dict) and _truthy(b.get("name"))]


_OPTION_FILTERS = {"tags": carousel_tag_options, "brands": carousel_brand_options}


def _has_slug(x: dict) -> bool:
    return x.get("slug") is not None and x.get("slug") != ""


def id_slug_list(raw: Any) -> list[dict]:
    """previewApi idSlugList over an already filtered list: ``[{id, slug}]`` for every object with a slug (an
    absent id stays absent, as JSON.stringify drops ``undefined``)."""
    out = []
    for x in raw if isinstance(raw, list) else []:
        if isinstance(x, dict) and _has_slug(x):
            out.append({**({"id": x["id"]} if "id" in x else {}), "slug": x["slug"]})
    return out


def option_lists_of(case: dict | None, fixture: dict | None = None) -> dict[str, Any]:
    """The raw carousel tag / brand lists a case's host holds: the case's ``option_lists`` (``{"tags": [...] |
    null, "brands": [...] | null}``; a missing or null kind = the list is unknown — not loaded, or its fetch
    failed) when it has one, else the lab fixture context's (``fixtures/context.json``)."""
    over = (case or {}).get("option_lists")
    if isinstance(over, dict):
        return {k: over.get(k) if isinstance(over.get(k), list) else None for k in OPTION_KINDS}
    fx = fixture if fixture is not None else load_context_fixture()
    return {k: fx.get(k) if isinstance(fx.get(k), list) else None for k in OPTION_KINDS}


def host_lists(case: dict | None, fixture: dict | None = None) -> dict[str, list[dict]]:
    """``{tags, brands}`` of the engine ctx (previewApi.buildCtx ``idSlugList(carouselTagOptions(v.tags))``…): an
    unknown list is ``[]``, as the browser host sends it."""
    raw = option_lists_of(case, fixture)
    return {k: id_slug_list(_OPTION_FILTERS[k](raw[k])) for k in OPTION_KINDS}


def option_ref_section_types(doc: Any) -> set[str]:
    """optionLists.optionRefSectionTypes: section types with a writable ``ref: tag|brand`` field, else the
    fallback (``productCarousel``)."""
    out: set[str] = set()
    sections = doc.get("sections") if isinstance(doc, dict) and isinstance(doc.get("sections"), dict) else {}
    for typ, spec in sections.items():
        fields = spec.get("fields") if isinstance(spec, dict) and isinstance(spec.get("fields"), dict) else {}
        if any(isinstance(f, dict) and f.get("type") == "ref" and f.get("ref") in OPTION_REF_KINDS
               and not _truthy(f.get("derived_from")) for f in fields.values()):
            out.add(typ)
    return out or set(OPTION_SECTION_FALLBACK)


def wants_option_lists(text: Any, types: list, doc: Any) -> bool:
    """optionLists.wantsOptionLists: the request names tags / brands / carousels, or one of ``types`` (the viewed
    page's, the named pages' and the focused section's types) carries a tag / brand ref."""
    if isinstance(text, str) and OPTION_TEXT_RE.search(text):
        return True
    ref_types = option_ref_section_types(doc)
    for t in types:
        s = "" if t is None else _jstr(t)
        if s and any(s == r or s.startswith(f"{r}-") for r in ref_types):
            return True
    return False


def _utf16(s: str) -> bytes:
    """A sort key in JavaScript's string order (UTF-16 code units)."""
    return s.encode("utf-16-be", "surrogatepass")


def option_list(kind: str, raw: Any, max_len: int = OPTION_LIST_MAX) -> tuple[list[dict], int]:
    """optionLists.optionList: the dropdown's options (visible tags / named brands) with a slug, ``name`` =
    plainLabel(name, else slug) or the slug, sorted by lower-cased name then slug (JavaScript ``<``), cut to
    ``max_len``; with the count before the cut."""
    rows = []
    for x in _OPTION_FILTERS[kind](raw):
        if not isinstance(x, dict) or not _has_slug(x):
            continue
        slug = js_str(x["slug"])
        name = x.get("name")
        name = plain_label(slug if name is None or name == "" else name) or slug
        rows.append({"slug": slug, "name": name})
    rows.sort(key=lambda r: (_utf16(r["name"].lower()), _utf16(r["slug"])))
    return rows[:max(0, max_len)], len(rows)


def _add_option_lists(out: dict, ex: dict, text: Any, types: list, doc: Any) -> None:
    """turnContext.addOptionLists: ``tags`` / ``brands`` (each with ``<kind>Truncated: {shown, total}`` when cut)
    when the host knows the list (an array; absent → left out, ``[]`` → sent empty) and the turn is about them."""
    kinds = [k for k in OPTION_KINDS if isinstance(ex.get(k), list)]
    if not kinds or not wants_option_lists(text, types, doc):
        return
    for kind in kinds:
        rows, total = option_list(kind, ex[kind])
        out[kind] = rows
        if total > len(rows):
            out[f"{kind}Truncated"] = {"shown": len(rows), "total": total}


def _category_row(c: dict) -> dict:
    """turnContext categories entry: ``{slug: str(slug || id), name: str(name || title || label), id?}`` — the id
    as text, only when it is neither null nor ``''``."""
    slug = c.get("slug") if _truthy(c.get("slug")) else c.get("id")
    name = next((c.get(k) for k in ("name", "title") if _truthy(c.get(k))), c.get("label"))
    row = {"slug": _jstr(slug), "name": _jstr(name)}
    if c.get("id") is not None and c.get("id") != "":
        row["id"] = _jstr(c["id"])
    return row


def context_of(state: dict, page: str = "home", extras: dict | None = None, manifest: Any = None,
               view: dict | None = None, text: str | None = None) -> dict:
    """The `context` member of POST /v1/turns (README §1.2), as
    `turnContext.buildTurnContext` derives it from the EditorState. `extras`
    supplies `categories` and `locations` (the fixture `context.json`);
    `manifest` (a Manifest or its doc) feeds the list outline; `view` is the
    host ctx focus (`tab`, `activeSectionId`, `globalSection`) — none = the
    editor tab with nothing open; `text` (the merchant's request) adds the
    pages it names to the outline (`named_pages`). A never-saved non-home
    page lists its template's default rows, as the browser does
    (`_page_rows`; the template is the host ctx's `templateId` when `view`
    carries one, else the state's)."""
    ex = extras or {}
    tpl = (view or {}).get("templateId") if isinstance(view, dict) and (view or {}).get("templateId") else (state.get("template") or {}).get("id")
    sections = _page_rows(state, page, tpl)
    layouts = state.get("layouts") if isinstance(state.get("layouts"), dict) else {}
    custom_pages = state.get("customPages") if isinstance(state.get("customPages"), list) else []
    focus = focus_summary(view)
    nav = nav_summaries(state.get("pageVisibility"), custom_pages)
    doc = manifest.doc if isinstance(manifest, Manifest) else manifest
    others = ([{"page": slug, "sections": _page_rows(state, slug, tpl)}
               for slug in named_pages(text, custom_pages, nav, _s(page))]
              if isinstance(doc, dict) and isinstance(text, str) and text else [])
    outline, truncated = build_outline(state, manifest, _s(page), sections, focus["sectionId"], OUTLINE_MAX, others)
    out = {
        "page": _s(page),
        "template": _s((state.get("template") or {}).get("id")),
        "layout": _s(layouts.get(page) or ""),
        "sections": section_summaries(sections),
        "categories": [_category_row(c) for c in ex.get("categories") or [] if isinstance(c, dict)],
        "customPages": [{"slug": _s(p.get("slug")), "title": _s(p.get("title") or p.get("name"))}
                        for p in custom_pages if isinstance(p, dict)],
        "nav": nav,
        "locations": [{"id": _s(l.get("id")), "name": _s(l.get("name") or l.get("title"))}
                      for l in ex.get("locations") or [] if isinstance(l, dict) and l.get("id") is not None],
        "focus": focus,
        "shopMenu": shop_menu_summary(state.get("globalSettings")),
        "outline": outline,
    }
    if truncated:
        out["outlineTruncated"] = True
    known = set(doc.get("sections") or {}) if isinstance(doc, dict) else set()
    ids = [r.get("id") if isinstance(r, dict) else None for r in sections]
    ids += [r.get("id") if isinstance(r, dict) else None for o in others for r in o.get("sections") or []]
    ids.append(focus["sectionId"])
    types = [lookup_section_type_id(_jstr(i), known) or section_type_id(_jstr(i)) for i in ids if _truthy(i)]
    _add_option_lists(out, ex, text, types, doc)
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    from .manifest import load
    ap = argparse.ArgumentParser(prog="souschef_lab.state", description="print a built EditorState or its turn context")
    ap.add_argument("--manifest", default=str(FIXTURES_DIR / "manifest.snapshot.json"))
    ap.add_argument("--template", default=DEFAULT_TEMPLATE)
    ap.add_argument("--plain", action="store_true", help="manifest defaults only, no probe seed")
    ap.add_argument("--context", metavar="PAGE", help="print context_of(state, PAGE) instead of the state")
    a = ap.parse_args(argv)
    m = load(a.manifest)
    st = build_state(m, template=a.template, rich=not a.plain)
    if a.context:
        print(json.dumps(context_of(st, a.context, load_context_fixture()), indent=1))
    else:
        print(json.dumps(st, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
