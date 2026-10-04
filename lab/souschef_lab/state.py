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
`customPages`, `nav` (from pageVisibility), `locations`.

Path helpers resolve a manifest field to its state path for a section row,
a global panel, branding or a page, and `gate_writes` computes the values
that open a field's `conditional_on` / `hidden_when` gate (docs/LAB.md §2).

Standard library only.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from . import FIXTURES_DIR
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
    slices, then manifest `default`s for whatever is still unset."""
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
    # The products page persists its reorderable pinned placeholder (schema.js PINNED_BY_PAGE).
    state["pageSectionsAll"].setdefault("products", [{"id": PINNED_BY_PAGE["products"]["id"], "enabled": True}])
    return state


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


def nav_summaries(page_visibility: dict | None) -> list[dict]:
    pv = page_visibility if isinstance(page_visibility, dict) else {}
    out = []
    for page, cfg in pv.items():
        if page == "_meta" or not isinstance(cfg, dict):
            continue
        if not cfg.get("inHeaderNav") or cfg.get("published") is False:
            continue
        out.append({"label": _s(cfg.get("headerNavName")) or page, "href": "/" if page == "home" else f"/{page}"})
    custom = (pv.get("_meta") or {}).get("headerNavCustomLinks") if isinstance(pv.get("_meta"), dict) else None
    if isinstance(custom, list):
        for link in custom:
            if not isinstance(link, dict):
                continue
            href = _s(link.get("url") or link.get("href"))
            if not href:
                continue
            out.append({"label": _s(link.get("label") or link.get("name")) or href, "href": href})
    return out


def context_of(state: dict, page: str = "home", extras: dict | None = None) -> dict:
    """The `context` member of POST /v1/turns (README §1.2), as
    `turnContext.buildTurnContext` derives it from the EditorState. `extras`
    supplies `categories` and `locations` (the fixture `context.json`)."""
    ex = extras or {}
    sections = sections_of(state, page)
    if page != "home" and not sections and page in PINNED_BY_PAGE:
        sections = [{"id": PINNED_BY_PAGE[page]["id"], "enabled": True}]
    layouts = state.get("layouts") if isinstance(state.get("layouts"), dict) else {}
    return {
        "page": _s(page),
        "template": _s((state.get("template") or {}).get("id")),
        "layout": _s(layouts.get(page) or ""),
        "sections": section_summaries(sections),
        "categories": [{"slug": _s(c.get("slug") or c.get("id")), "name": _s(c.get("name") or c.get("title") or c.get("label"))}
                       for c in ex.get("categories") or [] if isinstance(c, dict)],
        "customPages": [{"slug": _s(p.get("slug")), "title": _s(p.get("title") or p.get("name"))}
                        for p in state.get("customPages") or [] if isinstance(p, dict)],
        "nav": nav_summaries(state.get("pageVisibility")),
        "locations": [{"id": _s(l.get("id")), "name": _s(l.get("name") or l.get("title"))}
                      for l in ex.get("locations") or [] if isinstance(l, dict) and l.get("id") is not None],
    }


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
