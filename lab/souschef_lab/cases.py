"""Case generator (docs/LAB.md §2).

One JSONL row per case:

    {id, kind, target, fields:[key…], site_state:<EditorState>, context:<turn context>,
     request:"<NL text>", expected:{ops:[…], diff:[{path,before,after}]},
     holdout:bool, manifest_hash, template:"editorial", meta:{…}}

Kinds: set_field set_enum set_range toggle image list_add list_remove
list_reorder list_update reset layout multi clarify unsupported.

Everything is deterministic: a case's wording and values come from
`random.Random(f"{seed}:{case_id}")`, and `holdout` is
`sha1("<container>.<field or list>") % 100 < 20`, so paraphrases of one
target never straddle train and exam (thermometer's question-hash idea).

`expected.diff` is MEASURED: `measure(cases, engine)` replays
`expected.ops` through the browser engine (`engine.py`) on the case's
`site_state` and stores the leaf diff; an expected op the engine refuses is
a generator bug and the case goes to the rejects file instead.
`generate()` alone needs no node.

Type → op table (docs/LAB.md §2 "Generation rules"):

  text/richtext  set_fields, richtext wrapped in <p>
  bool           toggle: value = not current (current seeded in site_state)
  enum/select    set_enum: a member ≠ current; the request uses enum_labels
  range/number   set_range: min + k*step within [min,max], ≠ current; 1–100 without bounds
  color          set_field #rrggbb
  image          values[key] = asset.public_url and assets[key] = {alt, srcset};
                 second variant: pick_asset then set_fields
  ref            a slug / id from the context fixture
  keyed_by       `{locationId}` substituted with a fixture location id
  derived_from   excluded (E_READONLY); named only in `unsupported` asks
  json           excluded (no portable value to generate; counted in stats)

Hero `mirror.keys` are addressed as `edit_list update slides[0]` (engine
README §2 rule 6). Gated fields (`conditional_on` / `hidden_when`) have the
gate opened in `site_state` (meta.gate_writes); every 20th gated field also
gets a closed-gate variant (meta.gate_closed = true, meta.expect_warning =
"W_HIDDEN"). NOTE (measured 2026-10-01, validate.js:54 `isGated`): the
engine emits W_HIDDEN for every gated field whether or not the gate is open
in the state — the warning says "this field has a gate", not "the gate is
closed". The verifier/dataset must not treat it as a failure for open-gate
cases (docs/LAB.md §5 `warnings` keep rule needs that nuance).
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field as dc_field
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import FIXTURES_DIR
from .manifest import Container, Field, ListSpec, Manifest, WritableTarget, enumerate_targets, is_holdout, load, section_type_id

# schema.js KEYED_PREFIXES: a dotted key `<prefix>.<id>.<rest>` is validated against the named
# details panel's `item.<rest>` field (targets.keyedField) — NOT against the container's own
# `<prefix>.{locationId}.<rest>` manifest fields. Measured 2026-10-01: contentBlock
# `locationOverrides.loc-main.mode` / `.lat` → E_FIELD; only `address` exists on
# LocationDetailsPanel. Two-segment keys (`footerLocationEnabled.<id>`) never resolve.
KEYED_PREFIXES = {"section": {"locationOverrides": "LocationDetailsPanel"},
                  "global": {"categories": "CategoryDetailsPanel", "footerLocationOverrides": "LocationDetailsPanel"}}


def keyed_supported(manifest: Manifest, container: Container, key: str) -> bool:
    """Whether the engine will resolve a concrete keyed key (`locationOverrides.loc-main.address`)."""
    parts = key.split(".")
    if len(parts) < 3:
        return False
    panel_name = KEYED_PREFIXES.get(container.kind, {}).get(parts[0])
    panel = manifest.detail(panel_name) if panel_name else None
    return bool(panel and ".".join(parts[2:]) in panel.fields)  # panel field keys carry no `item.` prefix
from .state import (
    PINNED_BY_PAGE, slice_path, build_state, context_of, find_section, find_section_index, gate_writes, apply_gate_writes,
    fixture_asset_for, get_path, load_assets_fixture, load_context_fixture, load_layouts_fixture, read_field, resolve_key, sections_of,
    sections_of_type, sections_slice_path, set_path, type_of_section, write_values, section_title,
)

KINDS = ("set_field", "set_enum", "set_range", "toggle", "image", "list_add", "list_remove", "list_reorder",
         "list_update", "reset", "layout", "multi", "clarify", "unsupported")
SET_KINDS = ("set_field", "set_enum", "set_range", "toggle", "image")
SCHEMA_VERSION = 1
GATE_CLOSED_EVERY = 20
EXCLUDED_TYPES = ("json",)

SECTION_LABELS = {
    "heroBanner": "hero banner", "hero": "hero", "featuredCategories": "featured categories",
    "productCarousel": "product carousel", "testimonials": "customer reviews", "contentBlock": "content block",
    "gallery": "gallery", "blogSection": "blog section", "newsletter": "newsletter", "promotion": "promotion tiles",
    "sectionDivider": "section divider", "loyaltyForm": "loyalty form", "locations": "locations section",
    "content": "content section", "reviews": "reviews", "perksCards": "perks cards", "perksIcons": "perks icons",
    "milestones": "milestones", "blog": "blog",
}
PANEL_LABELS = {
    "BrandingSection": "the branding settings", "HeaderSection": "the header", "ShopMenuSection": "the shop menu",
    "FooterSection": "the footer", "TypographyColorsSection": "the typography and colours", "ButtonsSection": "the buttons",
    "ProductCardSection": "the product cards", "SocialMediaSection": "the social media links", "LayoutSection": "the site layout",
    "SidebarCartSection": "the sidebar cart", "InsiderBannerSection": "the insider banner", "NotificationSection": "the notification bar",
}
TEXT_BANK = [
    "Fresh picks every week", "Small batch, big flavour", "Shop the new drop", "Crafted with care since 2019",
    "Free delivery over $50", "Your neighbourhood dispensary", "Open late every Friday", "Hand-picked by our budtenders",
    "Visit us on Main Street", "Members save 10%", "New arrivals just landed", "Taste the difference",
]
URL_BANK = ["/products", "/about", "/faq", "/blog", "/locations", "/products?category=flower", "https://example.com/events"]
EMAIL_BANK = ["hello@example.com", "orders@example.com"]
PHONE_BANK = ["(604) 555-0142", "778-555-0199"]
PREMIUM_ASKS = [
    ("custom-code", "Add a custom JavaScript snippet to the home page that tracks scroll depth."),
    ("ab-test", "Set up an A/B test of two hero headlines and show me which converts better."),
    ("third-party-script", "Install the Hotjar tracking script in the site header."),
    ("new-section-type", "Add a countdown timer section above the hero."),
    ("new-section-type-2", "Put an Instagram feed widget under the gallery."),
    ("page-create", "Create a brand new 'Wholesale' page with a login wall."),
    ("checkout-logic", "Change the checkout so it charges a 3% fee on credit cards."),
    ("inventory", "Hide any product that has fewer than 3 units in stock."),
    ("seo-redirects", "Add a 301 redirect from /old-shop to /products."),
    ("email", "Send our newsletter subscribers an email about the weekend sale."),
    ("premium-slice", "Turn on the chat widget in the bottom corner."),
    ("premium-slice-2", "Add an announcement banner above the header that says 'Free delivery today'."),
    ("code-export", "Export the site's React source code so my developer can edit it."),
    ("dns", "Point the domain shop.example.com at this site."),
    ("payments", "Connect our Moneris account so customers can pay online."),
    ("user-accounts", "Delete the customer account for john@example.com."),
]


# ── the case ────────────────────────────────────────────────────────────────

@dataclass
class Case:
    id: str
    kind: str
    target: dict | None
    fields: list[str]
    site_state: dict
    context: dict
    request: str
    expected: dict  # {ops, diff, report_unsupported?, candidates?}
    holdout: bool
    manifest_hash: str
    template: str
    meta: dict = dc_field(default_factory=dict)
    # engine.engine_fingerprint of the engine that measured expected.diff (None until measured, or measured by a
    # stand-in engine without one). verify warns, dataset flags, and `cases remeasure` refreshes it.
    engine_fingerprint: str | None = None

    def to_row(self) -> dict:
        return asdict(self)

    @classmethod
    def from_row(cls, row: dict) -> "Case":
        return cls(**{k: row.get(k) for k in cls.__dataclass_fields__})  # type: ignore[arg-type]

    @property
    def browser_ops(self) -> list[dict]:
        """Expected ops minus `pick_asset` (never applied by the engine)."""
        return [o for o in self.expected.get("ops") or [] if o.get("op") != "pick_asset"]


def case_rng(seed: int | str, case_id: str) -> random.Random:
    return random.Random(f"{seed}:{case_id}")


def slug(s: str) -> str:
    """Case ids keep `/` as the segment separator; `#` is reserved for sample ids (`<case>#<k>`)."""
    return re.sub(r"[^A-Za-z0-9_./\-\[\]]+", "-", str(s)).strip("-")


# ── value generation ────────────────────────────────────────────────────────

def _hex(rng: random.Random) -> str:
    return "#%06x" % rng.randrange(0, 1 << 24)


def _text_for(field: Field, rng: random.Random) -> str:
    label = (field.label or "").lower()
    key = field.key.lower()
    if "email" in key or "email" in label:
        return rng.choice(EMAIL_BANK)
    if "phone" in key or "phone" in label:
        return rng.choice(PHONE_BANK)
    if key.endswith(("link", "url", "href")) or "link" in label.split("›")[-1] or "url" in label:
        return rng.choice(URL_BANK)
    if "font" in key and "size" not in key:
        return rng.choice(["Inter", "Playfair Display", "Lora", "Montserrat"])
    text = rng.choice(TEXT_BANK)
    return f"<p>{text}</p>" if field.type == "richtext" else text


def _number_for(field: Field, rng: random.Random, current: Any) -> int | float:
    lo, hi, step = field.min, field.max, field.step or 1
    if lo is None or hi is None:
        choices = [v for v in range(1, 101) if v != current]
        return rng.choice(choices)
    n = int((hi - lo) / step) if step else 0
    if n <= 0:
        return hi if current != hi else lo
    for _ in range(50):
        k = rng.randint(0, n)
        v = lo + k * step
        if isinstance(v, float) and float(v).is_integer():
            v = int(v)
        if v != current and lo <= v <= hi:
            return v
    return lo if current != lo else hi


def _enum_for(field: Field, rng: random.Random, current: Any) -> Any:
    members = [v for v in field.enum if v is not None and v != "" and v != current]
    if not members:
        members = [v for v in field.enum if v is not None and v != current]
    if not members:
        return None
    return rng.choice(members)


def _ref_for(field: Field, ctxf: dict, rng: random.Random, current: Any) -> Any:
    pool = {"category": [c["slug"] for c in ctxf.get("categories", [])], "tag": [t["slug"] for t in ctxf.get("tags", [])],
            "brand": [b["slug"] for b in ctxf.get("brands", [])], "location": [l["id"] for l in ctxf.get("locations", [])]}
    values = [v for v in pool.get(field.ref or "", []) if v != current] or ["flower"]
    return rng.choice(values)


def value_for(field: Field, rng: random.Random, *, current: Any = None, ctxf: dict | None = None, assets: list[dict] | None = None) -> Any:
    t = field.type
    if t == "bool":
        return not bool(current)
    if t in ("enum", "select"):
        if t == "select" and len(field.enum) < 2:
            return _text_for(field, rng)
        return _enum_for(field, rng, current)
    if t in ("range", "number"):
        return _number_for(field, rng, current)
    if t == "color":
        v = _hex(rng)
        return v if v != current else _hex(rng)
    if t == "image":
        return rng.choice(assets or load_assets_fixture())["public_url"]
    if t == "ref":
        return _ref_for(field, ctxf or {}, rng, current)
    return _text_for(field, rng)


def seed_value_for(field: Field, rng: random.Random, ctxf: dict, assets: list[dict]) -> Any:
    """A plausible CURRENT value for a field (so a reset / restore has a diff)."""
    if field.type == "bool":
        return True
    return value_for(field, rng, ctxf=ctxf, assets=assets)


# ── wording ─────────────────────────────────────────────────────────────────

def _words(key: str) -> str:
    """`taglineFontSize` → `tagline font size`; `locationOverrides.loc-main.city` → `city`."""
    segs = [seg for seg in key.split(".") if not (seg.startswith("{") and seg.endswith("}"))]
    last = segs[-1] if segs else key
    return re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", last).replace("_", " ").lower()


def _clean_label(label: str, key: str = "") -> str:
    parts = [p.strip() for p in (label or "").split("›")]
    text = parts[-1] if parts else ""
    text = re.sub(r"\{[^}]*\}", "", text)  # `Hide {item}`, `{locationId}` placeholders
    text = re.sub(r"\s*\(.*?\)\s*$", "", text).strip(" :-")
    if not text or text.lower() in ("hide", "show", "text size in pixels"):
        text = (_words(key) if key else "") or text
    return text.lower() or "setting"


def _ordinal(n: int) -> str:
    return ["first", "second", "third", "fourth", "fifth", "sixth"][n] if n < 6 else f"{n + 1}th"


# ── unambiguous field names (int-run1 finding B, 2026-10-04) ────────────────
#
# int-run1 (Claude Sonnet, 300 × 4) had 195 `no_answer` rows, most of them the model asking a sensible
# clarifying question because the REQUEST was ambiguous: "Clear the color on the checkout page" (checkout has
# four `Color` fields), "body text color → …" for `productDetail.reviewTextColor` (label `Body Text Color`,
# and 'body text' names another pair), "the third content block" (five content blocks, two titled alike),
# "address → …" for `locationOverrides.loc-main.address` (three locations). A request now names the field by
# its manifest LABEL — the text `lookup_fields` returns, group segments kept (`Tagline › Text size in pixels`)
# — qualified, when two or more fields of the same target share that label, by the type word (`Headline
# color` vs `Headline text`), then by the dotted parent of a nested key (`Color (sectionStyle)`), then both;
# a field no candidate makes unique in its target is SKIPPED (`skip:ambiguous-label`, the key recorded in
# `skipped_ambiguous`) rather than asked for ambiguously. `clarify` cases stay deliberately ambiguous about
# the SECTION (they name only the type), never about the field.

TYPE_WORDS = {"color": "color", "text": "text", "richtext": "text", "bool": "toggle", "image": "image",
              "range": "size", "number": "size", "select": "selection", "enum": "option", "ref": "selection"}


def label_text(f: Field, key: str | None = None) -> str:
    """The manifest label verbatim (`A › B` groups kept, parentheticals kept); an empty label or a
    `{item}` placeholder (`Hide {item}`, `{item} Color`) falls back to the key's words."""
    key = key or f.key
    label = f.spec.get("label") or ""
    if not label.strip():
        return _words(key)
    if "{" in label:  # `Hide {item}` → `Hide our-story`; `Overlay {item}` + overlayColor → `Overlay color`
        words = re.sub(r"\{[^}]*\}", _words(key), label).split()
        label = " ".join(w for i, w in enumerate(words) if i == 0 or w.lower() != words[i - 1].lower())
    return re.sub(r"\s+", " ", label).strip()


def _parent_key(key: str) -> str:
    segs = [s for s in key.split(".") if not (s.startswith("{") and s.endswith("}"))]
    return ".".join(segs[:-1])


def phrase_candidates(f: Field, key: str | None = None) -> list[str]:
    """Least to most qualified: label, label + type word, label (parent), label + type word (parent)."""
    key = key or f.key
    base = label_text(f, key)
    tw = TYPE_WORDS.get(f.type, f.type)
    parent = _parent_key(key)
    bare = bool(re.fullmatch(r"\s*(\{[^}]*\})?\s*", f.spec.get("label") or ""))  # no label, or `{item}` alone
    out = [] if bare and parent else [base, f"{base} {tw}"]  # `our-story` alone says nothing: start at the parent
    if parent:
        out += [f"{base} ({parent})", f"{base} {tw} ({parent})"]
    return out


def name_fields(scope: dict[str, Field]) -> dict[str, str | None]:
    """key → the first candidate no OTHER field of `scope` could also be called (case-insensitive, against
    every candidate of every other field), or None when none is — the ambiguity guard."""
    cands = {k: phrase_candidates(f, k) for k, f in scope.items()}
    lower = {k: {c.lower() for c in v} for k, v in cands.items()}
    out: dict[str, str | None] = {}
    for k, cs in cands.items():
        others: set[str] = set()
        for ok, v in lower.items():
            if ok != k:
                others |= v
        out[k] = next((c for c in cs if c.lower() not in others), None)
    return out


def scope_fields(manifest: Manifest, c: Container, page: str | None = None) -> dict[str, Field]:
    """Every field a request about `c` could mean: its fields, the hero's mirror item fields, a page's
    `_common` fields."""
    out = dict(c.fields)
    for key in c.mirror_keys:
        if key not in out:
            f = manifest.mirror_item_field(c, key)
            if f:
                out[key] = f
    if c.kind == "page":
        for k, f in manifest.common_page_fields(page or c.name).items():
            out.setdefault(k, f)
    return out


def keyed_phrase(f: Field, key: str, ctxf: dict) -> str:
    """`locationOverrides.loc-main.address` → "address override for the location 'Main Street' (loc-main)"."""
    ident = key.split(".")[1]
    pool, noun = ("locations", "location") if f.keyed_by == "location" else ("categories", "category")
    name = next((x.get("name") for x in ctxf.get(pool, []) if x.get("id") == ident or x.get("slug") == ident), None)
    leaf = _words(key)
    return f"{leaf} override for the {noun} '{name}' ({ident})" if name else f"{leaf} override for the {noun} {ident}"


def section_phrase(state: dict, section_id: str, rng: random.Random, page: str = "home") -> str:
    """Names the exact section: type label alone when the type is unique on the page (a unique title may
    stand in); with repeats, always the id (`the third content block (contentBlock-3)`). A title shared by
    two sections ('Why Choose Us?' on contentBlock-features AND -icons) is never used as the name."""
    row = find_section(state, section_id, page) or {"id": section_id}
    type_id = type_of_section(section_id)
    label = SECTION_LABELS.get(type_id, type_id)
    title = section_title(row)
    rows = sections_of(state, page)
    title_unique = bool(title) and sum(1 for s in rows if section_title(s) == title) == 1
    same = sections_of_type(state, type_id, page)
    options = []
    if len(same) <= 1:
        options += [f"the {label}"] + ([] if label.endswith("section") else [f"the {label} section"])
        if title_unique:
            options += [f"the '{title}' section"]
    else:
        idx = [s.get("id") for s in same].index(section_id)
        options += [f"the {_ordinal(idx)} {label} ({section_id})", f"the {label} {section_id}"]
        if title_unique:
            options += [f"the {label} titled '{title}' ({section_id})"]
    return rng.choice(options)


def container_phrase(c: Container, state: dict, rng: random.Random, section_id: str | None, page: str) -> str:
    if c.kind == "section":
        return section_phrase(state, section_id or c.name, rng, page)
    if c.kind == "global":
        return PANEL_LABELS.get(c.name, f"the {c.name} settings")
    if c.kind == "branding":
        return rng.choice(["the site branding", "branding", "the brand settings"])
    if c.kind == "page":
        return rng.choice([f"the {page} page", f"the '{page}' page"])  # the exact page id, never a paraphrase
    return c.name


def value_phrase(field: Field, value: Any, rng: random.Random, ctxf: dict, assets: list[dict]) -> str:
    t = field.type
    if t == "bool":
        return "on" if value else "off"
    if t in ("enum", "select") and field.enum_labels:
        return str(field.enum_labels.get(str(value), field.enum_labels.get(value, value)))
    if t in ("range", "number"):
        unit = "px" if "pixel" in (field.label or "").lower() or "px" in (field.label or "").lower() else ""
        return f"{value}{unit}"
    if t == "image":
        a = next((a for a in assets if a["public_url"] == value), None)
        return f"the image '{a['alt']}'" if a and a.get("alt") else f"the image at {value}"
    if t == "ref":
        for pool in ("categories", "tags", "brands", "locations"):
            for item in ctxf.get(pool, []):
                if item.get("slug") == value or item.get("id") == value:
                    return item.get("name", str(value))
        return str(value)
    if t == "richtext":
        return "'" + re.sub(r"<[^>]+>", "", str(value)) + "'"
    return f"'{value}'"


SET_TEMPLATES = [
    "Set the {field} of {where} to {value}.",
    "Change the {field} on {where} to {value}.",
    "In {where}, make the {field} {value}.",
    "Please update {where}: {field} should be {value}.",
    "Can you set {field} to {value} for {where}?",
    "{where} — {field} → {value}",
]
TOGGLE_ON_TEMPLATES = ["Turn on {field} for {where}.", "Enable the {field} on {where}.", "Switch {field} on in {where}.", "I want {field} enabled on {where}, please."]
TOGGLE_OFF_TEMPLATES = ["Turn off {field} for {where}.", "Disable the {field} on {where}.", "Switch {field} off in {where}.", "Please disable {field} on {where}."]
IMAGE_TEMPLATES = ["Use {value} as the {field} of {where}.", "Set {where}'s {field} to {value}.", "Swap the {field} on {where} for {value}.", "Change the {field} in {where} to {value}."]
PICK_TEMPLATES = ["Let me pick a new {field} for {where}.", "I want to choose a different {field} on {where} from my library.", "Change the {field} of {where} — I'll select the image.", "Open the image picker for the {field} in {where}."]
# reset = 'reset' / 'clear' + the field label (never "remove"/"unset", which read as delete-the-element or asked
# for a clarification in int-run1); restore = 'default' + the field label.
RESET_TEMPLATES = ["Clear the {field} on {where}.", "Reset the {field} of {where}.", "Please clear the {field} in {where}.", "{where}: reset the {field} (clear it)."]
RESTORE_TEMPLATES = ["Put the {field} of {where} back to its default.", "Restore the default {field} on {where}.", "Reset {where}'s {field} to the template default.", "Restore the {field} of {where} to its default value."]
LIST_ADD_TEMPLATES = ["Add a new {item} to {where}.", "Add another {item} in {where}.", "Please add one more {item} to {where}.", "{where}: add a {item}."]
LIST_REMOVE_TEMPLATES = ["Remove the {ordinal} {item} from {where}.", "Delete the {ordinal} {item} in {where}.", "Take the {ordinal} {item} out of {where}.", "{where}: drop the {ordinal} {item}."]
LIST_REORDER_TEMPLATES = ["Swap the first two {items} in {where}.", "Move the second {item} of {where} to the top.", "Reorder {where}: put the second {item} first.", "In {where}, make the second {item} come first."]
LIST_UPDATE_TEMPLATES = ["Set the {field} of the {ordinal} {item} in {where} to {value}.", "Change the {ordinal} {item}'s {field} on {where} to {value}.", "In {where}, update the {ordinal} {item}: {field} = {value}.", "For the {ordinal} {item} of {where}, make the {field} {value}."]
SECTION_ADD_TEMPLATES = ["Add a {label} section to the home page.", "Put a new {label} on the home page.", "Insert a {label} section on home.", "I want another {label} on the home page."]
SECTION_REMOVE_TEMPLATES = ["Remove {where} from the home page.", "Delete {where}.", "Take {where} off the home page.", "Get rid of {where}."]
SECTION_REORDER_TEMPLATES = ["Move {where} to the top of the home page.", "Make {where} the first section on home.", "Put {where} first on the home page.", "Reorder home so {where} comes first."]
SECTION_TOGGLE_TEMPLATES = ["Hide {where} for now.", "Disable {where} without deleting it.", "Turn {where} off on the home page.", "Switch {where} off."]
SECTION_DUP_TEMPLATES = ["Duplicate {where}.", "Make a copy of {where}.", "Clone {where} on the home page.", "Add a second copy of {where}."]
LAYOUT_TEMPLATES = ["Switch the {page} page to the {layout} layout.", "Use the {layout} layout for the {page} page.", "Change the {page} page layout to {layout}.", "Set {page} to the {layout} layout."]
PINNED_TEMPLATES = ["Reset the built-in product list on the products page.", "Restore the product list section on the products page to its default.", "Put the products page's product list back to default.", "Reset the Product List block on products."]
CLARIFY_TEMPLATES = ["Change the {field} of the {label} to {value}.", "Set the {label}'s {field} to {value}.", "Update the {field} on the {label} section to {value}.", "Make the {label} {field} {value}."]
DERIVED_TEMPLATES = ["Set the {field} on {where} to '{value}' directly.", "Change the {field} value of {where} to '{value}'.", "Override the {field} field in {where} with '{value}'.", "Write '{value}' into {where}'s {field}."]
JOINERS = [" and ", ", and also ", ". Also, ", "; then "]


def render(template: str, **slots: Any) -> str:
    text = template.format(**slots)
    text = text[0].upper() + text[1:] if text else text
    return text


# ── the generator ───────────────────────────────────────────────────────────

class Generator:
    def __init__(self, manifest: Manifest, *, seed: int = 0, template: str = "editorial", fixtures_dir: Path | None = None,
                 base_state: dict | None = None):
        self.m = manifest
        self.seed = seed
        self.template = template
        self.fixtures_dir = fixtures_dir
        self.ctxf = load_context_fixture(fixtures_dir)
        self.assets = load_assets_fixture(fixtures_dir)
        self.layouts = load_layouts_fixture(fixtures_dir)
        self.base = base_state if base_state is not None else build_state(manifest, template=template, fixtures_dir=fixtures_dir)
        self.stats: dict[str, int] = {}
        self.skipped_ambiguous: list[str] = []  # `<container>.<key>` of every field the ambiguity guard dropped
        self._ids: set[str] = set()
        self._names: dict[tuple[str, str, str | None], dict[str, str | None]] = {}

    # helpers
    def _bump(self, k: str) -> None:
        self.stats[k] = self.stats.get(k, 0) + 1

    def _field_name(self, c: Container, key: str, page: str | None = None) -> str | None:
        """The unambiguous phrase for `key` in container `c` (name_fields over scope_fields), None when the
        ambiguity guard rejects it — the caller bumps `skip:ambiguous-label` and skips the case."""
        scope_key = (c.kind, c.name, page if c.kind == "page" else None)
        names = self._names.get(scope_key)
        if names is None:
            names = self._names[scope_key] = name_fields(scope_fields(self.m, c, page))
        return names.get(key)

    def _skip_ambiguous(self, c: Container, key: str) -> None:
        self._bump("skip:ambiguous-label")
        self.skipped_ambiguous.append(f"{c.name}.{key}")

    def _state(self) -> dict:
        return copy.deepcopy(self.base)

    def _uid(self, raw: str) -> str:
        cid = slug(raw)
        if cid in self._ids:
            n = 2
            while f"{cid}~{n}" in self._ids:
                n += 1
            cid = f"{cid}~{n}"
        self._ids.add(cid)
        return cid

    def _case(self, cid: str, kind: str, target: dict | None, fields: list[str], state: dict, page: str, request: str,
              ops: list[dict], holdout_key: str, meta: dict, **expected_extra: Any) -> Case:
        meta = {"generator": "souschef_lab.cases", "schema": SCHEMA_VERSION, "seed": self.seed, "holdout_key": holdout_key, **meta}
        return Case(
            id=cid, kind=kind, target=target, fields=fields, site_state=state, context=context_of(state, page, self.ctxf),
            request=request, expected={"ops": ops, "diff": [], **expected_extra}, holdout=is_holdout(holdout_key),
            manifest_hash=self.m.hash, template=self.template, meta=meta,
        )

    def _open_gate(self, c: Container, f: Field, state: dict, section_id: str | None, page: str) -> dict:
        writes = gate_writes(self.m, c, f, self.assets)
        if writes:
            apply_gate_writes(self.m, c, state, writes, section_id=section_id, page=page)
        return writes

    def _current(self, t: WritableTarget, state: dict, key: str) -> Any:
        try:
            if t.via_list:
                return read_field(t.container, t.field, state, section_id=t.section_id, page=t.page, indexes=[0])
            return read_field(t.container, t.field, state, section_id=t.section_id, page=t.page, key=key if t.field.keyed_by else None)
        except KeyError:
            return None

    def _set_op(self, t: WritableTarget, key: str, value: Any, assets: dict | None = None) -> dict:
        if t.via_list:
            return {"op": "edit_list", "target": t.target(), "list": t.via_list, "action": "update", "index": 0, "item": {key: value}}
        op = {"op": "set_fields", "target": t.target(), "values": {key: value}}
        if assets:
            op["assets"] = assets
        return op

    @staticmethod
    def _image_op(t: WritableTarget, key: str, asset: dict) -> dict:
        """`set_fields {values:{key: url}, assets:{key:{alt, srcset}}}` for a fixture asset (LAB.md §2 image row)."""
        return {"op": "set_fields", "target": t.target(), "values": {key: asset["public_url"]},
                "assets": {key: {"alt": asset.get("alt", ""), "srcset": asset.get("srcset")}}}

    # ── field cases ──────────────────────────────────────────────────────
    def field_cases(self, targets: Iterable[WritableTarget]) -> Iterator[Case]:
        gated_seen = 0
        for t in targets:
            f = t.field
            if f.type in EXCLUDED_TYPES:
                self._bump(f"skip:type:{f.type}")
                continue
            if f.read_only:
                self._bump("skip:derived")
                continue
            kind = {"bool": "toggle", "enum": "set_enum", "select": "set_enum", "range": "set_range", "number": "set_range",
                    "image": "image"}.get(f.type, "set_field")
            if f.type == "select" and len(f.enum) < 2:
                kind = "set_field"
            key = f.key
            field_phrase = self._field_name(t.container, f.key, t.page)
            if field_phrase is None:
                self._skip_ambiguous(t.container, f.key)
                continue
            if f.keyed_by:
                ident = self.ctxf["locations"][0]["id"] if f.keyed_by == "location" else self.ctxf["categories"][0]["slug"]
                key = re.sub(r"\{[a-zA-Z]+\}", ident, f.key)
                if not keyed_supported(self.m, t.container, key):
                    self._bump("skip:keyed-not-resolvable")  # engine-gap (golden.test.js KNOWN_MISMATCHES)
                    continue
                field_phrase = keyed_phrase(f, key, self.ctxf)  # the concrete location/category, by name and id
            where = t.section_id or t.page if t.container.kind in ("section", "page") else t.container.name
            cid = self._uid(f"{kind}/{t.container.name}/{where}/{key}")
            rng = case_rng(self.seed, cid)
            state = self._state()
            gates = self._open_gate(t.container, f, state, t.section_id, t.page)
            current = self._current(t, state, key)
            if f.type == "bool" and current is None:
                current = False
                write_values(self.m, t.container, state, {key: current}, section_id=t.section_id, page=t.page)
            value = value_for(f, rng, current=current, ctxf=self.ctxf, assets=self.assets)
            if value is None:
                self._bump("skip:no-value")
                continue
            if value == current:
                self._bump("skip:same-as-current")
                continue
            meta = {"section_type": type_of_section(t.section_id) if t.section_id else None, "field_type": f.type,
                    "container_kind": t.container.kind, "gate_writes": gates, "gated": f.gated, "via_list": t.via_list,
                    "mirror": bool(t.via_list), "keyed_by": f.keyed_by, "current": current, "field_phrase": field_phrase}
            where_phrase = container_phrase(t.container, state, rng, t.section_id, t.page)
            if f.type == "image":
                # Image writes are always `set_fields` + `assets` (the form the tool schema advertises and the model
                # uses), also on a hero mirror key: the engine writes slides[0].<k> and the flat <k> for either form
                # (verify.py collapses them) and the asset's alt/srcset ride along as companions. Before 2026-10-04
                # the hero form was `edit_list update` WITHOUT assets, so the expected diff carried alt '' /
                # srcset null while the model (correctly) wrote the asset's — every hero image case failed.
                a = next(a for a in self.assets if a["public_url"] == value)
                op = self._image_op(t, key, a)
                request = render(rng.choice(IMAGE_TEMPLATES), where=where_phrase, field=field_phrase,
                                 value=value_phrase(f, value, rng, self.ctxf, self.assets))
                yield self._case(cid, kind, t.target(), [key], state, t.page, request, [op], t.key, {**meta, "variant": "url", "asset_id": a["id"]})
                # pick_asset variant: the model asks for the picker, then writes URL + alt/srcset of EXACTLY the
                # asset the collector's fake picker answers with (state.fixture_asset_for, keyed by this case id).
                cid2 = self._uid(f"{kind}/{t.container.name}/{where}/{key}/pick")
                picked = fixture_asset_for(self.assets, cid2, "image")
                if picked["public_url"] == current:
                    self._bump("skip:pick-same-as-current")
                    continue
                rng2 = case_rng(self.seed, cid2)
                state2 = self._state()
                self._open_gate(t.container, f, state2, t.section_id, t.page)
                pick = {"op": "pick_asset", "purpose": f"{field_phrase} for {where_phrase}", "asset_type": "image"}
                request2 = render(rng2.choice(PICK_TEMPLATES), where=where_phrase, field=field_phrase)
                yield self._case(cid2, kind, t.target(), [key], state2, t.page, request2, [pick, self._image_op(t, key, picked)], t.key,
                                 {**meta, "variant": "pick_asset", "asset_id": picked["id"]})
                continue
            op = self._set_op(t, key, value)
            if f.type == "bool":
                tmpl = rng.choice(TOGGLE_ON_TEMPLATES if value else TOGGLE_OFF_TEMPLATES)
                request = render(tmpl, where=where_phrase, field=field_phrase)
            else:
                request = render(rng.choice(SET_TEMPLATES), where=where_phrase, field=field_phrase,
                                 value=value_phrase(f, value, rng, self.ctxf, self.assets))
            yield self._case(cid, kind, t.target(), [key], state, t.page, request, [op], t.key, {**meta, "value": value})
            if f.gated:
                gated_seen += 1
                if gated_seen % GATE_CLOSED_EVERY == 0:
                    cidc = self._uid(f"{kind}/{t.container.name}/{where}/{key}/gate-closed")
                    statec = self._state()
                    yield self._case(cidc, kind, t.target(), [key], statec, t.page, request, [op], t.key,
                                     {**meta, "gate_writes": {}, "gate_closed": True, "expect_warning": "W_HIDDEN", "value": value})

    # ── reset / restore_default ──────────────────────────────────────────
    def reset_cases(self, targets: Iterable[WritableTarget]) -> Iterator[Case]:
        for t in targets:
            f = t.field
            if f.type in EXCLUDED_TYPES or f.read_only or f.keyed_by or t.via_list:
                continue
            field_phrase = self._field_name(t.container, f.key, t.page)
            if field_phrase is None:
                if f.has_unset or f.has_default:
                    self._skip_ambiguous(t.container, f.key)
                continue
            for mode, op_name, has in (("unset", "reset", f.has_unset), ("default", "restore_default", f.has_default)):
                if not has:
                    continue
                where = t.section_id or t.page if t.container.kind in ("section", "page") else t.container.name
                cid = self._uid(f"reset/{op_name}/{t.container.name}/{where}/{f.key}")
                rng = case_rng(self.seed, cid)
                state = self._state()
                gates = self._open_gate(t.container, f, state, t.section_id, t.page)
                expected_leaf = f.unset_value if mode == "unset" else f.default
                seeded = seed_value_for(f, rng, self.ctxf, self.assets)
                if seeded == expected_leaf:
                    seeded = value_for(f, rng, current=seeded, ctxf=self.ctxf, assets=self.assets)
                if seeded == expected_leaf or seeded is None:
                    self._bump("skip:reset-no-diff")
                    continue
                write_values(self.m, t.container, state, {f.key: seeded}, section_id=t.section_id, page=t.page)
                op = {"op": op_name, "target": t.target(), "keys": [f.key]}
                where_phrase = container_phrase(t.container, state, rng, t.section_id, t.page)
                request = render(rng.choice(RESET_TEMPLATES if mode == "unset" else RESTORE_TEMPLATES), where=where_phrase, field=field_phrase)
                meta = {"section_type": type_of_section(t.section_id) if t.section_id else None, "field_type": f.type, "mode": mode,
                        "container_kind": t.container.kind, "gate_writes": gates, "gated": f.gated, "seeded": seeded, "expected_leaf": expected_leaf,
                        "field_phrase": field_phrase}
                yield self._case(cid, "reset", t.target(), [f.key], state, t.page, request, [op], t.key, meta)

    # ── lists ────────────────────────────────────────────────────────────
    def _list_state_path(self, c: Container, l: ListSpec, state: dict, section_id: str | None, page: str, parents: list[int]) -> str:
        path = l.path
        for p in parents:
            path = path.replace("[]", f"[{p}]", 1)
        if c.kind == "section" and path.startswith("settings."):
            return f"{sections_slice_path(page)}[{find_section_index(state, section_id, page)}].{path}"
        return slice_path(path.replace("{page}", page))

    def _seed_items(self, l: ListSpec, rng: random.Random, n: int = 3) -> list:
        if l.orders:
            pool = {"categories": [c["slug"] for c in self.ctxf["categories"]], "locations": [x["id"] for x in self.ctxf["locations"]],
                    "pages": ["home", "products", "faq", "blog"], "paymentMethods": list(self.ctxf.get("paymentMethods", [])),
                    "cardChips": ["thc", "cbd", "weight", "brand"], "detailChips": ["thc", "cbd", "terpenes", "weight"],
                    "filters": ["category", "brand", "price", "thc"], "sortOptions": ["relevance", "price-asc", "price-desc", "newest"]}
            return list(pool.get(l.orders, [f"{l.orders}-{i}" for i in range(n)]))[:max(n, 2)]
        if l.scalar:
            return [a["public_url"] for a in self.assets[:n]]
        out = []
        for i in range(n):
            item: dict = {}
            for k, f in l.item_fields.items():
                if f.read_only or f.type in ("image", "json"):
                    continue
                if f.type == "bool":
                    item[k] = True
                elif f.type in ("enum", "select") and f.enum:
                    item[k] = next((v for v in f.enum if v not in (None, "")), f.enum[0])
                elif f.type in ("range", "number"):
                    item[k] = f.min if f.min is not None else 1
                elif f.type == "color":
                    item[k] = "#%06x" % (0x3366AA + i)
                elif f.type == "ref":
                    item[k] = self.ctxf["categories"][i % len(self.ctxf["categories"])]["slug"]
                else:
                    item[k] = f"{_clean_label(f.label, f.key).title()} {i + 1}"
            if not item:
                item = {"title": f"Item {i + 1}"}
            item.setdefault("id", f"{slug(l.key).replace('/', '-')}-{i + 1}")
            out.append(item)
        return out

    def _ensure_list(self, c: Container, l: ListSpec, state: dict, section_id: str | None, page: str, rng: random.Random,
                     parents: list[int], min_len: int = 2) -> tuple[str, list]:
        path = self._list_state_path(c, l, state, section_id, page, parents)
        cur = get_path(state, path)
        if not isinstance(cur, list) or len(cur) < min_len:
            seeded = self._seed_items(l, rng, max(min_len, 3))
            if isinstance(cur, list) and cur:
                seeded = cur + seeded[len(cur):]
            set_path(state, path, seeded)
            cur = seeded
        return path, cur

    def _add_item(self, l: ListSpec, rng: random.Random, item_names: dict[str, str | None] | None = None):
        """The `item` an add carries: a scalar for scalar lists (an asset url for image lists),
        else one text-ish item field filled (named by its unambiguous phrase). Returns (item, request suffix)."""
        if l.scalar:
            f = l.item_fields["value"]
            if f.type == "image":
                a = rng.choice(self.assets)
                return a["public_url"], f"using the image '{a['alt']}'" if a.get("alt") else ""
            text = rng.choice(TEXT_BANK)
            return text, f"with the text '{text}'"
        names = item_names if item_names is not None else name_fields(l.item_fields)
        for f in l.item_fields.values():
            if f.read_only or f.type not in ("text", "richtext") or names.get(f.key) is None:
                continue
            text = rng.choice(TEXT_BANK)
            value = f"<p>{text}</p>" if f.type == "richtext" else text
            return {f.key: value}, f"with the {names[f.key]} '{text}'"
        return None, ""

    def _item_noun(self, l: ListSpec) -> str:
        key = l.key.split("[].")[-1].split(".")[-1]
        nouns = {"slides": "slide", "bullets": "bullet", "images": "image", "items": "item", "faqs": "FAQ", "tiers": "tier",
                 "milestones": "milestone", "reviews": "review", "posts": "post", "certifications": "certification",
                 "footerCustomColumns": "footer column", "links": "link", "headerNavCustomLinks": "custom header link",
                 "footerNavCustomLinks": "custom footer link", "footerBottomCustomLinks": "bottom link", "faqCategories": "FAQ category",
                 "cardRibbonColorRules": "ribbon colour rule", "infoSections": "info section", "privacyContent": "privacy section",
                 "termsContent": "terms section", "returnRefundContent": "refund policy section"}
        return nouns.get(key, re.sub(r"([a-z])([A-Z])", r"\1 \2", key).lower().rstrip("s") or "item")

    def list_cases(self, containers: Iterable[tuple[Container, str | None, str]]) -> Iterator[Case]:
        """`containers`: (container, section_id, page) triples whose lists get cases."""
        for c, section_id, page in containers:
            for l in c.lists.values():
                if l.depth > 2:
                    self._bump("skip:list-depth")
                    continue
                if not l.item_fields:
                    self._bump("skip:list-no-item")  # companion parallels (`imagesAlt`, `imagesLink`): no item to write
                    continue
                where = section_id or page if c.kind in ("section", "page") else c.name
                hold_key = f"{c.name}.{l.key}"
                parents = [0] if l.depth == 2 else []
                noun = self._item_noun(l)
                target = c.target(page, section_id)
                base_meta = {"section_type": type_of_section(section_id) if section_id else None, "list": l.key, "list_ops": l.ops,
                             "orders": l.orders, "item_ref": l.item_ref, "container_kind": c.kind, "depth": l.depth}

                parent_list = c.lists.get(l.key.split("[]")[0]) if l.depth == 2 else None
                item_names = name_fields(l.item_fields)  # the ambiguity guard over the row's fields

                def prep(cid: str) -> tuple[random.Random, dict, str, list]:
                    rng = case_rng(self.seed, cid)
                    state = self._state()
                    if parent_list:
                        self._ensure_list(c, parent_list, state, section_id, page, rng, [], 1)
                    path, cur = self._ensure_list(c, l, state, section_id, page, rng, parents)
                    return rng, state, path, cur

                def where_phrase(state: dict, rng: random.Random) -> str:
                    """A nested list names its parent row too: `the first footer column of the footer`."""
                    base = container_phrase(c, state, rng, section_id, page)
                    if l.depth == 2:
                        return f"the {_ordinal(parents[0])} {self._item_noun(parent_list) if parent_list else 'item'} of {base}"
                    return base

                if "add" in l.ops:
                    cid = self._uid(f"list_add/{c.name}/{where}/{l.key}")
                    rng, state, path, cur = prep(cid)
                    op = {"op": "edit_list", "target": target, "list": l.key, "action": "add"}
                    if l.depth == 2:
                        op["index"] = parents
                    item_value, item_desc = self._add_item(l, rng, item_names)
                    if item_value is not None:
                        op["item"] = item_value
                    req = render(rng.choice(LIST_ADD_TEMPLATES), where=where_phrase(state, rng), item=noun)
                    if item_desc:
                        req = req.rstrip(".") + f" {item_desc}."
                    yield self._case(cid, "list_add", target, [l.key], state, page, req, [op], hold_key, {**base_meta, "len_before": len(cur), "item": item_value})
                if "remove" in l.ops:
                    cid = self._uid(f"list_remove/{c.name}/{where}/{l.key}")
                    rng, state, path, cur = prep(cid)
                    idx = len(cur) - 1
                    op = {"op": "edit_list", "target": target, "list": l.key, "action": "remove", "index": parents + [idx] if l.depth == 2 else idx}
                    req = render(rng.choice(LIST_REMOVE_TEMPLATES), where=where_phrase(state, rng), item=noun, ordinal=_ordinal(idx) if idx < 6 else "last")
                    yield self._case(cid, "list_remove", target, [l.key], state, page, req, [op], hold_key, {**base_meta, "index": idx, "len_before": len(cur)})
                if "reorder" in l.ops:
                    cid = self._uid(f"list_reorder/{c.name}/{where}/{l.key}")
                    rng, state, path, cur = prep(cid)
                    if l.orders:
                        order = [cur[1], cur[0]] + list(cur[2:])
                        if not all(isinstance(x, str) for x in order):
                            order = [x.get("id") if isinstance(x, dict) else x for x in order]
                    else:
                        order = [1, 0] + list(range(2, len(cur)))
                    op = {"op": "edit_list", "target": target, "list": l.key, "action": "reorder", "order": order}
                    if l.depth == 2:
                        op["index"] = parents
                    req = render(rng.choice(LIST_REORDER_TEMPLATES), where=where_phrase(state, rng), item=noun, items=noun + "s")
                    yield self._case(cid, "list_reorder", target, [l.key], state, page, req, [op], hold_key, {**base_meta, "order": order})
                # update: one item field (always allowed), skipping images / derived / json
                upd_fields = [f for f in l.item_fields.values() if not f.read_only and f.type not in ("image", "json") and f.key != "value"]
                if l.orders:
                    self._bump("skip:list-update-orders")  # id arrays: `update` has no object item to write
                    upd_fields = []
                if upd_fields and not l.scalar:
                    for f in upd_fields[:3]:
                        field_phrase = item_names.get(f.key)
                        if field_phrase is None:
                            self._skip_ambiguous(c, f"{l.key}[].{f.key}")
                            continue
                        cid = self._uid(f"list_update/{c.name}/{where}/{l.key}/{f.key}")
                        rng, state, path, cur = prep(cid)
                        idx = 1 if len(cur) > 1 else 0
                        cur_item = cur[idx] if isinstance(cur[idx], dict) else {}
                        value = value_for(f, rng, current=cur_item.get(f.key), ctxf=self.ctxf, assets=self.assets)
                        if value is None or value == cur_item.get(f.key):
                            self._bump("skip:list-update-value")
                            continue
                        op = {"op": "edit_list", "target": target, "list": l.key, "action": "update", "index": parents + [idx] if l.depth == 2 else idx, "item": {f.key: value}}
                        if f.type == "bool":
                            tmpl = rng.choice(TOGGLE_ON_TEMPLATES if value else TOGGLE_OFF_TEMPLATES)
                            req = render(tmpl, where=f"the {_ordinal(idx)} {noun} of {where_phrase(state, rng)}", field=field_phrase)
                        else:
                            req = render(rng.choice(LIST_UPDATE_TEMPLATES), where=where_phrase(state, rng), item=noun,
                                         ordinal=_ordinal(idx), field=field_phrase, value=value_phrase(f, value, rng, self.ctxf, self.assets))
                        yield self._case(cid, "list_update", target, [f"{l.key}[].{f.key}"], state, page, req, [op], f"{hold_key}[].{f.key}",
                                         {**base_meta, "index": idx, "field_type": f.type, "value": value, "field_phrase": field_phrase})

    def section_list_cases(self, page: str = "home") -> Iterator[Case]:
        """The page's own `sections` list: add / remove / reorder / duplicate / update{enabled}."""
        target = {"kind": "page", "page": page}
        hold = f"page.{page}.sections"
        meta = {"list": "sections", "container_kind": "page"}
        for type_id in ("promotion", "sectionDivider", "loyaltyForm", "locations"):
            cid = self._uid(f"list_add/page/{page}/sections/{type_id}")
            rng = case_rng(self.seed, cid)
            state = self._state()
            op = {"op": "edit_list", "target": target, "list": "sections", "action": "add", "item": {"id": type_id}}
            label = SECTION_LABELS.get(type_id, type_id)
            req = render(rng.choice(SECTION_ADD_TEMPLATES), label=label[:-len(" section")] if label.endswith(" section") else label)
            yield self._case(cid, "list_add", target, ["sections"], state, page, req, [op], hold, {**meta, "section_type": type_id})
        rows = sections_of(self.base, page)
        for action, templates, kind in (("remove", SECTION_REMOVE_TEMPLATES, "list_remove"), ("duplicate", SECTION_DUP_TEMPLATES, "list_add")):
            for row in (rows[4], rows[7]):  # testimonials, gallery: unique types, unambiguous wording
                cid = self._uid(f"{kind}/page/{page}/sections/{action}/{row['id']}")
                rng = case_rng(self.seed, cid)
                state = self._state()
                idx = find_section_index(state, row["id"], page)
                op = {"op": "edit_list", "target": target, "list": "sections", "action": action, "index": idx}
                req = render(rng.choice(templates), where=section_phrase(state, row["id"], rng, page))
                yield self._case(cid, kind, target, ["sections"], state, page, req, [op], hold, {**meta, "action": action, "sectionId": row["id"]})
        for row in (rows[7], rows[10]):  # gallery, blogSection → reorder to the top
            cid = self._uid(f"list_reorder/page/{page}/sections/{row['id']}")
            rng = case_rng(self.seed, cid)
            state = self._state()
            ids = [s["id"] for s in sections_of(state, page)]
            order = [row["id"]] + [i for i in ids if i != row["id"]]
            op = {"op": "edit_list", "target": target, "list": "sections", "action": "reorder", "order": order}
            req = render(rng.choice(SECTION_REORDER_TEMPLATES), where=section_phrase(state, row["id"], rng, page))
            yield self._case(cid, "list_reorder", target, ["sections"], state, page, req, [op], hold, {**meta, "order": order})
        for row in (rows[1], rows[12]):  # featuredCategories, newsletter → disable
            cid = self._uid(f"list_update/page/{page}/sections/{row['id']}/enabled")
            rng = case_rng(self.seed, cid)
            state = self._state()
            idx = find_section_index(state, row["id"], page)
            op = {"op": "edit_list", "target": target, "list": "sections", "action": "update", "index": idx, "item": {"enabled": False}}
            req = render(rng.choice(SECTION_TOGGLE_TEMPLATES), where=section_phrase(state, row["id"], rng, page))
            yield self._case(cid, "list_update", target, ["sections[].enabled"], state, page, req, [op], hold, {**meta, "sectionId": row["id"]})

    # ── layout ───────────────────────────────────────────────────────────
    def layout_cases(self) -> Iterator[Case]:
        pl = self.layouts["page_layouts"]
        names = self.layouts.get("layout_names", {})
        defaults = self.layouts.get("default_layouts", {})
        manifest_pages = set(self.m.pages) | {"home"}
        for page, layouts in pl.items():
            if page not in manifest_pages and page not in ("home", "locations", "location-details"):
                continue
            current = defaults.get(page)
            for layout in layouts:
                if layout == current:
                    continue
                cid = self._uid(f"layout/{page}/{layout}")
                rng = case_rng(self.seed, cid)
                state = self._state()
                op = {"op": "set_page_layout", "page": page, "layout": layout}
                req = render(rng.choice(LAYOUT_TEMPLATES), page=page.replace("-", " "), layout=names.get(page, {}).get(layout, layout))
                yield self._case(cid, "layout", {"kind": "page", "page": page}, ["layout"], state, page, req, [op], f"page.{page}.layout",
                                 {"layout": layout, "current": current, "container_kind": "page"})
        # reset_pinned_section once per page with a reorderable pinned row (products)
        for page, d in PINNED_BY_PAGE.items():
            if not d["reorderable"]:
                continue
            cid = self._uid(f"layout/{page}/reset_pinned/{d['id']}")
            rng = case_rng(self.seed, cid)
            state = self._state()
            state["pageSectionsAll"][page] = [{"id": d["id"], "enabled": False, "settings": {"stale": True}}]
            op = {"op": "reset_pinned_section", "page": page, "sectionId": d["id"]}
            yield self._case(cid, "layout", {"kind": "page", "page": page}, [d["id"]], state, page, rng.choice(PINNED_TEMPLATES), [op],
                             f"page.{page}.{d['id']}", {"pinned": d["id"], "container_kind": "page"})

    # ── multi ────────────────────────────────────────────────────────────
    def multi_cases(self, targets: list[WritableTarget], n: int = 60) -> Iterator[Case]:
        simple = [t for t in targets if t.container.kind == "section" and not t.via_list and not t.field.keyed_by and not t.field.gated
                  and t.field.type in ("text", "bool", "enum", "range", "number", "color") and not t.field.read_only
                  and self._field_name(t.container, t.field.key, t.page) is not None]  # ambiguous labels never join a multi
        by_section: dict[str, list[WritableTarget]] = {}
        for t in simple:
            by_section.setdefault(t.section_id or "", []).append(t)
        sections = sorted(by_section)
        rng0 = random.Random(f"{self.seed}:multi")
        for i in range(n):
            k = 2 if i % 3 else 3
            picked_sections = rng0.sample(sections, k)
            cid = self._uid(f"multi/{i:03d}/" + "+".join(picked_sections))
            rng = case_rng(self.seed, cid)
            state = self._state()
            ops, parts, keys, hold_keys = [], [], [], []
            for sid in picked_sections:
                t = rng.choice(by_section[sid])
                current = self._current(t, state, t.field.key)
                if t.field.type == "bool" and current is None:
                    current = False
                    write_values(self.m, t.container, state, {t.field.key: current}, section_id=sid, page=t.page)
                value = value_for(t.field, rng, current=current, ctxf=self.ctxf, assets=self.assets)
                if value is None or value == current:
                    continue
                ops.append(self._set_op(t, t.field.key, value))
                keys.append(f"{sid}.{t.field.key}")
                hold_keys.append(t.key)
                where = section_phrase(state, sid, rng, t.page)
                fl = self._field_name(t.container, t.field.key, t.page)
                if t.field.type == "bool":
                    parts.append(("turn on " if value else "turn off ") + f"{fl} for {where}")
                else:
                    parts.append(f"set the {fl} of {where} to {value_phrase(t.field, value, rng, self.ctxf, self.assets)}")
            if len(ops) < 2:
                self._bump("skip:multi-short")
                continue
            joiner = rng.choice(JOINERS)
            request = render(joiner.join(parts).rstrip(".") + ".")
            holdout = any(is_holdout(h) for h in hold_keys)
            case = self._case(cid, "multi", None, keys, state, "home", request, ops, "multi/" + "+".join(hold_keys),
                              {"targets": picked_sections, "hold_keys": hold_keys, "container_kind": "section"})
            case.holdout = holdout  # any constituent field in the exam → the whole case is exam
            yield case

    # ── clarify ──────────────────────────────────────────────────────────
    def clarify_cases(self) -> Iterator[Case]:
        for type_id in ("contentBlock", "productCarousel"):
            same = sections_of_type(self.base, type_id)
            if len(same) < 2:
                continue
            c = self.m.section(type_id)
            # deliberately ambiguous about the SECTION (type named, several rows); never about the field
            fields = [f for f in c.fields.values() if f.type in ("text", "range", "enum", "color") and not f.gated and not f.read_only
                      and self._field_name(c, f.key) is not None][:30]
            for f in fields:
                cid = self._uid(f"clarify/{type_id}/{f.key}")
                rng = case_rng(self.seed, cid)
                state = self._state()
                value = value_for(f, rng, ctxf=self.ctxf, assets=self.assets)
                vp = value_phrase(f, value, rng, self.ctxf, self.assets)
                field_phrase = self._field_name(c, f.key)
                req = render(rng.choice(CLARIFY_TEMPLATES), label=SECTION_LABELS.get(type_id, type_id), field=field_phrase, value=vp)
                yield self._case(cid, "clarify", None, [f.key], state, "home", req, [], f"clarify.{type_id}.{f.key}",
                                 {"section_type": type_id, "field_type": f.type, "candidates": [s["id"] for s in same], "container_kind": "section",
                                  "field_phrase": field_phrase},
                                 candidates=[s["id"] for s in same])

    # ── unsupported ──────────────────────────────────────────────────────
    def unsupported_cases(self) -> Iterator[Case]:
        for key, text in PREMIUM_ASKS:
            cid = self._uid(f"unsupported/{key}")
            state = self._state()
            yield self._case(cid, "unsupported", None, [], state, "home", text, [], f"unsupported.{key}", {"reason": key}, report_unsupported=True)
        # derived_from fields named directly
        seen = set()
        for c, f in self.m.iter_fields():
            if not f.read_only or c.kind != "section" or c.name in seen:
                continue
            rows = sections_of_type(self.base, c.name)
            if not rows:
                continue
            seen.add(c.name)
            sid = rows[0]["id"]
            cid = self._uid(f"unsupported/derived/{c.name}/{f.key}")
            rng = case_rng(self.seed, cid)
            state = self._state()
            req = render(rng.choice(DERIVED_TEMPLATES), where=section_phrase(state, sid, rng), field=f.key, value="12345")
            yield self._case(cid, "unsupported", c.target("home", sid), [f.key], state, "home", req, [], f"unsupported.{c.name}.{f.key}",
                             {"reason": "derived_from", "derived_from": f.derived_from, "section_type": c.name}, report_unsupported=True)

    # ── all ──────────────────────────────────────────────────────────────
    def all_cases(self, kinds: Iterable[str] | None = None) -> Iterator[Case]:
        wanted = set(kinds) if kinds else set(KINDS)
        rows = sections_of(self.base, "home")
        targets = enumerate_targets(self.m, rows)
        if wanted & set(SET_KINDS):
            for case in self.field_cases(targets):
                if case.kind in wanted:
                    yield case
        if "reset" in wanted:
            yield from self.reset_cases(targets)
        if wanted & {"list_add", "list_remove", "list_reorder", "list_update"}:
            containers: list[tuple[Container, str | None, str]] = []
            seen_types: set[str] = set()
            for row in rows:
                c = self.m.section(row["id"])
                if c and c.lists and c.name not in seen_types:
                    seen_types.add(c.name)
                    containers.append((c, row["id"], "home"))
            containers += [(c, None, "home") for c in self.m.containers("global") if c.lists]
            containers.append((self.m.branding, None, "home"))
            containers += [(self.m.page(p), None, p) for p in self.m.pages if self.m.page(p).lists]
            for case in self.list_cases(containers):
                if case.kind in wanted:
                    yield case
            for case in self.section_list_cases("home"):
                if case.kind in wanted:
                    yield case
        if "layout" in wanted:
            yield from self.layout_cases()
        if "multi" in wanted:
            yield from self.multi_cases(targets)
        if "clarify" in wanted:
            yield from self.clarify_cases()
        if "unsupported" in wanted:
            yield from self.unsupported_cases()


def generate(manifest: Manifest, *, seed: int = 0, template: str = "editorial", kinds: Iterable[str] | None = None,
             limit: int | None = None, fixtures_dir: Path | None = None, base_state: dict | None = None) -> list[Case]:
    g = Generator(manifest, seed=seed, template=template, fixtures_dir=fixtures_dir, base_state=base_state)
    out = []
    for case in g.all_cases(kinds):
        out.append(case)
        if limit and len(out) >= limit:
            break
    return out


# ── measuring expected.diff ─────────────────────────────────────────────────

def measure_one(case: Case, engine, ctx: dict | None = None) -> tuple[bool, dict]:
    """Replays `case.browser_ops` on `case.site_state`; fills `expected.diff`.
    Returns (accepted, info). A refused op or an empty diff rejects the case."""
    from .engine import EngineError, GOLDEN_CTX
    try:
        r = engine.run(case.site_state, case.browser_ops, ctx or GOLDEN_CTX)
    except EngineError as e:
        return False, {"reason": "engine_error", "detail": str(e)[:500]}
    info = {"ok": r.ok, "errors": r.errors, "warnings": r.warnings, "written": r.written, "elapsed_s": round(r.elapsed_s, 3), "cached": r.cached}
    if not r.ok:
        return False, {"reason": "op_refused", **info}
    diff = r.diff()
    if case.browser_ops and not diff:
        return False, {"reason": "empty_diff", **info}
    case.expected["diff"] = diff
    case.meta["measured"] = {"warnings": r.warnings, "written": r.written}
    fp = getattr(engine, "fingerprint", None)
    case.engine_fingerprint = str(fp) if fp else None
    if getattr(engine, "commit", None):
        case.meta["measured"]["engine_commit"] = engine.commit
    if case.meta.get("gate_closed") and not any(w.startswith("W_HIDDEN") for w in r.warnings):
        info["note"] = "expected W_HIDDEN not emitted"
    return True, info


def sample(cases: list[Case], n: int, *, seed: int = 0) -> list[Case]:
    """A stratified subset of about `n` cases: every kind keeps its share
    (at least one case when it has any), chosen by a seeded shuffle within
    the kind, returned in the original order. `n` ≥ len(cases) → all."""
    if n <= 0 or n >= len(cases):
        return list(cases)
    by_kind: dict[str, list[Case]] = {}
    for c in cases:
        by_kind.setdefault(c.kind, []).append(c)
    rng = random.Random(f"sample:{seed}:{n}")
    quota = {k: max(1, round(n * len(v) / len(cases))) for k, v in by_kind.items()}
    # Round-off can over- or under-shoot: trim the largest kinds, or top up the largest, until the total is n.
    while sum(quota.values()) > n:
        k = max(quota, key=lambda k: (quota[k], k))
        if quota[k] > 1:
            quota[k] -= 1
        else:
            break
    while sum(quota.values()) < n:
        k = max(by_kind, key=lambda k: (len(by_kind[k]) - quota[k], k))
        if quota[k] >= len(by_kind[k]):
            break
        quota[k] += 1
    chosen: set[str] = set()
    for k, v in by_kind.items():
        ids = [c.id for c in v]
        rng.shuffle(ids)
        chosen.update(ids[: quota[k]])
    return [c for c in cases if c.id in chosen]


def measure(cases: Iterable[Case], engine, *, ctx: dict | None = None, log=None, workers: int = 1) -> tuple[list[Case], list[dict]]:
    """Measure every case with browser ops through the engine (one node boot
    each, cached by request); `workers` > 1 runs that many node processes at
    once. Order is preserved; clarify/unsupported cases are accepted as-is."""
    todo = list(cases)
    accepted, rejects = [], []
    need = [c for c in todo if not (c.kind in ("clarify", "unsupported") or not c.browser_ops)]
    results: dict[str, tuple[bool, dict]] = {}
    if workers > 1 and len(need) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for case, res in zip(need, pool.map(lambda c: measure_one(c, engine, ctx), need)):
                results[case.id] = res
                if log:
                    log(case.id, res[0], res[1])
    for case in todo:
        if case.kind in ("clarify", "unsupported") or not case.browser_ops:
            accepted.append(case)
            continue
        if case.id in results:
            ok, info = results[case.id]
        else:
            ok, info = measure_one(case, engine, ctx)
            if log:
                log(case.id, ok, info)
        if ok:
            accepted.append(case)
        else:
            rejects.append({"id": case.id, "kind": case.kind, "fields": case.fields, "ops": case.expected.get("ops"), **info})
    return accepted, rejects


# ── io ──────────────────────────────────────────────────────────────────────

def write_jsonl(path: str | Path, rows: Iterable[dict | Case]) -> int:
    n = 0
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.to_row() if isinstance(r, Case) else r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_cases(path: str | Path) -> list[Case]:
    return [Case.from_row(r) for r in read_jsonl(path)]


def summarize(cases: list[Case]) -> dict:
    by_kind: dict[str, int] = {}
    by_type: dict[str, int] = {}
    hold = 0
    for c in cases:
        by_kind[c.kind] = by_kind.get(c.kind, 0) + 1
        ft = c.meta.get("field_type")
        if ft:
            by_type[ft] = by_type.get(ft, 0) + 1
        hold += c.holdout
    return {"cases": len(cases), "holdout": hold, "holdout_pct": round(100 * hold / len(cases), 1) if cases else 0.0,
            "by_kind": dict(sorted(by_kind.items())), "by_field_type": dict(sorted(by_type.items()))}


def diff_key(case: Case) -> set[tuple[str, str]]:
    """`{(path, canonical after)}` of a case's expected diff (what verify compares)."""
    from .engine import canonical
    return {(str(d.get("path")), canonical(d.get("after"))) for d in case.expected.get("diff") or []}


def remeasure(cases: list[Case], engine, *, workers: int = 1, log=None) -> dict:
    """Refresh every case's expected.diff / meta.measured / engine_fingerprint
    in place with `engine` (the cache key carries the engine fingerprint, so
    a diff cached under another engine is never reused). Returns
    `{accepted:[Case], rejects:[dict], changed:[id], unchanged:n, stale:n}`
    where `changed` lists the cases whose diff set differs from before and
    `stale` counts the cases that carried another engine's fingerprint."""
    fp = str(getattr(engine, "fingerprint", "") or "")
    before = {c.id: (diff_key(c), c.engine_fingerprint) for c in cases}
    stale = sum(1 for _, (_, old_fp) in before.items() if old_fp != (fp or None))
    accepted, rejects = measure(cases, engine, log=log, workers=workers)
    changed = [c.id for c in accepted if diff_key(c) != before[c.id][0]]
    return {"accepted": accepted, "rejects": rejects, "changed": changed,
            "unchanged": len(accepted) - len(changed), "stale": stale, "fingerprint": fp}


def main_remeasure(argv: list[str]) -> int:
    """`cases remeasure --cases X`: re-run every expected op through the
    current engine and rewrite X (rejects to X-rejects.jsonl, as `cases`)."""
    import argparse
    ap = argparse.ArgumentParser(prog="souschef_lab.cases remeasure", description="refresh expected diffs in place with the current engine")
    ap.add_argument("--cases", required=True, help="cases.jsonl to refresh in place")
    ap.add_argument("--manifest", default=str(FIXTURES_DIR / "manifest.snapshot.json"))
    ap.add_argument("--engine", help="headless.mjs path (default: $SOUS_CHEF_ENGINE / $COOKIE_CUTTER_DIR)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true", help="measure and report, write nothing")
    a = ap.parse_args(argv)
    from .engine import Engine, default_cache_dir
    eng = Engine(a.engine, manifest_path=a.manifest, cache_dir=default_cache_dir())
    path = Path(a.cases)
    cases = read_cases(path)
    done = [0]

    def log(cid: str, ok: bool, info: dict) -> None:
        done[0] += 1
        if done[0] % 25 == 0 or not ok:
            sys.stderr.write(f"[{done[0]}/{len(cases)}] {'ok ' if ok else 'REJECT'} {cid} {info.get('reason', '')} {info.get('errors', '') if not ok else ''}\n")
    t0 = time.monotonic()
    res = remeasure(cases, eng, workers=max(1, a.workers), log=log)
    report = {"cases": len(cases), "accepted": len(res["accepted"]), "rejects": len(res["rejects"]), "changed": len(res["changed"]),
              "unchanged": res["unchanged"], "stale_before": res["stale"], "engine_fingerprint": res["fingerprint"],
              "engine_commit": getattr(eng, "commit", None), "changed_ids": res["changed"], "engine": getattr(eng, "stats", dict)(),
              "measure_wall_s": round(time.monotonic() - t0, 1), "written": None if a.dry_run else str(path)}
    if not a.dry_run:
        tmp = path.with_name(path.name + ".tmp")
        write_jsonl(tmp, res["accepted"])
        tmp.replace(path)
        if res["rejects"]:
            write_jsonl(path.with_name(path.stem + "-rejects.jsonl"), res["rejects"])
    print(json.dumps(report, indent=1))
    return 0 if not res["rejects"] else 2


def main(argv: list[str] | None = None) -> int:
    import argparse
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["remeasure"]:
        return main_remeasure(argv[1:])
    ap = argparse.ArgumentParser(prog="souschef_lab.cases", description="generate (and measure) cases; `remeasure --cases X` refreshes expected diffs in place")
    ap.add_argument("--manifest", default=str(FIXTURES_DIR / "manifest.snapshot.json"))
    ap.add_argument("--out", required=True, help="cases.jsonl; rejects go beside it as cases-rejects.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--template", default="editorial")
    ap.add_argument("--kinds", help="comma-separated subset of kinds")
    ap.add_argument("--limit", type=int, help="stop after this many cases in generation order (kind by kind)")
    ap.add_argument("--sample", type=int, help="a stratified subset of about this many cases (every kind keeps its share)")
    ap.add_argument("--engine", help="headless.mjs path (default: $SOUS_CHEF_ENGINE / $COOKIE_CUTTER_DIR)")
    ap.add_argument("--no-measure", action="store_true", help="skip the engine; expected.diff stays empty")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    m = load(a.manifest)
    cases = generate(m, seed=a.seed, template=a.template, kinds=a.kinds.split(",") if a.kinds else None, limit=a.limit)
    if a.sample:
        cases = sample(cases, a.sample, seed=a.seed)
    rejects: list[dict] = []
    if not a.no_measure:
        from .engine import Engine, default_cache_dir
        eng = Engine(a.engine, manifest_path=a.manifest, cache_dir=default_cache_dir())
        done = [0]

        def log(cid: str, ok: bool, info: dict) -> None:
            done[0] += 1
            if done[0] % 25 == 0 or not ok:
                sys.stderr.write(f"[{done[0]}/{len(cases)}] {'ok ' if ok else 'REJECT'} {cid} {info.get('reason', '')} {info.get('errors', '') if not ok else ''}\n")
        t0 = time.monotonic()
        cases, rejects = measure(cases, eng, log=log, workers=max(1, a.workers))
        sys.stderr.write(json.dumps({"engine": eng.stats(), "measure_wall_s": round(time.monotonic() - t0, 1), "workers": a.workers}) + "\n")
    out = Path(a.out)
    n = write_jsonl(out, cases)
    nr = write_jsonl(out.with_name(out.stem + "-rejects.jsonl"), rejects)
    print(json.dumps({"written": n, "rejects": nr, "out": str(out), **summarize(cases)}, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
