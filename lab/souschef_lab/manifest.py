"""Manifest loader (docs/LAB.md §1 `manifest.py`).

The manifest is the editor-controls document the launcher builds
(`scripts/sous-chef-manifest`, snapshot in `service/testdata/`). This module:

- loads it and computes the service's README §1.1 hash in Python
  (`canonical_hash`), tested against the service's vector;
- wraps containers (sections, global panels, branding, pages, details) and
  their fields and lists in small dataclasses with the manifest's own
  vocabulary (`type`, `enum`, `min`/`max`/`step`, `default`, `unset_value`,
  `companions`, `aliases`, `conditional_on`, `hidden_when`, `derived_from`,
  `keyed_by`, `ref`, `suggested`; lists carry `item`/`item_ref`/`ops`/`orders`);
- enumerates the writable targets: one entry per (container, field) the
  engine will accept a write for — `derived_from` fields are read-only
  (engine README §2 rule 2, E_READONLY) and are excluded;
- derives the holdout bucket of a target from its container+field key
  (`sha1(key) % 100 < 20`), so paraphrases of one target never straddle
  train and exam (thermometer's question-hash idea, FACTS 1 §1).

Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Iterator

HASH_EXCLUDED = ("generated_from", "coverage")
HOLDOUT_PCT = 20
REPEATABLE_SUFFIX_RE = re.compile(r"-\d+$")  # schema.js REPEATABLE_SUFFIX_RE
# schema.js SECTION_PREFIX_FAMILIES: ids like `contentBlock-features` resolve to the family head
SECTION_PREFIX_FAMILIES = ("locations", "productCarousel", "contentBlock", "sectionDivider", "gallery", "milestones")
DETAIL_ITEM_PREFIX = "item."
DETAIL_REF_PREFIX = "details."
# Field types the manifest uses (counted on the 2026-10-01 snapshot).
FIELD_TYPES = ("color", "text", "bool", "enum", "select", "range", "number", "image", "richtext", "json", "ref")
NUMERIC_TYPES = ("range", "number")
ENUM_TYPES = ("enum", "select")
TEXT_TYPES = ("text", "richtext")

_MISSING = object()


# ── hash (README §1.1) ───────────────────────────────────────────────────────

def canonical_bytes(doc: dict) -> bytes:
    """`canonical(manifest minus generated_from and coverage)`: sorted keys,
    no whitespace, raw non-ASCII. Go's encoding/json formats numbers the way
    Python does for the integers and short floats the manifest holds; the
    service cross-checked this exact `json.dumps` form (README §1.1)."""
    body = {k: v for k, v in doc.items() if k not in HASH_EXCLUDED}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_hash(doc: dict) -> str:
    return hashlib.sha256(canonical_bytes(doc)).hexdigest()


def hash_file(path: str | Path) -> str:
    with Path(path).open("rb") as f:
        return canonical_hash(json.load(f))


# ── dataclasses ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Field:
    key: str
    spec: dict

    @property
    def path(self) -> str:
        return self.spec.get("path", "")

    @property
    def type(self) -> str:
        return self.spec.get("type", "")

    @property
    def label(self) -> str:
        return self.spec.get("label") or self.key

    @property
    def enum(self) -> list:
        return list(self.spec.get("enum") or [])

    @property
    def enum_labels(self) -> dict:
        return dict(self.spec.get("enum_labels") or {})

    @property
    def min(self):
        return self.spec.get("min")

    @property
    def max(self):
        return self.spec.get("max")

    @property
    def step(self):
        return self.spec.get("step")

    @property
    def has_default(self) -> bool:
        return "default" in self.spec

    @property
    def default(self):
        return self.spec.get("default")

    @property
    def has_unset(self) -> bool:
        return "unset_value" in self.spec

    @property
    def unset_value(self):
        return self.spec.get("unset_value")

    @property
    def companions(self) -> dict:
        return dict(self.spec.get("companions") or {})

    @property
    def aliases(self) -> dict:
        return dict(self.spec.get("aliases") or {})

    @property
    def conditional_on(self) -> dict | None:
        return self.spec.get("conditional_on")

    @property
    def hidden_when(self) -> dict | None:
        return self.spec.get("hidden_when")

    @property
    def gated(self) -> bool:
        return bool(self.conditional_on or self.hidden_when)

    @property
    def derived_from(self) -> str | None:
        return self.spec.get("derived_from")

    @property
    def read_only(self) -> bool:
        return self.derived_from is not None

    @property
    def keyed_by(self) -> str | None:
        return self.spec.get("keyed_by")

    @property
    def ref(self) -> str | None:
        return self.spec.get("ref")

    @property
    def suggested(self) -> list:
        return list(self.spec.get("suggested") or [])

    @property
    def writes(self) -> list:
        return list(self.spec.get("writes") or [])

    @property
    def is_list_item(self) -> bool:
        return "[]" in self.path


@dataclass(frozen=True)
class ListSpec:
    key: str  # the list key as ops spell it (`slides`, `slides[].bullets`, `pageVisibility.blog.posts`)
    spec: dict
    item_fields: dict[str, Field]  # list `item` ∪ details[item_ref].fields minus `item.`

    @property
    def path(self) -> str:
        return self.spec.get("path", "")

    @property
    def ops(self) -> list[str]:
        return list(self.spec.get("ops") or [])

    @property
    def orders(self) -> str | None:
        return self.spec.get("orders")

    @property
    def item_ref(self) -> str | None:
        return self.spec.get("item_ref")

    @property
    def depth(self) -> int:
        """1 for `slides`, 2 for `slides[].bullets` (targets.js listDepth)."""
        return len(self.key.split("[]"))

    @property
    def scalar(self) -> bool:
        """A list whose elements are values, not objects (gallery `images[]`)."""
        return list(self.item_fields) == ["value"] and self.item_fields["value"].path.endswith("[]")

    def allows(self, action: str) -> bool:
        return action == "update" or action in self.ops


@dataclass(frozen=True)
class Container:
    kind: str  # section | global | branding | page | details
    name: str  # heroBanner | HeaderSection | branding | faq | FaqDetailsPanel
    fields: dict[str, Field]
    lists: dict[str, ListSpec]
    pages: tuple[str, ...] = ()
    mirror: dict | None = None
    presets: list = dc_field(default_factory=list)
    editor_key: str | None = None

    @property
    def mirror_keys(self) -> list[str]:
        return list((self.mirror or {}).get("keys") or [])

    @property
    def mirror_from(self) -> str | None:
        return (self.mirror or {}).get("from")

    def target(self, page: str = "home", section_id: str | None = None) -> dict:
        """The engine target union (ops README §1) for this container."""
        if self.kind == "section":
            return {"kind": "section", "page": page, "sectionId": section_id or self.name}
        if self.kind == "global":
            return {"kind": "global", "panel": self.name}
        if self.kind == "branding":
            return {"kind": "branding"}
        if self.kind == "page":
            return {"kind": "page", "page": self.name}
        raise ValueError(f"{self.kind} containers are list items, not op targets")


@dataclass(frozen=True)
class WritableTarget:
    """One (container, field) the engine accepts a write for."""
    container: Container
    field: Field
    section_id: str | None = None  # the state row (`contentBlock-1`) for sections
    page: str = "home"
    via_list: str | None = None  # set when the field is a list item (hero mirror → `slides`)

    @property
    def key(self) -> str:
        """`<container>.<field>` — the holdout key (docs/LAB.md §2)."""
        return f"{self.container.name}.{self.field.key}"

    @property
    def holdout(self) -> bool:
        return is_holdout(self.key)

    def target(self) -> dict:
        return self.container.target(self.page, self.section_id)


# ── manifest ────────────────────────────────────────────────────────────────

class Manifest:
    def __init__(self, doc: dict, path: Path | None = None):
        self.doc = doc
        self.path = path
        self.hash = canonical_hash(doc)
        self._containers: dict[tuple[str, str], Container] = {}
        self._build()

    # construction
    def _build(self) -> None:
        d = self.doc
        for name, spec in (d.get("details") or {}).items():
            self._containers[("details", name)] = self._container("details", name, spec)
        for name, spec in (d.get("sections") or {}).items():
            self._containers[("section", name)] = self._container("section", name, spec)
        for name, spec in (d.get("globalPanels") or {}).items():
            self._containers[("global", name)] = self._container("global", name, spec)
        if d.get("branding"):
            self._containers[("branding", "branding")] = self._container("branding", "branding", d["branding"])
        for name, spec in (d.get("pages") or {}).items():
            self._containers[("page", name)] = self._container("page", name, spec)

    def _container(self, kind: str, name: str, spec: dict) -> Container:
        fields = {k: Field(k, v) for k, v in (spec.get("fields") or {}).items()}
        lists = {k: self._list(k, v) for k, v in (spec.get("lists") or {}).items()}
        return Container(
            kind=kind, name=name, fields=fields, lists=lists,
            pages=tuple(spec.get("pages") or ()), mirror=spec.get("mirror"),
            presets=list(spec.get("presets") or []), editor_key=spec.get("editorKey"),
        )

    def _list(self, key: str, spec: dict) -> ListSpec:
        items: dict[str, Field] = {k: Field(k, v) for k, v in (spec.get("item") or {}).items()}
        ref = spec.get("item_ref")
        if ref:
            panel = ref[len(DETAIL_REF_PREFIX):] if ref.startswith(DETAIL_REF_PREFIX) else ref
            detail = (self.doc.get("details") or {}).get(panel) or {}
            for k, v in (detail.get("fields") or {}).items():
                k2 = k[len(DETAIL_ITEM_PREFIX):] if k.startswith(DETAIL_ITEM_PREFIX) else k
                items[k2] = Field(k2, v)  # details panel wins (targets.resolveList)
        return ListSpec(key, spec, items)

    # lookups
    @property
    def version(self) -> int:
        return int(self.doc.get("version") or 0)

    @property
    def generated_from(self) -> dict:
        return dict(self.doc.get("generated_from") or {})

    def containers(self, kind: str | None = None) -> list[Container]:
        return [c for (k, _), c in self._containers.items() if kind is None or k == kind]

    def section(self, type_or_id: str) -> Container | None:
        t = self.type_of(type_or_id)
        return self._containers.get(("section", t)) if t else None

    def type_of(self, section_id: str) -> str | None:
        """targets.lookupSectionTypeId: strip `-<n>`; a `<family>-<anything>` id resolves to the family."""
        return lookup_section_type_id(section_id, set(self.doc.get("sections") or {}))

    def panel(self, name: str) -> Container | None:
        return self._containers.get(("global", name))

    def page(self, name: str) -> Container | None:
        return self._containers.get(("page", name))

    def detail(self, name: str) -> Container | None:
        name = name[len(DETAIL_REF_PREFIX):] if name.startswith(DETAIL_REF_PREFIX) else name
        return self._containers.get(("details", name))

    @property
    def branding(self) -> Container:
        return self._containers[("branding", "branding")]

    @property
    def section_types(self) -> list[str]:
        return [c.name for c in self.containers("section")]

    @property
    def panels(self) -> list[str]:
        return [c.name for c in self.containers("global")]

    @property
    def pages(self) -> list[str]:
        return [c.name for c in self.containers("page") if c.name != "_common"]

    def common_page_fields(self, page: str) -> dict[str, Field]:
        """`pages._common` with `{page}` substituted (targets.templateCommonFields)."""
        common = self._containers.get(("page", "_common"))
        if not common:
            return {}
        out = {}
        for k, f in common.fields.items():
            spec = dict(f.spec)
            spec["path"] = str(spec.get("path", "")).replace("{page}", page)
            out[k] = Field(k, spec)
        return out

    def page_fields(self, page: str) -> dict[str, Field]:
        c = self.page(page)
        out = dict(c.fields) if c else {}
        out.update(self.common_page_fields(page))
        return out

    def global_fields(self) -> dict[str, Field]:
        """Every global panel merged (ops README §1: `panel` only narrows validation)."""
        out: dict[str, Field] = {}
        for c in self.containers("global"):
            out.update(c.fields)
        return out

    # iterators
    def iter_fields(self) -> Iterator[tuple[Container, Field]]:
        for c in self.containers():
            for f in c.fields.values():
                yield c, f

    def iter_lists(self) -> Iterator[tuple[Container, ListSpec]]:
        for c in self.containers():
            for l in c.lists.values():
                yield c, l

    def iter_item_fields(self) -> Iterator[tuple[Container, ListSpec, Field]]:
        for c, l in self.iter_lists():
            for f in l.item_fields.values():
                yield c, l, f

    def counts(self) -> dict:
        n_fields = sum(len(c.fields) for c in self.containers()) + sum(
            len(l.spec.get("item") or {}) for _, l in self.iter_lists())
        return {
            "fields": n_fields,
            "lists": sum(len(c.lists) for c in self.containers()),
            "sections": len(self.containers("section")),
            "globalPanels": len(self.containers("global")),
            "pages": len(self.containers("page")),
            "details": len(self.containers("details")),
            "derived": sum(1 for _, f in self.iter_fields() if f.read_only)
            + sum(1 for _, _, f in self.iter_item_fields() if f.read_only),
        }

    # the hero mirror
    def mirror_item_field(self, container: Container, key: str) -> Field | None:
        """For a hero `mirror.keys` member, the `slides` item field it mirrors."""
        if not container.mirror or key not in container.mirror_keys:
            return None
        src = container.mirror_from or "slides[0]"
        list_key = re.sub(r"\[\d+\]$", "", src)
        l = container.lists.get(list_key)
        return l.item_fields.get(key) if l else None

    def resolve_field(self, container: Container, key: str) -> Field | None:
        """A top-level field, a hero mirror key or a page `_common` field."""
        if key in container.fields:
            return container.fields[key]
        m = self.mirror_item_field(container, key)
        if m:
            return m
        if container.kind == "page":
            return self.common_page_fields(container.name).get(key)
        return None


def load(path: str | Path) -> Manifest:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return Manifest(json.load(f), p)


# ── helpers ─────────────────────────────────────────────────────────────────

def section_type_id(section_id: str) -> str:
    """`contentBlock-2` → `contentBlock` (targets.sectionTypeId)."""
    return REPEATABLE_SUFFIX_RE.sub("", str(section_id or ""))


def lookup_section_type_id(section_id: str, known: set[str] | None = None) -> str | None:
    """`contentBlock-2` → `contentBlock`; `contentBlock-features` → `contentBlock` (a prefix
    family); an id that is no manifest section → None (targets.lookupSectionTypeId)."""
    stripped = section_type_id(section_id)
    if not stripped:
        return None
    if known is None or stripped in known:
        return stripped
    dash = stripped.find("-")
    if dash > 0:
        head = stripped[:dash]
        if head in SECTION_PREFIX_FAMILIES and head in known:
            return head
    return None


def holdout_bucket(key: str) -> int:
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % 100


def is_holdout(key: str, pct: int = HOLDOUT_PCT) -> bool:
    return holdout_bucket(key) < pct


def enumerate_targets(
    manifest: Manifest,
    sections: list[dict] | None = None,
    page: str = "home",
    *,
    pages: list[str] | None = None,
    include_global: bool = True,
    include_branding: bool = True,
    include_mirror: bool = True,
    include_keyed: bool = True,
) -> list[WritableTarget]:
    """Writable targets (FACTS 2 §2): one per field of each section row in
    `sections` (`[{id}]`, the state's rows for `page`), of each global panel,
    of branding, and of each page in `pages` (manifest pages by default).
    `derived_from` fields are excluded. Hero `mirror.keys` are included as
    `via_list='slides'` entries (the expected op is `edit_list update slides[0]`)."""
    out: list[WritableTarget] = []
    for row in sections or []:
        sid = str(row.get("id"))
        c = manifest.section(sid)
        if not c:
            continue
        for f in c.fields.values():
            if f.read_only:
                continue
            if f.keyed_by and not include_keyed:
                continue
            out.append(WritableTarget(c, f, sid, page))
        if include_mirror and c.mirror:
            list_key = re.sub(r"\[\d+\]$", "", c.mirror_from or "slides[0]")
            for key in c.mirror_keys:
                f = manifest.mirror_item_field(c, key)
                if f and not f.read_only and key not in c.fields:
                    out.append(WritableTarget(c, f, sid, page, via_list=list_key))
    if include_global:
        for c in manifest.containers("global"):
            for f in c.fields.values():
                if f.read_only or (f.keyed_by and not include_keyed):
                    continue
                out.append(WritableTarget(c, f, None, page))
    if include_branding:
        for f in manifest.branding.fields.values():
            if not f.read_only:
                out.append(WritableTarget(manifest.branding, f, None, page))
    for pg in (pages if pages is not None else manifest.pages):
        c = manifest.page(pg)
        if not c:
            continue
        for f in c.fields.values():
            if not f.read_only:
                out.append(WritableTarget(c, f, None, pg))
        for k, f in manifest.common_page_fields(pg).items():
            out.append(WritableTarget(c, f, None, pg))
    return out


def targets_by_type(targets: list[WritableTarget]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in targets:
        out[t.field.type] = out.get(t.field.type, 0) + 1
    return dict(sorted(out.items()))


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="souschef_lab.manifest", description="hash + counts of a manifest")
    ap.add_argument("manifest")
    ap.add_argument("--targets", action="store_true", help="also count writable targets for the default home rows")
    a = ap.parse_args(argv)
    m = load(a.manifest)
    out: dict[str, Any] = {"hash": m.hash, "generated_from": m.generated_from, **m.counts()}
    if a.targets:
        from .state import DEFAULT_HOME_SECTIONS  # local import: state depends on manifest
        ts = enumerate_targets(m, DEFAULT_HOME_SECTIONS)
        out["targets"] = len(ts)
        out["targets_holdout"] = sum(1 for t in ts if t.holdout)
        out["targets_by_type"] = targets_by_type(ts)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
