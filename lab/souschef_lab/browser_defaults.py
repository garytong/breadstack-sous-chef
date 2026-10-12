"""The launcher's own defaults for what a site has not saved (docs/LAB.md §2
"Browser defaults").

Two things the browser fills in that a saved config does not carry, and that
the lab used to get wrong:

* **A non-home page nothing was saved for** — `targets.resolveSectionSlot`
  (and so `turnContext.buildTurnContext`, the engine and the preview) uses
  `getDefaultPageSections(templateId, page)` (`src/data/pageSections.jsx`):
  the template's default rows, or `[]`. The lab used to put the page's pinned
  row there (`__pinned_faq` …), which the browser never sends.
* **The FAQ categories of a basic template** — PreviewPage.jsx seeds
  `faqCategories` with `getDefaultFaqForTemplate(template)` when nothing was
  saved, which for every basic template is `@breadstack/ui`'s
  `defaultBasicFaqCategories` (the `{name, items:[{q, a}]}` schema the basic
  editor and the manifest's `faqCategories[].items` use).

Both are JSX the lab cannot import, so `js/browser_defaults.mjs` loads them
through launcher/client's own vite config (as `headless.mjs` loads the
engine) and `fixtures/browser-defaults.json` keeps the dump with its source
identity. `python -m souschef_lab.browser_defaults --check` re-derives it
from `$COOKIE_CUTTER_DIR` and fails on any difference; `--write` refreshes it.

Standard library only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import FIXTURES_DIR

FIXTURE = "browser-defaults.json"
SCRIPT = Path(__file__).resolve().parent / "js" / "browser_defaults.mjs"
CLIENT_REL = Path("launcher/client")

# The source files the dump reads (their sha256 is recorded beside it).
SOURCES = (
    "launcher/client/src/data/pageSections.jsx",
    "launcher/client/src/pages/PreviewPage.jsx",
    "packages/ui/src/templates/default/DefaultFAQPage.jsx",
)

# Every page slug the lab may view or name: the manifest's pinned pages, the
# keys of pageSections.jsx DF_PAGE_SECTIONS / PLUTO_PAGE_SECTIONS, the text
# pages (resolvePageSectionKey → 'text-page') and the context fixture's
# custom pages. A slug outside the list resolves to no defaults (`[]`).
PAGES = (
    "home", "faq", "product-details", "products", "blog", "blog-detail",
    "privacy-policy", "terms-of-service", "return-refund", "disclaimer", "eula", "cookie-policy",
    "age-gate", "location-selection", "cart", "checkout", "account", "login",
    "order-confirmation", "order-details",
    "about", "contact", "careers", "delivery", "mail-order", "locations", "location-details", "loyalty",
    "events",
)

OTHER_TEMPLATE = "*"  # getDefaultPageSections for a template outside its map


ENV_OVERRIDE = "SOUS_CHEF_BROWSER_DEFAULTS"


def fixture_path(fixtures_dir: Path | None = None) -> Path:
    """The dump the lab reads: `fixtures_dir`'s when one is given, else
    $SOUS_CHEF_BROWSER_DEFAULTS when set (the pipeline's enginebundle dumps
    the defaults of the vendored engine snapshot there, DESIGN §6.2), else
    the committed fixture."""
    if fixtures_dir is None and os.environ.get(ENV_OVERRIDE):
        return Path(os.environ[ENV_OVERRIDE])
    return Path(fixtures_dir or FIXTURES_DIR) / FIXTURE


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load(fixtures_dir: Path | None = None) -> dict:
    return _load(str(fixture_path(fixtures_dir)))


def page_sections(template: Any, page: Any, fixtures_dir: Path | None = None) -> list[dict]:
    """`getDefaultPageSections(template, page) || []` (a fresh copy)."""
    pages = load(fixtures_dir)["pages"]
    by_page = pages.get(str(template or "")) or pages.get(OTHER_TEMPLATE) or {}
    rows = by_page.get(str(page or ""))
    return copy.deepcopy(rows) if isinstance(rows, list) else []


def faq_categories(template: Any, fixtures_dir: Path | None = None) -> list[dict] | None:
    """PreviewPage.jsx `getDefaultFaqForTemplate(template)` for a basic
    template (a fresh copy); None for any other template (not dumped)."""
    rows = load(fixtures_dir)["faqCategories"].get(str(template or ""))
    return copy.deepcopy(rows) if isinstance(rows, list) else None


# ── regeneration (needs node + the cookie-cutter checkout) ──────────────────

def cookie_cutter_dir(explicit: str | Path | None = None) -> Path | None:
    from .engine import DEFAULT_COOKIE_CUTTER_DIR
    for c in (explicit, os.environ.get("COOKIE_CUTTER_DIR"), DEFAULT_COOKIE_CUTTER_DIR):
        if c and (Path(c) / CLIENT_REL / "vite.config.js").is_file():
            return Path(c).resolve()
    return None


def source_identity(cc_dir: Path) -> dict:
    out: dict[str, Any] = {"files": {}}
    for rel in SOURCES:
        p = cc_dir / rel
        out["files"][rel] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    snap = cc_dir / "SNAPSHOT_OF"
    commit = snap.read_text(encoding="utf-8").strip() if snap.is_file() else None
    if not commit and shutil.which("git"):
        try:
            r = subprocess.run(["git", "-C", str(cc_dir), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20, check=False)
            commit = r.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            commit = None
    out["commit"] = commit
    return out


def generate(cc_dir: str | Path | None = None, node: str = "node", timeout: float = 300) -> dict:
    """Runs js/browser_defaults.mjs against the checkout and returns the
    fixture document (`source`, `pages`, `faqCategories`, `templates`)."""
    root = cookie_cutter_dir(cc_dir)
    if root is None:
        raise RuntimeError("no cookie-cutter checkout with launcher/client/vite.config.js (set COOKIE_CUTTER_DIR)")
    exe = shutil.which(node)
    if not exe:
        raise RuntimeError(f"{node} not on PATH")
    with tempfile.TemporaryDirectory(prefix="souschef-bd-") as td:
        pages_file = Path(td) / "pages.json"
        out_file = Path(td) / "out.json"
        pages_file.write_text(json.dumps(list(PAGES)), encoding="utf-8")
        r = subprocess.run([exe, str(SCRIPT), str(root / CLIENT_REL), str(pages_file), str(out_file)], cwd=str(root / CLIENT_REL),
                           capture_output=True, text=True, timeout=timeout, check=False)
        if r.returncode != 0:
            raise RuntimeError(f"browser_defaults.mjs exited {r.returncode}: {r.stderr.strip()[-2000:]}")
        dump = json.loads(out_file.read_text(encoding="utf-8"))
    return {
        "what": "launcher defaults for a never-saved site: pages[template][page] = getDefaultPageSections(template, page) "
                "(null = none, the browser then uses []), '*' = a template outside its map; faqCategories[template] = "
                "PreviewPage.jsx getDefaultFaqForTemplate. Regenerate: make -C lab browser-defaults (souschef_lab.browser_defaults --write)",
        "source": source_identity(root),
        "pages_dumped": list(PAGES),
        "templates": dump["templates"],
        "pages": dump["pages"],
        "faqCategories": dump["faqCategories"],
    }


def browser_contexts(cases: list[dict], cc_dir: str | Path | None = None, node: str = "node", timeout: float = 300) -> list[dict]:
    """`turnContext.buildTurnContext(state, ctx, extras)` — the BROWSER's
    context — for each `{state, ctx, extras}` (one node boot for all)."""
    root = cookie_cutter_dir(cc_dir)
    exe = shutil.which(node)
    if root is None or not exe:
        raise RuntimeError("needs node and a cookie-cutter checkout (COOKIE_CUTTER_DIR)")
    with tempfile.TemporaryDirectory(prefix="souschef-bc-") as td:
        cases_file, out_file = Path(td) / "cases.json", Path(td) / "out.json"
        cases_file.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
        r = subprocess.run([exe, str(SCRIPT), "--contexts", str(root / CLIENT_REL), str(cases_file), str(out_file)],
                           cwd=str(root / CLIENT_REL), capture_output=True, text=True, timeout=timeout, check=False)
        if r.returncode != 0:
            raise RuntimeError(f"browser_defaults.mjs --contexts exited {r.returncode}: {r.stderr.strip()[-2000:]}")
        return json.loads(out_file.read_text(encoding="utf-8"))


def _stable(v: Any) -> Any:
    """The dump minus what pageSections.jsx draws at random on every import:
    an `images` list of strings is compared by its length (pluto's
    location-details gallery samples its six images afresh on each load,
    measured 2026-10-10 on e7e90106)."""
    if isinstance(v, dict):
        return {k: (len(x) if k == "images" and isinstance(x, list) and all(isinstance(i, str) for i in x) else _stable(x))
                for k, x in v.items()}
    if isinstance(v, list):
        return [_stable(x) for x in v]
    return v


def same_content(a: dict, b: dict) -> bool:
    """Equal defaults (the source identity may differ — an unrelated edit to
    PreviewPage.jsx moves its hash without moving a default; random image
    draws are not compared, `_stable`)."""
    keys = ("templates", "pages", "faqCategories", "pages_dumped")
    return all(_stable(a.get(k)) == _stable(b.get(k)) for k in keys)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="souschef_lab.browser_defaults", description=__doc__.split("\n\n")[0])
    ap.add_argument("--cookie-cutter", help="the checkout (default $COOKIE_CUTTER_DIR, then the sibling checkout)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true", help=f"regenerate fixtures/{FIXTURE}")
    g.add_argument("--check", action="store_true", help="regenerate in memory and compare with the fixture (exit 1 on a difference)")
    a = ap.parse_args(argv)
    doc = generate(a.cookie_cutter)
    path = fixture_path()
    if a.write:
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        _load.cache_clear()
        print(f"browser-defaults: wrote {path} (commit {doc['source']['commit']})")
        return 0
    old = json.loads(path.read_text(encoding="utf-8"))
    if same_content(old, doc):
        print(f"browser-defaults: {path.name} matches {doc['source']['commit']}")
        return 0
    diff = [f"{t}/{p}" for t in doc["pages"] for p in doc["pages"][t]
            if _stable((old.get("pages") or {}).get(t, {}).get(p)) != _stable(doc["pages"][t][p])]
    faq = [t for t in doc["faqCategories"] if (old.get("faqCategories") or {}).get(t) != doc["faqCategories"][t]]
    print(f"browser-defaults: {path.name} is STALE against {doc['source']['commit']}: pages {diff[:20]}{' …' if len(diff) > 20 else ''}; faq {faq}")
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
