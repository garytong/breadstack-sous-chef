"""fields — the service's candidate-fields block, for the lab (docs/LAB.md §5
"Candidate fields").

A route with `fields_context` on (the default for every route kind since
2026-10-10 — before, only for `messages-compatible`, the student's; README §5) gets a `<candidate_fields>` block in the user turn,
between the context and the merchant's text. The block is computed by ONE
implementation, the Go package `service/internal/fields`; the lab never ports
it. This module builds the package's CLI (`service/cmd/sous-chef-fields`) once
per process into `lab/.cache/fields/` and shells out to it (`-jsonl`: one
process, the manifest parsed once), so a training row carries the very bytes
the service puts in the served turn. `render.context_block(context, text,
candidates)` places the block where `prompt.UserText` does.

    souschef-lab fields rerender --data data/merged-r123/dataset --manifest data/merged-r123/manifest.json \\
                                 --out data/merged-r123-fc/dataset
        re-render built train/val rows (dataset.to_hf's shape) with the block — or without it (--fields-context off)
        --system index | prefix  also re-renders system[1] (index) or system[0] + system[1] + tools (prefix) with the
                                 service's own code (service/cmd/sous-chef-prefix; SOUS_CHEF_PREFIX_BIN overrides)
    souschef-lab fields block --manifest M --context ctx.json --text "…"     one block, as the service would send it
    souschef-lab fields bin                                                   build the CLI, print its path

Needs `go` on PATH (or SOUS_CHEF_FIELDS_BIN = a built sous-chef-fields). The
tests that need it skip without one, as the serving-parity tests do
(SOUS_CHEF_LAB_REQUIRE_GO=1 makes them fail instead).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Iterable

from . import render

LAB_DIR = Path(__file__).resolve().parent.parent
SERVICE_DIR = LAB_DIR.parent / "service"
CACHE_DIR = Path(os.environ.get("SOUS_CHEF_LAB_CACHE") or LAB_DIR / ".cache") / "fields"
BIN_ENV = "SOUS_CHEF_FIELDS_BIN"
BIN_NAME = "sous-chef-fields"
PACKAGE = "./cmd/sous-chef-fields"
PREFIX_BIN_ENV = "SOUS_CHEF_PREFIX_BIN"
PREFIX_BIN_NAME = "sous-chef-prefix"
PREFIX_PACKAGE = "./cmd/sous-chef-prefix"
# What system[0]/system[1]/tools depend on besides the manifest (tests excluded).
PREFIX_SOURCE_DIRS = ("internal/prompt", "internal/toolgen", "internal/manifest", "internal/registry", "cmd/sous-chef-prefix")
# system[1] (prompt.indexHeader) starts with this line; a row's system turn is system[0] + system[1] joined with nothing.
INDEX_HEAD = "# Manifest index\n"
SYSTEM_MODES = ("keep", "index", "prefix")
# What the block's bytes depend on besides the manifest: these sources (tests excluded).
SOURCE_DIRS = ("internal/fields", "cmd/sous-chef-fields")
OPEN_TAG, CLOSE_TAG = "<candidate_fields>\n", "\n</candidate_fields>\n"
CONTEXT_OPEN, CONTEXT_CLOSE = "<context>\n", "\n</context>\n"
MANIFEST_NAME = "dataset-manifest.json"
SPLITS = ("train.jsonl", "val.jsonl")

_lock = threading.Lock()
_built: dict[str, Path] = {}


class FieldsError(RuntimeError):
    """The CLI is missing, could not be built, or failed."""


# --- the binary ---------------------------------------------------------------------------------------------------------

def available(service: Path = SERVICE_DIR) -> bool:
    """Whether a block can be computed here: SOUS_CHEF_FIELDS_BIN names a
    file, or `go` is on PATH and the service's sources are beside the lab."""
    env = os.environ.get(BIN_ENV)
    if env:
        return Path(env).is_file()
    return shutil.which("go") is not None and (service / "cmd" / BIN_NAME / "main.go").is_file()


def source_sha(service: Path = SERVICE_DIR, dirs: Iterable[str] = SOURCE_DIRS) -> str | None:
    """sha256 over the retrieval's Go sources (internal/fields and the CLI,
    tests excluded) and the files they embed (internal/fields/vocab.txt):
    the identity recorded beside a dataset, so two builds made with
    different retrieval code — or vocabulary — are distinguishable. None
    when the sources are not here (SOUS_CHEF_FIELDS_BIN on a machine
    without them). `dirs` = PREFIX_SOURCE_DIRS for the prefix's identity."""
    h = hashlib.sha256()
    found = False
    for d in dirs:
        for p in sorted([*(service / d).glob("*.go"), *(service / d).glob("*.txt")]):
            if p.name.endswith("_test.go"):
                continue
            found = True
            h.update(f"{d}/{p.name}\0".encode("utf-8"))
            h.update(p.read_bytes())
            h.update(b"\0")
    return h.hexdigest() if found else None


def binary(service: Path = SERVICE_DIR, cache: Path = CACHE_DIR) -> Path:
    """The built CLI: SOUS_CHEF_FIELDS_BIN when set, else `go build` into the
    cache dir — once per process (Go's own build cache makes a rebuild of
    unchanged sources a no-op, so a stale binary is never reused across
    processes). Raises FieldsError when it cannot be had."""
    return _go_binary(PACKAGE, BIN_NAME, BIN_ENV, service, cache)


def prefix_binary(service: Path = SERVICE_DIR, cache: Path = CACHE_DIR) -> Path:
    """The service's prefix CLI (service/cmd/sous-chef-prefix: system[0],
    system[1] and the tools exactly as registry.Compile builds them), built
    like `binary()`; SOUS_CHEF_PREFIX_BIN overrides."""
    return _go_binary(PREFIX_PACKAGE, PREFIX_BIN_NAME, PREFIX_BIN_ENV, service, cache)


def _go_binary(package: str, name: str, env_name: str, service: Path, cache: Path) -> Path:
    env = os.environ.get(env_name)
    if env:
        if not Path(env).is_file():
            raise FieldsError(f"{env_name}={env} is not a file")
        return Path(env)
    key = f"{service.resolve()}::{package}"
    with _lock:
        if key in _built and _built[key].is_file():
            return _built[key]
        if shutil.which("go") is None:
            raise FieldsError(f"`go` is not on PATH (and {env_name} is unset): {name} is the service's own Go code "
                              f"({package})")
        if not (service / package / "main.go").is_file():
            raise FieldsError(f"{service / package} is missing: the lab needs the service's sources beside it")
        cache.mkdir(parents=True, exist_ok=True)
        exe = cache / name
        tmp = cache / f".{name}.{os.getpid()}.tmp"
        env_vars = dict(os.environ)
        env_vars.setdefault("GOTOOLCHAIN", "auto")
        p = subprocess.run(["go", "build", "-o", str(tmp), package], cwd=service, capture_output=True, text=True,
                           env=env_vars, check=False)
        if p.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise FieldsError(f"go build {package} failed:\n{p.stderr.strip()[:2000]}")
        os.replace(tmp, exe)
        _built[key] = exe
        return exe


# --- blocks -------------------------------------------------------------------------------------------------------------

def _key(context: dict | None, text: str) -> str:
    return render.compact_json([context, text])


class Fields:
    """The candidate-fields blocks of turns on one manifest.

    `block(context, text)` is `fields.Candidates(...).Block` for the turn the
    service would see: the context object the browser sent and the TRIMMED
    merchant text (loop.candidateFields trims before the retrieval). "" when
    nothing matches — and, as in the service, when the retrieval cannot read
    the context (the turn then runs without a block). Answers are cached;
    `prime` computes many in one process."""

    def __init__(self, manifest: Path | str, exe: Path | str | None = None):
        self.manifest = Path(manifest)
        if not self.manifest.is_file():
            raise FieldsError(f"manifest {self.manifest} not found")
        self.exe = Path(exe) if exe else binary()
        self._cache: dict[str, str] = {}
        self._lock = threading.Lock()

    def identity(self) -> dict:
        """What a dataset records about the blocks it carries."""
        return {"cli": PACKAGE.removeprefix("./"), "source_sha256": source_sha(),
                "manifest_file": str(self.manifest),
                "manifest_file_sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest()}

    def _run(self, turns: list[tuple[dict | None, str]]) -> list[str]:
        lines = []
        for context, text in turns:
            row: dict = {"text": text}
            if context is not None:
                row["context"] = context
            lines.append(render.compact_json(row) + "\n")
        p = subprocess.run([str(self.exe), "-manifest", str(self.manifest), "-jsonl"], input="".join(lines),
                           capture_output=True, text=True, encoding="utf-8", check=False)
        if p.returncode != 0:
            raise FieldsError(f"{BIN_NAME} failed: {p.stderr.strip()[:500]}")
        out = [json.loads(x)["block"] for x in p.stdout.splitlines() if x.strip()]
        if len(out) != len(turns):
            raise FieldsError(f"{BIN_NAME} answered {len(out)} of {len(turns)} turns")
        return out

    def prime(self, turns: Iterable[tuple[dict | None, str]]) -> None:
        """Compute every turn not yet cached, in one process. A turn whose
        context the retrieval refuses gets "" (the service logs it and runs
        the turn without a block); a manifest the CLI refuses raises."""
        todo: dict[str, tuple[dict | None, str]] = {}
        for context, text in turns:
            text = render.go_trim_space(text or "")
            k = _key(context, text)
            if k not in self._cache and k not in todo:
                todo[k] = (context, text)
        if not todo:
            return
        keys, items = list(todo), list(todo.values())
        try:
            blocks = self._run(items)
        except FieldsError:
            if len(items) == 1:
                self._run([(None, "probe")])  # the manifest itself is refused → raise; else this one context is bad
                blocks = [""]
            else:
                blocks = []
                for item in items:  # find the bad turn(s); the rest keep their blocks
                    try:
                        blocks += self._run([item])
                    except FieldsError:
                        self._run([(None, "probe")])
                        blocks.append("")
        with self._lock:
            self._cache.update(zip(keys, blocks))

    def block(self, context: dict | None, text: str) -> str:
        text = render.go_trim_space(text or "")
        k = _key(context, text)
        if k not in self._cache:
            self.prime([(context, text)])
        return self._cache[k]


# --- the prefix (system[0], system[1], tools) --------------------------------------------------------------------------

class Prefix:
    """The service's prompt prefix for one manifest, from its own code
    (service/cmd/sous-chef-prefix → registry.Compile): `rules` (system[0]),
    `index` (system[1]), `tools` (the wire tool definitions) and the
    service's `manifest_hash`."""

    def __init__(self, manifest: Path | str, exe: Path | str | None = None):
        self.manifest = Path(manifest)
        if not self.manifest.is_file():
            raise FieldsError(f"manifest {self.manifest} not found")
        self.exe = Path(exe) if exe else prefix_binary()
        p = subprocess.run([str(self.exe), "-manifest", str(self.manifest)], capture_output=True, text=True,
                           encoding="utf-8", check=False)
        if p.returncode != 0:
            raise FieldsError(f"{PREFIX_BIN_NAME} failed: {p.stderr.strip()[:500]}")
        doc = json.loads(p.stdout)
        self.manifest_hash = str(doc["manifest_hash"])
        self.rules, self.index = (str(x) for x in doc["system"])
        self.tools = list(doc["tools"])

    def identity(self) -> dict:
        """What a dataset records about the system turn / tools it carries."""
        sha = lambda b: hashlib.sha256(b.encode("utf-8")).hexdigest()  # noqa: E731
        return {"cli": PREFIX_PACKAGE.removeprefix("./"), "source_sha256": source_sha(dirs=PREFIX_SOURCE_DIRS),
                "manifest_file": str(self.manifest), "manifest_hash": self.manifest_hash,
                "system0_sha256": sha(self.rules), "system1_sha256": sha(self.index),
                "tools_sha256": sha(render.compact_json(self.tools))}


def split_system(content: str) -> tuple[str, str]:
    """`(rules, index)` of a row's system turn (system[0] + system[1] joined
    with nothing, dataset.join_system): the index starts at its header line."""
    i = content.find(INDEX_HEAD)
    if i < 0 or content.find(INDEX_HEAD, i + 1) >= 0:
        raise ValueError("a system turn without exactly one manifest index header")
    return content[:i], content[i:]


def resystem_row(row: dict, prefix: Prefix, mode: str) -> tuple[bool, bool, bool]:
    """Re-render `row`'s system turn in place: `index` replaces system[1]
    only (the rules the row was collected with stay), `prefix` replaces
    system[0] and system[1] and the tools. Returns (rules, index, tools)
    changed."""
    from .dataset import hf_tools  # lazy: dataset imports this module

    if mode not in SYSTEM_MODES:
        raise ValueError(f"system mode must be one of {SYSTEM_MODES}")
    if mode == "keep":
        return False, False, False
    msgs = row.get("messages") or []
    if not msgs or msgs[0].get("role") != "system":
        raise ValueError(f"row {row.get('id')}: no system turn")
    rules, index = split_system(str(msgs[0]["content"]))
    new_rules = prefix.rules if mode == "prefix" else rules
    msgs[0]["content"] = new_rules + prefix.index
    tools_changed = False
    if mode == "prefix":
        new_tools = hf_tools(prefix.tools)
        tools_changed = new_tools != row.get("tools")
        row["tools"] = new_tools
    row.setdefault("meta", {})["system_rerendered"] = mode
    return new_rules != rules, prefix.index != index, tools_changed


# --- a rendered user turn, taken apart ----------------------------------------------------------------------------------

def split_user_text(content: str) -> tuple[str, dict | None, str, str]:
    """`(head, context, candidates, text)` of a user turn rendered by
    render.context_block / prompt.UserText: `head` is the literal
    `<context>\n{…}\n</context>\n` ("" without a context) — kept as bytes, so
    a re-render never re-encodes it — `context` the object in it, `candidates`
    the block (no trailing newline, "" when absent) and `text` the merchant's.
    `head + (candidates + "\n" if candidates) + text` is the turn again."""
    context: dict | None = None
    head, rest = "", content
    if content.startswith(CONTEXT_OPEN):
        end = content.find(CONTEXT_CLOSE, len(CONTEXT_OPEN))
        if end < 0:
            raise ValueError("a user turn opens <context> and never closes it")
        raw = content[len(CONTEXT_OPEN):end]
        if "\n" in raw:
            raise ValueError("the <context> of a user turn is not one line of compact JSON")
        context = json.loads(raw)
        if not isinstance(context, dict):
            raise ValueError("the <context> of a user turn is not a JSON object")
        head, rest = content[:end + len(CONTEXT_CLOSE)], content[end + len(CONTEXT_CLOSE):]
    candidates = ""
    if rest.startswith(OPEN_TAG):
        end = rest.find(CLOSE_TAG)
        if end < 0:
            raise ValueError("a user turn opens <candidate_fields> and never closes it")
        candidates = rest[:end + len(CLOSE_TAG) - 1]
        rest = rest[end + len(CLOSE_TAG):]
    return head, context, candidates, rest


def join_user_text(head: str, candidates: str, text: str) -> str:
    """The inverse of split_user_text (prompt.UserText's layout)."""
    candidates = render.go_trim_space(candidates)
    return head + (candidates + "\n" if candidates else "") + text


def has_block(content: str) -> bool:
    """Whether a rendered user turn carries a candidate-fields block."""
    try:
        return bool(split_user_text(str(content or ""))[2])
    except ValueError:
        return False


def first_user(hf: dict) -> dict | None:
    return next((m for m in hf.get("messages") or [] if m.get("role") == "user"), None)


def block_stats(rows: Iterable[dict]) -> dict:
    """How many built rows (dataset.to_hf's shape) carry the block in their
    first user turn — what train.py records in run.json."""
    n = with_block = 0
    for r in rows:
        n += 1
        u = first_user(r)
        with_block += bool(u and has_block(u.get("content")))
    return {"rows": n, "with_block": with_block}


# --- re-rendering a built dataset -----------------------------------------------------------------------------------------

def rerender_rows(rows: list[dict], fields: Fields | None, prefix: Prefix | None = None,
                  system: str = "keep") -> tuple[list[dict], dict]:
    """The rows with every first user turn re-rendered: its block replaced
    by the one `fields` computes now, or removed (`fields` None). With
    `prefix` and `system` index / prefix, the system turn (and for prefix
    the tools) too (`resystem_row`). Nothing else in a row changes but
    `meta.fields_context` / `fields_block_chars` / `system_rerendered`."""
    if system != "keep" and prefix is None:
        raise ValueError(f"system {system!r} needs a Prefix")
    parts = []
    for r in rows:
        u = first_user(r)
        if u is None:
            raise ValueError(f"row {r.get('id')}: no user turn")
        parts.append(split_user_text(str(u["content"])))
    if fields is not None:
        fields.prime((ctx, text) for _, ctx, _, text in parts)
    out, changed, with_block, chars = [], 0, 0, []
    sys_changed = {"rules": 0, "index": 0, "tools": 0, "any": 0}
    for r, (head, ctx, old, text) in zip(rows, parts):
        new = fields.block(ctx, text) if fields is not None else ""
        row = json.loads(json.dumps(r))
        u = first_user(row)
        u["content"] = join_user_text(head, new, text)
        meta = row.setdefault("meta", {})
        meta["fields_context"] = fields is not None
        meta["fields_block_chars"] = len(new)
        changed += new != old
        with_block += bool(new)
        if new:
            chars.append(len(new))
        if prefix is not None and system != "keep":
            flags = resystem_row(row, prefix, system)
            for k, f in zip(("rules", "index", "tools"), flags):
                sys_changed[k] += f
            sys_changed["any"] += any(flags)
        out.append(row)
    chars.sort()
    stats = {"rows": len(rows), "with_block": with_block, "changed": changed,
             "block_chars": {"p50": chars[len(chars) // 2], "p95": chars[min(len(chars) - 1, int(len(chars) * 0.95))],
                             "max": chars[-1]} if chars else None}
    if system != "keep":
        stats["system"] = {"mode": system, "changed": sys_changed}
    return out, stats


def manifest_file_hash(path: Path) -> str:
    """The service's hash of a manifest file (collect.manifest_hash) — what
    cases and rows carry as `manifest_hash`."""
    from .collect import manifest_hash  # lazy: collect imports this module lazily too

    try:
        return manifest_hash(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as e:
        raise FieldsError(f"manifest {path} cannot be read: {e}") from e


def check_manifest(fields: Fields, rows: Iterable[dict], declared: Iterable[str], where: str) -> dict:
    """Refuse a manifest that is not the rows' own: every row carries
    `meta.manifest_hash` (dataset.to_hf), and a block computed on another
    manifest lists that manifest's keys. `declared` = the hashes the dataset
    manifest names (`manifest_hashes`), checked too. Returns `{manifest_hash,
    rows_checked}`; rows without a hash (nothing built by this lab) cannot be
    checked and are counted, not refused."""
    have = manifest_file_hash(fields.manifest)
    want: dict[str, int] = {}
    checked = 0
    for r in rows:
        h = (r.get("meta") or {}).get("manifest_hash")
        if h:
            checked += 1
            if str(h) != have:
                want[str(h)] = want.get(str(h), 0) + 1
    for h in declared:
        if h and str(h) != have:
            want.setdefault(str(h), 0)
    if want:
        others = ", ".join(f"{h[:12]}… ({n} rows)" if n else f"{h[:12]}… ({MANIFEST_NAME})" for h, n in sorted(want.items()))
        raise FieldsError(f"{fields.manifest} hashes to {have[:12]}… but the rows of {where} were built on {others}: "
                          f"the blocks would list another manifest's keys — pass the dataset's own manifest with "
                          f"--manifest (nothing was written)")
    return {"manifest_hash": have, "rows_checked": checked}


def rerender(data: Path, out: Path, fields: Fields | None, prefix: Prefix | None = None, system: str = "keep") -> dict:
    """`data`/train.jsonl and val.jsonl re-rendered into `out`; every other
    file of `data` is copied as is, and dataset-manifest.json gains (or is
    created with) `fields_context` and `rerendered_from` (and `system_prompt`
    when the system turn is re-rendered too: `system` index / prefix). With
    `fields` or `prefix`, its manifest must be the rows' (`check_manifest`),
    else nothing is written."""
    data, out = Path(data), Path(out)
    if out.resolve() == data.resolve():
        raise FieldsError("--out must be another folder than --data (the source rows are kept)")
    if not any((data / s).is_file() for s in SPLITS):
        raise FieldsError(f"no train.jsonl / val.jsonl in {data}")
    try:
        manifest = json.loads((data / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest = {}
    if not isinstance(manifest, dict):
        manifest = {}
    splits = {s: [json.loads(line) for line in (data / s).read_text(encoding="utf-8").splitlines() if line.strip()]
              for s in SPLITS if (data / s).is_file()}
    declared = [manifest.get("manifest_hash"), *(manifest.get("manifest_hashes") or [])]
    checked = (check_manifest(fields, (r for rows in splits.values() for r in rows), [h for h in declared if h],
                              str(data)) if fields is not None else None)
    if system != "keep":
        if prefix is None:
            raise FieldsError(f"--system {system} needs the manifest's prefix")
        check_manifest(prefix, (r for rows in splits.values() for r in rows), [h for h in declared if h], str(data))
        if prefix.manifest_hash != manifest_file_hash(prefix.manifest):
            raise FieldsError(f"the service hashes {prefix.manifest} to {prefix.manifest_hash[:12]}…, the lab otherwise")
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"data": str(data), "out": str(out), "splits": {}}
    for p in sorted(data.iterdir()):
        if not p.is_file():
            continue
        if p.name in SPLITS:
            rows = splits[p.name]
            new, stats = rerender_rows(rows, fields, prefix, system)
            with (out / p.name).open("w", encoding="utf-8") as f:
                for r in new:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            report["splits"][p.name] = stats
        elif p.name != MANIFEST_NAME:
            shutil.copyfile(p, out / p.name)
    total = sum(s["rows"] for s in report["splits"].values())
    with_block = sum(s["with_block"] for s in report["splits"].values())
    manifest["fields_context"] = {"enabled": fields is not None, **(fields.identity() if fields is not None else {}),
                                  **(checked or {}), "rows": total, "rows_with_block": with_block}
    manifest["rerendered_from"] = str(data)
    if system != "keep":
        changed: dict[str, int] = {}
        for st in report["splits"].values():
            for k, v in st["system"]["changed"].items():
                changed[k] = changed.get(k, 0) + v
        manifest["system_prompt"] = {"rerendered": system, **prefix.identity(), "rows_changed": changed}
        report["system_prompt"] = manifest["system_prompt"]
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    report["fields_context"] = manifest["fields_context"]
    return report


# --- CLI ------------------------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="souschef-lab fields", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rerender", help="re-render a built dataset's user turns with (or without) the block")
    r.add_argument("--data", type=Path, required=True, help="folder with train.jsonl / val.jsonl (dataset.py's output)")
    r.add_argument("--out", type=Path, required=True, help="output folder (never the same as --data)")
    r.add_argument("--manifest", type=Path, help="the manifest the rows' cases were generated from (required unless off)")
    r.add_argument("--fields-context", choices=("on", "off"), default="on", help="off strips the block (default: on)")
    r.add_argument("--system", choices=SYSTEM_MODES, default="keep",
                   help="the system turn: keep (default) = as built; index = system[1] re-rendered by the service's "
                        "code (sous-chef-prefix), the rules kept; prefix = system[0] + system[1] + the tools re-rendered")
    b = sub.add_parser("block", help="print one turn's block")
    b.add_argument("--manifest", type=Path, required=True)
    b.add_argument("--context", type=Path, help="the turn context (JSON object)")
    b.add_argument("--text", required=True)
    sub.add_parser("bin", help="build the CLI and print its path")
    args = p.parse_args(argv)
    try:
        if args.cmd == "bin":
            print(binary())
            return 0
        if args.cmd == "block":
            ctx = json.loads(args.context.read_text(encoding="utf-8")) if args.context else None
            block = Fields(args.manifest).block(ctx, args.text)
            if block:
                print(block)
            return 0
        fields = None
        if args.fields_context == "on":
            if not args.manifest:
                p.error("rerender --fields-context on needs --manifest")
            fields = Fields(args.manifest)
        prefix = None
        if args.system != "keep":
            if not args.manifest:
                p.error(f"rerender --system {args.system} needs --manifest")
            prefix = Prefix(args.manifest)
        report = rerender(args.data, args.out, fields, prefix, args.system)
    except (FieldsError, ValueError) as e:
        print(f"fields: {e}", file=sys.stderr)
        return 1
    for name, s in report["splits"].items():
        c = s["block_chars"] or {}
        print(f"{name}: {s['rows']} rows · {s['with_block']} with a block · {s['changed']} changed"
              + (f" · block chars p50 {c['p50']} / p95 {c['p95']} / max {c['max']}" if c else ""))
    for name, s in report["splits"].items():
        if s.get("system"):
            ch = s["system"]["changed"]
            print(f"{name}: system {s['system']['mode']} · rows changed: rules {ch['rules']} · index {ch['index']} · "
                  f"tools {ch['tools']} · any {ch['any']}")
    fc = report["fields_context"]
    if fields is not None and not fc.get("rows_checked"):
        print(f"warning: no row carries meta.manifest_hash — {args.manifest} could not be checked against the rows",
              file=sys.stderr)
    print(f"wrote {args.out} (fields_context {'on' if fields is not None else 'off'}; {MANIFEST_NAME} records it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
