"""Subprocess wrapper around the browser engine's headless runner
(docs/LAB.md §1 `engine.py`; cookie-cutter
`launcher/client/src/sousChef/ops/headless.mjs`).

The runner applies one batch of ops to an EditorState with a manifest and
prints `{ok, results[{op, ok, error?, warnings, written}], changes[{slice,
written, diff[{path, before, after}]}], claims?}`; exit 0 = every op applied,
2 = at least one failed (JSON still printed), 1 = usage / load error. It
boots Vite's SSR module runner per process (measured here at ~7 s CPU,
~10 s wall on 2026-10-01 — recorded in `EngineResult.elapsed_s`, never
assumed), and it reads ONE request per boot: `--state/--ops` files or a
single JSON document on stdin. There is no stdin stream mode, so this
wrapper runs one boot per call and caches results by request hash (memory
+ optional on-disk cache under `lab/data/engine-cache/`), and `run_many`
fans calls out over a small thread pool (each is its own node process).

Location: `SOUS_CHEF_ENGINE` (the .mjs path) or `COOKIE_CUTTER_DIR` (the
checkout; default `../../breadstack-cookie-cutter` next to this repo).
`node` >= 20 (ESM + `fs.readSync` on fd 0). Nothing here imports anything
from the cookie-cutter repo; it only executes the script.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Iterable

from . import DATA_DIR, FIXTURES_DIR, LAB_DIR

ENGINE_REL_PATH = Path("launcher/client/src/sousChef/ops/headless.mjs")
DEFAULT_COOKIE_CUTTER_DIR = LAB_DIR.parent.parent / "breadstack-cookie-cutter"
MIN_NODE_MAJOR = 20
# golden.test.js CTX: a real BASIC_TEMPLATE_IDS member; `now` makes minted ids deterministic.
GOLDEN_CTX: dict = {"templateId": "editorial", "cdnUrl": "https://cdn.int.breadstack.io", "galleryFolder": "gal", "now": 1700000000000}
SERVER_TOOLS = ("lookup_fields", "describe_target", "report_unsupported")
ENGINE_OPS = ("set_fields", "edit_list", "reset", "restore_default", "set_page_layout", "reset_pinned_section", "pick_asset")


class EngineError(RuntimeError):
    """The runner exited 1 (usage / load error) or produced no JSON."""

    def __init__(self, message: str, *, returncode: int | None = None, stderr: str = ""):
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


@dataclass
class EngineResult:
    ok: bool
    results: list[dict]
    changes: list[dict]
    claims: dict | None = None
    state: dict | None = None
    returncode: int = 0
    elapsed_s: float = 0.0
    cached: bool = False
    stderr: str = ""

    @property
    def errors(self) -> list[dict]:
        return [r.get("error") for r in self.results if not r.get("ok")]

    @property
    def error_codes(self) -> list[str]:
        return [str((r.get("error") or {}).get("code")) for r in self.results if not r.get("ok")]

    @property
    def warnings(self) -> list[str]:
        out: list[str] = []
        for r in self.results:
            out.extend(r.get("warnings") or [])
        return out

    @property
    def written(self) -> list[str]:
        out: list[str] = []
        for r in self.results:
            out.extend(r.get("written") or [])
        return out

    def diff(self) -> list[dict]:
        """Every leaf across the touched slices: `[{path, before?, after?}]`."""
        out: list[dict] = []
        for c in self.changes:
            out.extend(c.get("diff") or [])
        return out

    def diff_set(self) -> set[tuple[str, str]]:
        """`{(path, canonical-json(after))}`; a removed leaf has after=None."""
        return {(d["path"], canonical(d.get("after"))) for d in self.diff()}

    def to_dict(self) -> dict:
        d = {"ok": self.ok, "results": self.results, "changes": self.changes}
        if self.claims is not None:
            d["claims"] = self.claims
        if self.state is not None:
            d["state"] = self.state
        return d

    @classmethod
    def from_output(cls, out: dict, *, returncode: int, elapsed_s: float, cached: bool = False, stderr: str = "") -> "EngineResult":
        return cls(ok=bool(out.get("ok")), results=list(out.get("results") or []), changes=list(out.get("changes") or []),
                   claims=out.get("claims"), state=out.get("state"), returncode=returncode, elapsed_s=elapsed_s,
                   cached=cached, stderr=stderr)


def canonical(v: Any) -> str:
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def request_hash(state: dict, ops: list[dict], ctx: dict | None, manifest_hash: str, engine_fingerprint: str = "") -> str:
    """The cache key: the request, the manifest and — when known — the engine
    that answers it, so a cached diff is never reused across engine versions."""
    h = hashlib.sha256()
    key = {"state": state, "ops": ops, "ctx": ctx, "manifest": manifest_hash}
    if engine_fingerprint:
        key["engine"] = engine_fingerprint
    h.update(canonical(key).encode("utf-8"))
    return h.hexdigest()


# ── engine identity ─────────────────────────────────────────────────────────

FINGERPRINT_SUFFIXES = (".js", ".mjs")
FINGERPRINT_SKIP = (".test.js", ".test.mjs")  # tests do not change what the engine writes
RESOLVE_SUFFIXES = (".js", ".jsx", ".mjs", ".ts", ".tsx")
INDEX_NAMES = tuple(f"index{s}" for s in RESOLVE_SUFFIXES)
WORKSPACE_SCOPE = "@breadstack/"  # workspace packages: not walked, their package.json version is hashed instead
# `import … from 'x'` / `export … from 'x'` / `import 'x'` at a statement start (multi-line import lists
# included: the char class crosses newlines and stops at the first quote), and `import('x')` anywhere —
# which also catches Vite's `runner.import('/src/…')` in headless.mjs, a path from the client root.
_STATIC_IMPORT_RE = re.compile(r"""^\s*(?:import|export)\b[^'"]*?\bfrom\s*['"]([^'"]+)['"]|^\s*import\s*['"]([^'"]+)['"]""",
                               re.MULTILINE)
_DYNAMIC_IMPORT_RE = re.compile(r"""\bimport\(\s*['"]([^'"]+)['"]\s*\)""")


def import_specifiers(text: str) -> list[str]:
    """Every module specifier a JS/JSX source imports statically or with a
    literal ``import('…')``, in order of appearance, duplicates kept."""
    out: list[str] = []
    for m in _STATIC_IMPORT_RE.finditer(text):
        out.append(m.group(1) or m.group(2))
    out += _DYNAMIC_IMPORT_RE.findall(text)
    return out


def resolve_import(spec: str, from_file: Path, client_root: Path) -> Path | None:
    """A relative (``./``, ``../``) or client-root-absolute (``/src/…``)
    specifier resolved to a file: as is, with a source suffix, or a
    directory's ``index.*``. Bare package names (``vite``, ``react``,
    ``node:fs``) give None."""
    if spec.startswith(("./", "../")):
        base = (from_file.parent / spec)
    elif spec.startswith("/"):
        base = client_root / spec.lstrip("/")
    else:
        return None
    candidates = [base, *(base.with_name(base.name + s) for s in RESOLVE_SUFFIXES), *(base / i for i in INDEX_NAMES)]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def workspace_package_version(spec: str, start: Path) -> str | None:
    """``<name>@<version>`` of a ``@breadstack/*`` package as the nearest
    ``node_modules`` above ``start`` resolves it (a workspace symlink is
    followed); None when it cannot be found."""
    parts = spec.split("/")
    name = "/".join(parts[:2]) if spec.startswith("@") else parts[0]
    for d in [start, *start.parents]:
        pkg = d / "node_modules" / name / "package.json"
        if pkg.is_file():
            try:
                version = json.loads(pkg.read_text(encoding="utf-8")).get("version")
            except (OSError, ValueError):
                version = None
            return f"{name}@{version or '?'}"
    return None


def import_closure(script: Path) -> tuple[list[Path], list[str]]:
    """The STATIC IMPORT CLOSURE of ``headless.mjs``: every source file the
    runner reaches through relative / client-root specifiers, walked
    recursively (``ops/*.js`` → ``../../data/pageSections.jsx``,
    ``../../components/basicEditor/constants.jsx``, ``../../utils/…``,
    ``../../lib/cdnUrl.js``, …), plus the ``@breadstack/*`` workspace
    packages met on the way as ``name@version`` (their trees are not walked).
    Returns ``(files, packages)`` sorted."""
    client_root = script.parents[3] if len(script.parents) > 3 else script.parent  # ops → sousChef → src → client
    seen: set[Path] = set()
    packages: set[str] = set()
    todo = [script.resolve()]
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for spec in import_specifiers(text):
            if spec.startswith(WORKSPACE_SCOPE):
                v = workspace_package_version(spec, f.parent)
                if v:
                    packages.add(v)
                continue
            target = resolve_import(spec, f, client_root)
            if target is not None and target not in seen and not target.name.endswith(FINGERPRINT_SKIP):
                todo.append(target)
    return sorted(seen), sorted(packages)


def git_head(repo: str | Path) -> str | None:
    """The checkout's HEAD commit, None when git or the repo is unavailable."""
    exe = shutil.which("git")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20,
                             check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", sha) else None


def engine_fingerprint(path: str | Path | None = None) -> dict | None:
    """Who the engine is (LAB.md §2): ``{fingerprint, files, packages,
    commit, dir}`` where ``fingerprint`` is sha256 over the name and content
    of every source in the static import closure of ``headless.mjs``
    (``import_closure``: the ``ops/*.js`` modules AND what they import from
    outside ``ops/`` — ``src/data/pageSections.jsx``,
    ``src/components/basicEditor/constants.jsx``, …) together with every
    other ``ops/*.js`` / ``*.mjs`` source (tests excluded), the
    ``@breadstack/*`` package versions met, and the cookie-cutter HEAD commit
    when git can tell it. ``files`` are paths relative to the client root.
    Stamped on cases, collector rows, dataset rows and manifests, eval runs
    and verify results, and part of the engine cache key. None when no
    engine is found."""
    script = find_engine(path)
    if script is None:
        return None
    ops_dir = script.parent
    client_root = script.parents[3] if len(script.parents) > 3 else ops_dir
    closure, packages = import_closure(script)
    files = sorted({*closure, *(p.resolve() for p in ops_dir.iterdir()
                                if p.is_file() and p.suffix in FINGERPRINT_SUFFIXES and not p.name.endswith(FINGERPRINT_SKIP))})

    def rel(p: Path) -> str:
        try:
            return p.relative_to(client_root).as_posix()
        except ValueError:
            return p.as_posix()

    h = hashlib.sha256()
    names = [rel(p) for p in files]
    for name, p in zip(names, files):
        h.update(f"{name}\0".encode("utf-8"))
        h.update(hashlib.sha256(p.read_bytes()).hexdigest().encode("ascii"))
        h.update(b"\n")
    for pkg in packages:
        h.update(f"package {pkg}\n".encode("utf-8"))
    commit = git_head(script.parents[5]) if len(script.parents) > 5 else None  # ops → sousChef → src → client → launcher → repo
    if commit:
        h.update(f"HEAD {commit}\n".encode("ascii"))
    return {"fingerprint": h.hexdigest(), "files": names, "packages": packages, "commit": commit, "dir": str(ops_dir)}


# ── discovery ───────────────────────────────────────────────────────────────

def find_engine(explicit: str | Path | None = None) -> Path | None:
    """The headless.mjs path: explicit → $SOUS_CHEF_ENGINE → $COOKIE_CUTTER_DIR → the sibling checkout."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("SOUS_CHEF_ENGINE"):
        candidates.append(Path(os.environ["SOUS_CHEF_ENGINE"]))
    if os.environ.get("COOKIE_CUTTER_DIR"):
        candidates.append(Path(os.environ["COOKIE_CUTTER_DIR"]) / ENGINE_REL_PATH)
    candidates.append(DEFAULT_COOKIE_CUTTER_DIR / ENGINE_REL_PATH)
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def node_version(node: str = "node") -> tuple[int, ...] | None:
    exe = shutil.which(node)
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=20, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.match(r"v?(\d+)(?:\.(\d+))?(?:\.(\d+))?", out)
    return tuple(int(x) for x in m.groups() if x is not None) if m else None


def engine_available(path: str | Path | None = None, node: str = "node") -> bool:
    p = find_engine(path)
    v = node_version(node)
    return bool(p) and bool(v) and v[0] >= MIN_NODE_MAJOR and (p.parent.parent.parent.parent / "node_modules").is_dir()


def unavailable_reason(path: str | Path | None = None, node: str = "node") -> str:
    p = find_engine(path)
    if not p:
        return "headless.mjs not found (set SOUS_CHEF_ENGINE or COOKIE_CUTTER_DIR)"
    v = node_version(node)
    if not v:
        return f"{node} not on PATH"
    if v[0] < MIN_NODE_MAJOR:
        return f"node {'.'.join(map(str, v))} < {MIN_NODE_MAJOR}"
    if not (p.parent.parent.parent.parent / "node_modules").is_dir():
        return f"{p.parent.parent.parent.parent}/node_modules missing (npm install in launcher/client)"
    return ""


# ── the wrapper ─────────────────────────────────────────────────────────────

class Engine:
    """One node boot per `run`; results cached by request hash."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        manifest_path: str | Path | None = None,
        node: str = "node",
        timeout_s: float = 180.0,
        cache_dir: str | Path | None = None,
        memory_cache: bool = True,
        env: dict | None = None,
    ):
        found = find_engine(path)
        if not found:
            raise EngineError(unavailable_reason(path, node))
        self.path = found
        self.client_root = found.parent.parent.parent.parent  # launcher/client
        self.cookie_cutter_dir = self.client_root.parent.parent
        self.node = node
        self.timeout_s = timeout_s
        # Absolute: node runs with cwd=launcher/client, so a relative `--manifest fixtures/…` would miss there.
        self.manifest_path = (Path(manifest_path) if manifest_path else (FIXTURES_DIR / "manifest.snapshot.json")).resolve()
        if not self.manifest_path.exists():
            raise EngineError(f"manifest not found: {self.manifest_path}")
        self.manifest_hash = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        ident = engine_fingerprint(self.path) or {}
        self.fingerprint: str = str(ident.get("fingerprint") or "")
        self.commit: str | None = ident.get("commit")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if cache_dir is None and os.environ.get("SOUS_CHEF_ENGINE_CACHE"):
            self.cache_dir = Path(os.environ["SOUS_CHEF_ENGINE_CACHE"])
        self._mem: dict[str, dict] | None = {} if memory_cache else None
        self._lock = threading.Lock()
        self.env = env
        self.calls = 0
        self.boots = 0
        self.boot_seconds: list[float] = []

    # cache
    def _cache_get(self, key: str) -> dict | None:
        if self._mem is not None:
            with self._lock:
                hit = self._mem.get(key)
            if hit is not None:
                return hit
        if self.cache_dir:
            p = self.cache_dir / f"{key}.json"
            if p.is_file():
                try:
                    return json.loads(p.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    return None
        return None

    def _cache_put(self, key: str, value: dict) -> None:
        if self._mem is not None:
            with self._lock:
                self._mem[key] = value
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_dir / f"{key}.{os.getpid()}.{threading.get_ident()}.tmp"  # unique: parallel writers never clash
            tmp.write_text(json.dumps(value), encoding="utf-8")
            tmp.replace(self.cache_dir / f"{key}.json")

    # execution
    def _spawn(self, args: list[str], stdin_text: str | None) -> tuple[int, str, str, float]:
        env = dict(os.environ)
        env["COOKIE_CUTTER_DIR"] = str(self.cookie_cutter_dir)
        env.setdefault("NODE_NO_WARNINGS", "1")
        if self.env:
            env.update(self.env)
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                [self.node, str(self.path), *args], input=stdin_text, capture_output=True, text=True,
                cwd=str(self.client_root), env=env, timeout=self.timeout_s, check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise EngineError(f"engine timed out after {self.timeout_s}s", stderr=str(e.stderr or "")) from e
        elapsed = time.monotonic() - t0
        with self._lock:
            self.boots += 1
            self.boot_seconds.append(elapsed)
        return proc.returncode, proc.stdout, proc.stderr, elapsed

    @staticmethod
    def _parse(returncode: int, stdout: str, stderr: str) -> dict:
        if returncode == 1 or not stdout.strip():
            raise EngineError(f"engine exit {returncode}: {stderr.strip()[-2000:]}", returncode=returncode, stderr=stderr)
        try:
            return json.loads(stdout)
        except ValueError as e:
            raise EngineError(f"engine printed no JSON (exit {returncode}): {stdout[:300]!r}", returncode=returncode, stderr=stderr) from e

    def run(self, state: dict, ops: list[dict] | dict, ctx: dict | None = None, *, print_state: bool = False,
            use_cache: bool = True) -> EngineResult:
        """stdin mode: one `{state, ops, ctx}` document; the manifest goes by `--manifest <path>`."""
        ops_list = ops if isinstance(ops, list) else [ops]
        ctx = GOLDEN_CTX if ctx is None else ctx
        key = request_hash(state, ops_list, ctx, self.manifest_hash + (":state" if print_state else ""), self.fingerprint)
        with self._lock:
            self.calls += 1
        if use_cache:
            hit = self._cache_get(key)
            if hit is not None:
                return EngineResult.from_output(hit, returncode=0 if hit.get("ok") else 2, elapsed_s=0.0, cached=True)
        req = json.dumps({"state": state, "ops": ops_list, "ctx": ctx})
        # The result goes through `--out <file>`, never the stdout pipe: headless.mjs calls process.exit right
        # after process.stdout.write, which truncates a piped stdout at 64 KiB (measured 2026-10-01: a
        # --print-state reply of 300 KB arrived as 65,511 bytes). The pipe stays as the fallback.
        with tempfile.TemporaryDirectory(prefix="souschef-engine-") as tmp:
            out_path = Path(tmp) / "result.json"
            args = ["--manifest", str(self.manifest_path), "--out", str(out_path)] + (["--print-state"] if print_state else [])
            rc, out, err, elapsed = self._spawn(args, req)
            if rc != 1 and out_path.is_file():
                try:
                    parsed = json.loads(out_path.read_text(encoding="utf-8"))
                except ValueError as e:
                    raise EngineError(f"engine wrote unreadable JSON to --out (exit {rc})", returncode=rc, stderr=err) from e
            else:
                parsed = self._parse(rc, out, err)
        if use_cache:
            self._cache_put(key, parsed)
        return EngineResult.from_output(parsed, returncode=rc, elapsed_s=elapsed, stderr=err)

    def run_files(self, state_path: str | Path, ops_path: str | Path, ctx_path: str | Path | None = None, *,
                  out_path: str | Path | None = None) -> EngineResult:
        """file mode: `--state s.json --ops ops.json [--ctx c.json] [--out r.json]` (uncached)."""
        args = ["--state", str(state_path), "--ops", str(ops_path), "--manifest", str(self.manifest_path)]
        if ctx_path:
            args += ["--ctx", str(ctx_path)]
        if out_path:
            args += ["--out", str(out_path)]
        with self._lock:
            self.calls += 1
        rc, out, err, elapsed = self._spawn(args, None)
        if out_path:
            if rc == 1:
                raise EngineError(f"engine exit 1: {err.strip()[-2000:]}", returncode=rc, stderr=err)
            parsed = json.loads(Path(out_path).read_text(encoding="utf-8"))
        else:
            parsed = self._parse(rc, out, err)
        return EngineResult.from_output(parsed, returncode=rc, elapsed_s=elapsed, stderr=err)

    def run_many(self, requests: Iterable[tuple[dict, list[dict], dict | None]], *, workers: int = 1) -> list[EngineResult]:
        """Sequential (or `workers` parallel node processes) runs; order preserved."""
        reqs = list(requests)
        if workers <= 1:
            return [self.run(s, o, c) for s, o, c in reqs]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda r: self.run(r[0], r[1], r[2]), reqs))

    def stats(self) -> dict:
        bs = self.boot_seconds
        return {
            "fingerprint": self.fingerprint, "commit": self.commit,
            "calls": self.calls, "boots": self.boots, "cache_hits": self.calls - self.boots,
            "boot_s_mean": round(sum(bs) / len(bs), 3) if bs else None,
            "boot_s_max": round(max(bs), 3) if bs else None,
        }


SERVER_SCRIPT = Path(__file__).resolve().parent / "js" / "engine_server.mjs"


class _Server:
    """One `node js/engine_server.mjs` process: the engine booted once,
    answering one request at a time (the pool hands a server to one caller)."""

    def __init__(self, owner: "PersistentEngine"):
        import queue as _q
        self.owner = owner
        env = dict(os.environ)
        env["COOKIE_CUTTER_DIR"] = str(owner.cookie_cutter_dir)
        env.setdefault("NODE_NO_WARNINGS", "1")
        if owner.env:
            env.update(owner.env)
        t0 = time.monotonic()
        self.proc = subprocess.Popen([owner.node, str(SERVER_SCRIPT), str(owner.client_root), str(owner.manifest_path)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                     encoding="utf-8", cwd=str(owner.client_root), env=env, bufsize=1)
        self.lines: "_q.Queue[str | None]" = _q.Queue()
        self.err: list[str] = []
        threading.Thread(target=self._pump, daemon=True).start()
        threading.Thread(target=self._pump_err, daemon=True).start()
        ready = self._line(owner.boot_timeout_s)
        if not ready or not json.loads(ready).get("ready"):
            self.close()
            raise EngineError(f"engine server did not start: {''.join(self.err)[-2000:]}", returncode=1, stderr="".join(self.err))
        self.boot_s = time.monotonic() - t0
        self.n = 0
        self.broken = False

    def _pump(self) -> None:
        for line in self.proc.stdout:  # type: ignore[union-attr]
            self.lines.put(line)
        self.lines.put(None)

    def _pump_err(self) -> None:
        for line in self.proc.stderr:  # type: ignore[union-attr]
            self.err.append(line)
            del self.err[:-200]

    def _line(self, timeout: float) -> str | None:
        import queue as _q
        try:
            return self.lines.get(timeout=timeout)
        except _q.Empty:
            return None

    def alive(self) -> bool:
        return self.proc.poll() is None and not self.broken

    def request(self, state: dict, ops: list[dict], ctx: dict, timeout: float) -> tuple[int, dict]:
        self.n += 1
        with tempfile.TemporaryDirectory(prefix="souschef-engine-") as tmp:
            out = Path(tmp) / "result.json"
            line = json.dumps({"id": self.n, "state": state, "ops": ops, "ctx": ctx, "out": str(out)})
            try:
                self.proc.stdin.write(line + "\n")  # type: ignore[union-attr]
                self.proc.stdin.flush()  # type: ignore[union-attr]
            except (BrokenPipeError, OSError) as e:
                self.broken = True
                raise EngineError(f"engine server is gone: {e}; {''.join(self.err)[-1000:]}", returncode=1) from e
            reply = self._line(timeout)
            if reply is None:
                self.broken = True
                raise EngineError(f"engine server: no answer in {timeout}s (or it exited): {''.join(self.err)[-1000:]}",
                                  returncode=1)
            doc = json.loads(reply)
            if doc.get("id") != self.n:
                self.broken = True
                raise EngineError(f"engine server answered request {doc.get('id')} for {self.n}", returncode=1)
            if doc.get("exit") == 1:
                raise EngineError(f"engine exit 1: {doc.get('error')}", returncode=1, stderr=str(doc.get("error")))
            return int(doc.get("exit") or 0), json.loads(out.read_text(encoding="utf-8"))

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self.proc.kill()


class PersistentEngine(Engine):
    """`Engine` with the engine booted once per server process instead of
    once per call: a pool of up to `size` `js/engine_server.mjs` processes
    (each answers one request at a time; a caller takes a free one, a new one
    boots when none is free and the pool is not full). Same `run` contract,
    same cache keys and the same fingerprint as `Engine`: the server runs
    headless.mjs's steps on the same modules (tests/test_engine_server.py
    holds the two byte-equal on real requests), so a cached answer is the
    same answer whichever produced it. A request that crashes the engine
    raises EngineError like headless.mjs's exit 1; a server that dies or
    stops answering is replaced on the next call. `close()` ends the pool."""

    def __init__(self, path: str | Path | None = None, *, size: int = 4, boot_timeout_s: float = 300.0, **kw):
        super().__init__(path, **kw)
        if not SERVER_SCRIPT.is_file():
            raise EngineError(f"{SERVER_SCRIPT} is missing")
        import queue as _q
        self.size = max(1, int(size))
        self.boot_timeout_s = boot_timeout_s
        self._free: "_q.Queue[_Server]" = _q.Queue()
        self._started = 0
        self._pool_lock = threading.Lock()
        self.server_boots: list[float] = []

    def _take(self) -> _Server:
        import queue as _q
        try:
            return self._free.get_nowait()
        except _q.Empty:
            pass
        with self._pool_lock:
            grow = self._started < self.size
            if grow:
                self._started += 1
        if grow:
            try:
                srv = _Server(self)
            except Exception:
                with self._pool_lock:
                    self._started -= 1
                raise
            with self._lock:
                self.server_boots.append(srv.boot_s)
            return srv
        return self._free.get()

    def _give(self, srv: _Server, ok: bool) -> None:
        if ok and srv.alive():
            self._free.put(srv)
            return
        srv.close()
        with self._pool_lock:
            self._started -= 1

    def run(self, state: dict, ops: list[dict] | dict, ctx: dict | None = None, *, print_state: bool = False,
            use_cache: bool = True) -> EngineResult:
        ops_list = ops if isinstance(ops, list) else [ops]
        ctx = GOLDEN_CTX if ctx is None else ctx
        key_state = request_hash(state, ops_list, ctx, self.manifest_hash + ":state", self.fingerprint)
        key_plain = request_hash(state, ops_list, ctx, self.manifest_hash, self.fingerprint)
        key = key_state if print_state else key_plain
        with self._lock:
            self.calls += 1
        if use_cache:
            hit = self._cache_get(key)
            if hit is not None:
                return EngineResult.from_output(hit, returncode=0 if hit.get("ok") else 2, elapsed_s=0.0, cached=True)
        srv = self._take()
        t0 = time.monotonic()
        try:
            rc, full = srv.request(state, ops_list, ctx, self.timeout_s)
        except EngineError:
            self._give(srv, ok=srv.alive())  # an exit-1 answer leaves the server sound; a timeout does not
            raise
        except Exception:
            self._give(srv, ok=False)
            raise
        self._give(srv, ok=True)
        elapsed = time.monotonic() - t0
        with self._lock:
            self.boots += 1  # an engine evaluation (not a boot here; `server_boots` counts those)
            self.boot_seconds.append(elapsed)
        plain = {k: v for k, v in full.items() if k != "state"}
        if use_cache:
            self._cache_put(key_state, full)
            self._cache_put(key_plain, plain)
        return EngineResult.from_output(full if print_state else plain, returncode=rc, elapsed_s=elapsed)

    def close(self) -> None:
        import queue as _q
        while True:
            try:
                srv = self._free.get_nowait()
            except _q.Empty:
                break
            srv.close()
        with self._pool_lock:
            self._started = 0

    def stats(self) -> dict:
        d = super().stats()
        d.update({"servers": self.size, "server_boots": len(self.server_boots),
                  "server_boot_s_max": round(max(self.server_boots), 1) if self.server_boots else None})
        return d


def default_cache_dir() -> Path:
    return DATA_DIR / "engine-cache"


def write_request_files(dir_path: str | Path, state: dict, ops: list[dict], ctx: dict | None = None) -> tuple[Path, Path, Path | None]:
    d = Path(dir_path)
    d.mkdir(parents=True, exist_ok=True)
    s, o = d / "state.json", d / "ops.json"
    s.write_text(json.dumps(state), encoding="utf-8")
    o.write_text(json.dumps(ops), encoding="utf-8")
    c = None
    if ctx is not None:
        c = d / "ctx.json"
        c.write_text(json.dumps(ctx), encoding="utf-8")
    return s, o, c


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="souschef_lab.engine", description="run one request through headless.mjs")
    ap.add_argument("--engine", help="headless.mjs path (else $SOUS_CHEF_ENGINE / $COOKIE_CUTTER_DIR)")
    ap.add_argument("--manifest", default=str(FIXTURES_DIR / "manifest.snapshot.json"))
    ap.add_argument("--state", help="EditorState JSON (required unless --check / --fingerprint)")
    ap.add_argument("--ops", help="ops JSON list (required unless --check / --fingerprint)")
    ap.add_argument("--ctx")
    ap.add_argument("--files", action="store_true", help="file mode instead of stdin mode")
    ap.add_argument("--print-state", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--check", action="store_true", help="only report whether the engine is available")
    ap.add_argument("--fingerprint", action="store_true", help="print the engine's identity (sha256 over the import closure of headless.mjs + @breadstack versions + HEAD)")
    a = ap.parse_args(argv)
    if a.check:
        ok = engine_available(a.engine)
        print(json.dumps({"available": ok, "engine": str(find_engine(a.engine) or ""), "node": node_version(), "reason": unavailable_reason(a.engine),
                          "fingerprint": (engine_fingerprint(a.engine) or {}).get("fingerprint")}))
        return 0 if ok else 2
    if a.fingerprint:
        ident = engine_fingerprint(a.engine)
        print(json.dumps(ident or {"error": unavailable_reason(a.engine)}, indent=1))
        return 0 if ident else 2
    if not a.state or not a.ops:
        ap.error("--state and --ops are required")
    eng = Engine(a.engine, manifest_path=a.manifest, cache_dir=None if a.no_cache else default_cache_dir())
    if a.files:
        r = eng.run_files(a.state, a.ops, a.ctx)
    else:
        state = json.loads(Path(a.state).read_text(encoding="utf-8"))
        ops = json.loads(Path(a.ops).read_text(encoding="utf-8"))
        ctx = json.loads(Path(a.ctx).read_text(encoding="utf-8")) if a.ctx else None
        r = eng.run(state, ops, ctx, print_state=a.print_state, use_cache=not a.no_cache)
    print(json.dumps({**r.to_dict(), "elapsed_s": round(r.elapsed_s, 3), "cached": r.cached}, indent=1))
    return 0 if r.ok else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
