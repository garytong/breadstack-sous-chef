"""enginebundle — the vendored launcher engine for jobs in GCP (DESIGN §6.2,
docs/DRIFT.md "Engine refresh").

The engine (cookie-cutter `launcher/client/src/sousChef/ops/headless.mjs`,
loaded through the launcher's own vite config) and the browser defaults the
lab mirrors need a cookie-cutter tree with `node_modules`. GCP has no
Bitbucket access, so a person (later: cookie-cutter's CI) vendors one
snapshot per launcher commit into the pipeline bucket and every job fetches
the one the deployed manifest was generated from:

  engine/index.json                {"snapshots": {<sha>: {fingerprint, created_at, tar_sha256, size,
                                    browser_defaults_sha256, node}}, "newest": <sha>, "updated_at"}
  engine/<sha>/snapshot.tar.zst    `launcher/client` + `packages/` + the workspace's node_modules
                                    (npm ci --ignore-scripts), SNAPSHOT_OF = sha
  engine/<sha>/engine.json         {sha, fingerprint, node, created_at, browser_defaults_sha256, tar_sha256, size}
  engine/<sha>/browser-defaults.json   the dump of the snapshot (browser_defaults.generate)

  pack     a snapshot folder → <out>/{snapshot.tar.zst, engine.json, browser-defaults.json}
           (lab/tools/engine-vendor.sh builds the folder from a checkout with git archive + npm ci)
  publish  <out> → the bucket, and the index updated (conditional on its generation)
  ensure   at job start: choose the snapshot for a launcher sha (prefix match, ≥ 7 hex), else the
           newest (status `stale`); download, check tar_sha256, extract, point COOKIE_CUTTER_DIR /
           ENGINE / SOUS_CHEF_ENGINE at it, dump (or copy) the browser defaults to
           $SOUS_CHEF_BROWSER_DEFAULTS, and assert the extracted engine's fingerprint equals
           engine.json's

    python -m souschef_lab.enginebundle pack --snapshot DIR --sha SHA --out DIR
    python -m souschef_lab.enginebundle publish --bucket gs://… --dir DIR
    python -m souschef_lab.enginebundle ensure --bucket gs://… [--sha SHA] [--dir /tmp/engine]
    python -m souschef_lab.enginebundle status --bucket gs://… --sha SHA
    python -m souschef_lab.enginebundle index --bucket gs://…

Standard library only (tar and zstd are the system's: the lab runtime image ships both).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import gcsio
from .engine import ENGINE_REL_PATH, engine_fingerprint

INDEX_KEY = "engine/index.json"
TAR_NAME = "snapshot.tar.zst"
META_NAME = "engine.json"
DEFAULTS_NAME = "browser-defaults.json"
SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")


class BundleError(RuntimeError):
    pass


@dataclass
class Bundle:
    sha: str
    status: str  # current | stale | unknown
    dir: Path
    engine: Path
    fingerprint: str
    browser_defaults: Path
    wanted: str | None = None
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"sha": self.sha, "status": self.status, "wanted": self.wanted, "dir": str(self.dir),
                "engine": str(self.engine), "fingerprint": self.fingerprint,
                "browser_defaults": str(self.browser_defaults)}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def same_sha(a: Any, b: Any) -> bool:
    """Two commit ids name the same commit: one is a prefix of the other and
    the shorter has ≥ 7 hex digits (manifest `generated_from.sha` is short)."""
    a, b = str(a or "").lower(), str(b or "").lower()
    if not (SHA_RE.match(a) and SHA_RE.match(b)):
        return False
    return a.startswith(b) or b.startswith(a)


def choose(index: dict | None, want: str | None) -> tuple[str | None, str]:
    """(sha to use, status): the snapshot of `want` → current; another one
    (the newest) → stale; no index or no snapshot → (None, unknown)."""
    snaps = (index or {}).get("snapshots") or {}
    if not snaps:
        return None, "unknown"
    if want and want != "newest":
        for sha in snaps:
            if same_sha(sha, want):
                return sha, "current"
    newest = (index or {}).get("newest")
    if newest not in snaps:
        newest = max(snaps, key=lambda s: str(snaps[s].get("created_at") or ""))
    if not want or want == "newest":
        return newest, "current"
    return newest, "stale"


def status_for(index: dict | None, want: str | None) -> str:
    """`current` | `stale` (no snapshot of `want`) | `unknown` (no snapshot at
    all, or the deployed manifest names no commit)."""
    if not want:
        return "unknown"
    sha, status = choose(index, want)
    return status if sha else "unknown"


# ── pack / publish (a person, or cookie-cutter CI) ───────────────────────────

def _outside_links(root: Path) -> list[str]:
    """Top-level node_modules symlinks that point outside the snapshot (a
    workstation snapshot like cc-snapshot-r4 links the live checkout's)."""
    out = []
    for p in [root / "node_modules", root / "launcher" / "client" / "node_modules", root / "packages" / "ui" / "node_modules"]:
        if p.is_symlink():
            target = p.resolve()
            if root.resolve() not in target.parents:
                out.append(f"{p.relative_to(root)} -> {target}")
    return out


def tar_zstd(src: Path, out: Path, dereference: bool = False) -> None:
    if not shutil.which("zstd"):
        raise BundleError("zstd is not on PATH")
    args = ["tar", "--zstd", "-cf", str(out), "--sort=name", "--owner=0", "--group=0", "--numeric-owner"]
    if dereference:
        args.append("--dereference")
    r = subprocess.run([*args, "-C", str(src), "."], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise BundleError(f"tar exited {r.returncode}: {r.stderr.strip()[-500:]}")


def untar_zstd(tar: Path, dest: Path) -> None:
    if not shutil.which("zstd"):
        raise BundleError("zstd is not on PATH")
    dest.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["tar", "--zstd", "-xf", str(tar), "-C", str(dest), "--no-same-owner"], capture_output=True,
                       text=True, check=False)
    if r.returncode != 0:
        raise BundleError(f"tar -x exited {r.returncode}: {r.stderr.strip()[-500:]}")


def node_version(node: str = "node") -> str | None:
    exe = shutil.which(node)
    if not exe:
        return None
    r = subprocess.run([exe, "--version"], capture_output=True, text=True, check=False)
    return r.stdout.strip() or None


def pack(snapshot: str | Path, sha: str, out: str | Path, *, dereference: bool = False, dump_defaults: bool = True,
         node: str = "node") -> dict:
    """The bundle of a snapshot folder (`launcher/client`, `packages/`,
    `node_modules`): the zstd tar, the dumped browser defaults, engine.json."""
    from . import browser_defaults as BD

    root = Path(snapshot).resolve()
    sha = str(sha).strip().lower()
    if not SHA_RE.match(sha):
        raise BundleError(f"{sha!r} is not a commit id")
    headless = root / ENGINE_REL_PATH
    if not headless.is_file():
        raise BundleError(f"{root} has no {ENGINE_REL_PATH}")
    links = _outside_links(root)
    if links and not dereference:
        raise BundleError("node_modules link outside the snapshot (" + "; ".join(links)
                          + "): vendor with lab/tools/engine-vendor.sh (npm ci), or pass --dereference")
    snap_of = root / "SNAPSHOT_OF"
    if not snap_of.is_file() or not same_sha(snap_of.read_text(encoding="utf-8").strip(), sha):
        snap_of.write_text(sha + "\n", encoding="utf-8")
    fp = engine_fingerprint(headless)
    if fp is None:
        raise BundleError(f"no engine fingerprint for {headless}")
    if fp.get("commit"):
        raise BundleError(f"{root} is inside a git checkout (HEAD {fp['commit'][:12]} enters the fingerprint): "
                          "vendor from `git archive`, not from a working tree")
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    defaults_sha = None
    if dump_defaults:
        doc = BD.generate(root, node=node)
        (out_dir / DEFAULTS_NAME).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        defaults_sha = sha256_file(out_dir / DEFAULTS_NAME)
    tar = out_dir / TAR_NAME
    tar_zstd(root, tar, dereference=dereference)
    meta = {"sha": sha, "fingerprint": fp["fingerprint"], "node": node_version(node), "created_at": gcsio.iso(gcsio.utc_now()),
            "browser_defaults_sha256": defaults_sha, "tar_sha256": sha256_file(tar), "size": tar.stat().st_size,
            "files": len(fp.get("files") or []), "packages": fp.get("packages") or []}
    (out_dir / META_NAME).write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return meta


def publish(b: gcsio.Bucket, bundle_dir: str | Path, *, make_newest: bool = True) -> dict:
    d = Path(bundle_dir)
    meta = json.loads((d / META_NAME).read_text(encoding="utf-8"))
    sha = meta["sha"]
    if sha256_file(d / TAR_NAME) != meta["tar_sha256"]:
        raise BundleError(f"{d / TAR_NAME} does not match engine.json's tar_sha256")
    b.upload(d / TAR_NAME, f"engine/{sha}/{TAR_NAME}")
    if (d / DEFAULTS_NAME).is_file():
        b.upload(d / DEFAULTS_NAME, f"engine/{sha}/{DEFAULTS_NAME}")
    b.upload(d / META_NAME, f"engine/{sha}/{META_NAME}")  # last: its presence says the bundle is complete
    for _ in range(5):
        gen = b.generation(INDEX_KEY)
        idx = b.read_json(INDEX_KEY, None) or {"snapshots": {}}
        idx.setdefault("snapshots", {})[sha] = {k: meta.get(k) for k in ("fingerprint", "created_at", "tar_sha256", "size",
                                                                         "browser_defaults_sha256", "node")}
        if make_newest or not idx.get("newest"):
            idx["newest"] = sha
        idx["updated_at"] = gcsio.iso(gcsio.utc_now())
        try:
            b.write_json(INDEX_KEY, idx, if_generation=0 if gen is None else gen)
            return idx
        except gcsio.PreconditionFailed:
            continue
    raise BundleError("engine/index.json kept changing under the update")


# ── ensure (every job start) ─────────────────────────────────────────────────

def ensure(b: gcsio.Bucket, want: str | None = "newest", *, workdir: str | Path | None = None, redump: bool = True,
           set_env: bool = True, defaults_out: str | Path | None = None, node: str = "node") -> Bundle:
    """The snapshot for `want` (a launcher sha, or "newest") extracted and
    checked; the environment pointed at it (`set_env`)."""
    from . import browser_defaults as BD

    idx = b.read_json(INDEX_KEY, None)
    sha, status = choose(idx, want)
    if sha is None:
        raise BundleError(f"no vendored engine snapshot in {b.uri('engine/')} (make -C lab engine-vendor SHA=… CC=… UPLOAD=1)")
    base = Path(workdir or os.environ.get("SOUS_CHEF_ENGINE_DIR") or Path(tempfile.gettempdir()) / "engine")
    dest = base / sha
    meta = b.read_json(f"engine/{sha}/{META_NAME}")
    marker = dest / ".bundle.json"
    if not (marker.is_file() and json.loads(marker.read_text(encoding="utf-8")).get("tar_sha256") == meta["tar_sha256"]):
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        tar = base / f"{sha}.tar.zst"
        b.download(f"engine/{sha}/{TAR_NAME}", tar)
        got = sha256_file(tar)
        if got != meta["tar_sha256"]:
            raise BundleError(f"engine/{sha}/{TAR_NAME}: sha256 {got[:12]}… is not engine.json's {meta['tar_sha256'][:12]}…")
        untar_zstd(tar, dest)
        tar.unlink()
        marker.write_text(json.dumps(meta), encoding="utf-8")
    headless = dest / ENGINE_REL_PATH
    fp = engine_fingerprint(headless)
    if fp is None or fp["fingerprint"] != meta["fingerprint"]:
        raise BundleError(f"the extracted engine's fingerprint {(fp or {}).get('fingerprint', '?')[:12]}… is not "
                          f"engine.json's {meta['fingerprint'][:12]}…")
    out = Path(defaults_out) if defaults_out else dest.parent / f"{sha}.browser-defaults.json"
    stored = b.read_json(f"engine/{sha}/{DEFAULTS_NAME}", None)
    if redump:
        doc = BD.generate(dest, node=node)
        if stored is not None and not BD.same_content(stored, doc):
            raise BundleError(f"the browser defaults dumped from engine/{sha} differ from its stored {DEFAULTS_NAME}")
    elif stored is not None:
        doc = stored
    else:
        raise BundleError(f"engine/{sha} has no {DEFAULTS_NAME} and redump is off")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    bundle = Bundle(sha=sha, status=status, dir=dest, engine=headless, fingerprint=meta["fingerprint"], browser_defaults=out,
                    wanted=None if want == "newest" else want, meta=meta)
    if set_env:
        activate(bundle)
    return bundle


def activate(bundle: Bundle) -> None:
    """Point the lab (engine discovery, browser defaults) at the bundle."""
    from . import browser_defaults as BD

    os.environ["COOKIE_CUTTER_DIR"] = str(bundle.dir)
    os.environ["ENGINE"] = str(bundle.engine)
    os.environ["SOUS_CHEF_ENGINE"] = str(bundle.engine)
    os.environ[BD.ENV_OVERRIDE] = str(bundle.browser_defaults)
    BD._load.cache_clear()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="souschef_lab.enginebundle", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack", help="a snapshot folder → <out>/{snapshot.tar.zst, engine.json, browser-defaults.json}")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--sha", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dereference", action="store_true", help="follow node_modules links (a workstation snapshot)")
    p.add_argument("--no-defaults", action="store_true", help="skip the browser-defaults dump (not for real bundles)")
    u = sub.add_parser("publish", help="upload a packed bundle and update engine/index.json")
    u.add_argument("--bucket", default=os.environ.get("PIPELINE_BUCKET"))
    u.add_argument("--dir", required=True)
    u.add_argument("--not-newest", action="store_true", help="add it without making it the newest")
    e = sub.add_parser("ensure", help="fetch, check and extract the snapshot for a sha (or the newest)")
    e.add_argument("--bucket", default=os.environ.get("PIPELINE_BUCKET"))
    e.add_argument("--sha", default="newest")
    e.add_argument("--dir", help="extract under this folder (default $SOUS_CHEF_ENGINE_DIR or $TMPDIR/engine)")
    e.add_argument("--no-redump", action="store_true", help="use the stored browser-defaults.json instead of dumping")
    s = sub.add_parser("status", help="current | stale | unknown for a launcher sha")
    s.add_argument("--bucket", default=os.environ.get("PIPELINE_BUCKET"))
    s.add_argument("--sha", required=True)
    i = sub.add_parser("index", help="print engine/index.json")
    i.add_argument("--bucket", default=os.environ.get("PIPELINE_BUCKET"))
    a = ap.parse_args(argv)
    try:
        if a.cmd == "pack":
            meta = pack(a.snapshot, a.sha, a.out, dereference=a.dereference, dump_defaults=not a.no_defaults)
            print(json.dumps(meta, indent=1))
            return 0
        b = gcsio.bucket(a.bucket)
        if a.cmd == "publish":
            idx = publish(b, a.dir, make_newest=not a.not_newest)
            print(f"engine bundle published; newest {idx['newest']}; {len(idx['snapshots'])} snapshot(s)")
            return 0
        if a.cmd == "ensure":
            bundle = ensure(b, a.sha, workdir=a.dir, redump=not a.no_redump)
            print(json.dumps(bundle.as_dict(), indent=1))
            return 0
        if a.cmd == "status":
            print(status_for(b.read_json(INDEX_KEY, None), a.sha))
            return 0
        print(json.dumps(b.read_json(INDEX_KEY, None), indent=1))
        return 0
    except (BundleError, gcsio.StoreError, RuntimeError) as e:
        print(f"enginebundle {a.cmd}: {e}", file=sys.stderr)
        return 2 if isinstance(e, BundleError) else 1


if __name__ == "__main__":
    raise SystemExit(main())
