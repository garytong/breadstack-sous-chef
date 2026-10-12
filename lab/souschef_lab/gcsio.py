"""gcsio — the pipeline bucket, for the drift runner and the retrain stages
(docs/DRIFT.md, docs/RETRAIN.md; the layout is DESIGN I2).

One small interface over two backends:

  gs://<bucket>[/<prefix>]   the real bucket, through the `gcloud storage` CLI
                             (in the lab runtime image; no Google client library,
                             the lab stays standard-library only)
  <dir> | file://<dir>       a local folder with the same layout (DRY=1 runs, the
                             tests, and a playbook rehearsal on a workstation)

    from souschef_lab import gcsio
    b = gcsio.bucket("gs://breadstack-sous-chef-int-pipeline")
    b.read_json("state/model.json", None)       # default instead of NotFound
    b.write_json("drift/dr-…/result.json", {...})
    b.rsync_up(local_dir, "runs/rt-…/collect")  # and rsync_down
    with b.lock("gpu-eval", run_id, ttl_s=4 * 3600) as held: ...

Writes: one object upload is atomic on GCS (readers see the old object or the
new one, never half of it), so a JSON document is uploaded in one request; the
local backend writes a temp file and renames it, which is the same guarantee.

Locks (`state/locks/<name>.json` = {holder, run_id, expires_at, updated_at}):
created with `--if-generation-match=0` (only when absent), taken over only when
expired and only with `--if-generation-match=<the generation read>`, released
with the same condition. A lost race raises LockHeld; nothing ever deletes a
lock that another run holds.

Never prints or logs object contents; errors carry the gcloud exit and the
last line of its stderr with URLs only.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

_MISSING = object()
LOCK_PREFIX = "state/locks"


class StoreError(RuntimeError):
    """A storage call failed (not a missing object)."""


class NotFound(StoreError):
    """The object does not exist."""


class PreconditionFailed(StoreError):
    """A conditional write lost (the generation moved)."""


class LockHeld(StoreError):
    """Another run holds the lock (`holder` = its lock document)."""

    def __init__(self, name: str, holder: dict | None):
        super().__init__(f"lock {name} is held by {(holder or {}).get('run_id') or '?'} until {(holder or {}).get('expires_at') or '?'}")
        self.name = name
        self.holder = holder or {}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s: Any) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _key(k: str) -> str:
    k = str(k).strip("/")
    if not k or ".." in k.split("/"):
        raise StoreError(f"bad object key {k!r}")
    return k


class Bucket:
    """The interface both backends implement."""

    url: str

    # bytes
    def read_bytes(self, key: str) -> bytes:  # pragma: no cover - interface
        raise NotImplementedError

    def write_bytes(self, key: str, data: bytes, *, if_generation: int | None = None) -> None:  # pragma: no cover
        raise NotImplementedError

    def generation(self, key: str) -> int | None:  # pragma: no cover
        raise NotImplementedError

    def delete(self, key: str, *, if_generation: int | None = None) -> bool:  # pragma: no cover
        raise NotImplementedError

    def list(self, prefix: str) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    def upload(self, local: str | Path, key: str) -> None:  # pragma: no cover
        raise NotImplementedError

    def download(self, key: str, local: str | Path) -> Path:  # pragma: no cover
        raise NotImplementedError

    def rsync_up(self, local_dir: str | Path, prefix: str) -> None:  # pragma: no cover
        raise NotImplementedError

    def rsync_down(self, prefix: str, local_dir: str | Path) -> Path:  # pragma: no cover
        raise NotImplementedError

    def uri(self, key: str) -> str:
        return f"{self.url.rstrip('/')}/{_key(key)}"

    # JSON and text
    def exists(self, key: str) -> bool:
        return self.generation(key) is not None

    def read_json(self, key: str, default: Any = _MISSING) -> Any:
        try:
            raw = self.read_bytes(key)
        except NotFound:
            if default is _MISSING:
                raise
            return default
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise StoreError(f"{self.uri(key)} is not JSON: {e}") from None

    def write_json(self, key: str, obj: Any, *, if_generation: int | None = None, indent: int | None = 1) -> None:
        data = (json.dumps(obj, ensure_ascii=False, indent=indent, sort_keys=False) + "\n").encode("utf-8")
        self.write_bytes(key, data, if_generation=if_generation)

    def read_text(self, key: str, default: Any = _MISSING) -> Any:
        try:
            return self.read_bytes(key).decode("utf-8")
        except NotFound:
            if default is _MISSING:
                raise
            return default

    def write_text(self, key: str, text: str) -> None:
        self.write_bytes(key, text.encode("utf-8"))

    def read_jsonl(self, key: str) -> list[dict]:
        return [json.loads(line) for line in self.read_text(key).splitlines() if line.strip()]

    # locks
    def lock_key(self, name: str) -> str:
        return f"{LOCK_PREFIX}/{_key(name)}.json"

    def read_lock(self, name: str) -> dict | None:
        return self.read_json(self.lock_key(name), None)

    def acquire(self, name: str, run_id: str, ttl_s: float, *, holder: str = "", now: datetime | None = None) -> dict:
        """Take the lock or raise LockHeld. An expired lock is taken over
        (conditionally on the generation read); the same run re-acquiring
        refreshes it."""
        now = now or utc_now()
        key = self.lock_key(name)
        doc = {"holder": holder or run_id, "run_id": run_id, "expires_at": iso(now + timedelta(seconds=ttl_s)),
               "updated_at": iso(now), "acquired_at": iso(now)}
        for _ in range(3):
            gen = self.generation(key)
            if gen is None:
                try:
                    self.write_json(key, doc, if_generation=0)
                    return doc
                except PreconditionFailed:
                    continue
            cur = self.read_json(key, None) or {}
            exp = parse_iso(cur.get("expires_at"))
            if cur.get("run_id") != run_id and exp is not None and exp > now:
                raise LockHeld(name, cur)
            if cur.get("run_id") == run_id:
                doc["acquired_at"] = cur.get("acquired_at") or doc["acquired_at"]
            try:
                self.write_json(key, doc, if_generation=gen)
                return doc
            except PreconditionFailed:
                continue
        raise LockHeld(name, self.read_lock(name))

    def refresh(self, name: str, run_id: str, ttl_s: float, *, now: datetime | None = None) -> dict:
        """Push the expiry out (a long stage calls it as it progresses)."""
        return self.acquire(name, run_id, ttl_s, now=now)

    def release(self, name: str, run_id: str) -> bool:
        """Delete the lock when this run holds it; False when it does not."""
        key = self.lock_key(name)
        gen = self.generation(key)
        if gen is None:
            return False
        cur = self.read_json(key, None) or {}
        if cur.get("run_id") != run_id:
            return False
        try:
            return self.delete(key, if_generation=gen)
        except PreconditionFailed:
            return False

    @contextlib.contextmanager
    def lock(self, name: str, run_id: str, ttl_s: float, *, holder: str = "") -> Iterator[dict]:
        doc = self.acquire(name, run_id, ttl_s, holder=holder)
        try:
            yield doc
        finally:
            self.release(name, run_id)


# ── local folder ─────────────────────────────────────────────────────────────

class LocalBucket(Bucket):
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.url = f"file://{self.root}"

    def path(self, key: str) -> Path:
        return self.root / _key(key)

    @contextlib.contextmanager
    def _guard(self) -> Iterator[None]:
        lf = self.root / ".gcsio.lock"
        with open(lf, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def read_bytes(self, key: str) -> bytes:
        p = self.path(key)
        if not p.is_file():
            raise NotFound(self.uri(key))
        return p.read_bytes()

    def generation(self, key: str) -> int | None:
        p = self.path(key)
        return p.stat().st_mtime_ns if p.is_file() else None

    def write_bytes(self, key: str, data: bytes, *, if_generation: int | None = None) -> None:
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        with self._guard():
            if if_generation is not None:
                gen = self.generation(key)
                if (if_generation == 0 and gen is not None) or (if_generation != 0 and gen != if_generation):
                    raise PreconditionFailed(f"{self.uri(key)}: generation {gen} is not {if_generation}")
            fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=str(p.parent))
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            # a rewrite in the same nanosecond would keep the generation: move it on
            old = self.generation(key)
            os.replace(tmp, p)
            if old is not None and self.generation(key) == old:
                os.utime(p, ns=(old + 1, old + 1))

    def delete(self, key: str, *, if_generation: int | None = None) -> bool:
        p = self.path(key)
        with self._guard():
            gen = self.generation(key)
            if gen is None:
                return False
            if if_generation is not None and gen != if_generation:
                raise PreconditionFailed(f"{self.uri(key)}: generation {gen} is not {if_generation}")
            p.unlink()
            return True

    def list(self, prefix: str) -> list[str]:
        prefix = str(prefix).strip("/")
        base = self.root / prefix if prefix else self.root
        if base.is_file():
            return [prefix]
        if not base.is_dir():
            # a name prefix (`manifests/bf78`)
            parent = base.parent
            if not parent.is_dir():
                return []
            return sorted(p.relative_to(self.root).as_posix() for p in parent.rglob("*")
                          if p.is_file() and p.relative_to(self.root).as_posix().startswith(prefix) and not p.name.startswith(".tmp-"))
        return sorted(p.relative_to(self.root).as_posix() for p in base.rglob("*")
                      if p.is_file() and not p.name.startswith((".tmp-", ".gcsio")))

    def upload(self, local: str | Path, key: str) -> None:
        self.write_bytes(key, Path(local).read_bytes())

    def download(self, key: str, local: str | Path) -> Path:
        out = Path(local)
        out.parent.mkdir(parents=True, exist_ok=True)
        src = self.path(key)
        if not src.is_file():
            raise NotFound(self.uri(key))
        shutil.copyfile(src, out)
        return out

    def rsync_up(self, local_dir: str | Path, prefix: str) -> None:
        src = Path(local_dir)
        for p in sorted(src.rglob("*")):
            if p.is_file():
                rel = p.relative_to(src).as_posix()
                dst = self.path(f"{prefix.strip('/')}/{rel}")
                if not dst.is_file() or dst.read_bytes() != p.read_bytes():
                    self.write_bytes(f"{prefix.strip('/')}/{rel}", p.read_bytes())

    def rsync_down(self, prefix: str, local_dir: str | Path) -> Path:
        out = Path(local_dir)
        out.mkdir(parents=True, exist_ok=True)
        base = prefix.strip("/")
        for key in self.list(base + "/"):
            rel = key[len(base) + 1:]
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.path(key), dst)
        return out


# ── GCS through gcloud ───────────────────────────────────────────────────────

Runner = Callable[[list[str], bytes | None], subprocess.CompletedProcess]


def _run_gcloud(args: list[str], stdin: bytes | None = None) -> subprocess.CompletedProcess:
    exe = shutil.which("gcloud")
    if not exe:
        raise StoreError("gcloud is not on PATH (the lab runtime image ships it)")
    return subprocess.run([exe, *args], input=stdin, capture_output=True, timeout=1800, check=False)


def _last_line(b: bytes) -> str:
    lines = [ln for ln in b.decode("utf-8", "replace").splitlines() if ln.strip()]
    return lines[-1][:400] if lines else ""


_NOT_FOUND = ("No URLs matched", "matched no objects", "not found", "NotFound", "404", "does not exist")
_PRECONDITION = ("PreconditionFailed", "Precondition", "412", "conditionNotMet", "At least one of the pre-conditions")


class GcsBucket(Bucket):
    def __init__(self, url: str, runner: Runner | None = None):
        if not url.startswith("gs://"):
            raise StoreError(f"{url} is not gs://…")
        self.url = url.rstrip("/")
        self.run = runner or _run_gcloud

    def _call(self, args: list[str], stdin: bytes | None = None, *, key: str = "") -> subprocess.CompletedProcess:
        r = self.run(["storage", *args, "--quiet"] if args[0] not in ("cat",) else ["storage", *args], stdin)
        if r.returncode == 0:
            return r
        err = r.stderr if isinstance(r.stderr, bytes) else str(r.stderr or "").encode()
        last = _last_line(err)
        text = err.decode("utf-8", "replace")[-4000:]  # `cat` names the URL on a line of its own after the error
        if any(s in text for s in _PRECONDITION):
            raise PreconditionFailed(f"{self.uri(key) if key else args}: {last}")
        if any(s in text for s in _NOT_FOUND):
            raise NotFound(self.uri(key) if key else " ".join(args[:2]))
        raise StoreError(f"gcloud storage {args[0]} exited {r.returncode}: {last}")

    def read_bytes(self, key: str) -> bytes:
        r = self._call(["cat", self.uri(key)], key=key)
        return r.stdout if isinstance(r.stdout, bytes) else str(r.stdout).encode()

    def generation(self, key: str) -> int | None:
        try:
            r = self._call(["objects", "describe", self.uri(key), "--format=json"], key=key)
        except NotFound:
            return None
        out = r.stdout.decode("utf-8") if isinstance(r.stdout, bytes) else str(r.stdout)
        try:
            d = json.loads(out)
        except json.JSONDecodeError:
            raise StoreError(f"describe {self.uri(key)}: not JSON") from None
        g = d.get("generation")
        return int(g) if g is not None else None

    def write_bytes(self, key: str, data: bytes, *, if_generation: int | None = None) -> None:
        with tempfile.NamedTemporaryFile(prefix="gcsio-", delete=False) as f:
            f.write(data)
            tmp = f.name
        try:
            args = ["cp", tmp, self.uri(key)]
            if if_generation is not None:
                args.append(f"--if-generation-match={int(if_generation)}")
            self._call(args, key=key)
        finally:
            os.unlink(tmp)

    def delete(self, key: str, *, if_generation: int | None = None) -> bool:
        args = ["rm", self.uri(key)]
        if if_generation is not None:
            args.append(f"--if-generation-match={int(if_generation)}")
        try:
            self._call(args, key=key)
        except NotFound:
            return False
        return True

    def list(self, prefix: str) -> list[str]:
        prefix = str(prefix).lstrip("/")
        if not prefix.strip("/"):
            pattern = f"{self.url}/**"
        elif prefix.endswith("/"):  # a folder: never `runs/rt-10/…` for `runs/rt-1/`
            pattern = f"{self.uri(prefix)}/**"
        else:  # a name prefix (`manifests/bf788701`)
            pattern = f"{self.uri(prefix)}**"
        try:
            r = self._call(["ls", pattern], key=prefix)
        except NotFound:
            return []
        out = r.stdout.decode("utf-8") if isinstance(r.stdout, bytes) else str(r.stdout)
        base = self.url + "/"
        return sorted(ln.strip()[len(base):] for ln in out.splitlines() if ln.strip().startswith(base) and not ln.strip().endswith("/"))

    def upload(self, local: str | Path, key: str) -> None:
        self._call(["cp", str(local), self.uri(key)], key=key)

    def download(self, key: str, local: str | Path) -> Path:
        out = Path(local)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._call(["cp", self.uri(key), str(out)], key=key)
        return out

    def rsync_up(self, local_dir: str | Path, prefix: str) -> None:
        self._call(["rsync", "--recursive", str(local_dir), self.uri(prefix)], key=prefix)

    def rsync_down(self, prefix: str, local_dir: str | Path) -> Path:
        out = Path(local_dir)
        out.mkdir(parents=True, exist_ok=True)
        try:
            self._call(["rsync", "--recursive", self.uri(prefix), str(out)], key=prefix)
        except NotFound:
            pass
        return out


def bucket(url: str | Path | None = None, *, runner: Runner | None = None) -> Bucket:
    """`gs://…` → GCS; a path or `file://…` → a local folder. None reads
    $PIPELINE_BUCKET."""
    u = str(url or os.environ.get("PIPELINE_BUCKET") or "")
    if not u:
        raise StoreError("no bucket: pass --bucket or set PIPELINE_BUCKET")
    if u.startswith("gs://"):
        return GcsBucket(u, runner=runner)
    if u.startswith("file://"):
        return LocalBucket(u[len("file://"):])
    if is_bare_bucket_name(u) and (os.environ.get("PIPELINE_JOB") or not Path(u).exists()):
        # a bare bucket name (what the workflows' config carries) is GCS, never a folder of that name in the
        # job's working directory, where every write would silently vanish with the container
        return GcsBucket(f"gs://{u}", runner=runner)
    if os.environ.get("PIPELINE_JOB"):
        raise StoreError(f"bucket {u!r}: a pipeline job writes to gs://… only (PIPELINE_JOB is set)")
    return LocalBucket(u)


_BUCKET_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]")


def is_bare_bucket_name(u: str) -> bool:
    """`breadstack-sous-chef-int-pipeline`: no slash and GCS's naming rules (a relative folder is `./name`)."""
    return bool(_BUCKET_NAME.fullmatch(u)) and "/" not in u and not u.endswith((".json", ".jsonl"))


# module-level helpers (DESIGN I2 names) ---------------------------------------

def read_json(b: Bucket, key: str, default: Any = _MISSING) -> Any:
    return b.read_json(key, default)


def write_json(b: Bucket, key: str, obj: Any) -> None:
    b.write_json(key, obj)


def rsync_up(b: Bucket, local_dir: str | Path, prefix: str) -> None:
    b.rsync_up(local_dir, prefix)


def rsync_down(b: Bucket, prefix: str, local_dir: str | Path) -> Path:
    return b.rsync_down(prefix, local_dir)


def lock(b: Bucket, name: str, run_id: str, ttl_s: float):
    return b.lock(name, run_id, ttl_s)


def exists(b: Bucket, key: str) -> bool:
    return b.exists(key)


class Syncer:
    """`rsync_up` of a local folder every `every_s` seconds on a thread, and
    once more at stop (r4's collection logs: a retried task resumes from
    them). Errors are counted, never raised, until the final sync."""

    def __init__(self, b: Bucket, local_dir: str | Path, prefix: str, every_s: float = 60.0):
        import threading
        self.b, self.local, self.prefix, self.every = b, Path(local_dir), prefix, every_s
        self.errors = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.wait(self.every):
            try:
                self.b.rsync_up(self.local, self.prefix)
            except Exception:  # noqa: BLE001 — a transient storage error must not kill the stage
                self.errors += 1

    def __enter__(self) -> "Syncer":
        self.local.mkdir(parents=True, exist_ok=True)
        self._t.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._t.join(timeout=5)
        self.b.rsync_up(self.local, self.prefix)


def wait_for(predicate: Callable[[], bool], timeout_s: float, every_s: float = 5.0) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(every_s)
    return predicate()
