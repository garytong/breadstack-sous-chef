"""harness — the deployed Sous Chef service, alone, inside a job (DESIGN §1.3;
the in-job equivalent of tools/vm-solo.sh `serve`).

    with Harness({"current": eval_url, "candidate": cand_url}, run_id=…) as h:
        client = h.client()                # collect.Client on the local service, a random per-run key
        … exam rows through h.url …

It starts:

  1. Postgres (the server binaries in $PG_BIN, Postgres 16 in the lab image): `initdb` into
     <workdir>/pg as the container's user (trust auth, no password), listening on 127.0.0.1 only
     (port 5433 when free, else any free port), its socket in <workdir>.
  2. The DEPLOYED service binary ($SOUS_CHEF_BIN, /opt/sous-chef/bin/sous-chef in the image —
     built from the same tree as the int service image) with PORT, MIGRATE_AT_BOOT=on, a random
     SERVICE_KEYS, BUILD_ID=<build>-harness-<run>, MODEL_CALL_TIMEOUT=180s, MAX_CONCURRENT_TURNS=4,
     no ANTHROPIC_API_KEY and no THERMOMETER_*: nothing leaves the job but the model calls.
  3. MODEL_ROUTES = one messages-compatible route per arm, exactly the deployed blueberry-muffin route
     (infra/envs/int/main.tf `inhouse_route`) MINUS `fallback`: the student alone, never Claude:
       {"kind":"messages-compatible","url":<GPU service>,"auth":"id-token","model":"sous-chef",
        "max_tokens":1024,"stream":false,"temperature":0.3,"prime":"on","fast_finish":"on","report_as":<arm>}
     The service mints ID tokens itself (model/idtoken.go: the metadata server of the Cloud Run Job).
  4. `/version` asserted: every arm's route_info has fields_context true and fallback null.

`auth="none"` and `route_overrides` exist for the lab's own tests (a fake model on 127.0.0.1).
Logs: <workdir>/{postgres.log, service.log}. Standard library only.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

DEFAULT_BIN = "/opt/sous-chef/bin/sous-chef"
DEFAULT_PG_PORT = 5433
ROUTE_DEFAULTS = {"kind": "messages-compatible", "auth": "id-token", "model": "sous-chef", "max_tokens": 1024,
                  "stream": False, "temperature": 0.3, "prime": "on", "fast_finish": "on"}


class HarnessError(RuntimeError):
    pass


def free_port(prefer: int | None = None) -> int:
    if prefer:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", prefer))
                return prefer
            except OSError:
                pass
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def pg_bin_dir() -> Path:
    for c in (os.environ.get("PG_BIN"), "/usr/lib/postgresql/16/bin"):
        if c and (Path(c) / "pg_ctl").is_file():
            return Path(c)
    exe = shutil.which("pg_ctl")
    if exe:
        return Path(exe).parent
    raise HarnessError("no Postgres server binaries (set PG_BIN; the lab image has /usr/lib/postgresql/16/bin)")


def service_bin(explicit: str | Path | None = None) -> Path:
    for c in (explicit, os.environ.get("SOUS_CHEF_BIN"), DEFAULT_BIN):
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return Path(c)
    raise HarnessError("no sous-chef service binary (set SOUS_CHEF_BIN; the lab image has /opt/sous-chef/bin/sous-chef)")


def baked_build_id(default: str = "dev") -> str:
    p = Path(os.environ.get("SOUS_CHEF_BUILD_ID_FILE") or "/opt/sous-chef/BUILD_ID")
    return p.read_text(encoding="utf-8").strip() if p.is_file() else default


def route_spec(url: str, report_as: str, *, auth: str = "id-token", overrides: dict | None = None) -> dict:
    r = {**ROUTE_DEFAULTS, "url": url, "auth": auth, "report_as": report_as}
    r.update(overrides or {})
    r.pop("fallback", None)
    return r


def _get(url: str, headers: dict | None = None, timeout: float = 5.0) -> tuple[int, bytes]:
    req = urlrequest.Request(url)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urlrequest.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urlerror.HTTPError as e:
        return e.code, e.read()


class Harness:
    def __init__(self, routes: dict[str, str], *, run_id: str = "local", workdir: str | Path | None = None,
                 binary: str | Path | None = None, build_id: str | None = None, auth: str = "id-token",
                 route_overrides: dict | None = None, default_route: str | None = None, port: int | None = None,
                 pg_port: int | None = None, model_call_timeout: str = "180s", max_concurrent_turns: int = 4,
                 extra_env: dict | None = None, boot_timeout_s: float = 90.0):
        if not routes:
            raise HarnessError("at least one route (arm → GPU service URL)")
        for name in routes:
            if not name.replace("-", "").isalnum() or not name.islower():
                raise HarnessError(f"route name {name!r}: lowercase letters, digits and dashes")
        self.routes = dict(routes)
        self.run_id = run_id
        self.own_workdir = workdir is None
        self.workdir = Path(workdir or tempfile.mkdtemp(prefix="souschef-harness-"))
        self.binary = service_bin(binary)
        self.build = build_id or baked_build_id()
        self.auth = auth
        self.route_overrides = route_overrides or {}
        self.default_route = default_route or next(iter(self.routes))
        self.port = port
        self.pg_port = pg_port
        self.model_call_timeout = model_call_timeout
        self.max_concurrent_turns = max_concurrent_turns
        self.extra_env = extra_env or {}
        self.boot_timeout_s = boot_timeout_s
        self.key = "lab-harness-" + secrets.token_urlsafe(24)
        self.pg_dir = self.workdir / "pg"
        self.proc: subprocess.Popen | None = None
        self._log = None
        self._pg_started = False
        self.version: dict = {}

    # ── public ──
    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def build_id(self) -> str:
        return f"{self.build}-harness-{self.run_id}"

    def model_routes(self) -> dict[str, dict]:
        return {name: route_spec(u, name, auth=self.auth, overrides=self.route_overrides) for name, u in self.routes.items()}

    def env(self) -> dict[str, str]:
        keep = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "TMPDIR", "TZ", "LANG", "GCE_METADATA_HOST")}
        return {**keep, "PORT": str(self.port), "DATABASE_URL": self.database_url, "MIGRATE_AT_BOOT": "on",
                "SERVICE_KEYS": self.key, "BUILD_ID": self.build_id, "MODEL_ROUTES": json.dumps(self.model_routes()),
                "MODEL_ROUTE_ALLOWLIST": ",".join(self.routes), "DEFAULT_ROUTE": self.default_route,
                "MODEL_CALL_TIMEOUT": self.model_call_timeout, "MAX_CONCURRENT_TURNS": str(self.max_concurrent_turns),
                "TURN_TTL": "15m", "TRACE_RETENTION_DAYS": "30", **self.extra_env}

    @property
    def database_url(self) -> str:
        return f"postgres://souschef@127.0.0.1:{self.pg_port}/souschef?sslmode=disable"

    def client(self, build_id: str = "lab-drift", timeout_s: float = 300.0):
        from .collect import Client
        return Client(self.url, self.key, timeout_s=timeout_s, build_id=build_id)

    # ── lifecycle ──
    def __enter__(self) -> "Harness":
        try:
            self.start_postgres()
            self.start_service()
            self.check_version()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start_postgres(self) -> None:
        pg = pg_bin_dir()
        self.pg_port = self.pg_port or free_port(DEFAULT_PG_PORT)
        if not (self.pg_dir / "PG_VERSION").is_file():
            self.pg_dir.mkdir(parents=True, exist_ok=True)
            r = subprocess.run([str(pg / "initdb"), "-D", str(self.pg_dir), "-U", "souschef", "-A", "trust", "-E", "UTF8",
                                "--no-instructions"], capture_output=True, text=True, check=False)
            if r.returncode != 0:
                raise HarnessError(f"initdb exited {r.returncode}: {r.stderr.strip()[-800:]}")
        # a Unix socket path is limited to ~107 bytes: a deep workdir would fail at start, so the socket gets a
        # short folder of its own under /tmp (TCP on 127.0.0.1 is what the service uses)
        sock = Path(tempfile.mkdtemp(prefix="pgs-", dir="/tmp" if Path("/tmp").is_dir() else None))
        self._sock_dir = sock
        opts = f"-p {self.pg_port} -c listen_addresses=127.0.0.1 -k {sock} -c fsync=off -c max_connections=60"
        r = subprocess.run([str(pg / "pg_ctl"), "-D", str(self.pg_dir), "-l", str(self.workdir / "postgres.log"), "-o", opts,
                            "-w", "-t", "60", "start"], capture_output=True, text=True, check=False)
        if r.returncode != 0:
            raise HarnessError(f"pg_ctl start exited {r.returncode}: {r.stderr.strip()[-400:]} "
                               f"(log {self.workdir / 'postgres.log'})")
        self._pg_started = True
        psql = pg / "psql"
        if not psql.is_file():
            psql = Path(shutil.which("psql") or "psql")
        r = subprocess.run([str(psql), "-h", "127.0.0.1", "-p", str(self.pg_port), "-U", "souschef", "-d", "postgres",
                            "-tAc", "SELECT 1 FROM pg_database WHERE datname='souschef'"], capture_output=True, text=True, check=False)
        if r.stdout.strip() != "1":
            r = subprocess.run([str(psql), "-h", "127.0.0.1", "-p", str(self.pg_port), "-U", "souschef", "-d", "postgres",
                                "-c", "CREATE DATABASE souschef"], capture_output=True, text=True, check=False)
            if r.returncode != 0:
                raise HarnessError(f"CREATE DATABASE souschef: {r.stderr.strip()[-400:]}")

    def start_service(self, attempts: int = 3) -> None:
        fixed = self.port is not None
        for attempt in range(1, attempts + 1):
            try:
                self._start_service()
                return
            except HarnessError as e:
                if fixed or "address already in use" not in str(e) or attempt == attempts:
                    raise
                self.port = None  # a port taken between the probe and the bind: another one

    def _start_service(self) -> None:
        self.port = self.port or free_port(8091)
        if self._log is None:
            self._log = open(self.workdir / "service.log", "ab")
        self.proc = subprocess.Popen([str(self.binary)], env=self.env(), stdout=self._log, stderr=subprocess.STDOUT,
                                     cwd=str(self.workdir))
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.boot_timeout_s:
            if self.proc.poll() is not None:
                raise HarnessError(f"the service exited {self.proc.returncode} at boot: {self.log_tail()}")
            try:
                status, _ = _get(self.url + "/healthz", timeout=2)
                if status == 200:
                    return
            except (urlerror.URLError, OSError):
                pass
            time.sleep(0.5)
        raise HarnessError(f"the service did not answer /healthz in {self.boot_timeout_s:.0f}s: {self.log_tail()}")

    def check_version(self) -> dict:
        status, raw = _get(self.url + "/version", headers={"Authorization": f"Bearer {self.key}"}, timeout=10)
        if status != 200:
            raise HarnessError(f"/version answered {status}")
        doc = json.loads(raw.decode("utf-8"))
        info = doc.get("route_info") or {}
        for name in self.routes:
            ri = info.get(name)
            if not isinstance(ri, dict):
                raise HarnessError(f"/version has no route_info for {name} (a service build before route_info?)")
            if ri.get("fields_context") is not True or ri.get("fallback") is not None:
                raise HarnessError(f"/version route {name}: fields_context {ri.get('fields_context')!r}, "
                                   f"fallback {ri.get('fallback')!r} (want true and null)")
        if doc.get("build_id") not in (None, self.build_id):
            raise HarnessError(f"/version build_id {doc.get('build_id')!r} is not {self.build_id!r}")
        self.version = doc
        return doc

    def log_tail(self, n: int = 15) -> str:
        p = self.workdir / "service.log"
        if not p.is_file():
            return ""
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
        from .pipeline_common import scrub
        return scrub(" | ".join(lines), extra=[self.key])[-2000:]

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        self.proc = None
        if self._log is not None:
            self._log.close()
            self._log = None
        if self._pg_started:
            try:
                subprocess.run([str(pg_bin_dir() / "pg_ctl"), "-D", str(self.pg_dir), "-m", "fast", "-w", "stop"],
                               capture_output=True, text=True, timeout=60, check=False)
            except (HarnessError, subprocess.SubprocessError, OSError):
                pass
            self._pg_started = False
        sock = getattr(self, "_sock_dir", None)
        if sock is not None:
            shutil.rmtree(sock, ignore_errors=True)
            self._sock_dir = None

    def describe(self) -> dict[str, Any]:
        return {"url": self.url, "build_id": self.build_id, "routes": {n: {k: v for k, v in r.items() if k != "url"} | {"url": r["url"]}
                                                                        for n, r in self.model_routes().items()},
                "pg_port": self.pg_port, "workdir": str(self.workdir)}
