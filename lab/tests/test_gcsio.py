"""gcsio: the local backend (DRY, tests) and the gcloud command lines of the
GCS backend (a fake runner: no network)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from souschef_lab import gcsio  # noqa: E402


def test_local_json_list_rsync_and_missing(tmp_path):
    b = gcsio.bucket(tmp_path / "b")
    assert b.read_json("state/x.json", None) is None
    with pytest.raises(gcsio.NotFound):
        b.read_json("state/x.json")
    b.write_json("state/x.json", {"a": 1})
    assert b.read_json("state/x.json") == {"a": 1} and b.exists("state/x.json")
    src = tmp_path / "src"
    (src / "d").mkdir(parents=True)
    (src / "d" / "f.txt").write_text("hi")
    b.rsync_up(src, "runs/r1/collect")
    assert b.list("runs/r1/") == ["runs/r1/collect/d/f.txt"]
    out = b.rsync_down("runs/r1/collect", tmp_path / "back")
    assert (out / "d" / "f.txt").read_text() == "hi"
    assert b.list("manifests/zz") == []
    b.write_json("manifests/abcdef12xyz.json", {})
    assert b.list("manifests/abcdef12") == ["manifests/abcdef12xyz.json"]
    with pytest.raises(gcsio.StoreError):
        b.read_json("../etc/passwd")


def test_local_generation_preconditions(tmp_path):
    b = gcsio.bucket(f"file://{tmp_path / 'b'}")
    b.write_json("k.json", {"v": 1}, if_generation=0)
    with pytest.raises(gcsio.PreconditionFailed):
        b.write_json("k.json", {"v": 2}, if_generation=0)
    g = b.generation("k.json")
    b.write_json("k.json", {"v": 3}, if_generation=g)
    assert b.generation("k.json") != g
    with pytest.raises(gcsio.PreconditionFailed):
        b.delete("k.json", if_generation=g)


def test_locks_take_over_only_when_expired(tmp_path):
    b = gcsio.bucket(tmp_path / "b")
    now = gcsio.utc_now()
    b.acquire("gpu-eval", "dr-1", 60, now=now)
    with pytest.raises(gcsio.LockHeld) as e:
        b.acquire("gpu-eval", "rt-2", 60, now=now)
    assert e.value.holder["run_id"] == "dr-1"
    b.acquire("gpu-eval", "dr-1", 600, now=now)  # the holder refreshes
    from datetime import timedelta
    b.acquire("gpu-eval", "rt-2", 60, now=now + timedelta(seconds=601))  # expired: taken over
    assert not b.release("gpu-eval", "dr-1") and b.release("gpu-eval", "rt-2")
    with b.lock("retrain", "rt-3", 60) as doc:
        assert doc["run_id"] == "rt-3" and b.read_lock("retrain")["run_id"] == "rt-3"
    assert b.read_lock("retrain") is None


# gcloud 585's stderr for each case, verbatim from the int bucket (read-only probes, 2026-10-12) or, for the
# conditional writes we may not probe, from its source (api_lib/storage/errors.py: "HTTPError {code}: {message}").
# gcloud names the object in its errors (describe url-encodes it), so object names that contain "412" or "404"
# are the point of these tests.
def _describe_404(url: str) -> bytes:
    from urllib.parse import quote
    b, _, k = url[len("gs://"):].partition("/")
    return f"ERROR: (gcloud.storage.objects.describe) gs://{b}/{quote(k, safe='')} not found: 404.\n".encode()


def _no_match(cmd: str, url: str) -> bytes:
    return f"ERROR: (gcloud.storage.{cmd}) The following URLs matched no objects or files:\n{url}\n".encode()


def _http(cmd: str, code: int, msg: str) -> bytes:
    return f"ERROR: (gcloud.storage.{cmd}) HTTPError {code}: {msg}\n".encode()


DENIED = ("souschef-drift@breadstack-sous-chef-int.iam.gserviceaccount.com does not have storage.objects.get access "
          "to the Google Cloud Storage object. Permission 'storage.objects.get' denied on resource (or it may not exist).")
PRECOND = "At least one of the pre-conditions you specified did not hold."


class FakeGcloud:
    def __init__(self):
        self.calls: list[list[str]] = []
        self.objects: dict[str, tuple[bytes, int]] = {}
        self.types: dict[str, str] = {}
        self.fail: dict[str, bytes] = {}  # url → stderr of an injected failure (rc 1)

    def __call__(self, args: list[str], stdin: bytes | None = None) -> subprocess.CompletedProcess:
        self.calls.append(args)
        cmd = args[1]
        def done(out=b"", rc=0, err=b""):
            return subprocess.CompletedProcess(args, rc, out, err)
        url = args[3] if cmd == "objects" else args[2]
        if url in self.fail:
            return done(rc=1, err=self.fail[url])
        if cmd == "cat":
            o = self.objects.get(args[2])
            return done(o[0]) if o else done(rc=1, err=_no_match("cat", args[2]))
        if cmd == "objects":
            o = self.objects.get(args[3])
            return done(json.dumps({"generation": str(o[1])}).encode()) if o else done(rc=1, err=_describe_404(args[3]))
        if cmd == "cp":
            src, dst = args[2], args[3]
            if src.startswith("gs://"):
                if src not in self.objects:
                    return done(rc=1, err=_no_match("cp", src))
                Path(dst).write_bytes(self.objects[src][0])
                return done()
            cond = [a for a in args if a.startswith("--if-generation-match=")]
            if cond:
                want = int(cond[0].split("=")[1])
                have = self.objects.get(dst, (b"", 0))[1]
                if want != have:
                    return done(rc=1, err=f"Copying file://{src} to {dst}\n".encode() + _http("cp", 412, PRECOND))
            data = Path(src).read_bytes()
            self.objects[dst] = (data, self.objects.get(dst, (b"", 0))[1] + 1)
            ctype = [a.split("=", 1)[1] for a in args if a.startswith("--content-type=")]
            self.types[dst] = ctype[0] if ctype else "application/octet-stream"
            return done()
        if cmd == "rm":
            if args[2] not in self.objects:
                return done(rc=1, err=_no_match("rm", args[2]))
            cond = [a for a in args if a.startswith("--if-generation-match=")]
            if cond and int(cond[0].split("=")[1]) != self.objects[args[2]][1]:
                return done(rc=1, err=_http("rm", 412, PRECOND))
            del self.objects[args[2]]
            return done()
        if cmd == "ls":
            prefix = args[2].rstrip("*").rstrip("/")
            hits = [k for k in sorted(self.objects) if k.startswith(prefix)]
            if not hits:
                return done(rc=1, err=b"ERROR: (gcloud.storage.ls) One or more URLs matched no objects.\n")
            return done("\n".join(hits).encode())
        if cmd == "rsync":
            src = args[3]
            if src.startswith("gs://") and not any(k.startswith(src.rstrip("/") + "/") for k in self.objects):
                return done(rc=1, err=f"ERROR: (gcloud.storage.rsync) Did not find existing container at: {src}\n".encode())
            return done()
        return done()


def test_gcs_backend_command_lines():
    fake = FakeGcloud()
    b = gcsio.bucket("gs://bkt", runner=fake)
    assert b.read_json("state/m.json", None) is None
    b.write_json("state/m.json", {"a": 1})
    assert b.read_json("state/m.json") == {"a": 1}
    b.acquire("gpu-eval", "dr-1", 60)
    cp = [c for c in fake.calls if c[1] == "cp" and c[3].endswith("locks/gpu-eval.json")]
    assert "--if-generation-match=0" in cp[0]
    with pytest.raises(gcsio.LockHeld):
        b.acquire("gpu-eval", "rt-2", 60)
    assert b.release("gpu-eval", "dr-1")
    rm = [c for c in fake.calls if c[1] == "rm"][-1]
    assert rm[2] == "gs://bkt/state/locks/gpu-eval.json" and any(a.startswith("--if-generation-match=") for a in rm)
    assert b.list("state/") == ["state/m.json"]
    assert fake.calls[-1][2] == "gs://bkt/state/**"
    b.list("manifests/bf78")
    assert fake.calls[-1][2] == "gs://bkt/manifests/bf78**"
    b.rsync_up("/tmp/x", "runs/r")
    assert fake.calls[-1][:4] == ["storage", "rsync", "--recursive", "/tmp/x"]


def test_a_bare_bucket_name_is_gcs_never_a_folder(tmp_path, monkeypatch):
    """The workflows' config carries the bucket's bare name: a job must never write to a folder of that name."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PIPELINE_JOB", raising=False)
    b = gcsio.bucket("breadstack-sous-chef-int-pipeline")
    assert isinstance(b, gcsio.GcsBucket) and b.url.rstrip("/") == "gs://breadstack-sous-chef-int-pipeline"
    assert not (tmp_path / "breadstack-sous-chef-int-pipeline").exists()
    assert isinstance(gcsio.bucket(str(tmp_path / "local")), gcsio.LocalBucket)
    monkeypatch.setenv("PIPELINE_JOB", "souschef-drift")
    with pytest.raises(gcsio.StoreError, match="gs://"):
        gcsio.bucket(str(tmp_path / "local"))


# ── error classification (finding ST-1: never by digits inside an object name) ─────────────────────────────────────

CAPTURED = {  # gcloud 585 against gs://breadstack-sous-chef-int-pipeline and a bucket we may not read, 2026-10-12
    "describe_missing": ("ERROR: (gcloud.storage.objects.describe) gs://breadstack-sous-chef-int-pipeline/"
                         "baselines%2Fsha256%3Aab412cd404%2Fbaseline.json not found: 404.\n", "not_found"),
    "cat_missing": ("ERROR: (gcloud.storage.cat) The following URLs matched no objects or files:\n"
                    "gs://breadstack-sous-chef-int-pipeline/runs/dr-20261013-0404-poll/x412.json\n", "not_found"),
    "ls_no_match": ("ERROR: (gcloud.storage.ls) One or more URLs matched no objects.\n", "not_found"),
    "rsync_missing": ("ERROR: (gcloud.storage.rsync) Did not find existing container at: "
                      "gs://breadstack-sous-chef-int-pipeline/runs/nope-0412\n", "not_found"),
    "cat_403": ("ERROR: (gcloud.storage.cat) HTTPError 403: The billing account for the owning project is disabled in "
                "state absent. This command is authenticated as someone@example.com which is the active account "
                "specified by the [core/account] property\n", "error"),
    "ls_403": ("ERROR: (gcloud.storage.ls) [someone@example.com] does not have permission to access b instance [test] "
               "(or it may not exist): someone@example.com does not have storage.objects.list access to the Google "
               "Cloud Storage bucket. Permission 'storage.objects.list' denied on resource "
               "'//storage.googleapis.com/projects/_/buckets/test' (or it may not exist). This command is "
               "authenticated as someone@example.com which is the active account specified by the [core/account] "
               "property.\n", "error"),
    "cp_412": ("Copying file:///tmp/gcsio-0404abcd to gs://b/state/locks/gpu-eval.json\n"
               "ERROR: (gcloud.storage.cp) HTTPError 412: " + PRECOND + "\n", "precondition"),
    "rm_412_on_404_name": ("ERROR: (gcloud.storage.rm) HTTPError 412: " + PRECOND + "\n", "precondition"),
    "503_exhausted": ("ERROR: (gcloud.storage.cat) HTTPError 503: Backend Error\n", "error"),
    "usage": ("ERROR: (gcloud.storage) Invalid choice: 'objectz'.\n", "error"),
}


@pytest.mark.parametrize("name", sorted(CAPTURED))
def test_classify_real_gcloud_texts(name):
    text, want = CAPTURED[name]
    assert gcsio.classify_gcloud_error(text.encode()) == want


@pytest.mark.parametrize("name", ["baselines/sha256:ab412cd404/baseline.json", "runs/dr-20261013-0404-poll/result.json",
                                  "manifests/0412404412.json", "runs/rt-20260412-0412-404/run.json"])
def test_missing_objects_whose_names_hold_412_or_404_read_as_missing(name):
    fake = FakeGcloud()
    b = gcsio.bucket("gs://breadstack-sous-chef-int-pipeline", runner=fake)
    assert b.read_json(name, None) is None
    with pytest.raises(gcsio.NotFound):
        b.read_json(name)
    assert b.generation(name) is None and not b.exists(name)
    assert b.delete(name) is False
    assert b.list(name.rsplit("/", 1)[0] + "/") == []


def test_rsync_down_of_a_missing_prefix_is_empty_not_an_error(tmp_path):
    fake = FakeGcloud()
    b = gcsio.bucket("gs://bkt", runner=fake)
    out = b.rsync_down("runs/nope-0412/collect", tmp_path / "out")
    assert out.is_dir() and list(out.iterdir()) == []


def test_real_errors_on_names_holding_404_are_never_missing(tmp_path):
    """A 403 (or a 5xx after gcloud's retries) must raise: read_json(key, default) would otherwise hand the
    default to a caller that then rebuilds a baseline, or skips a heartbeat, on a read that never happened."""
    fake = FakeGcloud()
    b = gcsio.bucket("gs://bkt", runner=fake)
    key = "runs/dr-20261013-0404-poll/result.json"
    fake.fail[f"gs://bkt/{key}"] = _http("cat", 403, DENIED)
    with pytest.raises(gcsio.StoreError) as e:
        b.read_json(key, None)
    assert not isinstance(e.value, (gcsio.NotFound, gcsio.PreconditionFailed))
    fake.fail[f"gs://bkt/{key}"] = _http("objects.describe", 503, "Backend Error")
    with pytest.raises(gcsio.StoreError) as e:
        b.generation(key)
    assert not isinstance(e.value, gcsio.NotFound)
    # cp names both ends in its progress line before the error
    fake.fail[f"gs://bkt/{key}"] = (f"Copying gs://bkt/{key} to file:///tmp/w-0412/result.json\n".encode()
                                    + _http("cp", 403, DENIED))
    with pytest.raises(gcsio.StoreError) as e:
        b.download(key, tmp_path / "w-0404" / "x.json")
    assert not isinstance(e.value, (gcsio.NotFound, gcsio.PreconditionFailed))


def test_conditional_writes_on_names_holding_404_lose_as_precondition():
    fake = FakeGcloud()
    b = gcsio.bucket("gs://bkt", runner=fake)
    b.write_json("state/locks/dr-0404.json", {"v": 1}, if_generation=0)
    with pytest.raises(gcsio.PreconditionFailed):
        b.write_json("state/locks/dr-0404.json", {"v": 2}, if_generation=0)
    with pytest.raises(gcsio.PreconditionFailed):
        b.delete("state/locks/dr-0404.json", if_generation=99)
    # a lock taken over by a run whose id holds both numbers
    b.acquire("gpu-eval-0404", "rt-20260412-0412", 60)
    with pytest.raises(gcsio.LockHeld):
        b.acquire("gpu-eval-0404", "dr-20261013-0404-poll", 60)
    assert b.release("gpu-eval-0404", "rt-20260412-0412")


# ── what the uploads look like ───────────────────────────────────────────────────────────────────────────────────

def test_json_is_uploaded_as_application_json_and_strict(tmp_path):
    fake = FakeGcloud()
    b = gcsio.bucket("gs://bkt", runner=fake)
    b.write_json("drift/dr-1/result.json", {"loss": float("nan"), "xs": [float("inf"), 1.5], "t": (1, float("-inf"))})
    assert fake.types["gs://bkt/drift/dr-1/result.json"] == "application/json"
    raw = fake.objects["gs://bkt/drift/dr-1/result.json"][0].decode()
    assert "NaN" not in raw and "Infinity" not in raw
    assert json.loads(raw) == {"loss": None, "xs": [None, 1.5], "t": [1, None]}
    # every .json key through write_bytes too (retrain's Store), jsonl and text by their extension
    b.write_bytes("runs/r/model/meta.json", b"{}\n")
    b.write_bytes("runs/r/cases.jsonl", b"{}\n")
    b.write_bytes("runs/r/train/DONE", b"x\n")
    assert fake.types["gs://bkt/runs/r/model/meta.json"] == "application/json"
    assert fake.types["gs://bkt/runs/r/cases.jsonl"] == "application/x-ndjson"
    assert fake.types["gs://bkt/runs/r/train/DONE"] == "application/octet-stream"
    # the local backend writes the same bytes
    lb = gcsio.bucket(tmp_path / "b")
    lb.write_json("x.json", {"v": float("nan")})
    assert json.loads((tmp_path / "b" / "x.json").read_text()) == {"v": None}


def test_gcloud_runs_without_parallel_composite_uploads(monkeypatch):
    monkeypatch.delenv("CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED", raising=False)
    assert gcsio.gcloud_env()["CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED"] == "False"
    monkeypatch.setenv("CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED", "True")  # an explicit choice wins
    assert gcsio.gcloud_env()["CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED"] == "True"
    seen = {}

    def fake_run(argv, **kw):
        seen.update(kw)
        return subprocess.CompletedProcess(argv, 0, b"", b"")
    monkeypatch.delenv("CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED")
    monkeypatch.setattr(gcsio.shutil, "which", lambda _: "/usr/bin/gcloud")
    monkeypatch.setattr(gcsio.subprocess, "run", fake_run)
    gcsio._run_gcloud(["storage", "ls", "gs://bkt/**"])
    assert seen["env"]["CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED"] == "False"
