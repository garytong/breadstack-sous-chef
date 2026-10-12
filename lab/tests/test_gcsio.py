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


class FakeGcloud:
    def __init__(self):
        self.calls: list[list[str]] = []
        self.objects: dict[str, tuple[bytes, int]] = {}

    def __call__(self, args: list[str], stdin: bytes | None = None) -> subprocess.CompletedProcess:
        self.calls.append(args)
        cmd = args[1]
        def done(out=b"", rc=0, err=b""):
            return subprocess.CompletedProcess(args, rc, out, err)
        if cmd == "cat":
            o = self.objects.get(args[2])
            return done(o[0]) if o else done(rc=1, err=b"ERROR: (gcloud.storage.cat) The following URLs matched no objects or files:\n"
                                                + args[2].encode() + b"\n")
        if cmd == "objects":
            o = self.objects.get(args[3])
            return done(json.dumps({"generation": str(o[1])}).encode()) if o else done(rc=1, err=b"ERROR: gs://x not found: 404")
        if cmd == "cp":
            src, dst = args[2], args[3]
            cond = [a for a in args if a.startswith("--if-generation-match=")]
            if cond:
                want = int(cond[0].split("=")[1])
                have = self.objects.get(dst, (b"", 0))[1]
                if want != have:
                    return done(rc=1, err=b"ERROR: HTTPError 412: At least one of the pre-conditions you specified did not hold.")
            data = Path(src).read_bytes()
            self.objects[dst] = (data, self.objects.get(dst, (b"", 0))[1] + 1)
            return done()
        if cmd == "rm":
            if args[2] not in self.objects:
                return done(rc=1, err=b"ERROR: No URLs matched")
            del self.objects[args[2]]
            return done()
        if cmd == "ls":
            prefix = args[2].rstrip("*").rstrip("/")
            return done("\n".join(k for k in sorted(self.objects) if k.startswith(prefix)).encode())
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
