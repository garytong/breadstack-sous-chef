#!/usr/bin/env python3
"""Unit tests of e2e_dry's fake Google APIs: they must answer as the real ones do (audit 2026-10-12), independent of
how the workflows are written. No network, no lab jobs, no thermometer.

    python3 -m unittest infra/modules/pipeline/tests/test_e2e_fakes.py
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e_dry as E  # noqa: E402
import wfsim  # noqa: E402

RUN = "https://run.googleapis.com/v2/"
GCS = "https://storage.googleapis.com/storage/v1/b/" + E.BUCKET
UPLOAD = "https://storage.googleapis.com/upload/storage/v1/b/" + E.BUCKET + "/o"
AR = f"https://artifactregistry.googleapis.com/v1/projects/{E.PROJECT}/locations/{E.NE2}/repositories/sous-chef/packages/"
OAUTH = {"type": "OAuth2"}
T0 = 1_800_000_000.0


class FakeBase(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        work = Path(self.td.name)
        self.w = E.World(work)
        self.w.bucket.mkdir(parents=True)
        self.w.logs.mkdir(parents=True)
        self.w.s = {"clock": T0, **E.gcp_fixture(), "executions": {}, "vertex": {}, "builds": {}, "started": [],
                    "posted": [], "phases": {}}
        self.clock = wfsim.VirtualClock(T0)
        self.g = E.FakeGCP(self.w, int_url="http://int.local", int_key="k", therm_url="http://therm.local",
                           therm_key="tk", snapshot=work, interp_ref={"clock": self.clock})
        self.g.ops_allowed = False

    def tearDown(self):
        self.td.cleanup()

    def call(self, method, url, query=None, body=None, headers=None, auth=OAUTH):
        code, out, hdr = self.g(method, url, dict(query or {}), dict(headers or {}), body, auth)
        hdr = {wfsim.canonical_header(k): v for k, v in hdr.items()}
        return code, wfsim.Interpreter.response_body(out, hdr, "test"), hdr

    def svc_url(self, name, region=E.GPU):
        return f"{RUN}projects/{E.PROJECT}/locations/{region}/services/{name}"

    def get_svc(self, name, region=E.GPU):
        code, s, _ = self.call("GET", self.svc_url(name, region))
        self.assertEqual(code, 200)
        return s


class CloudRunServices(FakeBase):
    def test_patch_is_a_not_done_operation_the_workflow_account_cannot_read(self):
        s = self.get_svc("sous-chef-model-eval")
        s["template"]["containers"][0]["image"] = f"{E.MODEL_REPO}/model@{E.R4_DIGEST}"
        code, op, _ = self.call("PATCH", self.svc_url("sous-chef-model-eval"), body=s)
        self.assertEqual((code, op["done"]), (200, False))
        self.assertRegex(op["name"], r"^projects/[^/]+/locations/us-east4/operations/[0-9a-f-]+$")
        code, err, _ = self.call("GET", RUN + op["name"])
        self.assertEqual(code, 403)
        self.assertIn("run.operations.get", err["error"]["message"])

    def test_the_resource_reconciles_on_the_clock(self):
        s = self.get_svc("sous-chef-model-eval")
        old = s["latestReadyRevision"]
        s["template"]["containers"][0]["image"] = f"{E.MODEL_REPO}/model@{E.R4_DIGEST}"
        self.call("PATCH", self.svc_url("sous-chef-model-eval"), body=s)
        now = self.get_svc("sous-chef-model-eval")
        self.assertTrue(now["reconciling"])
        self.assertEqual((now["generation"], now["observedGeneration"]), ("2", "1"))
        self.assertEqual(now["terminalCondition"]["state"], "CONDITION_RECONCILING")
        self.assertEqual(now["latestCreatedRevision"], old)  # not created yet: ready == created proves nothing
        self.clock.advance(E.RECONCILE_S["gpu"])
        done = self.get_svc("sous-chef-model-eval")
        self.assertNotIn("reconciling", done)
        self.assertEqual(done["observedGeneration"], done["generation"])
        self.assertEqual(done["terminalCondition"]["state"], "CONDITION_SUCCEEDED")
        self.assertNotEqual(done["latestReadyRevision"], old)
        self.assertEqual(done["trafficStatuses"][0]["revision"], E.short_rev(done["latestReadyRevision"]))
        code, rev, _ = self.call("GET", RUN + done["latestReadyRevision"])
        self.assertIn("@sha256:", rev["containers"][0]["image"])

    def test_identical_patch_is_a_noop_unless_forced(self):
        s = self.get_svc("sous-chef-model-eval")
        self.call("PATCH", self.svc_url("sous-chef-model-eval"), body=s)
        self.assertEqual(self.get_svc("sous-chef-model-eval")["generation"], "1")
        self.call("PATCH", self.svc_url("sous-chef-model-eval") + "?forceNewRevision=true", body=s)
        self.clock.advance(E.RECONCILE_S["gpu"])
        after = self.get_svc("sous-chef-model-eval")
        self.assertEqual(after["generation"], "2")
        self.assertNotEqual(after["latestReadyRevision"], s["latestReadyRevision"])

    def test_failed_revision_keeps_the_old_one_serving(self):
        self.g.fail_next["revision-fail:sous-chef-model-eval"] = 1
        s = self.get_svc("sous-chef-model-eval")
        old = s["latestReadyRevision"]
        s["template"]["containers"][0]["image"] = f"{E.MODEL_REPO}/model@{E.R4_DIGEST}"
        self.call("PATCH", self.svc_url("sous-chef-model-eval"), body=s)
        self.clock.advance(E.RECONCILE_S["gpu-failed"])
        after = self.get_svc("sous-chef-model-eval")
        self.assertEqual(after["latestReadyRevision"], old)
        self.assertNotEqual(after["latestCreatedRevision"], old)
        self.assertEqual(after["terminalCondition"]["state"], "CONDITION_FAILED")
        # a failed reconciliation: reconciling ends, observedGeneration keeps the last serving generation (v2 reference)
        self.assertFalse(after.get("reconciling"))
        self.assertEqual(after["observedGeneration"], s["generation"])
        self.assertNotEqual(after["generation"], s["generation"])
        self.assertEqual(after["trafficStatuses"][0]["revision"], E.short_rev(old))
        # the same template again: nothing happens without forceNewRevision
        self.call("PATCH", self.svc_url("sous-chef-model-eval"), body=after)
        self.clock.advance(E.RECONCILE_S["gpu"])
        self.assertEqual(self.get_svc("sous-chef-model-eval")["latestReadyRevision"], old)

    def test_stale_etag_and_extra_fields(self):
        s = self.get_svc("sous-chef-model")
        self.g.fail_next["run-etag-race:sous-chef-model"] = 1
        s["template"]["containers"][0]["image"] = f"{E.MODEL_REPO}/model@{E.R4_DIGEST}"
        code, err, _ = self.call("PATCH", self.svc_url("sous-chef-model"), body=s)
        self.assertEqual(code, 409)
        s = self.get_svc("sous-chef-model")
        s["template"]["scaling"]["maxInstanceCount"] = 9
        with self.assertRaises(wfsim.SimError):
            self.call("PATCH", self.svc_url("sous-chef-model"), body=s)

    def test_pinned_traffic_gives_the_new_revision_nothing(self):
        prod = self.w.s["services"]["sous-chef-model"]
        old = E.short_rev(prod["latestReadyRevision"])
        prod["traffic"] = [{"type": "TRAFFIC_TARGET_ALLOCATION_TYPE_REVISION", "revision": old, "percent": 100}]
        s = self.get_svc("sous-chef-model")
        s["template"]["containers"][0]["image"] = f"{E.MODEL_REPO}/model@{E.R4_DIGEST}"
        self.call("PATCH", self.svc_url("sous-chef-model"), body=s)
        self.clock.advance(E.RECONCILE_S["gpu"])
        after = self.get_svc("sous-chef-model")
        self.assertNotEqual(E.short_rev(after["latestReadyRevision"]), old)
        self.assertEqual([E.short_rev(n) for n, _ in self.g.serving("sous-chef-model")], [old])
        # setting traffic to LATEST 100 is allowed in the body
        after["traffic"] = E.LATEST_100
        self.call("PATCH", self.svc_url("sous-chef-model"), body=after)
        self.clock.advance(E.RECONCILE_S["gpu"])
        self.assertEqual(self.g.serving("sous-chef-model")[0][0], self.get_svc("sous-chef-model")["latestReadyRevision"])

    def test_preview_revision_is_ready_but_not_serving(self):
        old = self.w.s["services"]["sous-chef"]["latestReadyRevision"]
        new = self.g.deploy_preview("sous-chef", f"{E.LAB_REPO}/sous-chef:{E.PREVIEW_TAG}", E.PREVIEW_TAG)
        s = self.get_svc("sous-chef", E.NE2)
        self.assertEqual(s["latestReadyRevision"], new)
        self.assertEqual(self.g.serving("sous-chef"), [(old, 100)])
        tagged = [t for t in s["trafficStatuses"] if t.get("tag") == E.PREVIEW_TAG]
        self.assertEqual(len(tagged), 1)
        self.assertNotIn("percent", tagged[0])


class CloudRunJobs(FakeBase):
    def job_url(self, j="souschef-drift"):
        return f"{RUN}projects/{E.PROJECT}/locations/{E.NE2}/jobs/{j}"

    def test_job_patch_reconciles_and_identical_is_a_noop(self):
        code, j, _ = self.call("GET", self.job_url())
        j["template"]["template"]["containers"][0]["image"] = f"{E.LAB_REPO}/sous-chef-lab:{E.SERVICE_TAG}"
        code, op, _ = self.call("PATCH", self.job_url(), body=j)
        self.assertFalse(op["done"])
        _, now, _ = self.call("GET", self.job_url())
        self.assertTrue(now["reconciling"])
        self.clock.advance(E.RECONCILE_S["job"])
        _, done, _ = self.call("GET", self.job_url())
        self.assertFalse(done.get("reconciling"))
        self.assertEqual(done["observedGeneration"], done["generation"])
        self.call("PATCH", self.job_url(), body=done)
        _, again, _ = self.call("GET", self.job_url())
        self.assertEqual(again["generation"], done["generation"])

    def test_execution_has_no_completion_until_it_ends(self):
        name = f"projects/{E.PROJECT}/locations/{E.NE2}/jobs/souschef-drift/executions/souschef-drift-0001"
        self.w.s["executions"][name] = {"name": name, "job": "souschef-drift", "args": ["drift"], "exit_code": 0,
                                        "image": f"{E.LAB_REPO}/sous-chef-lab:{E.SERVICE_TAG}", "start": T0, "end": T0 + 60}
        _, ex, _ = self.call("GET", RUN + name)
        self.assertNotIn("completionTime", ex)
        _, tl, _ = self.call("GET", RUN + name + "/tasks")
        self.assertNotIn("lastAttemptResult", tl["tasks"][0])
        self.clock.advance(61)
        _, ex, _ = self.call("GET", RUN + name)
        self.assertEqual((ex["succeededCount"], ex["conditions"][0]["state"]), (1, "CONDITION_SUCCEEDED"))
        _, tl, _ = self.call("GET", RUN + name + "/tasks")
        self.assertEqual(tl["tasks"][0]["lastAttemptResult"], {"status": {}})  # exitCode 0 omitted

    def test_cancel(self):
        name = f"projects/{E.PROJECT}/locations/{E.NE2}/jobs/souschef-collect/executions/souschef-collect-0001"
        self.w.s["executions"][name] = {"name": name, "job": "souschef-collect", "args": ["collect"], "exit_code": 0,
                                        "image": f"{E.LAB_REPO}/sous-chef-lab:{E.SERVICE_TAG}", "start": T0, "end": T0 + 9999}
        code, _, _ = self.call("POST", RUN + name + ":cancel")
        self.assertEqual(code, 200)
        _, ex, _ = self.call("GET", RUN + name)
        self.assertEqual(ex["cancelledCount"], 1)
        self.assertIn("completionTime", ex)


class ArtifactRegistry(FakeBase):
    def test_versions(self):
        d = "sha256:" + "3" * 64
        _, v, _ = self.call("GET", AR + "sous-chef/versions/" + d)
        self.assertNotIn("relatedTags", v)
        _, v, _ = self.call("GET", AR + "sous-chef/versions/" + d.replace(":", "%3A"), {"view": "FULL"})
        names = [t["name"].rsplit("/", 1)[1] for t in v["relatedTags"]]
        self.assertEqual(names[0], "latest")
        self.assertEqual(set(names), {"latest", E.LABONLY_TAG, E.SERVICE_TAG})
        code, v, _ = self.call("GET", AR + "sous-chef/versions/" + E.UNTAGGED_DIGEST, {"view": "FULL"})
        self.assertEqual(code, 200)
        self.assertNotIn("relatedTags", v)
        code, _, _ = self.call("GET", AR + "sous-chef/versions/sha256:" + "9" * 64, {"view": "FULL"})
        self.assertEqual(code, 404)


class CloudBuild(FakeBase):
    def step(self, name, release_args=("--build-context", "model=/workspace/stage")):
        return {"id": "build", "name": name, "args": ["build", *release_args, "-t", "$_IMAGE", "model-image"]}

    def test_docker_versions(self):
        self.assertEqual(E.docker_major("gcr.io/cloud-builders/docker"), 20)
        self.assertEqual(E.docker_major("gcr.io/cloud-builders/docker:latest"), 20)
        self.assertEqual(E.docker_major("gcr.io/cloud-builders/docker@sha256:6c9d05f41f8e721e05f7d3d1b5c29904ac2595d1b67de093f59420b6e8146ecf"), 29)
        self.assertEqual(E.docker_major("mirror.gcr.io/library/docker:28-cli@sha256:" + "a" * 64), 28)
        self.assertIsNone(E.docker_major("gcr.io/google.com/cloudsdktool/cloud-sdk:slim"))

    def test_builder_problems(self):
        P = E.FakeGCP.builder_problem
        self.assertIn("unknown flag: --build-context", P([self.step("gcr.io/cloud-builders/docker")], 20))
        self.assertIn("legacy worker", P([self.step("docker:28-cli")], 20))
        self.assertIn("too old", P([self.step("gcr.io/cloud-builders/docker")], 29))
        self.assertIsNone(P([self.step("gcr.io/cloud-builders/docker@sha256:6c9d05f41f8e721e05f7d3d1b5c29904ac2595d1b67de093f59420b6e8146ecf")], 29))
        self.assertIsNone(P([{"id": "stage", "name": "gcr.io/google.com/cloudsdktool/cloud-sdk", "args": ["-c", "x"]}], 20))

    def test_lifecycle_push_and_immutable_tag(self):
        key = f"{E.GPU}/sous-chef-model/model:c-x"
        base = {"status": "QUEUED", "_final": "TIMEOUT", "_detail": "timeout", "_push": [key, "sha256:" + "a" * 64],
                "_t_work": T0 + 20, "_t_done": T0 + 100, "_image": f"{E.MODEL_REPO}/model:c-x", "id": "b1"}
        self.w.s["builds"]["b1"] = dict(base)
        self.g.tick()
        self.assertEqual(self.w.s["builds"]["b1"]["status"], "QUEUED")
        self.clock.advance(30)
        self.g.tick()
        self.assertEqual(self.w.s["builds"]["b1"]["status"], "WORKING")
        self.clock.advance(100)
        self.g.tick()
        self.assertEqual(self.w.s["builds"]["b1"]["status"], "TIMEOUT")
        self.assertEqual(self.w.s["tags"][key], "sha256:" + "a" * 64)  # pushed before it timed out
        self.w.s["builds"]["b2"] = {**base, "id": "b2", "_final": "SUCCESS", "_detail": "", "_push": [key, "sha256:" + "b" * 64],
                                    "_t_work": self.clock.now() + 1, "_t_done": self.clock.now() + 2}
        self.clock.advance(5)
        self.g.tick()
        self.assertEqual(self.w.s["builds"]["b2"]["status"], "FAILURE")
        self.assertIn("immutable", self.w.s["builds"]["b2"]["statusDetail"])


class Vertex(FakeBase):
    def create(self, strategy, run="rt-1", region="us-central1", wait=None, labels=None):
        sched = {"strategy": strategy}
        if wait is not None:
            sched["maxWaitDuration"] = f"{wait}s"
        body = {"displayName": f"train-{run}", "labels": labels or {}, "jobSpec": {"scheduling": sched, "workerPoolSpecs": [{
            "machineSpec": {"machineType": "a3-highgpu-1g", "acceleratorType": "NVIDIA_H100_80GB", "acceleratorCount": 1},
            "replicaCount": 1, "containerSpec": {"imageUri": f"{E.LAB_REPO}/sous-chef-trainer:{E.SERVICE_TAG}",
                                                 "args": [run, "full"], "env": [{"name": "PIPELINE_BUCKET", "value": f"gs://{E.BUCKET}"}]}}]}}
        code, job, _ = self.call("POST", f"https://{region}-aiplatform.googleapis.com/v1/projects/{E.PROJECT}/locations/{region}/customJobs", body=body)
        self.assertEqual(code, 200)
        return job["name"]

    def get(self, name, region="us-central1"):
        return self.call("GET", f"https://{region}-aiplatform.googleapis.com/v1/{name}")[1]

    def test_flex_never_starts_and_its_message_matches_no_regex(self):
        name = self.create("FLEX_START", wait=14400)
        self.assertEqual(self.get(name)["state"], "JOB_STATE_QUEUED")
        self.clock.advance(60)
        self.assertEqual(self.get(name)["state"], "JOB_STATE_PENDING")
        self.clock.advance(14400)
        j = self.get(name)
        self.assertEqual(j["state"], "JOB_STATE_FAILED")
        self.assertNotIn("startTime", j)
        src = (E.WF_DIR / "retrain.yaml").read_text()
        for rx in re.findall(r'text\.match_regex\(msg, "([^"]+)"\)', src):
            self.assertIsNone(re.search(rx.replace("(?i)", ""), j["error"]["message"], re.I), rx)

    def test_spot_preempted_past_six_restarts_is_a_stockout(self):
        self.g.fail_next["vertex-preempt"] = 7
        name = self.create("SPOT")
        self.clock.advance(300)
        j = self.get(name)
        self.assertEqual(j["state"], "JOB_STATE_RUNNING")
        self.assertIn("startTime", j)
        self.clock.advance(1800 + 7 * 600)
        j = self.get(name)
        self.assertEqual(j["state"], "JOB_STATE_FAILED")
        self.assertIn("Resources are insufficient", j["error"]["message"])
        prog = json.loads((self.w.bucket / "runs/rt-1/train/progress.json").read_text())
        self.assertEqual(prog["status"], "interrupted")

    def test_list_by_label(self):
        base = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{E.PROJECT}/locations/us-central1/customJobs"
        a = self.create("SPOT", labels={"create": "abc-1"})
        self.create("SPOT", labels={"create": "abc-2"})
        code, out, _ = self.call("GET", base, query={"filter": "labels.create=abc-1"})
        self.assertEqual((code, [j["name"] for j in out["customJobs"]]), (200, [a]))
        self.assertEqual(self.call("GET", base, query={"filter": "labels.create=nope"})[1], {})

    def test_cancel_passes_through_cancelling_and_overlap_is_recorded(self):
        self.g.fail_next["vertex-pending"] = 1
        a = self.create("SPOT")
        self.clock.advance(3600)
        self.assertEqual(self.get(a)["state"], "JOB_STATE_PENDING")
        self.call("POST", f"https://us-central1-aiplatform.googleapis.com/v1/{a}:cancel")
        self.assertEqual(self.get(a)["state"], "JOB_STATE_CANCELLING")
        self.create("SPOT", region="us-east4")  # the next rung before the cancel finished
        self.assertEqual(len(self.w.s["vertex_overlap"]), 1)
        self.clock.advance(120)
        self.assertEqual(self.get(a)["state"], "JOB_STATE_CANCELLED")


class Storage(FakeBase):
    def obj(self, name):
        return GCS + "/o/" + name.replace("/", "%2F")

    def test_media_bytes_headers_and_missing(self):
        code, meta, _ = self.call("POST", UPLOAD, {"uploadType": "media", "name": "state/x.json"}, {"a": 1},
                                  {"Content-Type": "application/json"})
        self.assertEqual(code, 200)
        code, body, hdr = self.call("GET", self.obj("state/x.json"), {"alt": "media"})
        self.assertEqual(body, {"a": 1})
        self.assertEqual(hdr["X-Goog-Generation"], meta["generation"])
        (self.w.bucket / "runs").mkdir()
        (self.w.bucket / "runs" / "lab.json").write_text('{"b": 2}')  # written by gcloud: octet-stream
        _, body, _ = self.call("GET", self.obj("runs/lab.json"), {"alt": "media"})
        self.assertEqual(body, b'{"b": 2}')
        code, body, hdr = self.call("GET", self.obj("state/nope-0412.json"), {"alt": "media"})
        self.assertEqual((code, body), (404, f"No such object: {E.BUCKET}/state/nope-0412.json"))
        code, body, _ = self.call("GET", self.obj("state/nope.json"))
        self.assertEqual(body["error"]["errors"][0]["reason"], "notFound")
        code, _, _ = self.call("GET", GCS + "/o/state/x.json", {"alt": "media"})  # unencoded slash
        self.assertEqual(code, 404)

    def test_list_pages(self):
        for k in range(5):
            self.call("POST", UPLOAD, {"uploadType": "media", "name": f"wf/outbox/{k}.json"}, {"k": k}, {"Content-Type": "application/json"})
        names, token, pages = [], None, 0
        while True:
            q = {"prefix": "wf/outbox/", "maxResults": "2", "fields": "items(name),nextPageToken"}
            if token:
                q["pageToken"] = token
            _, body, _ = self.call("GET", GCS + "/o", q)
            pages += 1
            names += [i["name"] for i in body.get("items", [])]
            token = body.get("nextPageToken")
            if not token:
                break
        self.assertEqual((pages, len(names)), (3, 5))
        _, body, _ = self.call("GET", GCS + "/o", {"prefix": "nothing/", "fields": "items(name),nextPageToken"})
        self.assertEqual(body, {})

    def test_race_between_read_and_conditional_delete(self):
        self.call("POST", UPLOAD, {"uploadType": "media", "name": "state/locks/gpu-eval.json"}, {"run_id": "other"},
                  {"Content-Type": "application/json"})
        self.g.race("state/locks/gpu-eval.json", lambda d: {**d, "refreshed": True})
        _, _, hdr = self.call("GET", self.obj("state/locks/gpu-eval.json"), {"alt": "media"})
        code, body, _ = self.call("DELETE", self.obj("state/locks/gpu-eval.json"), {"ifGenerationMatch": hdr["X-Goog-Generation"]})
        self.assertEqual(code, 412)
        self.assertEqual(body["error"]["errors"][0]["reason"], "conditionNotMet")


class Faults(FakeBase):
    def test_one_off_and_lost_response(self):
        self.g.inject("GET", r"/o/infra%2Fconfig\.json$", 500)
        (self.w.bucket / "infra").mkdir()
        (self.w.bucket / "infra" / "config.json").write_text("{}")
        url = GCS + "/o/infra%2Fconfig.json"
        self.assertEqual(self.call("GET", url, {"alt": "media"})[0], 500)
        self.assertEqual(self.call("GET", url, {"alt": "media"})[0], 200)
        self.g.inject("POST", r"/upload/", 503, after=True)
        code, _, _ = self.call("POST", UPLOAD, {"uploadType": "media", "name": "state/hb.json", "ifGenerationMatch": "0"}, {"x": 1},
                               {"Content-Type": "application/json"})
        self.assertEqual(code, 503)
        self.assertTrue((self.w.bucket / "state" / "hb.json").is_file())  # applied
        code, _, _ = self.call("POST", UPLOAD, {"uploadType": "media", "name": "state/hb.json", "ifGenerationMatch": "0"}, {"x": 1},
                               {"Content-Type": "application/json"})
        self.assertEqual(code, 412)  # the retry meets its own first attempt

    def test_thermometer_key_with_a_newline(self):
        code, body, _ = self.call("GET", "http://therm.local/api/ingest/souschef/decisions/c-x", headers={"X-API-Key": "tk\n"}, auth=None)
        self.assertEqual(code, 401)
        self.g.raw_error_bodies = True
        code, body, _ = self.call("GET", "http://therm.local/api/ingest/souschef/decisions/c-x", headers={"X-API-Key": "tk\n"}, auth=None)
        self.assertIsInstance(body, str)

    def test_operations_readable_only_with_the_grant(self):
        self.g.ops_allowed = True
        s = self.get_svc("sous-chef-model-eval")
        s["template"]["containers"][0]["image"] = f"{E.MODEL_REPO}/model@{E.R4_DIGEST}"
        _, op, _ = self.call("PATCH", self.svc_url("sous-chef-model-eval"), body=s)
        _, cur, _ = self.call("GET", RUN + op["name"])
        self.assertFalse(cur["done"])
        self.clock.advance(E.RECONCILE_S["gpu"])
        _, cur, _ = self.call("GET", RUN + op["name"])
        self.assertTrue(cur["done"])
        self.assertIsInstance(E.iam_grants_run_operations(), bool)

    def test_ttl_max_reads_variables_tf(self):
        self.assertGreaterEqual(E.ttl_max(), 1)


if __name__ == "__main__":
    unittest.main()
