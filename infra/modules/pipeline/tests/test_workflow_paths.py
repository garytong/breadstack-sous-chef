#!/usr/bin/env python3
"""wfsim tests of the lib / drift / retrain subworkflow paths the 2026-10-12 audit changed (no network): Cloud Run
waits on the resource (never operations/{id}), one execution per jobs:run, locks and retries, resolve_images,
notify_result's memory cap, the decision wait's step cost, Vertex adoption / cancel / classification / lost creates,
the model build's pushed-tag recheck, the thermometer's raw-text 404, and stage_rollout's traffic guard and error
classes. The full end-to-end lives in e2e_dry.py; these pin the edge cases it reaches only by chance.

    python3 -m unittest infra/modules/pipeline/tests/test_workflow_paths.py
"""
from __future__ import annotations

import copy
import json
import sys
import unittest
import urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent
WF = HERE.parent / "workflows"
sys.path.insert(0, str(HERE))
import wfsim  # noqa: E402

CFG = {"project": "p", "bucket": "b", "jobs_region": "ne2", "thermometer_url": "https://therm.example",
       "thermometer_secret": "s", "decision_ttl_days": 14, "smoke_ttl_days": 1, "workflows_region": "us-central1",
       "lab_repo": {"location": "ne2", "repository": "sous-chef", "image": "ne2-docker.pkg.dev/p/sous-chef/sous-chef-lab",
                    "trainer": "ne2-docker.pkg.dev/p/sous-chef/sous-chef-trainer"},
       "int_service": {"name": "sous-chef", "region": "ne2"},
       "prod_model": {"name": "sous-chef-model", "region": "us-east4", "url": "https://prod"},
       "model_repo": {"location": "us-east4", "repository": "sous-chef-model",
                      "image": "us-east4-docker.pkg.dev/p/sous-chef-model/model"},
       "trainer": {"timeout_s": 3600, "disk_gb": 200, "service_account": "t@p", "ladder": [
           {"name": "flex", "region": "us-central1", "machine_type": "a3", "accelerator_type": "H",
            "strategy": "FLEX_START", "attempts": 1, "max_wait_s": 0},
           {"name": "spot", "region": "us-central1", "machine_type": "a3", "accelerator_type": "H",
            "strategy": "SPOT", "attempts": 2}]}}
CTX = {"run_id": "rt-x", "candidate_id": "c-rt-x", "smoke": False, "dry_req": False, "prefix": "runs/rt-x/",
       "mode": "full", "imgs": {"trainer": "tr:t", "tag": "t"}}
JOB = "projects/p/locations/ne2/jobs/j"


class Router:
    """A bucket (name -> (object, generation)) plus handlers tried first: pred(method, url, query) -> fn."""

    def __init__(self):
        self.gcs: dict = {}
        self.gen = 100
        self.handlers: list = []
        self.log: list = []

    def put(self, name, obj):
        self.gen += 1
        self.gcs[name] = (copy.deepcopy(obj), self.gen)

    def __call__(self, method, url, query, headers, body, auth):
        self.log.append((method, url, dict(query)))
        for pred, fn in self.handlers:
            if pred(method, url, query):
                return fn(method, url, query, headers, body)
        u = urllib.parse.urlsplit(url)
        if u.netloc != "storage.googleapis.com":
            raise wfsim.SimError(f"no fake {method} {url}")
        p = u.path
        if p.startswith("/upload/"):
            name = query["name"]
            g = query.get("ifGenerationMatch")
            cur = self.gcs.get(name)
            if g is not None and ((g == "0" and cur) or (g != "0" and (not cur or str(cur[1]) != g))):
                return 412, {"error": {"code": 412}}, {}
            self.put(name, body)
            return 200, {"name": name, "generation": str(self.gen)}, {}
        if p.endswith("/o"):
            pre = query.get("prefix", "")
            names = sorted(k for k in self.gcs if k.startswith(pre))
            mx = int(query.get("maxResults", 1000))
            start = int(query.get("pageToken", 0) or 0)
            page = names[start:start + mx]
            out = {"items": [{"name": n} for n in page]} if page else {}
            if start + mx < len(names):
                out["nextPageToken"] = str(start + mx)
            return 200, out, {}
        name = urllib.parse.unquote(p.split("/o/", 1)[1])
        cur = self.gcs.get(name)
        if method == "GET":
            if not cur:
                return 404, "No such object: " + name, {}
            if query.get("alt") == "media":
                return 200, copy.deepcopy(cur[0]), {"X-Goog-Generation": str(cur[1])}
            return 200, {"name": name, "generation": str(cur[1])}, {}
        if method == "DELETE":
            g = query.get("ifGenerationMatch")
            if not cur:
                return 404, "No such object", {}
            if g is not None and str(cur[1]) != g:
                return 412, {"error": {"code": 412}}, {}
            self.gcs.pop(name)
            return 204, None, {}
        raise wfsim.SimError(f"no fake {method} {url}")


def interp(name, router, env=None, on_sleep=None):
    e = {"GOOGLE_CLOUD_WORKFLOW_EXECUTION_ID": "8a0cc8ec-1111-2222-3333-444455556666"}
    e.update(env or {})
    return wfsim.Interpreter.from_files(str(WF / f"{name}.yaml"), str(WF / "lib.yaml"), http=router,
                                        secrets=lambda a: "tm_key\n", env=e, clock=wfsim.VirtualClock(1_791_000_000),
                                        on_sleep=on_sleep)


def raised(test, fn):
    with test.assertRaises(wfsim.WfError) as cm:
        fn()
    return cm.exception.value


class Svc:
    """A Cloud Run service that reconciles over three GETs after a PATCH; GET operations/{id} is 403."""

    def __init__(self, fail_next=False, traffic=None, settles=True, moves_traffic=True):
        n = "projects/p/locations/us-east4/services/sous-chef-model"
        self.s = {"name": n, "generation": "8", "observedGeneration": "8",
                  "latestReadyRevision": n + "/revisions/m-1", "latestCreatedRevision": n + "/revisions/m-1",
                  "terminalCondition": {"type": "Ready", "state": "CONDITION_SUCCEEDED"},
                  "template": {"containers": [{"image": "img@sha256:a"}]},
                  "traffic": traffic or [{"type": "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST", "percent": 100}],
                  "trafficStatuses": [{"type": "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST", "revision": "m-1", "percent": 100}]}
        self.n = 1
        self.fail_next, self.settles, self.moves_traffic = fail_next, settles, moves_traffic
        self.pending = 0
        self.patches: list = []
        self.ops_read = 0

    def handle(self, m, u, q, h, b):
        path = urllib.parse.urlsplit(u).path
        if "/operations/" in path:
            self.ops_read += 1
            return 403, {"error": {"code": 403, "message": "Permission 'run.operations.get' denied"}}, {}
        if "/revisions/" in path:
            rev = path.rsplit("/", 1)[1]
            return 200, {"name": rev, "containers": [{"image": "img@sha256:" + rev}],
                         "createTime": "2026-10-13T00:00:00.123456Z"}, {}
        if m == "GET":
            if self.pending and self.settles:
                self.pending -= 1
                if self.pending == 0:
                    self.settle()
            return 200, copy.deepcopy(self.s), {}
        if m == "PATCH":
            self.patches.append(dict(q))
            if b["template"] == self.s["template"] and q.get("forceNewRevision") != "true":
                return 200, {"name": "projects/p/locations/us-east4/operations/x", "done": False}, {}
            self.s = copy.deepcopy(b)
            self.s["generation"] = str(int(self.s["generation"]) + 1)
            self.s["reconciling"] = True
            self.s["terminalCondition"] = {"type": "Ready", "state": "CONDITION_RECONCILING"}
            self.n += 1
            self.s["latestCreatedRevision"] = self.s["name"] + f"/revisions/m-{self.n}"
            self.pending = 3
            return 200, {"name": "projects/p/locations/us-east4/operations/x", "done": False,
                         "metadata": {"@type": "Service", "generation": self.s["generation"]}}, {}
        raise wfsim.SimError(f"svc {m} {u}")

    def settle(self):
        self.s.pop("reconciling", None)
        if self.fail_next:  # observedGeneration and latestReadyRevision keep the last serving state (v2 reference)
            self.fail_next = False
            self.s["terminalCondition"] = {"type": "Ready", "state": "CONDITION_FAILED", "message": "no L4"}
            return
        self.s["observedGeneration"] = self.s["generation"]
        self.s["latestReadyRevision"] = self.s["latestCreatedRevision"]
        self.s["terminalCondition"] = {"type": "Ready", "state": "CONDITION_SUCCEEDED"}
        if self.moves_traffic and any(t.get("type") == "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST" for t in self.s.get("traffic") or []):
            self.s["trafficStatuses"] = [{"type": "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST",
                                          "revision": self.s["latestReadyRevision"].rsplit("/", 1)[1], "percent": 100}]


class Helpers(unittest.TestCase):
    def test_small_helpers(self):
        it = interp("drift", Router())
        st = it.call_sub("run_stamp", {})
        self.assertTrue(st.endswith("-8a0cc8") and len(st) == len("20261012-0310-8a0cc8"), st)
        self.assertEqual(it.call_sub("err_class", {"e": {"code": 404, "tags": ["HttpError"]}}), "code 404")
        self.assertEqual(it.call_sub("therm_key", {"cfg": CFG}), "tm_key")
        for code, want in ((408, True), (500, True), (503, True), (404, False), (412, False)):
            self.assertIs(it.call_sub("retry_transient", {"e": {"code": code, "tags": ["HttpError"]}}), want, code)
        self.assertIs(it.call_sub("retry_transient", {"e": {"tags": ["ConnectionFailedError"]}}), True)
        self.assertLess(abs(it.call_sub("api_time", {"s": "2026-10-13T04:43:10.336255123Z"})
                            - wfsim.parse_time("2026-10-13T04:43:10.336255Z")), 1e-6)

    def test_gcs_read_retries_a_500(self):
        r = Router()
        r.put("x.json", {"a": 1})
        n = {"c": 0}

        def once500(m, u, q, h, b):
            n["c"] += 1
            if n["c"] == 1:
                return 500, "boom", {}
            r.handlers.pop(0)
            return r(m, u, q, h, b, None)
        r.handlers.append((lambda m, u, q: "x.json" in u and m == "GET", once500))
        self.assertEqual(interp("drift", r).call_sub("gcs_read", {"bucket": "b", "name": "x.json"}), {"a": 1})

    def test_locks(self):
        r = Router()
        it = interp("drift", r)
        self.assertTrue(it.call_sub("lock_acquire", {"bucket": "b", "name": "L", "run_id": "A", "ttl_s": 60})["acquired"])
        it2 = interp("drift", r, env={"GOOGLE_CLOUD_WORKFLOW_EXECUTION_ID": "99999999-0000"})
        g2 = it2.call_sub("lock_acquire", {"bucket": "b", "name": "L", "run_id": "B", "ttl_s": 60})
        self.assertTrue(not g2["acquired"] and g2["holder"]["run_id"] == "A")
        it2.clock.advance(120)
        self.assertTrue(it2.call_sub("lock_acquire", {"bucket": "b", "name": "L", "run_id": "B", "ttl_s": 60})["acquired"])
        self.assertIs(it.call_sub("lock_release", {"bucket": "b", "name": "L", "run_id": "A"}), False)

        def bump(m, u, q, h, b):  # someone rewrites the lock between our read and our delete
            r.handlers.pop(0)
            r.put("L", r.gcs["L"][0])
            return r(m, u, q, h, b, None)
        r.handlers.append((lambda m, u, q: m == "DELETE", bump))
        self.assertIs(it2.call_sub("lock_release", {"bucket": "b", "name": "L", "run_id": "B"}), False)
        self.assertIs(it2.call_sub("lock_refresh", {"bucket": "b", "name": "L", "run_id": "B", "ttl_s": 60}), True)

    def test_gcs_list_max(self):
        r = Router()
        for i in range(130):
            r.put(f"wf/outbox/{i:04d}.json", {"id": str(i)})
        it = interp("drift", r)
        self.assertEqual(len(it.call_sub("gcs_list", {"bucket": "b", "prefix": "wf/outbox/", "max": 50})), 50)
        self.assertEqual(len(it.call_sub("gcs_list", {"bucket": "b", "prefix": "wf/outbox/"})), 130)


class CloudRunService(unittest.TestCase):
    def test_failed_revision_then_forced(self):
        sv = Svc(fail_next=True)
        r = Router()
        r.handlers.append((lambda m, u, q: "run.googleapis.com" in u, sv.handle))
        it = interp("retrain", r)
        e = raised(self, lambda: it.call_sub("service_set_image", {"cfg": CFG, "svc": CFG["prod_model"], "image": "img@sha256:b", "note": "c-1"}))
        self.assertEqual(e.get("error_class"), "revision_not_ready")
        self.assertIn("no L4", e.get("message", ""))
        out = it.call_sub("service_set_image", {"cfg": CFG, "svc": CFG["prod_model"], "image": "img@sha256:b", "note": "c-1"})
        self.assertTrue(out["changed"] and out["revision"].endswith("m-3"))
        self.assertEqual(sv.patches[-1].get("forceNewRevision"), "true")
        self.assertEqual(sv.ops_read, 0, "no GET operations/{id}")

    def test_wait_target_without_operation_metadata(self):
        # an operation without metadata.generation: gen_before + 1 is the floor, not 0
        sv = Svc()
        orig = sv.handle

        def bare(m, u, q, h, b):
            code, body, hd = orig(m, u, q, h, b)
            if m == "PATCH":
                body = {"name": body["name"], "done": False}
            return code, body, hd
        r = Router()
        r.handlers.append((lambda m, u, q: "run.googleapis.com" in u, bare))
        out = interp("retrain", r).call_sub("service_set_image", {"cfg": CFG, "svc": CFG["prod_model"], "image": "img@sha256:b", "note": "c"})
        self.assertTrue(out["changed"] and out["revision"].endswith("m-2"), out)

    def test_serving_revision_and_traffic(self):
        sv = Svc()
        r = Router()
        r.handlers.append((lambda m, u, q: "run.googleapis.com" in u, sv.handle))
        it = interp("retrain", r)
        it.call_sub("service_set_image", {"cfg": CFG, "svc": CFG["prod_model"], "image": "img@sha256:b", "note": "c-1"})
        sr = it.call_sub("service_ready_image", {"cfg": CFG, "svc": CFG["prod_model"]})
        self.assertTrue(sr["revision"].endswith("m-2") and sr["percent"] == 100)
        sv.s["latestReadyRevision"] = sv.s["name"] + "/revisions/m-9"  # a --no-traffic preview at 0 %
        sv.s["trafficStatuses"].append({"type": "TRAFFIC_TARGET_ALLOCATION_TYPE_REVISION", "revision": "m-9", "tag": "p-x"})
        self.assertTrue(it.call_sub("service_ready_image", {"cfg": CFG, "svc": CFG["prod_model"]})["revision"].endswith("m-2"))
        self.assertIs(it.call_sub("traffic_latest", {"s": sv.s}), True)
        pin = {"traffic": [{"type": "TRAFFIC_TARGET_ALLOCATION_TYPE_REVISION", "revision": "m-1", "percent": 100}]}
        self.assertIs(it.call_sub("traffic_latest", {"s": pin}), False)
        self.assertIs(it.call_sub("traffic_latest", {"s": {}}), True)

    def test_job_set_image_409_once(self):
        class Job:
            def __init__(self):
                self.j = {"name": JOB, "generation": "2", "observedGeneration": "2",
                          "terminalCondition": {"type": "Ready", "state": "CONDITION_SUCCEEDED"},
                          "template": {"template": {"containers": [{"image": "old"}]}}}
                self.conflict, self.pending = True, 0

            def handle(self, m, u, q, h, b):
                if "/operations/" in u:
                    return 403, "denied", {}
                if m == "GET":
                    if self.pending:
                        self.pending -= 1
                        if not self.pending:
                            self.j.pop("reconciling", None)
                            self.j["observedGeneration"] = self.j["generation"]
                    return 200, copy.deepcopy(self.j), {}
                if self.conflict:
                    self.conflict = False
                    return 409, {"error": {"code": 409}}, {}
                self.j = copy.deepcopy(b)
                self.j["generation"], self.j["reconciling"], self.pending = "3", True, 2
                return 200, {"name": "projects/p/locations/ne2/operations/y", "done": False}, {}
        jb = Job()
        r = Router()
        r.handlers.append((lambda m, u, q: "run.googleapis.com" in u, jb.handle))
        self.assertIs(interp("drift", r).call_sub("job_set_image", {"cfg": CFG, "job": "j", "image": "new"}), True)
        self.assertEqual(jb.j["template"]["template"]["containers"][0]["image"], "new")


class RunJob(unittest.TestCase):
    """jobs:run has no request id: one execution per start, whatever its answer."""

    def router(self, run_answers, *, job_get_fails_after=None, ex_body=None):
        st = {"execs": [], "gets": 0}
        r = Router()

        def job(m, u, q, h, b):
            st["gets"] += 1
            if job_get_fails_after is not None and st["gets"] > job_get_fails_after:
                return 503, "unavailable", {}
            last = st["execs"][-1] if st["execs"] else "j-old"
            return 200, {"name": JOB, "latestCreatedExecution": {"name": last, "createTime": "2026-10-13T00:00:00Z"}}, {}

        def run(m, u, q, h, b):
            code, applied = run_answers.pop(0) if run_answers else (200, True)
            if applied:
                st["execs"].append(f"j-{len(st['execs']) + 1}")
            if code != 200:
                return code, {"error": {"code": code}}, {}
            return 200, {"metadata": {"name": f"{JOB}/executions/{st['execs'][-1]}"}}, {}

        def ex(m, u, q, h, b):
            return 200, ex_body or {"completionTime": "x", "succeededCount": 1,
                                    "conditions": [{"type": "Completed", "state": "CONDITION_SUCCEEDED"}]}, {}
        r.handlers.append((lambda m, u, q: u.endswith("/jobs/j") and m == "GET", job))
        r.handlers.append((lambda m, u, q: u.endswith(":run"), run))
        r.handlers.append((lambda m, u, q: "/executions/" in u, ex))
        return r, st

    def go(self, r):
        return interp("drift", r).call_sub("run_job", {"cfg": CFG, "job": "j", "args": [], "env": {}, "timeout_s": 60})

    def test_lost_answer_takes_the_execution_it_made(self):
        r, st = self.router([(503, True)])
        o = self.go(r)
        self.assertTrue(o["ok"], o)
        self.assertEqual(st["execs"], ["j-1"], "no second execution")
        self.assertEqual(o["execution"], f"{JOB}/executions/j-1")

    def test_not_applied_is_resent(self):
        r, st = self.router([(500, False), (502, False)])
        o = self.go(r)
        self.assertTrue(o["ok"] and st["execs"] == ["j-1"], (o, st))

    def test_never_starts(self):
        r, st = self.router([(503, False)] * 4)
        o = self.go(r)
        self.assertTrue(not o["ok"] and o["execution"] is None and not o.get("no_retry") and not st["execs"], o)

    def test_4xx_is_not_retried(self):
        r, st = self.router([(400, False)])
        o = self.go(r)
        self.assertTrue(not o["ok"] and o.get("no_retry") is True and "refused" in o["message"], o)

    def test_unknown_outcome_is_not_retried(self):
        r, st = self.router([(503, True)], job_get_fails_after=1)
        o = self.go(r)
        self.assertTrue(not o["ok"] and o.get("no_retry") is True and "unknown" in o["message"], o)

    def test_unreadable_job_before_start(self):
        r, st = self.router([], job_get_fails_after=0)
        o = self.go(r)
        self.assertTrue(not o["ok"] and not st["execs"], o)

    def test_cancelled_and_condition_only(self):
        r, _ = self.router([], ex_body={"completionTime": "x", "cancelledCount": 1,
                                        "conditions": [{"type": "Completed", "state": "CONDITION_FAILED"}]})
        o = self.go(r)
        self.assertTrue(o.get("cancelled") is True and not o["ok"])
        r, _ = self.router([], ex_body={"completionTime": "x", "conditions": [{"type": "Completed", "state": "CONDITION_SUCCEEDED"}]})
        self.assertTrue(self.go(r)["ok"])

    def test_stage_job_does_not_retry_no_retry(self):
        r, st = self.router([(503, True)], job_get_fails_after=1)
        it = interp("retrain", r)
        e = raised(self, lambda: it.call_sub("stage_job", {"cfg": CFG, "ctx": CTX, "stage": "collect", "job": "j", "args": ["collect"],
                                                           "timeout_s": 60, "attempts": 3}))
        self.assertEqual(e.get("attempt"), 1, e)
        self.assertEqual(len([x for x in r.log if x[1].endswith(":run")]), 1)


class ResolveImages(unittest.TestCase):
    @staticmethod
    def router(related, have):
        r = Router()
        r.handlers.append((lambda m, u, q: "run.googleapis.com" in u and "/revisions/" in u,
                           lambda *a: (200, {"containers": [{"image": "ne2-docker.pkg.dev/p/sous-chef/sous-chef@sha256:f6"}],
                                             "createTime": "2026-10-13T00:00:00Z"}, {})))
        r.handlers.append((lambda m, u, q: "run.googleapis.com" in u,
                           lambda *a: (200, {"name": "projects/p/locations/ne2/services/sous-chef",
                                             "latestReadyRevision": "projects/p/locations/ne2/services/sous-chef/revisions/r1",
                                             "trafficStatuses": [{"type": "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST", "revision": "r1", "percent": 100}]}, {})))

        def ver(m, u, q, h, b):
            out = {"name": "v"}
            if related is not None and q.get("view") == "FULL":
                out["relatedTags"] = [{"name": f"projects/p/locations/ne2/repositories/sous-chef/packages/sous-chef/tags/{t}"} for t in related]
            return 200, out, {}
        r.handlers.append((lambda m, u, q: "/versions/" in u, ver))

        def tag(m, u, q, h, b):
            pkg, t = urllib.parse.unquote(u).split("/packages/")[1].split("/tags/")
            if (pkg, t) in have:
                return 200, {"version": f"x/versions/sha256:{pkg[-3:]}{t}"}, {}
            return 404, {"error": {"code": 404}}, {}
        r.handlers.append((lambda m, u, q: "/tags/" in u, tag))
        return r

    def test_untagged(self):
        it = interp("drift", self.router(None, set()))
        self.assertEqual(raised(self, lambda: it.call_sub("resolve_images", {"cfg": CFG})).get("error_class"), "untagged_service_image")

    def test_prefers_lab_and_trainer(self):
        it = interp("drift", self.router(["latest", "a1", "b2"], {("sous-chef-lab", "a1"), ("sous-chef-lab", "b2"), ("sous-chef-trainer", "b2")}))
        o = it.call_sub("resolve_images", {"cfg": CFG})
        self.assertTrue(o["tag"] == "b2" and o["lab_ok"] and o["trainer_ok"], o)
        it = interp("drift", self.router(["latest", "a1"], {("sous-chef-lab", "a1")}))
        o = it.call_sub("resolve_images", {"cfg": CFG})
        self.assertTrue(o["tag"] == "a1" and o["lab_ok"] and not o["trainer_ok"], o)


class NotifyAndDecide(unittest.TestCase):
    def test_details_capped_ids_by_index(self):
        r, posted = Router(), []
        r.handlers.append((lambda m, u, q: "therm.example" in u, lambda m, u, q, h, b: (posted.append(b) or (201, {"stored": True}, {}))))
        it = interp("drift", r)
        res = {"notifications": [{"kind": "drift_notice", "title": "t", "summary": "s", "details": {"blob": "x" * 60000}},
                                 {"kind": "heartbeat", "title": "h", "summary": "h"},
                                 {"kind": "drift_notice", "title": "t2", "summary": "s2", "details": {"k": 1}}],
               "verdict": "ok", "model": {"tag": "r4", "digest": "d", "training_manifest": "m"}}
        self.assertEqual(it.call_sub("notify_result", {"cfg": CFG, "res": res, "run_id": "dr-x"}), 2)
        self.assertEqual([p["id"] for p in posted], ["dr-x/drift_notice/0", "dr-x/drift_notice/2"])
        self.assertTrue(posted[0]["details"].get("truncated") is True and posted[0]["details"].get("verdict") == "ok")
        self.assertEqual(posted[1]["details"].get("k"), 1)
        self.assertLess(it.peak_memory, 512 * 1024)

    def test_decision_wait_steps_per_day(self):
        r, state, marks = Router(), {"s": "pending"}, []

        def therm(m, u, q, h, b):
            if m == "POST":
                return 201, {"stored": True}, {}
            return 200, {"candidate_id": "c-rt-x", "run_id": "rt-x", "notification_id": "rt-x/candidate_ready/1", "state": state["s"],
                         "smoke": False, "bar_pass": True, "created_at": "2026-10-01T00:00:00Z", "expires_at": "2026-11-02T00:00:00Z",
                         "decided_at": "2026-10-05T00:00:00Z", "decided_by": "x"}, {}
        r.handlers.append((lambda m, u, q: "therm.example" in u, therm))
        r.put("runs/rt-x/decision.json", {"bar_pass": True, "results": {"candidate": {"digest": "sha256:a"}}})
        r.put("runs/rt-x/result-eval.json", {"status": "ok"})
        r.put("state/locks/retrain.json", {"holder": "", "run_id": "rt-x", "expires_at": "2099-01-01T00:00:00Z"})

        def on_sleep(i, s):
            marks.append(i.steps)
            if len(marks) == 49:
                state["s"] = "approved"
        d = interp("retrain", r, on_sleep=on_sleep).call_sub("stage_decide", {"cfg": CFG, "ctx": CTX})
        self.assertEqual(d["state"], "approved")
        self.assertLess((marks[48] - marks[0]) * 15, 30_000, "a 14-day wait (+1) stays far below 100,000 steps")


class VX:
    """Vertex customJobs: per-create behaviours; cancel takes `cancel_polls` GETs (None: never ends)."""

    def __init__(self, plan, cancel_polls=2, lose_create_answer=0):
        self.plan, self.jobs, self.n, self.cancels = plan, {}, 0, []
        self.cancel_polls, self.lose = cancel_polls, lose_create_answer
        self.filters: list = []
        self.on_end = None  # on_end(job): what the trainer left in the bucket when the job ended

    def handle(self, m, u, q, h, b):
        path = urllib.parse.urlsplit(u).path
        if m == "GET" and path.endswith("/customJobs"):
            self.filters.append(q.get("filter"))
            want = (q.get("filter") or "").split("=", 1)[1].strip('"')
            hits = [self.view(j) for j in self.jobs.values() if j["labels"].get("create") == want]
            return 200, ({"customJobs": hits} if hits else {}), {}
        if m == "POST" and path.endswith("/customJobs"):
            beh = self.plan.pop(0)
            self.n += 1
            name = f"projects/p/locations/us-central1/customJobs/{self.n}"
            self.jobs[name] = {"name": name, "beh": beh, "polls": 0, "state": "JOB_STATE_PENDING", "labels": dict(b.get("labels") or {})}
            if self.lose > 0:
                self.lose -= 1
                return 503, {"error": {"code": 503}}, {}
            return 200, {"name": name}, {}
        name = path[len("/v1/"):]
        if name.endswith(":cancel"):
            name = name[:-len(":cancel")]
            self.cancels.append(name)
            self.jobs[name]["cancelling"] = self.cancel_polls
            return 200, {}, {}
        j = self.jobs[name]
        j["polls"] += 1
        beh = j["beh"]
        if j.get("cancelling", "no") != "no":
            if j["cancelling"] is not None:
                j["cancelling"] -= 1
            j["state"] = ("JOB_STATE_SUCCEEDED" if beh == "late_success" else "JOB_STATE_CANCELLED") \
                if j["cancelling"] is not None and j["cancelling"] <= 0 else "JOB_STATE_CANCELLING"
        elif beh == "flex_fail":
            j["state"], j["error"] = "JOB_STATE_FAILED", {"code": 8, "message": "something undocumented"}
        elif beh in ("pending", "late_success"):
            j["state"] = "JOB_STATE_PENDING"
        elif beh == "run_then_ok":
            j["state"] = "JOB_STATE_SUCCEEDED" if j["polls"] >= 3 else "JOB_STATE_RUNNING"
            j["startTime"] = "2026-10-03T00:00:00Z"
        elif beh in ("exit1", "flex_14"):
            # exit1: the container exited 1 (Vertex's own text); flex_14: Flex-start's wait ran out (code 14), with a
            # startTime though no machine was ever provisioned (2026-10-14, job 463589157793431552)
            j["state"], j["startTime"] = "JOB_STATE_FAILED", "2026-10-03T00:00:00Z"
            j["error"] = {"code": 3, "message": "The replica workerpool0-0 exited with a non-zero status of 1."} \
                if beh == "exit1" else {"code": 14, "message": "Job failed."}
            if self.on_end:
                self.on_end(j)
        return 200, self.view(j), {}

    @staticmethod
    def view(j):
        return {k: v for k, v in j.items() if k in ("name", "state", "error", "startTime", "labels")}


def vrouter(vx):
    r = Router()

    def h(m, u, q, hd, b):
        out = vx.handle(m, u, q, hd, b)
        if any(j["state"] == "JOB_STATE_SUCCEEDED" for j in vx.jobs.values()):
            r.put("runs/rt-x/train/DONE", {"ok": True})
        return out
    r.handlers.append((lambda m, u, q: "aiplatform.googleapis.com" in u, h))
    r.put("state/locks/retrain.json", {"holder": "", "run_id": "rt-x", "expires_at": "2099-01-01T00:00:00Z"})
    return r


SPOT_ONLY = copy.deepcopy(CFG)
SPOT_ONLY["trainer"]["ladder"] = [CFG["trainer"]["ladder"][1]]


class Vertex(unittest.TestCase):
    def test_flex_failure_without_start_is_capacity(self):
        vx = VX(["flex_fail", "run_then_ok"])
        r = vrouter(vx)
        out = interp("retrain", r).call_sub("stage_train", {"cfg": CFG, "ctx": CTX})
        vj = r.gcs["runs/rt-x/train/vertex.json"][0]
        self.assertTrue(out["state"] == "succeeded" and vj["attempts"][0]["state"] == "capacity", vj["attempts"])
        self.assertNotIn("wf/runs/rt-x/train/current.json", r.gcs)

    def test_cancel_awaited_late_success_taken(self):
        vx = VX(["late_success"])
        out = interp("retrain", vrouter(vx)).call_sub("stage_train", {"cfg": SPOT_ONLY, "ctx": CTX})
        self.assertTrue(out["state"] == "succeeded" and vx.cancels, out)

    def test_cancel_that_never_ends_starts_no_other_job(self):
        vx = VX(["pending", "run_then_ok"], cancel_polls=None)
        r = vrouter(vx)
        e = raised(self, lambda: interp("retrain", r).call_sub("stage_train", {"cfg": SPOT_ONLY, "ctx": CTX}))
        self.assertEqual(e.get("error_class"), "train_failed", e)
        self.assertEqual(vx.n, 1, "no second job beside the one still cancelling")
        self.assertIn("wf/runs/rt-x/train/current.json", r.gcs, "kept for the round's except and a resume")

    def test_running_job_is_adopted(self):
        vx = VX([])
        name = "projects/p/locations/us-central1/customJobs/77"
        vx.jobs[name] = {"name": name, "beh": "run_then_ok", "polls": 0, "state": "JOB_STATE_RUNNING", "labels": {}}
        r = vrouter(vx)
        r.put("wf/runs/rt-x/train/current.json", {"job": name, "base": "https://us-central1-aiplatform.googleapis.com/v1/",
                                                  "rung": "spot", "try_n": 1, "created_at": "2026-10-03T00:00:00Z"})
        out = interp("retrain", r).call_sub("stage_train", {"cfg": CFG, "ctx": CTX})
        self.assertTrue(out["state"] == "succeeded" and out["job"].endswith("/77") and vx.n == 0, out)

    def test_lost_create_answer_is_looked_up(self):
        vx = VX(["run_then_ok", "run_then_ok"], lose_create_answer=1)
        out = interp("retrain", vrouter(vx)).call_sub("stage_train", {"cfg": SPOT_ONLY, "ctx": CTX})
        self.assertTrue(out["state"] == "succeeded" and vx.n == 1, (out, vx.n))
        self.assertTrue(vx.filters and vx.filters[0].startswith('labels.create="') and vx.filters[0].endswith('"'), vx.filters)

    def attempt(self, beh, on_end, rung=None):
        vx = VX([beh])
        r = vrouter(vx)
        vx.on_end = lambda j: on_end(r, j)
        out = interp("retrain", r).call_sub("vertex_attempt", {"cfg": CFG, "ctx": CTX, "rung": rung or CFG["trainer"]["ladder"][1],
                                                              "try_n": 1})
        return out, vx, r

    def test_restarted_failure_stays_fatal(self):
        # Vertex restarted the container after exit 1; the restart rewrote progress.json as `interrupted` (2026-10-14).
        # The trainer's sticky train/failed-<nonce>.json for THIS job decides: fatal, never retried as a preemption
        def end(r, j):
            n = j["labels"]["create"]
            r.put("runs/rt-x/train/progress.json", {"status": "interrupted", "step": "pull base", "nonce": n,
                                                     "updated_at": "2099-01-01T00:00:00Z"})
            r.put(f"runs/rt-x/train/failed-{n}.json", {"status": "failed", "error": "export q4_k_m: export", "nonce": n})
        out, vx, r = self.attempt("exit1", end)
        self.assertEqual(out["state"], "fatal", out)
        self.assertIn("export q4_k_m", out["message"])
        n = next(iter(vx.jobs.values()))["labels"]["create"]
        self.assertTrue(n and n.startswith("8a0cc8ec-"), n)

    def test_the_nonce_reaches_the_trainer(self):
        vx = VX(["run_then_ok"])
        bodies = []
        h = vx.handle

        def spy(m, u, q, hd, b):
            if m == "POST" and u.endswith("/customJobs"):
                bodies.append(b)
            return h(m, u, q, hd, b)
        vx.handle = spy
        interp("retrain", vrouter(vx)).call_sub("stage_train", {"cfg": SPOT_ONLY, "ctx": CTX})
        env = {e["name"]: e["value"] for e in bodies[0]["jobSpec"]["workerPoolSpecs"][0]["containerSpec"]["env"]}
        self.assertEqual(env["TRAIN_NONCE"], bodies[0]["labels"]["create"])
        self.assertEqual(env["TRAIN_IMAGE"], "tr:t")

    def test_another_jobs_progress_is_not_this_attempts(self):
        # a leftover container of an earlier job wrote `interrupted` while this job ran: not this job's preemption
        def end(r, j):
            r.put("runs/rt-x/train/progress.json", {"status": "interrupted", "step": "SFT", "nonce": "other-1",
                                                     "updated_at": "2099-01-01T00:00:00Z"})
        out, _, _ = self.attempt("exit1", end)
        self.assertEqual(out["state"], "unknown", out)

    def test_own_interrupted_progress_is_a_preemption(self):
        def end(r, j):
            r.put("runs/rt-x/train/progress.json", {"status": "interrupted", "step": "SFT", "nonce": j["labels"]["create"],
                                                     "updated_at": "2000-01-01T00:00:00Z"})
        out, _, _ = self.attempt("exit1", end)
        self.assertEqual(out["state"], "preempted", out)

    def test_flex_wait_code_14_is_capacity(self):
        out, _, _ = self.attempt("flex_14", lambda r, j: None, rung=CFG["trainer"]["ladder"][0])
        self.assertEqual(out["state"], "capacity", out)

    def test_stop_current(self):
        vx = VX([])
        name = "projects/p/locations/us-central1/customJobs/9"
        vx.jobs[name] = {"name": name, "beh": "pending", "polls": 0, "state": "JOB_STATE_PENDING", "labels": {}}
        r = vrouter(vx)
        r.put("wf/runs/rt-x/train/current.json", {"job": name, "base": "https://us-central1-aiplatform.googleapis.com/v1/", "rung": "spot"})
        it = interp("retrain", r)
        s = it.call_sub("vertex_stop_current", {"cfg": CFG, "run_id": "rt-x"})
        self.assertTrue("JOB_STATE_CANCELLED" in s and "wf/runs/rt-x/train/current.json" not in r.gcs, s)
        self.assertEqual(it.call_sub("vertex_stop_current", {"cfg": CFG, "run_id": "rt-x"}), "no training job")


class RunRecord(unittest.TestCase):
    def test_resume_moves_the_failure_and_links_the_execution(self):
        r = Router()
        old = {"run_id": "rt-x", "status": "failed", "images": {"tag": "t0"}, "stages": {}, "resumes": [],
               "failed": {"stage": "train", "error_class": "train_failed"}}
        rec = interp("retrain", r).call_sub("run_record", {"cfg": CFG, "ctx": CTX, "existing": old, "resume_from": "train"})
        self.assertNotIn("failed", rec)
        self.assertEqual(rec["status"], "running")
        self.assertEqual(rec["resumes"][-1]["previous_failure"]["error_class"], "train_failed")
        self.assertEqual(rec["resumes"][-1]["previous_images"], {"tag": "t0"})
        self.assertEqual(rec["execution_url"], "https://console.cloud.google.com/workflows/workflow/us-central1/souschef-retrain"
                                               "/execution/8a0cc8ec-1111-2222-3333-444455556666?project=p")
        self.assertEqual(r.gcs["runs/rt-x/run.json"][0], rec)


    def test_resume_refuses_another_service_tag(self):
        it = interp("retrain", Router())
        run = {"images": {"tag": "e5a8e05"}}
        call = lambda tag, frm, at: it.call_sub("images_unchanged", {"run": run, "imgs": {"tag": tag}, "resume_from": frm,
                                                                      "start_at": at})
        e = raised(self, lambda: call("f00ba47", "train", 3))
        self.assertEqual(e.get("error_class"), "images_changed", e)
        self.assertTrue(call("e5a8e05", "train", 3))   # the same tag, re-pointed
        self.assertTrue(call("f00ba47", "dataset", 2))  # before the dataset: rebuilt with the new images
        self.assertTrue(call("f00ba47", "decide", 7))   # the decision wait runs no image
        self.assertTrue(call("f00ba47", "", 0))         # a new round
        self.assertTrue(it.call_sub("images_unchanged", {"run": None, "imgs": {"tag": "t"}, "resume_from": "", "start_at": 0}))


class Build(unittest.TestCase):
    def run_build(self, status):
        r, st = Router(), {"tag": None, "builds": 0}
        cfg = copy.deepcopy(CFG)
        cfg["build"] = {"spec": "infra/cb.json", "region": "us-east4", "service_account": "b@p"}
        r.put("infra/cb.json", {"steps": []})
        r.put("runs/rt-x/model/context.tar.gz", {"x": 1})

        def ar(m, u, q, h, b):
            return (200, {"version": "x/versions/" + st["tag"]}, {}) if st["tag"] else (404, {"error": {"code": 404}}, {})

        def cb(m, u, q, h, b):
            if m == "POST":
                st["builds"] += 1
                return 200, {"metadata": {"build": {"id": f"b{st['builds']}"}}}, {}
            st["tag"] = "sha256:pushed"  # the tag holds an image (this build's push, or a twin's), then this build ends badly
            return 200, {"status": status, "id": "b1"}, {}
        r.handlers.append((lambda m, u, q: "/tags/" in u, ar))
        r.handlers.append((lambda m, u, q: "cloudbuild.googleapis.com" in u, cb))
        d = interp("retrain", r).call_sub("stage_build", {"cfg": cfg, "ctx": CTX})
        return d, st, r

    def test_pushed_tag_reused_on_any_failure(self):
        for status in ("TIMEOUT", "FAILURE"):
            d, st, r = self.run_build(status)
            self.assertTrue(d == "sha256:pushed" and st["builds"] == 1, (status, d, st))
            self.assertEqual(r.gcs["runs/rt-x/model/image.json"][0]["digest"], "sha256:pushed")


class Thermometer(unittest.TestCase):
    def test_raw_text_404(self):
        r = Router()
        r.handlers.append((lambda m, u, q: "therm.example" in u, lambda *a: (404, '{"error":"no decision","code":"NOT_FOUND"}', {})))
        o = interp("retrain", r).call_sub("thermometer_decision", {"cfg": CFG, "candidate_id": "c-1"})
        self.assertTrue(o["found"] is False and o["error"] is None, o)
        r = Router()
        r.handlers.append((lambda m, u, q: "therm.example" in u, lambda *a: (404, "404 page not found", {})))
        o = interp("retrain", r).call_sub("thermometer_decision", {"cfg": CFG, "candidate_id": "c-1"})
        self.assertTrue(o["found"] is False and o["error"] is not None, o)


class Rollout(unittest.TestCase):
    C = dict(copy.deepcopy(CFG), rollout_enabled=True, l4_headroom=True)

    def setup(self, sv):
        r = Router()
        r.put("runs/rt-x/model/image.json", {"image": "x:c-rt-x", "digest": "sha256:new"})
        r.put("runs/rt-x/decision.json", {"results": {"candidate": {"digest": "sha256:new"}}})
        r.handlers.append((lambda m, u, q: "/tags/" in u, lambda *a: (200, {"version": "x/versions/sha256:new"}, {})))
        r.handlers.append((lambda m, u, q: "run.googleapis.com" in u, sv.handle))
        return r

    def roll(self, r):
        return interp("retrain", r).call_sub("stage_rollout", {"cfg": self.C, "ctx": CTX, "d": {"decided_by": "g"}})

    def test_pinned_traffic_refused(self):
        pinned = [{"type": "TRAFFIC_TARGET_ALLOCATION_TYPE_REVISION", "revision": "m-1", "percent": 100},
                  {"type": "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"}]
        sv = Svc(traffic=pinned)
        r = self.setup(sv)
        e = raised(self, lambda: self.roll(r))
        self.assertTrue(e.get("error_class") == "traffic_pinned" and not sv.patches, e)
        self.assertNotIn("state/locks/gpu-eval.json", r.gcs)

    def test_failed_revision_keeps_its_class(self):
        sv = Svc(fail_next=True)
        r = self.setup(sv)
        e = raised(self, lambda: self.roll(r))
        ro = r.gcs["wf/runs/rt-x/rollout.json"][0]
        self.assertEqual(e.get("error_class"), "revision_not_ready", e)
        self.assertEqual(e.get("pipeline_stage"), "rollout")
        self.assertTrue(ro["state"] == "unconfirmed" and ro["old_image"] == "img@sha256:m-1", ro)

    def test_traffic_not_moved(self):
        sv = Svc(moves_traffic=False)
        r = self.setup(sv)
        e = raised(self, lambda: self.roll(r))
        self.assertEqual(e.get("error_class"), "traffic_not_moved", e)
        self.assertEqual(r.gcs["wf/runs/rt-x/rollout.json"][0]["state"], "unconfirmed")

    def test_unsettled_is_unconfirmed_and_resume_keeps_rollback(self):
        sv = Svc(settles=False)
        r = self.setup(sv)
        e = raised(self, lambda: self.roll(r))
        ro = r.gcs["wf/runs/rt-x/rollout.json"][0]
        self.assertEqual(e.get("error_class"), "rollout_unconfirmed", e)
        self.assertTrue(ro["state"] == "unconfirmed" and ro["old_image"] == "img@sha256:m-1", ro)
        # production did take it after all; a resume must not patch again, and the rollback still names m-1's image
        sv.settles = True
        sv.pending = 1
        sv.handle("GET", "https://run.googleapis.com/v2/" + sv.s["name"], {}, {}, None)
        sv.s["template"]["containers"][0]["image"] = "us-east4-docker.pkg.dev/p/sous-chef-model/model@sha256:new"
        n = len(sv.patches)
        try:
            self.roll(r)
        except (wfsim.WfError, wfsim.SimError):
            pass  # the decision job after the rollout has no fake here
        ro = r.gcs["wf/runs/rt-x/rollout.json"][0]
        self.assertEqual(len(sv.patches), n, "no second PATCH")
        self.assertTrue(ro.get("old_image") == "img@sha256:m-1" and "sha256:m-1" in ro.get("rollback", ""), ro)


if __name__ == "__main__":
    unittest.main()
