#!/usr/bin/env python3
"""Unit tests of wfsim's Workflows limits and runtime behaviour (https://docs.cloud.google.com/workflows/quotas,
.../docs/reference/syntax/variables, .../syntax/retrying, .../syntax/error-types, .../docs/http-requests).

    python3 -m unittest infra/modules/pipeline/tests/test_wfsim_limits.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wfsim  # noqa: E402


def sim(src: str, http=None, **kw) -> wfsim.Interpreter:
    return wfsim.Interpreter(yaml.safe_load(src), http=http, secrets=lambda a: "s", env={}, clock=wfsim.VirtualClock(0),
                             strict=False, **kw)


class StepLimit(unittest.TestCase):
    def test_endless_loop_is_an_uncatchable_resource_limit(self):
        s = sim("""
main:
  steps:
    - guard:
        try:
          steps:
            - spin:
                assign:
                  - x: 1
                next: spin
        except:
          as: e
          steps:
            - caught:
                return: "caught"
""")
        with self.assertRaises(wfsim.ResourceLimitError):
            s.run()
        self.assertEqual(s.steps, wfsim.STEP_LIMIT + 1)

    def test_subworkflow_call_counts_the_call_and_its_steps(self):
        s = sim("""
main:
  steps:
    - a:
        call: sub
    - b:
        return: 1
sub:
  steps:
    - x:
        assign:
          - y: 1
    - z:
        assign:
          - y: 2
""")
        s.run()
        self.assertEqual(s.steps, 4)  # a, x, z, b

    def test_each_retry_attempt_counts(self):
        calls = []

        def http(method, url, query, headers, body, auth):
            calls.append(url)
            return (503, "busy", {}) if len(calls) < 3 else (200, b'{"ok": true}', {"content-type": "application/json"})
        s = sim("""
main:
  steps:
    - get:
        try:
          call: http.get
          args:
            url: https://example.com/x
          result: r
        retry: ${http.default_retry}
    - done:
        return: ${r.body.ok}
""", http=http)
        self.assertIs(s.run(), True)
        self.assertEqual(s.steps, 2 + 2)  # get, done + two retries


class Memory(unittest.TestCase):
    def test_sizes_follow_the_documented_formula(self):
        self.assertEqual(wfsim.wsize("This is a string of 28 bytes"), 36)
        self.assertEqual(wfsim.wsize(1), 8)
        self.assertEqual(wfsim.wsize([1, "ab"]), 8 + 8 + 10)
        self.assertEqual(wfsim.wsize({"k": 1}), 8 + 9 + 8)

    def test_copies_in_every_live_frame_count_and_cannot_be_caught(self):
        prog = """
main:
  steps:
    - make:
        assign:
          - big: '${text.replace_all("%s", "x", "xxxxxxxxxx")}'
    - guard:
        try:
          call: down
          args:
            v: ${big}
            n: 3
        except:
          as: e
          steps:
            - caught:
                return: "caught"
    - done:
        return: "ok"
down:
  params: [v, n]
  steps:
    - more:
        switch:
          - condition: ${n > 0}
            call: down
            args:
              v: ${v}
              n: ${n - 1}
    - done:
        return: ${len(v)}
"""
        small = sim(prog % ("x" * 9000))  # 90 KB: five frames x 90 KB = 450 KB < 512 KB
        self.assertEqual(small.run(), "ok")
        self.assertGreater(small.peak_memory, 400_000)
        big = sim(prog % ("x" * 11000))  # 110 KB: five copies > 512 KB
        with self.assertRaises(wfsim.ResourceLimitError):
            big.run()

    def test_for_loop_iteration_variables_are_freed(self):
        s = sim("""
main:
  steps:
    - loop:
        for:
          value: i
          range: [1, 50]
          steps:
            - tmp:
                assign:
                  - local_big: '${text.replace_all("%s", "x", "xxxxxxxxxx")}'
    - done:
        return: "ok"
""" % ("x" * 20000))
        self.assertEqual(s.run(), "ok")
        self.assertLess(s.peak_memory, 300_000)

    def test_a_string_over_256_kb(self):
        s = sim("""
main:
  steps:
    - make:
        assign:
          - s: '${text.replace_all("%s", "x", "xxxxxxxxxx")}'
""" % ("x" * 30000))
        with self.assertRaises(wfsim.ResourceLimitError):
            s.run()

    def test_in_place_assignment_is_resized(self):
        s = sim("""
main:
  steps:
    - init:
        assign:
          - m: {}
    - grow:
        for:
          value: i
          range: [1, 30]
          steps:
            - put:
                assign:
                  - m[string(i)]: '${text.replace_all("%s", "x", "xxxxxxxxxx")}'
""" % ("x" * 2000))
        with self.assertRaises(wfsim.ResourceLimitError):
            s.run()


class Retry(unittest.TestCase):
    def router(self, codes):
        seq = list(codes)

        def http(method, url, query, headers, body, auth):
            c = seq.pop(0) if seq else 200
            return (c, {"error": c} if c >= 400 else {"ok": True}, {"Content-Type": "application/json"})
        return http

    def test_custom_predicate_subworkflow(self):
        s = sim("""
main:
  steps:
    - get:
        try:
          call: http.get
          args:
            url: https://example.com/x
          result: r
        retry:
          predicate: ${retry_pred}
          max_retries: 3
          backoff:
            initial_delay: 1
            max_delay: 10
            multiplier: 2
    - done:
        return: ${r.code}
retry_pred:
  params: [e]
  steps:
    - judge:
        return: ${default(map.get(e, "code"), 0) in [408, 429, 500, 502, 503, 504]}
""", http=self.router([500, 408]))
        self.assertEqual(s.run(), 200)

    def test_default_predicates(self):
        I = wfsim.Interpreter
        s = sim("main:\n  steps:\n    - a:\n        return: 1\n")
        idem = {"predicate": "http.default_retry_predicate"}
        non = {"predicate": "http.default_retry_predicate_non_idempotent"}
        self.assertFalse(s._should_retry(idem, {"code": 500, "tags": ["HttpError"]}))
        self.assertFalse(s._should_retry(idem, {"code": 408, "tags": ["HttpError"]}))
        self.assertTrue(s._should_retry(idem, {"tags": ["ConnectionFailedError"]}))
        self.assertTrue(s._should_retry(non, {"tags": ["ConnectionFailedError"]}))
        self.assertFalse(s._should_retry(non, {"code": 502, "tags": ["HttpError"]}))
        self.assertTrue(I.STD_PREDICATES)

    def test_500_is_not_retried_by_the_default_policy(self):
        s = sim("""
main:
  steps:
    - get:
        try:
          call: http.get
          args:
            url: https://example.com/x
          result: r
        retry: ${http.default_retry}
""", http=self.router([500]))
        with self.assertRaises(wfsim.WfError) as cm:
            s.run()
        self.assertEqual(cm.exception.value["code"], 500)


class Responses(unittest.TestCase):
    def run_get(self, body, headers):
        s = sim("""
main:
  steps:
    - get:
        try:
          call: http.get
          args:
            url: https://example.com/x
          result: r
        except:
          as: e
          steps:
            - err:
                return: ${e}
    - done:
        return: ${r}
""", http=lambda *a: (200 if not isinstance(body, str) or "No such" not in body else 404, body, headers))
        return s.run()

    def test_body_parsed_by_content_type_and_headers_canonical(self):
        r = self.run_get(b'{"a": 1}', {"content-type": "application/json; charset=UTF-8", "x-goog-generation": "7"})
        self.assertEqual(r["body"], {"a": 1})
        self.assertEqual(r["headers"]["X-Goog-Generation"], "7")
        r = self.run_get(b'{"a": 1}', {"Content-Type": "application/octet-stream"})
        self.assertEqual(r["body"], b'{"a": 1}')
        r = self.run_get(b"hello", {"Content-Type": "text/plain"})
        self.assertEqual(r["body"], "hello")

    def test_error_body_text(self):
        e = self.run_get("No such object: b/o", {"Content-Type": "text/html; charset=UTF-8"})
        self.assertEqual(e["code"], 404)
        self.assertEqual(e["body"], "No such object: b/o")
        self.assertIn("Content-Type", e["headers"])

    def test_response_over_2_mb(self):
        with self.assertRaises(wfsim.ResourceLimitError):
            self.run_get(b"x" * (2 * 1024 * 1024 + 1), {"Content-Type": "application/octet-stream"})


class Misc(unittest.TestCase):
    def test_url_encode_escapes_tilde(self):
        s = sim("main:\n  steps:\n    - a:\n        return: '${text.url_encode(\"wf/outbox/a~b c\")}'\n")
        self.assertEqual(s.run(), "wf%2Foutbox%2Fa%7Eb%20c")

    def test_json_decode_rejects_nan(self):
        ok = sim("main:\n  steps:\n    - a:\n        return: '${json.decode(\"[1.5]\")}'\n")
        self.assertEqual(ok.run(), [1.5])
        s = sim("main:\n  steps:\n    - a:\n        return: '${json.decode(\"[NaN]\")}'\n")
        with self.assertRaises(wfsim.WfError) as cm:
            s.run()
        self.assertIn("NaN", str(cm.exception.value))

    def test_call_depth(self):
        s = sim("""
main:
  steps:
    - go:
        try:
          call: rec
          args:
            n: 0
        except:
          as: e
          steps:
            - caught:
                return: ${e.tags}
rec:
  params: [n]
  steps:
    - again:
        call: rec
        args:
          n: ${n + 1}
""")
        self.assertEqual(s.run(), ["RecursionError"])
        self.assertEqual(s.max_depth, wfsim.CALL_DEPTH_LIMIT + 1)


if __name__ == "__main__":
    unittest.main()
