#!/usr/bin/env python3
"""Unit tests of the strict Workflows checker (lint_workflows.py) and of wfsim's use of it.

    python3 -m unittest infra/modules/pipeline/tests/test_lint_workflows.py

The valid cases are Google's own documented examples (reference/syntax/*, reference/stdlib/*); the invalid
ones are what the docs say a deploy rejects, plus the exact construct of the 2026-10-13 deploy error.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import lint_workflows as L  # noqa: E402
import wfsim  # noqa: E402


def rules(findings, severity="error"):
    return sorted({f.rule for f in findings if f.severity == severity})


def check(yaml_text: str):
    return L.check_text(textwrap.dedent(yaml_text))


def expr_rules(expr: str, extra_vars=("a", "b", "x", "y", "e", "lst", "m", "input", "args")) -> list[str]:
    """Rules an expression trips inside a one-step main that defines extra_vars."""
    assigns = "".join(f"\n          - {v}: null" for v in extra_vars)
    text = f"main:\n  steps:\n    - init:\n        assign:{assigns}\n    - out:\n        return: {json.dumps(expr)}\n"
    return rules(check(text))


class Grammar(unittest.TestCase):
    # The real 2026-10-13 error: "parse error: in workflow 'drift_after', step 'start': extraneous input '{'
    # expecting {'[', LOGICAL_NOT, '(', ')', '-', TRUE, FALSE, NULL, NUM_FLOAT, NUM_INT, STRING, IDENTIFIER}"
    REAL = ('json.encode_to_string({"mode": "full", "trigger": {"drift_run": run_id, "reason": '
            'default(map.get(res, "retrain_reason"), "drift")}})')

    def test_the_real_failure_is_rejected(self):
        with self.assertRaises(L.ExprSyntaxError) as cm:
            L.parse_expression(self.REAL)
        self.assertEqual(cm.exception.rule, "map-literal")
        self.assertEqual(cm.exception.pos, self.REAL.index("{"))

    def test_the_real_failure_in_its_step(self):
        fs = check("""
        main:
          steps:
            - go:
                call: drift_after
                args: {run_id: "r", res: {}}
        drift_after:
          params: [run_id, res]
          steps:
            - start:
                call: http.post
                args:
                  url: https://example.com
                  body:
                    argument: '${json.encode_to_string({"mode": "full", "trigger": {"drift_run": run_id, "reason": default(map.get(res, "retrain_reason"), "drift")}})}'
                result: ex
            - done:
                return: ${ex.body}
        """)
        errs = [f for f in fs if f.severity == "error"]
        self.assertEqual([(f.rule, f.subworkflow, f.step, f.line) for f in errs],
                         [("map-literal", "drift_after", "start", 15)])

    def test_every_map_literal_form_is_rejected(self):
        for e in ['default(args, {})', 'default(map.get(a, "trigger"), {"drift_run": null, "reason": "manual"})',
                  'list.concat(lst, {"name": x, "value": string(y)})', 'default(map.get(m, "tasks"), [{}])[0]',
                  '{}', '{"a": 1}', 'json.encode({})', '[1, {"a": 2}]']:
            with self.subTest(e=e):
                with self.assertRaises(L.ExprSyntaxError) as cm:
                    L.parse_expression(e)
                self.assertEqual(cm.exception.rule, "map-literal")

    def test_documented_expressions_parse(self):
        # every one is from a Google doc page (syntax/*, stdlib/*)
        for e in ['"Hello " + first_name + " " + last_name + " from " + country + "."',
                  'not("Nur" in list)', "'SystemError' in e.tags", 'my_true and my_false', 'my_true or my_false',
                  'not my_true', 'temperature_C * 9/5 + 32', 'num % 2 != 0', 'text.split(location, "-")[0]',
                  'map.get(my2DMap, ["key1", "key2"])', '[minTemp, maxTemp]', 'math.abs(-123)',
                  'math.max(123,-456.7)', 'default(map.get(input, "region"), "northamerica-northeast1")',
                  '\n"SELECT * from " + table +\n"\\nWHERE id=" + string(id)\n', 'http.default_retry',
                  'sys.now()', 'uuid.generate()', 'len(array) > i', 'name_message("Ada","Lovelace")',
                  'monthsList.body[2]', 'translation.data.translations[0].translatedText', 'var.key["special!key"]',
                  'my_list[len(my_list) - 1]', 'json.decode(base64.decode(m.receivedMessages[0].message.data))',
                  'time.format(1720788633,"Europe/Paris")', 'text.substring("hello", -10, 10)', 'null', 'TRUE',
                  'False', 'x // 2', 'if(a, 1, 2)', 'currentTime.body.dayOfWeek == "Saturday" or '
                  'currentTime.body.dayOfWeek == "Sunday"', '"a: " +my_string', 'a AND b OR NOT c']:
            with self.subTest(e=e):
                L.parse_expression(e)

    def test_what_the_grammar_does_not_have(self):
        for e, why in [('a ? b : c', "ternary"), ('x = 1', "assignment"), ('a && b', "C operators"),
                       ('a || b', "C operators"), ('!a', "! negation"), ('lambda x: x', "lambda"),
                       ('a[1:2]', "slicing"), ('f(x)(y)', "calling a call"), ('(a.b)(1)', "calling a group"),
                       ('a.in', "keyword as field"), ('"unterminated', "string"), ('1e5', "exponent"),
                       ('', "empty"), ('a b', "two operands"), ('a}${b', "two expressions in one string"),
                       ('+1', "unary plus"), ('And', "mixed-case keyword is an identifier: fine")]:
            with self.subTest(e=e, why=why):
                if why.endswith("fine"):
                    L.parse_expression(e)
                else:
                    with self.assertRaises(L.ExprSyntaxError):
                        L.parse_expression(e)

    def test_not_binds_tighter_than_comparison(self):
        n = L.parse_expression('not a in b')
        self.assertEqual((n.kind, n.a, n.b.kind), ("bin", "in", "not"))
        n = L.parse_expression('not(a in b)')
        self.assertEqual(n.kind, "not")
        n = L.parse_expression('a + b * c == d and not e or f')
        self.assertEqual(n.kind, "or")
        self.assertEqual(n.a.kind, "and")
        self.assertEqual(n.a.a.a, "==")

    def test_assignment_targets(self):
        for t in ['my_list[0]', 'my_list[idx + 1]', 'my_list[len(my_list) - 1]', 'my_map.Key1', 'my_map["Key2"]',
                  'my_map[key_str + "4"]', 'my_list[0][1]', 'my_nested_map.NestedMapKey.Key2', 'x']:
            with self.subTest(t=t):
                L.parse_target(t)
        for t in ['my_map.{}', '1x', 'a b', 'f(x)', '"a"']:
            with self.subTest(t=t):
                with self.assertRaises(L.ExprSyntaxError):
                    L.parse_target(t)

    def test_escapes(self):
        self.assertEqual(L.parse_expression(r'"a\"b\\c\ndA"').a, 'a"b\\c\ndA')
        p = L.ExprParser(r'"\d+"')
        p.parse()
        self.assertTrue(p.warnings)


class Rules(unittest.TestCase):
    def test_documented_workflow_is_clean(self):
        fs = check("""
        main:
          params: [args]
          steps:
            - init:
                assign:
                  - e: null
                  - my_list: [0, 1, 2]
                  - my_map: {"Key1": "hello"}
                  - dynamicKey: "MiddleName"
                  - person:
                      FirstName: "John"
                      ${dynamicKey}: "Apple"
                  - my_list[0]: "Value0"
                  - my_map[dynamicKey + "x"]: ${len(my_list)}
                  - sum: 0
                  - empty: {}
            - loop:
                for:
                  value: v
                  index: i
                  range: ${[1, 9]}
                  steps:
                    - add:
                        assign:
                          - sum: ${sum + v + i}
                    - check:
                        switch:
                          - condition: ${v > 5}
                            next: break
                          - condition: ${not(v in my_list)}
                            next: continue
            - read:
                try:
                  call: http.get
                  args:
                    url: https://host.com/api
                    query:
                      a: ${string(sum)}
                  result: r
                retry:
                  predicate: ${custom_predicate}
                  max_retries: 5
                  backoff:
                    initial_delay: 2
                    max_delay: 60
                    multiplier: 2
                except:
                  as: e
                  steps:
                    - known:
                        switch:
                          - condition: ${e.code == 404}
                            return: "not found"
            - sub:
                call: name_message
                args:
                  first_name: "Ada"
                  last_name: "Lovelace"
                result: out
            - inline:
                assign:
                  - out2: ${name_message("Sherlock", "Holmes") + " " + out}
                  - secret_ok: ${default(map.get(args, "x"), empty)}
            - log:
                call: sys.log
                args:
                  text: '${"sum: " + string(sum)}'
                  severity: INFO
            - secret:
                call: googleapis.secretmanager.v1.projects.secrets.versions.accessString
                args:
                  secret_id: foo
            - done:
                return: ${[out2, e, r, http.default_retry, sys.get_env("GOOGLE_CLOUD_PROJECT_ID")]}
        name_message:
          params: [first_name, last_name, country: "England"]
          steps:
            - prepareMessage:
                return: ${"Hello " + first_name + " " + last_name + " from " + country + "."}
        custom_predicate:
          params: [e]
          steps:
            - what:
                switch:
                  - condition: ${e.code == 500}
                    return: true
            - otherwise:
                return: false
        """)
        self.assertEqual([str(f) for f in fs if f.severity == "error"], [])

    def test_undefined_symbol(self):  # troubleshooting#deployment_errors
        fs = check("""
        main:
          steps:
            - step1:
                assign:
                  - varA: "Hello"
                  - varB: "World"
            - step2:
                return: ${varC + varB}
        """)
        self.assertEqual(rules(fs), ["undefined-symbol"])
        self.assertIn("'varC'", fs[0].message)

    def test_except_scope(self):  # syntax/variables "Share a variable before an except block"
        bad = """
        main:
          steps:
            - fail:
                try:
                  call: http.get
                  args:
                    url: https://foo.bar
                  result: nope
                except:
                  as: e
            - returnError:
                return: ${e}
        """
        self.assertEqual(rules(check(bad)), ["out-of-scope"])
        good = bad.replace("    steps:\n            - fail:", "    steps:\n            - init:\n                assign:\n"
                           "                  - e: null\n            - fail:")
        self.assertEqual(rules(check(good)), [])

    def test_loop_scope(self):  # syntax/iteration: loop variables die with the loop
        fs = check("""
        main:
          steps:
            - init:
                assign:
                  - workflowScope: foo
            - outerLoop:
                for:
                  value: outerLoopValue
                  in: [1, 2]
                  steps:
                    - a:
                        assign:
                          - outerLoopScope: ${workflowScope}
                          - workflowScope: ${outerLoopValue}
            - final:
                return: ${[workflowScope, outerLoopScope, outerLoopValue]}
        """)
        errs = [f for f in fs if f.severity == "error"]
        self.assertEqual(sorted(f.message.split("'")[1] for f in errs), ["outerLoopScope", "outerLoopValue"])
        self.assertEqual(rules(fs), ["out-of-scope"])

    def test_try_for_known_issue(self):
        fs = check("""
        main:
          steps:
            - t:
                try:
                  for:
                    value: v
                    range: [1, 2]
                    steps:
                      - log:
                          call: sys.log
                          args:
                            data: ${v}
                retry: ${http.default_retry}
        """)
        self.assertIn("try-for", rules(fs))

    def test_jumps(self):
        fs = check("""
        main:
          steps:
            - loop:
                for:
                  value: v
                  in: [1, 2]
                  steps:
                    - out:
                        next: after
                    - break:
                        assign:
                          - x: 1
            - after:
                next: nowhere
            - stray:
                next: continue
        """)
        msgs = " | ".join(f.message for f in fs if f.rule == "jump")
        self.assertIn("into or out of a for loop", msgs)
        self.assertIn("a step named 'break' inside a for loop", msgs)
        self.assertIn("'nowhere' names no step", msgs)
        self.assertIn("continue outside a for loop", msgs)

    def test_functions(self):
        cases = {
            'http.get("https://x")': "blocking-in-expression",
            'sys.log("x")': "blocking-in-expression",
            'text.substring("a", 1)': "arity",
            'len()': "arity",
            'map.get(m)': "arity",
            'bool(x)': "unknown-function",
            'map.keys(m)': "unknown-function",
            'nosuch(1)': "unknown-function",
            'text.nope': "unknown-stdlib-member",
            'not x in lst': "not-precedence",
            'not x == y': "not-precedence",
            'undefined_thing': "undefined-symbol",
        }
        for e, rule in cases.items():
            with self.subTest(e=e):
                self.assertEqual(expr_rules("${" + e + "}"), [rule])
        for e in ['if(x, 1, 2)', 'sys.get_env("A", "b")', 'json.encode_to_string(x, 2)', 'time.format(1)',
                  'not x', 'not(x in lst)', 'retry.always', 'http.default_retry_predicate']:
            with self.subTest(e=e):
                self.assertEqual(expr_rules("${" + e + "}"), [])

    def test_subworkflow_in_expression(self):
        fs = check("""
        main:
          steps:
            - a:
                assign:
                  - ok: ${pure(1)}
                  - few: ${pure()}
                  - blocking: ${talks(1)}
        pure:
          params: [x, y: 2]
          steps:
            - r:
                return: ${x + y}
        talks:
          params: [x]
          steps:
            - l:
                call: log
                args: {text: "${string(x)}"}
        log:
          params: [text]
          steps:
            - w:
                call: sys.log
                args: {text: "${text}"}
        """)
        errs = sorted((f.rule, f.text) for f in fs if f.severity == "error")
        self.assertEqual(errs, [("arity", "${pure()}"), ("blocking-in-expression", "${talks(1)}")])

    def test_call_args(self):
        fs = check("""
        main:
          steps:
            - a:
                call: sub
                args: {x: 1, nope: 2}
            - b:
                call: sub
                args: {y: 1}
            - c:
                call: http.get
                args: {uri: "x"}
            - d:
                call: no.such.thing
            - e:
                call: googleapis.secretmanager.v1.projects.secrets.versions.accessString
                args: {secretId: "x"}
        sub:
          params: [x, y: 1]
          steps:
            - r:
                return: ${x}
        """)
        msgs = [f.message for f in fs if f.severity == "error"]
        self.assertTrue(any("sub has no params ['nope']" in m for m in msgs), msgs)
        self.assertTrue(any("sub needs ['x']" in m for m in msgs), msgs)
        self.assertTrue(any("http.get has no params ['uri']" in m for m in msgs), msgs)
        self.assertTrue(any("http.get needs ['url']" in m for m in msgs), msgs)
        self.assertTrue(any("'no.such.thing'" in m for m in msgs), msgs)
        self.assertTrue(any("accessString needs ['secret_id']" in m for m in msgs), msgs)

    def test_limits(self):
        long = "${" + " + ".join(['"aaaaaaaaaa"'] * 30) + "}"
        self.assertGreater(len(long) - 3, L.EXPRESSION_LIMIT)
        self.assertEqual(expr_rules(long), ["expression-length"])
        many = "".join(f"\n          - v{i}: {i}" for i in range(51))
        fs = L.check_text(f"main:\n  steps:\n    - a:\n        assign:{many}\n    - a:\n        return: 1\n")
        self.assertEqual(rules(fs), ["duplicate-step", "limit"])
        big = {"main": {"steps": [{f"s{i}": {"assign": [{"x": "y" * 1000}]}} for i in range(140)]}}
        self.assertIn("source-size", rules(L.check_program(big)))

    def test_param_default_literal(self):  # 2026-10-13 validation deploy: "default value for parameter 'query' must be a literal"
        fs = check("""
        main:
          steps:
            - go:
                call: sub
                args:
                  url: u
                result: r
            - done:
                return: ${r}
        sub:
          params: [url, query: {}, skip: ["heartbeat"], a: null, b: 1, c: "s", d: true]
          steps:
            - done:
                return: ${url}
        """)
        lit = [f for f in fs if f.rule == "param-default-literal"]
        self.assertEqual(sorted(f.message.split("'")[1] for f in lit), ["query", "skip"])
        self.assertTrue(all(f.severity == "error" for f in lit))

    def test_reserved_variable_env(self):  # 2026-10-13 deploy: "cannot assign to a reserved variable 'env'"
        fs = check("""
        main:
          params: [args]
          steps:
            - prep:
                call: sys.get_env
                args:
                  name: PIPELINE_CONFIG
                result: env
            - set:
                assign:
                  - env: {}
                  - env.job: 1
                  - env["k"]: 2
            - loop:
                for:
                  value: env
                  index: env
                  in: [1]
                  steps:
                    - noop:
                        assign:
                          - z: 1
            - guard:
                try:
                  call: sys.log
                  args:
                    text: hi
                except:
                  as: env
                  steps:
                    - re:
                        raise: ${env}
            - out:
                return: ${env}
        """)
        errs = [f for f in fs if f.rule == "reserved-variable"]
        self.assertEqual(len(errs), 7)  # result, 3 assign targets, for value + index, except as
        self.assertEqual(rules(fs), ["reserved-variable"])
        # reading a parameter named env is what Google accepted (drift_job, run_job, ...)
        ok = check("""
        main:
          params: [env]
          steps:
            - out:
                return: ${env.x}
        """)
        self.assertEqual(rules(ok), [])

    def test_warnings(self):
        fs = check("""
        main:
          steps:
            - a:
                assign:
                  - name: "x"
                  - s: hello-${name}
                  - r: ${text.match_regex(name, "\\d")}
        """)
        self.assertEqual(rules(fs), [])
        self.assertEqual(rules(fs, "warning"), ["embedded-expression", "string-escape"])

    def test_line_mapping_across_joined_files(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "wf.yaml", Path(d) / "lib.yaml"
            a.write_text("main:\n  steps:\n    - go:\n        call: helper\n")
            b.write_text("# lib\nhelper:\n  steps:\n    - bad:\n        return: ${default(x, {})}\n")
            fs = L.check_definition("wf", [a, b])
        errs = [f for f in fs if f.severity == "error"]
        self.assertEqual([(f.file, f.line, f.subworkflow, f.step, f.rule) for f in errs],
                         [("lib.yaml", 5, "helper", "bad", "map-literal"), ("lib.yaml", 5, "helper", "bad",
                                                                           "undefined-symbol")])

    def test_compiled_json_is_terraforms(self):
        tf = shutil.which("terraform")
        if not tf:
            self.skipTest("no terraform")
        wf = HERE.parent / "workflows"
        with tempfile.TemporaryDirectory() as d:
            Path(d, "main.tf").write_text(
                'locals {\n  src = jsonencode(yamldecode(join("\\n", [file("%s"), file("%s")])))\n}\n'
                % (wf / "drift.yaml", wf / "lib.yaml"))
            out = subprocess.run([tf, "console", "-no-color"], input="jsonencode(local.src)\n", cwd=d,
                                 capture_output=True, text=True, timeout=300)
        if out.returncode != 0:
            self.skipTest(f"terraform console: {out.stderr[:200]}")
        want = json.loads(json.loads(out.stdout.strip()))
        self.assertEqual(L.compiled_json(L.load_source("drift").program), want)

    def test_real_definitions_load(self):
        for name in L.DEFINITIONS:
            src = L.load_source(name)
            self.assertIn("main", src.program)
            L.Checker(src).run()


class Simulator(unittest.TestCase):
    def interp(self, program, **kw):
        return wfsim.Interpreter(program, http=None, secrets=None, env={}, clock=wfsim.VirtualClock(), **kw)

    def test_refuses_what_google_refuses(self):
        prog = {"main": {"params": ["args"], "steps": [{"init": {"assign": [{"a": "${default(args, {})}"}]}}]}}
        with self.assertRaises(wfsim.SimError) as cm:
            self.interp(prog)
        self.assertIn("map-literal", str(cm.exception))
        # even unchecked, evaluating the expression uses the same parser
        with self.assertRaises(wfsim.SimError):
            self.interp(prog, strict=False).run({})

    def test_same_semantics_as_googles_parser(self):
        prog = {"main": {"steps": [{"r": {"return": "${[not(1 == 2), 'a' + \"b\", TRUE, 7 // 2, -2 * 3]}"}}]}}
        self.assertEqual(self.interp(prog).run(None), [True, "ab", True, 3, -6])
        bad = {"main": {"steps": [{"r": {"return": "${not 1 == 2}"}}]}}
        with self.assertRaises(wfsim.WfError):  # (not 1) == 2: not on an integer
            self.interp(bad, strict=False).run(None)

    def test_from_files_names_the_file_and_line(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "wf.yaml", Path(d) / "lib.yaml"
            a.write_text("main:\n  steps:\n    - go:\n        return: ${helper()}\n")
            b.write_text("helper:\n  steps:\n    - bad:\n        return: ${[{}]}\n")
            with self.assertRaises(wfsim.SimError) as cm:
                wfsim.Interpreter.from_files(str(a), str(b), http=None, secrets=None, env={},
                                             clock=wfsim.VirtualClock())
        self.assertIn("lib.yaml:4", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
