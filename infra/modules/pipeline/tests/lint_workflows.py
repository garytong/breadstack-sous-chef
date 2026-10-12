#!/usr/bin/env python3
"""Offline lint of the pipeline's Cloud Workflows definitions.

There is no offline `gcloud workflows deploy --dry-run`, so this checks what
a deploy would reject, and the mistakes a deploy would not catch, on the
definitions exactly as workflows.tf assembles them (<name>.yaml + lib.yaml):

  - YAML parses; every subworkflow has `steps`; each step is a one-key map
    whose body is a known step kind
  - step names are unique within a subworkflow
  - every `call:` names a subworkflow of the same definition or a standard
    library / connector function
  - subworkflow calls pass only declared params and every param without a
    default
  - a switch step with its own `next` has no entry that falls through (no
    next / return / raise): what Workflows does then is not documented
  - every `next:` names a step of the same subworkflow, or end / break /
    continue, and none jumps out of a `for` loop
  - every function inside ${...} is a standard-library function: Workflows
    has no user functions in expressions, so a subworkflow there is a bug
  - every variable an expression reads is defined somewhere in the
    subworkflow (a param, an assign target, a `result`, a `for` value or
    index, an `except` name)
  - no unquoted YAML 1.1 boolean word (y, n, yes, no, on, off) as a key or
    value: Terraform's yamldecode, which makes the deployed JSON, is YAML 1.1
  - with --schema <workflows.json> (GoogleCloudPlatform/workflows-samples,
    jsonschema/workflows.json): the JSON-schema validation the samples' CI
    runs

  python3 infra/modules/pipeline/tests/lint_workflows.py [--schema FILE]

Exit 1 on any finding.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent.parent / "workflows"
DEFINITIONS = {"drift": ["drift.yaml", "lib.yaml"], "retrain": ["retrain.yaml", "lib.yaml"]}

STEP_KINDS = {"assign", "call", "switch", "for", "parallel", "raise", "return", "next", "steps", "try"}
STEP_KEYS = STEP_KINDS | {"args", "result", "retry", "except"}
# Standard library and connector call targets the definitions use.
CALL_PREFIXES = ("http.", "sys.", "googleapis.", "events.", "experimental.")
# Functions allowed inside expressions (Workflows standard library).
FUNCTIONS = {
    "default", "keys", "len", "string", "int", "double", "bool", "get_type", "not",
    "map.get", "map.merge", "map.merge_nested", "map.delete",
    "list.concat", "list.prepend",
    "json.encode", "json.encode_to_string", "json.decode",
    "text.substring", "text.replace_all", "text.replace_all_regex", "text.match_regex", "text.find_all",
    "text.find_all_regex", "text.split", "text.to_lower", "text.to_upper", "text.url_encode", "text.url_encode_plus",
    "text.url_decode", "text.encode", "text.decode",
    "time.format", "time.parse", "sys.now", "sys.get_env",
    "base64.encode", "base64.decode", "math.floor", "math.abs", "math.max", "math.min",
    "hash.compute_checksum", "uuid.generate",
}
KEYWORDS = {"and", "or", "not", "in", "true", "false", "null", "True", "False", "None"}
EXPR = re.compile(r"\$\{(.*)\}\s*$", re.S)


def load(name: str) -> dict:
    text = "\n".join((HERE / f).read_text() for f in DEFINITIONS[name])
    return yaml.safe_load(text)


def strip_strings(expr: str) -> str:
    """The expression with string literals blanked out."""
    out, i, n = [], 0, len(expr)
    while i < n:
        c = expr[i]
        if c == '"':
            j = i + 1
            while j < n and expr[j] != '"':
                j += 2 if expr[j] == "\\" else 1
            out.append('""')
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def expressions(node, path=""):
    """(path, expression text) for every ${...} string under node."""
    if isinstance(node, str):
        m = EXPR.match(node.strip())
        if m:
            yield path, m.group(1)
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from expressions(v, f"{path}.{k}")
            if isinstance(k, str) and EXPR.match(k.strip()):
                yield f"{path}.<key>", EXPR.match(k.strip()).group(1)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from expressions(v, f"{path}[{i}]")


def walk_steps(steps, visit, path):
    """visit(name, body, path) for every step, nested ones included."""
    if not isinstance(steps, list):
        visit(None, steps, path + " (steps is not a list)")
        return
    for i, item in enumerate(steps):
        if not isinstance(item, dict) or len(item) != 1:
            visit(None, item, f"{path}[{i}] (a step must be a one-key map)")
            continue
        (name, body), = item.items()
        p = f"{path}/{name}"
        visit(name, body, p)
        if not isinstance(body, dict):
            continue
        if "steps" in body:
            walk_steps(body["steps"], visit, p)
        if "for" in body and isinstance(body["for"], dict):
            walk_steps(body["for"].get("steps"), visit, p + "/for")
        if "switch" in body:
            for j, cond in enumerate(body["switch"] or []):
                if isinstance(cond, dict) and "steps" in cond:
                    walk_steps(cond["steps"], visit, f"{p}/switch[{j}]")
        if "try" in body:
            t = body["try"]
            if isinstance(t, dict) and "steps" in t:
                walk_steps(t["steps"], visit, p + "/try")
        if "except" in body and isinstance(body["except"], dict) and "steps" in body["except"]:
            walk_steps(body["except"]["steps"], visit, p + "/except")
        if "parallel" in body:
            par = body["parallel"]
            for bi, br in enumerate(par.get("branches", []) or []):
                (bn, bb), = br.items()
                walk_steps(bb.get("steps"), visit, f"{p}/branch:{bn}")
            if "for" in par:
                walk_steps(par["for"].get("steps"), visit, p + "/pfor")


def loop_jumps(steps, where: str, errors: list[str]) -> None:
    """A `next` inside a for loop may only name a step of that loop's body
    (or break / continue / end): Workflows does not jump out of a loop."""

    def names_and_nexts(body_steps):
        names, nexts = set(), []

        def visit(name, body, path):
            if name is not None:
                names.add(name)
            if isinstance(body, dict):
                if "next" in body:
                    nexts.append((body["next"], path))
                for cond in body.get("switch", []) or []:
                    if isinstance(cond, dict) and "next" in cond:
                        nexts.append((cond["next"], path))

        walk_steps(body_steps, visit, "")
        return names, nexts

    def visit_outer(name, body, path):
        if isinstance(body, dict) and isinstance(body.get("for"), dict):
            names, nexts = names_and_nexts(body["for"].get("steps"))
            for target, p in nexts:
                if target not in ("break", "continue", "end") and target not in names:
                    errors.append(f"{where}{path}/for{p}: next -> {target!r} leaves the for loop")

    walk_steps(steps, visit_outer, "")


def params_of(sub: dict) -> tuple[list[str], set[str]]:
    """(all param names, required param names)."""
    names, required = [], set()
    for p in sub.get("params", []) or []:
        if isinstance(p, dict):
            (k, _), = p.items()
            names.append(k)
        else:
            names.append(p)
            required.add(p)
    return names, required


def defined_names(sub: dict) -> set[str]:
    names = set(params_of(sub)[0])

    def visit(name, body, path):
        if not isinstance(body, dict):
            return
        for a in body.get("assign", []) or []:
            if isinstance(a, dict):
                for k in a:
                    names.add(re.split(r"[.\[]", str(k), maxsplit=1)[0])
        if "result" in body:
            names.add(body["result"])
        if isinstance(body.get("for"), dict):
            names.add(body["for"].get("value"))
            if body["for"].get("index"):
                names.add(body["for"]["index"])
        for cond in body.get("switch", []) or []:
            if isinstance(cond, dict):
                for a in cond.get("assign", []) or []:
                    if isinstance(a, dict):
                        for k in a:
                            names.add(re.split(r"[.\[]", str(k), maxsplit=1)[0])
        t = body.get("try")
        if isinstance(t, dict) and "result" in t:
            names.add(t["result"])
        e = body.get("except")
        if isinstance(e, dict) and e.get("as"):
            names.add(e["as"])

    walk_steps(sub.get("steps"), visit, "")
    return names


def lint(name: str, doc: dict) -> list[str]:
    errors: list[str] = []
    subs = doc
    if "main" not in subs:
        errors.append(f"{name}: no main")
    for sname, sub in subs.items():
        where = f"{name}:{sname}"
        if not isinstance(sub, dict) or "steps" not in sub:
            errors.append(f"{where}: a subworkflow needs `steps`")
            continue
        seen: dict[str, str] = {}
        calls: list[tuple[str, dict, str]] = []
        nexts: list[tuple[str, str]] = []

        def visit(step, body, path):
            if step is None:
                errors.append(f"{where}{path}")
                return
            if step in seen:
                errors.append(f"{where}: duplicate step name {step!r} ({seen[step]} and {path})")
            seen[step] = path
            if not isinstance(body, dict):
                errors.append(f"{where}{path}: step body is not a map")
                return
            unknown = set(body) - STEP_KEYS
            if unknown:
                errors.append(f"{where}{path}: unknown step keys {sorted(unknown)}")
            if not (set(body) & STEP_KINDS):
                errors.append(f"{where}{path}: no step kind ({sorted(STEP_KINDS)})")
            if "call" in body:
                calls.append((body["call"], body.get("args") or {}, path))
            t = body.get("try")
            if isinstance(t, dict) and "call" in t:
                calls.append((t["call"], t.get("args") or {}, path + "/try"))
            if "next" in body:
                nexts.append((body["next"], path))
            for cond in body.get("switch", []) or []:
                if not isinstance(cond, dict) or "condition" not in cond:
                    errors.append(f"{where}{path}: a switch entry needs a condition")
                    continue
                extra = set(cond) - {"condition", "next", "steps", "assign", "return", "raise", "call", "args", "result"}
                if extra:
                    errors.append(f"{where}{path}: switch entry keys {sorted(extra)}")
                if "next" in cond:
                    nexts.append((cond["next"], path))
                if "call" in cond:
                    calls.append((cond["call"], cond.get("args") or {}, path + "/switch"))
                kinds = set(cond) & {"next", "steps", "assign", "return", "raise", "call"}
                if len(kinds - {"next"}) > 1:
                    errors.append(f"{where}{path}: switch entry mixes {sorted(kinds)}")
                if "steps" in cond and "next" in cond:
                    errors.append(f"{where}{path}: switch entry with both steps and next (ambiguous)")
                if "next" in body and not (set(cond) & {"next", "return", "raise"}):
                    # undocumented: does a matched entry without next go to the step's next, or on to the
                    # following step? (the dry end-to-end run found one that read it both ways)
                    errors.append(f"{where}{path}: switch entry without next/return/raise in a step that has a next (ambiguous)")

        walk_steps(sub["steps"], visit, "")
        loop_jumps(sub["steps"], where, errors)

        for target, args, path in calls:
            if not isinstance(target, str):
                errors.append(f"{where}{path}: call target {target!r}")
            elif target.startswith(CALL_PREFIXES):
                pass
            elif target not in subs:
                errors.append(f"{where}{path}: call to unknown subworkflow {target!r}")
            else:
                names, required = params_of(subs[target])
                given = set(args)
                extra = given - set(names)
                missing = required - given
                if extra:
                    errors.append(f"{where}{path}: {target} has no params {sorted(extra)}")
                if missing:
                    errors.append(f"{where}{path}: {target} needs {sorted(missing)}")
        for target, path in nexts:
            if target in ("end", "break", "continue"):
                continue
            if target not in seen:
                errors.append(f"{where}{path}: next -> unknown step {target!r}")

        defined = defined_names(sub)
        for path, expr in expressions(sub.get("steps"), ""):
            bare = strip_strings(expr)
            for m in re.finditer(r"(?<![\w.\"])([A-Za-z_][\w.]*)\s*(\()?", bare):
                ident, call = m.group(1), m.group(2)
                if call and ident in KEYWORDS:
                    continue
                if call:
                    if ident not in FUNCTIONS:
                        errors.append(f"{where}{path}: function {ident}() is not a standard-library function: {expr[:120]}")
                    continue
                root = ident.split(".")[0]
                if root in KEYWORDS or root in ("http", "sys", "time", "text", "json", "map", "list", "math", "base64"):
                    continue
                if root not in defined:
                    errors.append(f"{where}{path}: {root!r} is never defined in {sname}: {expr[:120]}")
    return errors


# Words YAML 1.1 reads as booleans when unquoted. Terraform's yamldecode
# (which turns these files into the deployed JSON) is YAML 1.1: a step,
# variable or value spelled like this silently becomes true / false.
YAML11_BOOL = re.compile(r"^\s*(?:-\s+)?(?:(y|Y|n|N|yes|Yes|YES|no|No|NO|on|On|ON|off|Off|OFF)\s*:|[^#'\"]*?:\s+(y|Y|n|N|yes|Yes|YES|no|No|NO|on|On|ON|off|Off|OFF)\s*(?:#.*)?$)")


def yaml11_booleans() -> list[str]:
    found = []
    for f in sorted({f for fs in DEFINITIONS.values() for f in fs}):
        for i, line in enumerate((HERE / f).read_text().splitlines(), 1):
            if not line.lstrip().startswith("#") and YAML11_BOOL.match(line):
                found.append(f"{f}:{i}: a YAML 1.1 boolean word (Terraform's yamldecode makes it true/false): {line.strip()}")
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", type=Path, help="workflows-samples jsonschema/workflows.json")
    ns = ap.parse_args()
    errors: list[str] = yaml11_booleans()
    for name in DEFINITIONS:
        try:
            doc = load(name)
        except yaml.YAMLError as e:
            errors.append(f"{name}: YAML: {e}")
            continue
        errors += lint(name, doc)
        if ns.schema:
            import jsonschema
            schema = json.loads(ns.schema.read_text())
            # The schema's first oneOf branch is the subworkflow form; its
            # leaf errors say what is wrong (the oneOf error itself does not).
            form = dict(schema["oneOf"][0], definitions=schema["definitions"])
            v = jsonschema.Draft7Validator(form)

            def leaves(err):
                if err.context:
                    for c in err.context:
                        yield from leaves(c)
                else:
                    yield err

            found = set()
            for err in v.iter_errors(doc):
                for leaf in leaves(err):
                    found.add(f"{name}: schema: {'/'.join(map(str, leaf.absolute_path))}: {leaf.message[:300]}")
            errors += sorted(found)
    for e in errors:
        print(e)
    print(f"{'FAIL' if errors else 'ok'}: {len(errors)} finding(s) in {', '.join(DEFINITIONS)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
