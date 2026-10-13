#!/usr/bin/env python3
"""Strict offline checker for the pipeline's Cloud Workflows definitions.

There is no offline `gcloud workflows deploy --dry-run`. This checks the
definitions exactly as workflows.tf deploys them (yamldecode of
<name>.yaml + lib.yaml, then every string value) against the Workflows syntax
as Google documents it (the doc page of each rule is in RULES):

  expressions   every ${...} string (values and dynamic map keys) is tokenized
                and parsed by a recursive-descent parser for the documented
                grammar (see ExprParser): literals (int, float, "..." / '...'
                strings, true/True/TRUE, false/False/FALSE, null), names,
                .field, [index], calls of dotted names, list literals, and the
                operators by the documented precedence (() . [] not unary-
                * / // % + - comparisons/in and or). There are NO map literals
                inside ${} (the 2026-10-13 deploy error), no ternary, no
                lambdas, no assignment. Assignment targets (`a.b[i]: v`) are
                parsed with the target grammar.
  names         every identifier is a variable in scope or a subworkflow name
                (Google rejects anything else at deploy: "symbol 'x' is
                neither a variable nor a sub-workflow name"); for bodies and
                except blocks are child scopes (a variable first set there is
                gone outside)
  functions     every function inside an expression is a non-blocking
                standard-library function with its documented arity, or a
                subworkflow (positional args, no blocking steps); every
                `call:` target is a standard-library function, a connector or
                a subworkflow, with named args the callee accepts and every
                required one given
  structure     step shapes (assign / call / switch / for / try / retry /
                except / raise / return / steps / parallel / next), unique
                step names, jumps (none into or out of a for loop, no step
                named break/continue in a loop), the known issue of `for`
                directly under `try`, YAML 1.1 boolean words, duplicate keys
  limits        compiled JSON size (128 KB), expression length (400 chars),
                50 assignments per step, 50 conditions per switch, 10 branches
                per parallel step, parallel depth 2, static call depth 20

  python3 infra/modules/pipeline/tests/lint_workflows.py [--json OUT] [--warnings] [--schema FILE]

Exit 1 on any error-severity finding (warnings print with --warnings and never
fail the run). wfsim.py parses with this module's parser and refuses a
definition with error findings, so the dry end-to-end fails the same way.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml

HERE = Path(__file__).resolve().parent.parent / "workflows"
DEFINITIONS = {"drift": ["drift.yaml", "lib.yaml"], "retrain": ["retrain.yaml", "lib.yaml"]}

# Documented limits (https://cloud.google.com/workflows/quotas).
SOURCE_LIMIT = 128 * 1024          # "Source code size: 128 KB"
EXPRESSION_LIMIT = 400             # "Expression length: 400 characters"
ASSIGN_LIMIT = 50                  # "Assignments per step"
SWITCH_LIMIT = 50                  # "Conditions per switch"
BRANCH_LIMIT = 10                  # "Branches per step"
PARALLEL_DEPTH = 2                 # "Parallel depth"
CALL_DEPTH = 20                    # "Maximum call stack depth"

DOC = "https://cloud.google.com/workflows/docs/"
RULES = {
    # rule: (severity, doc)
    "syntax": ("error", DOC + "reference/syntax/expressions"),
    "map-literal": ("error", DOC + "reference/syntax/expressions (and the 2026-10-13 deploy error)"),
    "string-escape": ("warning", DOC + "reference/syntax/expressions"),
    "embedded-expression": ("warning", DOC + "troubleshooting#resource_name_must_conform_to_rfc_1123"),
    "expression-length": ("error", DOC + "quotas#resource_limits"),
    "expression-not-allowed": ("error", DOC + "reference/syntax/expressions"),
    "undefined-symbol": ("error", DOC + "troubleshooting#deployment_errors"),
    "out-of-scope": ("error", DOC + "reference/syntax/variables#variable-scope"),
    "unknown-function": ("error", DOC + "reference/stdlib/overview"),
    "unknown-stdlib-member": ("error", DOC + "reference/stdlib/overview"),
    "arity": ("error", DOC + "reference/stdlib/overview"),
    "blocking-in-expression": ("error", DOC + "reference/syntax/expressions#blocking-calls"),
    "connector-in-expression": ("warning", DOC + "reference/googleapis/secretmanager/Overview"),
    "not-precedence": ("error", DOC + "reference/syntax/datatypes#order-of-operations"),
    "call-target": ("error", DOC + "reference/syntax/calls"),
    "unverified-connector": ("warning", DOC + "reference/googleapis"),
    "call-args": ("error", DOC + "reference/syntax/subworkflows"),
    "step-shape": ("error", DOC + "reference/syntax"),
    "duplicate-step": ("error", DOC + "reference/syntax/steps#step-names"),
    "jump": ("error", DOC + "reference/syntax/iteration#jump-within-a-loop"),
    "switch-fallthrough": ("warning", DOC + "reference/syntax/conditions"),
    "try-for": ("error", DOC + "issues#placement_of_for_directly_after_try"),
    "variable-name": ("error", DOC + "reference/syntax/variables#variable-names"),
    "reserved-variable": ("error", "undocumented: the 2026-10-13 deploy error \"cannot assign to a reserved variable 'env'\""),
    "namespace-shadow": ("warning", DOC + "reference/syntax/lists#add-a-new-element-to-the-end-of-a-list"),
    "param-default-expression": ("error", DOC + "reference/syntax/subworkflows#default-parameters"),
    "param-default-literal": ("error", "the 2026-10-13 validation deploy error \"default value for parameter 'query' must be a literal\""),
    "limit": ("error", DOC + "quotas"),
    "source-size": ("error", DOC + "quotas#resource_limits"),
    "call-depth": ("error", DOC + "quotas#step_limits"),
    "yaml": ("error", "YAML (Terraform yamldecode)"),
    "yaml11-bool": ("error", "YAML 1.1 (Terraform yamldecode)"),
    "schema": ("error", "GoogleCloudPlatform/workflows-samples jsonschema"),
}

# ── the standard library (reference/stdlib/overview, reference/stdlib/expression-helpers, one page per function) ──
# name -> ordered params; a trailing "?" marks an optional one. Positional order is the documented order.
STDLIB: dict[str, list[str]] = {
    "default": ["val", "defaultVal"],
    "if": ["condition", "ifTrue", "ifFalse"],
    "keys": ["map"],
    "len": ["value"],
    "get_type": ["arg"],
    "int": ["value"],
    "double": ["value"],
    "string": ["value"],
    "base64.decode": ["data", "padding?"],
    "base64.encode": ["data", "padding?"],
    "events.create_callback_endpoint": ["http_callback_method?"],
    "events.await_callback": ["callback", "timeout?"],
    "experimental.executions.map": ["workflow_id", "arguments", "timeout?", "location?", "project_id?"],
    "experimental.executions.run": ["workflow_id", "location?", "project_id?", "argument?"],
    "hash.compute_checksum": ["data", "algorithm"],
    "hash.compute_hmac": ["key", "data", "algorithm"],
    "http.get": ["url", "timeout?", "headers?", "query?", "auth?", "private_service_name?"],
    "http.delete": ["url", "timeout?", "body?", "headers?", "query?", "auth?", "private_service_name?"],
    "http.post": ["url", "timeout?", "body?", "headers?", "query?", "auth?", "private_service_name?"],
    "http.put": ["url", "timeout?", "body?", "headers?", "query?", "auth?", "private_service_name?"],
    "http.patch": ["url", "timeout?", "body?", "headers?", "query?", "auth?", "private_service_name?"],
    "http.request": ["method", "url", "timeout?", "body?", "headers?", "query?", "auth?", "private_service_name?"],
    "http.default_retry_predicate": ["exception"],
    "http.default_retry_predicate_non_idempotent": ["exception"],
    "json.decode": ["data"],
    "json.encode": ["data", "indent?"],
    "json.encode_to_string": ["data", "indent?"],
    "list.concat": ["objs", "val"],
    "list.prepend": ["objs", "val"],
    "map.delete": ["map", "key"],
    "map.get": ["map", "keys"],
    "map.merge": ["first", "second"],
    "map.merge_nested": ["first", "second"],
    "math.abs": ["x"],
    "math.floor": ["x"],
    "math.max": ["x", "y"],
    "math.min": ["x", "y"],
    "retry.always": ["exception"],
    "retry.never": ["exception"],
    "sys.get_env": ["name", "default?"],
    "sys.now": [],
    "sys.log": ["data?", "severity?", "text?", "json?", "timeout?"],
    "sys.sleep": ["seconds"],
    "sys.sleep_until": ["time"],
    "text.decode": ["data", "charset?"],
    "text.encode": ["text", "charset?"],
    "text.find_all": ["source", "substr"],
    "text.find_all_regex": ["source", "regexp"],
    "text.match_regex": ["source", "regexp"],
    "text.replace_all": ["source", "substr", "repl"],
    "text.replace_all_regex": ["source", "regexp", "repl"],
    "text.split": ["source", "separator"],
    "text.substring": ["source", "start", "end"],
    "text.to_lower": ["source"],
    "text.to_upper": ["source"],
    "text.url_decode": ["source"],
    "text.url_encode": ["source"],
    "text.url_encode_plus": ["source"],
    "time.format": ["seconds", "timezone?"],
    "time.parse": ["value"],
    "uuid.generate": [],
}
# Blocking functions: call steps only (syntax/expressions#blocking-calls; sleep_until by the same token).
BLOCKING = {"events.await_callback", "experimental.executions.map", "experimental.executions.run",
            "http.get", "http.delete", "http.post", "http.put", "http.patch", "http.request",
            "sys.log", "sys.sleep", "sys.sleep_until"}
# Documented objects, and functions usable as values (retry predicates).
STD_OBJECTS = {"http.default_retry", "http.default_retry_non_idempotent", "retry.default_backoff",
               "http.default_retry_predicate", "http.default_retry_predicate_non_idempotent",
               "retry.always", "retry.never"}
STD_MODULES = {"base64", "events", "experimental", "hash", "http", "json", "list", "map", "math", "retry", "sys",
               "text", "time", "uuid", "googleapis"}
# Connectors the definitions call, with their documented args (reference/googleapis/secretmanager/...).
CONNECTORS: dict[str, list[str]] = {
    "googleapis.secretmanager.v1.projects.secrets.versions.accessString": ["secret_id", "version?", "project_id?"],
}
CONNECTOR_COMMON = {"connector_params"}  # reference/googleapis "Invoke a connector call"


def arity(name: str) -> tuple[int, int]:
    ps = STDLIB[name]
    return sum(1 for p in ps if not p.endswith("?")), len(ps)


# ── source loading with line numbers ────────────────────────────────────────────────────────────────────────────────────


class PDict(dict):
    """A dict that remembers its own source line, each value's (`lines[key]`) and each key's (`klines[key]`)."""

    def __init__(self, *a: Any, **kw: Any):
        super().__init__(*a, **kw)
        self.line = 1
        self.lines: dict[Any, int] = {}
        self.klines: dict[Any, int] = {}


class PList(list):
    def __init__(self, *a: Any):
        super().__init__(*a)
        self.line = 1
        self.lines: list[int] = []


def _construct(loader: yaml.SafeLoader, node: yaml.Node, dups: list[tuple[int, str]]) -> Any:
    if isinstance(node, yaml.MappingNode):
        loader.flatten_mapping(node)
        d = PDict()
        d.line = node.start_mark.line + 1
        for kn, vn in node.value:
            k = _construct(loader, kn, dups)
            if isinstance(k, (dict, list)):
                k = json.dumps(k)
            if k in d:
                dups.append((kn.start_mark.line + 1, str(k)))
            d[k] = _construct(loader, vn, dups)
            d.lines[k] = vn.start_mark.line + 1
            d.klines[k] = kn.start_mark.line + 1
        return d
    if isinstance(node, yaml.SequenceNode):
        lst = PList()
        lst.line = node.start_mark.line + 1
        for vn in node.value:
            lst.append(_construct(loader, vn, dups))
            lst.lines.append(vn.start_mark.line + 1)
        return lst
    return loader.construct_object(node, deep=True)


def wrap(v: Any) -> Any:
    """A plain loaded value as PDict / PList (every line 1)."""
    if isinstance(v, dict):
        d = PDict()
        for k, x in v.items():
            d[k] = wrap(x)
            d.lines[k] = d.klines[k] = 1
        return d
    if isinstance(v, list):
        lst = PList(wrap(x) for x in v)
        lst.lines = [1] * len(lst)
        return lst
    return v


@dataclass
class Source:
    """One deployed definition: <name>.yaml + lib.yaml joined with "\\n", as workflows.tf does."""
    name: str
    files: list[tuple[str, int]]           # (file name, its first line in the joined text)
    text: str
    program: Any = None
    duplicates: list[tuple[int, str]] = field(default_factory=list)

    def where(self, line: int) -> tuple[str, int]:
        cur = self.files[0]
        for f in self.files:
            if line >= f[1]:
                cur = f
        return cur[0], line - cur[1] + 1


def load_source(name: str, paths: list[Path] | None = None) -> Source:
    paths = paths or [HERE / f for f in DEFINITIONS[name]]
    texts = [Path(p).read_text(encoding="utf-8") for p in paths]
    files, line = [], 1
    for p, t in zip(paths, texts):
        files.append((Path(p).name, line))
        line += t.count("\n") + 1          # join("\n", ...) puts one more newline between files
    return load_text(name, "\n".join(texts), files)


def load_text(name: str, text: str, files: list[tuple[str, int]] | None = None) -> Source:
    src = Source(name, files or [(f"{name}.yaml", 1)], text)
    loader = yaml.SafeLoader(text)
    try:
        node = loader.get_single_node()
        src.program = _construct(loader, node, src.duplicates) if node is not None else None
    finally:
        loader.dispose()
    return src


def _plain(v: Any) -> Any:
    if isinstance(v, dict):
        return {(k if isinstance(k, str) else json.dumps(k)): _plain(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_plain(x) for x in v]
    return v


def compiled_json(program: Any) -> str:
    """What Terraform's jsonencode makes of it: compact JSON, keys sorted, <, >, &, U+2028/9 escaped."""
    s = json.dumps(_plain(program), separators=(",", ":"), sort_keys=True, ensure_ascii=False)
    for a, b in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026"), (" ", "\\u2028"), (" ", "\\u2029")):
        s = s.replace(a, b)
    return s


# ── the expression language: tokenizer ──────────────────────────────────────────────────────────────────────────────────


class ExprSyntaxError(Exception):
    def __init__(self, message: str, pos: int, rule: str = "syntax"):
        super().__init__(message)
        self.pos = pos
        self.rule = rule


@dataclass
class Tok:
    kind: str          # NUM_INT NUM_FLOAT STRING TRUE FALSE NULL AND OR NOT IN IDENT OP BAD EOF
    value: Any
    pos: int
    text: str


KEYWORDS = {
    "true": "TRUE", "True": "TRUE", "TRUE": "TRUE",
    "false": "FALSE", "False": "FALSE", "FALSE": "FALSE",
    "null": "NULL",
    "and": "AND", "AND": "AND", "or": "OR", "OR": "OR", "not": "NOT", "NOT": "NOT",
    "in": "IN",
}
_OPS2 = ("//", "==", "!=", "<=", ">=")
_OPS1 = "+-*/%<>()[].,"
_ESC = {'"': '"', "'": "'", "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
_DIGITS = "0123456789"


def _ident_char(c: str) -> bool:
    return c.isascii() and (c.isalnum() or c == "_")


def tokenize(src: str, warnings: list[tuple[int, str]] | None = None) -> list[Tok]:
    out: list[Tok] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c in " \t\r\n":
            i += 1
            continue
        start = i
        if c in _DIGITS:
            while i < n and src[i] in _DIGITS:
                i += 1
            kind = "NUM_INT"
            if i + 1 < n and src[i] == "." and src[i + 1] in _DIGITS:
                i += 1
                while i < n and src[i] in _DIGITS:
                    i += 1
                kind = "NUM_FLOAT"
            if i < n and _ident_char(src[i]):
                raise ExprSyntaxError(f"malformed number {src[start:i + 1]!r} (exponents and suffixes are not "
                                      "documented)", start)
            text = src[start:i]
            out.append(Tok(kind, int(text) if kind == "NUM_INT" else float(text), start, text))
            continue
        if c in "\"'":
            q, i, buf = c, i + 1, []
            while True:
                if i >= n:
                    raise ExprSyntaxError("unterminated string literal", start)
                ch = src[i]
                if ch == q:
                    i += 1
                    break
                if ch == "\\":
                    if i + 1 >= n:
                        raise ExprSyntaxError("unterminated string literal", start)
                    e = src[i + 1]
                    if e in _ESC:
                        buf.append(_ESC[e])
                        i += 2
                    elif e == "u" and re.fullmatch(r"[0-9a-fA-F]{4}", src[i + 2:i + 6]):
                        buf.append(chr(int(src[i + 2:i + 6], 16)))
                        i += 6
                    else:
                        if warnings is not None:
                            warnings.append((i, f"escape \\{e} in a string literal is not a documented escape"))
                        buf.append("\\" + e)
                        i += 2
                    continue
                buf.append(ch)
                i += 1
            out.append(Tok("STRING", "".join(buf), start, src[start:i]))
            continue
        if c.isascii() and (c.isalpha() or c == "_"):
            while i < n and _ident_char(src[i]):
                i += 1
            text = src[start:i]
            out.append(Tok(KEYWORDS.get(text, "IDENT"), text, start, text))
            continue
        two = src[i:i + 2]
        if two in _OPS2:
            out.append(Tok("OP", two, start, two))
            i += 2
            continue
        if c in _OPS1:
            out.append(Tok("OP", c, start, c))
            i += 1
            continue
        out.append(Tok("BAD", c, start, c))
        i += 1
    out.append(Tok("EOF", None, n, "<end>"))
    return out


# ── the expression language: parser ─────────────────────────────────────────────────────────────────────────────────────


@dataclass
class Node:
    """An AST node. kinds: lit(a=value) name(a) list(a=[nodes]) attr(a=base, b=field) index(a=base, b=key)
    call(a=callee node, b=[args]) and/or(a, b) not(a) neg(a) bin(a=op, b=left, c=right)."""
    kind: str
    a: Any = None
    b: Any = None
    c: Any = None
    pos: int = 0
    paren: bool = False


OPERAND_START = "expecting one of '[', not, '(', '-', true, false, null, a number, a string, an identifier"
CMP_OPS = ("<", "<=", ">", ">=", "==", "!=")


class ExprParser:
    """Recursive descent for the documented grammar (reference/syntax/expressions, reference/syntax/datatypes
    "Order of operations": () > . > [] > not > unary - > * / // % > + - > comparisons, in > and > or):

        expr     := or
        or       := and ( OR and )*
        and      := cmp ( AND cmp )*
        cmp      := add ( ('<'|'<='|'>'|'>='|'=='|'!='|IN) add )*
        add      := mul ( ('+'|'-') mul )*
        mul      := unary ( ('*'|'/'|'//'|'%') unary )*
        unary    := ('-' | NOT) unary | postfix          -- not binds tighter than any binary operator
        postfix  := primary ( '.' IDENT | '[' expr ']' | '(' [args] ')' )*   -- calls only on dotted names
        primary  := NUM_INT | NUM_FLOAT | STRING | TRUE | FALSE | NULL | IDENT | '(' expr ')' | '[' [args] ']'
        args     := expr ( ',' expr )*

    There is no '{': Workflows has no map literal inside ${} (the deployed parser's expected set after '(' is
    {'[', LOGICAL_NOT, '(', ')', '-', TRUE, FALSE, NULL, NUM_FLOAT, NUM_INT, STRING, IDENTIFIER}).
    """

    def __init__(self, src: str):
        self.src = src
        self.warnings: list[tuple[int, str]] = []
        self.toks = tokenize(src, self.warnings)
        self.i = 0

    @property
    def tok(self) -> Tok:
        return self.toks[self.i]

    def at(self, kind: str, val: Any = None) -> bool:
        t = self.tok
        return t.kind == kind and (val is None or t.value == val)

    def take(self) -> Tok:
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect(self, kind: str, val: Any, what: str) -> Tok:
        if not self.at(kind, val):
            self.fail(f"expecting {what}")
        return self.take()

    def fail(self, expecting: str) -> None:
        t = self.tok
        if t.kind == "BAD" and t.value == "{":
            raise ExprSyntaxError("extraneous '{': there are no map literals inside ${...}, not even {}", t.pos,
                                  "map-literal")
        if t.kind == "BAD" and t.value == "}":
            raise ExprSyntaxError("unexpected '}' (two ${...} in one string? Workflows has no interpolation)", t.pos)
        what = "end of expression" if t.kind == "EOF" else repr(t.text)
        raise ExprSyntaxError(f"unexpected {what}; {expecting}", t.pos)

    def parse(self) -> Node:
        if self.at("EOF"):
            self.fail(OPERAND_START)
        n = self.or_()
        if not self.at("EOF"):
            self.fail("expecting an operator or the end of the expression")
        return n

    def or_(self) -> Node:
        n = self.and_()
        while self.at("OR"):
            p = self.take().pos
            n = Node("or", n, self.and_(), pos=p)
        return n

    def and_(self) -> Node:
        n = self.cmp()
        while self.at("AND"):
            p = self.take().pos
            n = Node("and", n, self.cmp(), pos=p)
        return n

    def cmp(self) -> Node:
        n = self.add()
        while (self.tok.kind == "OP" and self.tok.value in CMP_OPS) or self.at("IN"):
            t = self.take()
            n = Node("bin", "in" if t.kind == "IN" else t.value, n, self.add(), pos=t.pos)
        return n

    def add(self) -> Node:
        n = self.mul()
        while self.at("OP", "+") or self.at("OP", "-"):
            t = self.take()
            n = Node("bin", t.value, n, self.mul(), pos=t.pos)
        return n

    def mul(self) -> Node:
        n = self.unary()
        while self.tok.kind == "OP" and self.tok.value in ("*", "/", "//", "%"):
            t = self.take()
            n = Node("bin", t.value, n, self.unary(), pos=t.pos)
        return n

    def unary(self) -> Node:
        if self.at("OP", "-"):
            p = self.take().pos
            return Node("neg", self.unary(), pos=p)
        if self.at("NOT"):
            p = self.take().pos
            return Node("not", self.unary(), pos=p)
        return self.postfix()

    def _field(self) -> Tok:
        if not self.at("IDENT"):
            if self.tok.kind in ("TRUE", "FALSE", "NULL", "AND", "OR", "NOT", "IN"):
                raise ExprSyntaxError(f"keyword {self.tok.text!r} cannot be a field name after '.'; use "
                                      f"[\"{self.tok.text}\"]", self.tok.pos)
            self.fail("expecting a field name after '.'")
        return self.take()

    def postfix(self) -> Node:
        n = self.primary()
        while True:
            if self.at("OP", "."):
                self.take()
                t = self._field()
                n = Node("attr", n, t.value, pos=t.pos)
            elif self.at("OP", "["):
                p = self.take().pos
                idx = self.or_()
                self.expect("OP", "]", "']'")
                n = Node("index", n, idx, pos=p)
            elif self.at("OP", "("):
                if dotted(n) is None:
                    raise ExprSyntaxError("only a function or subworkflow name can be called", self.tok.pos)
                self.take()
                args: list[Node] = []
                if not self.at("OP", ")"):
                    args.append(self.or_())
                    while self.at("OP", ","):
                        self.take()
                        args.append(self.or_())
                self.expect("OP", ")", "',' or ')'")
                n = Node("call", n, args, pos=n.pos)
            else:
                return n

    def primary(self) -> Node:
        t = self.tok
        if t.kind in ("NUM_INT", "NUM_FLOAT", "STRING"):
            self.take()
            return Node("lit", t.value, pos=t.pos)
        if t.kind in ("TRUE", "FALSE", "NULL"):
            self.take()
            return Node("lit", {"TRUE": True, "FALSE": False, "NULL": None}[t.kind], pos=t.pos)
        if t.kind == "IDENT":
            self.take()
            return Node("name", t.value, pos=t.pos)
        if self.at("OP", "("):
            self.take()
            n = self.or_()
            self.expect("OP", ")", "')'")
            n.paren = True
            return n
        if self.at("OP", "["):
            p = self.take().pos
            items: list[Node] = []
            if not self.at("OP", "]"):
                items.append(self.or_())
                while self.at("OP", ","):
                    self.take()
                    items.append(self.or_())
            self.expect("OP", "]", "',' or ']'")
            return Node("list", items, pos=p)
        self.fail(OPERAND_START)
        raise AssertionError("unreachable")

    def parse_target(self) -> Node:
        """An assignment target: IDENT ( '.' IDENT | '[' expr ']' )* (syntax/lists, syntax/maps)."""
        if not self.at("IDENT"):
            self.fail("expecting a variable name")
        t = self.take()
        n = Node("name", t.value, pos=t.pos)
        while not self.at("EOF"):
            if self.at("OP", "."):
                self.take()
                f = self._field()
                n = Node("attr", n, f.value, pos=f.pos)
            elif self.at("OP", "["):
                p = self.take().pos
                idx = self.or_()
                self.expect("OP", "]", "']'")
                n = Node("index", n, idx, pos=p)
            else:
                self.fail("expecting '.', '[' or the end of the target")
        return n


def parse_expression(src: str) -> Node:
    """Parse the text inside ${...}; raises ExprSyntaxError."""
    return ExprParser(src).parse()


def parse_target(src: str) -> Node:
    return ExprParser(src).parse_target()


def maps_as_lists(src: str) -> str:
    """`{"k": v, ...}` -> `["k", v, ...]` outside string literals (only to keep checking past a map literal)."""
    out, i, n, depth = [], 0, len(src), []
    while i < n:
        c = src[i]
        if c in "\"'":
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            out.append(src[i:j + 1])
            i = j + 1
            continue
        if c in "{[(":
            depth.append(c)
            out.append("[" if c == "{" else c)
        elif c in "}])":
            opener = depth.pop() if depth else ""
            out.append("]" if c == "}" and opener == "{" else c)
        elif c == ":" and depth and depth[-1] == "{":
            out.append(",")
        else:
            out.append(c)
        i += 1
    return "".join(out)


def dotted(n: Node) -> str | None:
    if n.paren:
        return None
    if n.kind == "name":
        return n.a
    if n.kind == "attr":
        base = dotted(n.a)
        return f"{base}.{n.b}" if base else None
    return None


def target_root(n: Node) -> Node:
    while n.kind in ("attr", "index"):
        n = n.a
    return n


EXPR_RE = re.compile(r"^\$\{(.*)\}$", re.S)


def expr_text(v: Any) -> str | None:
    """The expression inside a ${...} string value (Workflows evaluates a string whose whole text is ${...})."""
    if isinstance(v, str):
        m = EXPR_RE.match(v)
        if m:
            return m.group(1)
    return None


# ── findings ────────────────────────────────────────────────────────────────────────────────────────────────────────────


@dataclass
class Finding:
    workflow: str
    file: str
    line: int
    subworkflow: str
    step: str
    rule: str
    message: str
    text: str
    fix: str = ""

    @property
    def severity(self) -> str:
        return RULES[self.rule][0]

    def key(self) -> tuple:
        return (self.file, self.line, self.subworkflow, self.step, self.rule, self.message, self.text)

    def as_dict(self) -> dict:
        return {"line": self.line, "subworkflow": self.subworkflow, "step": self.step, "text": self.text,
                "rule": self.rule, "severity": self.severity, "message": self.message, "fix": self.fix,
                "doc": RULES[self.rule][1]}

    def __str__(self) -> str:
        ctx = self.subworkflow + (f"/{self.step}" if self.step else "")
        txt = self.text if len(self.text) <= 200 else self.text[:197] + "..."
        return (f"{self.file}:{self.line}: {self.severity} [{self.rule}] {ctx}: {self.message}"
                + (f"\n    {txt}" if txt else "") + (f"\n    fix: {self.fix}" if self.fix else ""))


# ── scopes ──────────────────────────────────────────────────────────────────────────────────────────────────────────────


class Scope:
    """A subworkflow's variables; for bodies, except blocks and parallel branches are child scopes
    (syntax/variables#variable-scope). Resolution is flow-insensitive within a scope."""

    def __init__(self, parent: "Scope | None", kind: str):
        self.parent = parent
        self.kind = kind
        self.names: set[str] = set()
        self.children: list[Scope] = []
        if parent:
            parent.children.append(self)

    def visible(self, name: str) -> bool:
        s: Scope | None = self
        while s:
            if name in s.names:
                return True
            s = s.parent
        return False

    def defined_below(self, name: str) -> bool:
        return any(name in c.names or c.defined_below(name) for c in self.children)

    def root(self) -> "Scope":
        s = self
        while s.parent:
            s = s.parent
        return s

    def all_names(self) -> set[str]:
        out = set(self.names)
        for c in self.children:
            out |= c.all_names()
        return out


# Names a workflow may read (as a param) but not assign: Google rejects `result: env` with "cannot assign to a
# reserved variable 'env'" (2026-10-13 deploy error, 3 of the 17 in ops.json). Undocumented; params named env
# were accepted. Applies to every way of setting a variable: assign targets, call result, except as, for value /
# index. Path writes (env.a.b: x) are flagged too, a deliberate over-approximation: Google's error list did not
# include drift_night/mode's `env.job_env.NIGHT_FALLBACK`, but it also skipped other real errors (a for body).
RESERVED_VARIABLES = {"env"}
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
STEP_KINDS = {"assign", "call", "switch", "for", "parallel", "raise", "return", "steps", "try", "next"}
STEP_KEYS = STEP_KINDS | {"args", "result", "retry", "except"}
SWITCH_ENTRY_KEYS = {"condition", "next", "steps", "assign", "return", "raise", "call", "args", "result"}
NEXT_SPECIAL = {"end", "break", "continue"}


def fix_for_map_literal(expr: str) -> str:
    """A hint per map-literal pattern in the expression (all of them: deploy stops at the first)."""
    hints = []
    rest = expr
    if re.search(r"map\.get\(default\([^,()]+,\s*\{\}\)", rest):
        hints.append("map.get(default(x, {}), k) -> map.get(x, k): map.get returns null when its first argument is "
                     "not a map (reference/stdlib/map/get); NB wfsim's map.get still raises TypeError on null")
        rest = re.sub(r"map\.get\(default\([^,()]+,\s*\{\}\)", "map.get(x", rest)
    if re.search(r"json\.encode(_to_string)?\(\s*\{", rest):
        hints.append("build the map as YAML structure in an assign step (leaves may be ${...}) and encode that "
                     "variable")
    elif re.search(r"list\.(concat|prepend)\([^{]*\{", rest):
        hints.append("assign the new element as a YAML map in an earlier assign item, then list.concat(lst, thatVar)")
    elif "[{" in rest:
        hints.append("a list literal cannot hold a map: assign the map as YAML first and use [thatVar]")
    elif "{" in rest:
        hints.append("a map default: assign it as YAML first (`- empty: {}` / `- dflt: {k: v}` in an assign list) and "
                     "use ${default(x, empty)}, or branch on x == null in a switch")
    return "; ".join(hints) or ("write the map as YAML/JSON structure (an assign value or call arg whose leaves are "
                                "${...}) and refer to it")


class Checker:
    def __init__(self, src: Source, *, compiled: str | None = None):
        self.src = src
        self.wf = src.name
        self.program = src.program
        self.findings: list[Finding] = []
        self.compiled = compiled
        self.sub = ""

    def add(self, line: int, sub: str, step: str, rule: str, message: str, text: str = "", fix: str = "") -> None:
        f, ln = self.src.where(line or 1)
        self.findings.append(Finding(self.wf, f, ln, sub, step, rule, message, text, fix))

    # -- entry
    def run(self) -> list[Finding]:
        prog = self.program
        for line, key in self.src.duplicates:
            self.add(line, "", "", "yaml", f"duplicate key {key!r} (PyYAML keeps the later one; Terraform's "
                                           "yamldecode may refuse it or keep either)", key)
        if not isinstance(prog, dict):
            self.add(1, "", "", "step-shape", "the definition is not a map of subworkflows")
            return self.findings
        if not isinstance(prog, PDict):
            prog = self.program = wrap(prog)
        if "main" not in prog:
            self.add(1, "", "", "step-shape", "no main workflow")
        self.subs = {k: v for k, v in prog.items() if isinstance(k, str)}
        for k in prog:
            if not isinstance(k, str):
                self.add(prog.klines.get(k, 1), str(k), "", "yaml", f"a subworkflow name that is not a string: {k!r}")
        self.params: dict[str, tuple[list[str], set[str]]] = {n: self._params(n, s) for n, s in self.subs.items()}
        self.blocking = self._blocking_subs()
        for name, sub in self.subs.items():
            if not isinstance(sub, dict) or "steps" not in sub:
                self.add(getattr(sub, "line", prog.lines.get(name, 1)), name, "", "step-shape",
                         "a subworkflow needs `steps`")
                continue
            extra = set(sub) - {"params", "steps"}
            if extra:
                self.add(sub.line, name, "", "step-shape", f"unknown subworkflow keys {sorted(map(str, extra))}")
            self.check_sub(name, sub)
        self.check_call_depth()
        if self.compiled is None:
            self.compiled = compiled_json(prog)
        size = len(self.compiled.encode("utf-8"))
        if size > SOURCE_LIMIT:
            self.add(1, "", "", "source-size", f"compiled JSON is {size} bytes, over the 128 KB source limit",
                     f"{size} bytes", "move rarely-used logic out, or split the workflow")
        return self.findings

    # -- params
    def _params(self, name: str, sub: Any) -> tuple[list[str], set[str]]:
        names: list[str] = []
        required: set[str] = set()
        if not isinstance(sub, dict) or sub.get("params") is None:
            return names, required
        ps = sub["params"]
        line = sub.lines.get("params", sub.line)
        if not isinstance(ps, list):
            self.add(line, name, "", "step-shape", "params must be a list")
            return names, required
        for i, p in enumerate(ps):
            pl = ps.lines[i] if i < len(ps.lines) else line
            if isinstance(p, str):
                pn = p
                required.add(p)
            elif isinstance(p, dict) and len(p) == 1:
                (pn, default), = p.items()
                if isinstance(default, (dict, list)):
                    # Google refuses to build it: "default value for parameter 'query' must be a literal"
                    # (2026-10-13). Default to null and substitute inside the subworkflow instead.
                    self.add(pl, name, "", "param-default-literal",
                             f"default value for parameter {pn!r} must be a literal (null, a number, a string or a bool)",
                             repr(default)[:80])
                for t in self._strings(default):
                    if "${" in t:
                        self.add(pl, name, "", "param-default-expression",
                                 "an expression as a param default is not documented (defaults are literals)", t)
            else:
                self.add(pl, name, "", "step-shape", f"a param is a name or a one-key map, not {p!r}")
                continue
            if not isinstance(pn, str) or not IDENT_RE.match(pn):
                self.add(pl, name, "", "variable-name", f"param name {pn!r} is not a valid variable name")
                continue
            names.append(pn)
        if name == "main" and len(names) > 1:
            self.add(line, name, "", "step-shape", "main takes at most one param (the runtime argument)")
        return names, required

    @staticmethod
    def _strings(v: Any) -> Iterator[str]:
        if isinstance(v, str):
            yield v
        elif isinstance(v, dict):
            for k, x in v.items():
                if isinstance(k, str):
                    yield k
                yield from Checker._strings(x)
        elif isinstance(v, list):
            for x in v:
                yield from Checker._strings(x)

    # -- blocking subworkflows (an expression cannot call one: syntax/subworkflows)
    @staticmethod
    def calls_in(steps: Any) -> Iterator[str]:
        for _, body in iter_steps(steps):
            if not isinstance(body, dict):
                continue
            if isinstance(body.get("call"), str):
                yield body["call"]
            if isinstance(body.get("switch"), list):
                for cond in body["switch"]:
                    if isinstance(cond, dict) and isinstance(cond.get("call"), str):
                        yield cond["call"]

    def _blocking_subs(self) -> set[str]:
        direct = {n: set(self.calls_in(s.get("steps"))) if isinstance(s, dict) else set() for n, s in self.subs.items()}
        blocking = {n for n, calls in direct.items()
                    if any(c in BLOCKING or c.startswith(("googleapis.", "experimental.")) for c in calls)}
        changed = True
        while changed:
            changed = False
            for n, calls in direct.items():
                if n not in blocking and calls & blocking:
                    blocking.add(n)
                    changed = True
        return blocking

    def check_call_depth(self) -> None:
        graph = {n: {c for c in self.calls_in(s.get("steps")) if c in self.subs} if isinstance(s, dict) else set()
                 for n, s in self.subs.items()}
        memo: dict[str, int] = {}
        INF = 10 ** 6

        def depth(n: str, stack: frozenset) -> int:
            if n in stack:
                return INF
            if n not in memo:
                memo[n] = 1 + max((depth(c, stack | {n}) for c in graph.get(n, ())), default=0)
            return memo[n]

        if "main" in graph:
            d = depth("main", frozenset())
            if d >= INF:
                self.add(1, "main", "", "call-depth", "recursive subworkflow calls: the call depth is unbounded")
            elif d > CALL_DEPTH:
                self.add(1, "main", "", "call-depth", f"static subworkflow call depth {d} > {CALL_DEPTH}")

    # -- one subworkflow
    def check_sub(self, name: str, sub: dict) -> None:
        self.sub = name
        steps = sub["steps"]
        sline = sub.lines.get("steps", sub.line)
        if not isinstance(steps, list) or not steps:
            self.add(sline, name, "", "step-shape", "steps must be a non-empty list")
            return
        root = Scope(None, "subworkflow")
        root.names.update(self.params[name][0])
        self.scope_of: dict[int, Scope] = {}
        self.collect(steps, root)
        self.step_names: dict[str, int] = {}
        self.loop_of: dict[str, int | None] = {}
        self.jumps: list[tuple[str, int | None, int, str]] = []
        self.shadow_calls: set[str] = set()
        self.walk_steps(steps, root, sline, loop=None, in_loop=False, pdepth=0)
        for target, loop, line, step in self.jumps:
            if target in NEXT_SPECIAL:
                if target in ("break", "continue") and loop is None:
                    self.add(line, name, step, "jump", f"next: {target} outside a for loop", target)
                continue
            if target not in self.step_names:
                self.add(line, name, step, "jump", f"next: {target!r} names no step of {name}", target)
            elif self.loop_of[target] != loop:
                self.add(line, name, step, "jump", f"next: {target!r} jumps into or out of a for loop (only jumps "
                                                   "between steps of the same loop body are allowed)", target)
        for n in sorted(root.all_names() & STD_MODULES):
            used = n in self.shadow_calls
            self.add(sub.lines.get("params", sline) if n in self.params[name][0] else sline, name, "",
                     "namespace-shadow",
                     f"variable {n!r} has the name of the standard-library module {n}"
                     + (f", and {name} calls {n}.…() functions: which one Workflows resolves is undocumented"
                        if used else ""), n, f"rename the variable (the docs: don't use `list` as a list name)")

    # pass 1: the variables each scope defines
    def collect(self, steps: Any, scope: Scope) -> None:
        if not isinstance(steps, list):
            return
        for item in steps:
            if isinstance(item, dict) and len(item) == 1:
                (_, body), = item.items()
                if isinstance(body, dict):
                    self._collect_body(body, scope)

    @staticmethod
    def _define(scope: Scope, name: Any) -> None:
        if isinstance(name, str) and not scope.visible(name):
            scope.names.add(name)

    def _collect_assign(self, items: Any, scope: Scope) -> None:
        if isinstance(items, list):
            for a in items:
                if isinstance(a, dict) and len(a) == 1:
                    (k, _), = a.items()
                    if isinstance(k, str) and IDENT_RE.match(k.strip()):
                        self._define(scope, k.strip())

    def _collect_for(self, f: dict, scope: Scope) -> None:
        child = Scope(scope, "for")
        self.scope_of[id(f)] = child
        for k in ("value", "index"):
            if isinstance(f.get(k), str):
                child.names.add(f[k])
        self.collect(f.get("steps"), child)

    def _collect_body(self, body: dict, scope: Scope) -> None:
        self._collect_assign(body.get("assign"), scope)
        if isinstance(body.get("result"), str):
            self._define(scope, body["result"])
        if "steps" in body:
            self.collect(body["steps"], scope)
        sw = body.get("switch")
        if isinstance(sw, list):
            for cond in sw:
                if isinstance(cond, dict):
                    self._collect_assign(cond.get("assign"), scope)
                    if isinstance(cond.get("result"), str):
                        self._define(scope, cond["result"])
                    if "steps" in cond:
                        self.collect(cond["steps"], scope)
        if isinstance(body.get("for"), dict):
            self._collect_for(body["for"], scope)
        t = body.get("try")
        if isinstance(t, dict):
            self._collect_body(t, scope)
        e = body.get("except")
        if isinstance(e, dict):
            child = Scope(scope, "except")
            self.scope_of[id(e)] = child
            if isinstance(e.get("as"), str):
                child.names.add(e["as"])
            self.collect(e.get("steps"), child)
        par = body.get("parallel")
        if isinstance(par, dict):
            for br in par.get("branches") or []:
                if isinstance(br, dict) and len(br) == 1:
                    (_, bb), = br.items()
                    if isinstance(bb, dict):
                        child = Scope(scope, "branch")
                        self.scope_of[id(bb)] = child
                        self.collect(bb.get("steps"), child)
            if isinstance(par.get("for"), dict):
                self._collect_for(par["for"], scope)

    # pass 2: shapes, names, expressions
    def walk_steps(self, steps: Any, scope: Scope, line: int, *, loop: int | None, in_loop: bool, pdepth: int) -> None:
        if not isinstance(steps, list):
            self.add(line, self.sub, "", "step-shape", "steps must be a list")
            return
        if not steps:
            self.add(line, self.sub, "", "step-shape", "steps must not be empty")
        for i, item in enumerate(steps):
            il = steps.lines[i] if isinstance(steps, PList) and i < len(steps.lines) else line
            if not isinstance(item, dict) or len(item) != 1:
                self.add(il, self.sub, "", "step-shape", "a step must be a one-key map {name: body}",
                         json.dumps(item, default=str)[:160])
                continue
            (sname, body), = item.items()
            step = str(sname)
            bl = item.lines.get(sname, il)
            if not isinstance(sname, str):
                self.add(il, self.sub, step, "yaml", f"step name {sname!r} is not a string (YAML 1.1 typed it)")
            if step in self.step_names:
                self.add(il, self.sub, step, "duplicate-step",
                         f"step name {step!r} is already used at {':'.join(map(str, self.src.where(self.step_names[step])))}"
                         " (step names are unique per subworkflow, nested ones included)", step)
            else:
                self.step_names[step] = il
                self.loop_of[step] = loop
            if in_loop and step in ("break", "continue"):
                self.add(il, self.sub, step, "jump", f"a step named {step!r} inside a for loop", step)
            if not isinstance(body, dict):
                self.add(bl, self.sub, step, "step-shape", "a step body must be a map")
                continue
            self.check_step(step, body, scope, bl, loop=loop, in_loop=in_loop, pdepth=pdepth)

    def check_step(self, step: str, body: dict, scope: Scope, line: int, *, loop: int | None, in_loop: bool,
                   pdepth: int) -> None:
        def ln(k: str) -> int:
            return body.lines.get(k, line)

        unknown = set(body) - STEP_KEYS
        if unknown:
            self.add(line, self.sub, step, "step-shape", f"unknown step keys {sorted(map(str, unknown))}")
        kinds = set(body) & STEP_KINDS
        if not kinds:
            self.add(line, self.sub, step, "step-shape", f"no step kind (one of {sorted(STEP_KINDS)})")
        if len(kinds - {"next"}) > 1:
            self.add(line, self.sub, step, "step-shape", f"a step mixes {sorted(kinds - {'next'})}")
        if ("args" in body or "result" in body) and "call" not in body:
            self.add(line, self.sub, step, "step-shape", "args/result without call")
        if ("retry" in body or "except" in body) and "try" not in body:
            self.add(line, self.sub, step, "step-shape", "retry/except without try")
        if "assign" in body:
            self.check_assign(step, body["assign"], scope, ln("assign"))
        if "call" in body:
            self.check_call(step, body, scope, line)
        if "return" in body:
            self.check_value(step, body["return"], scope, ln("return"))
        if "raise" in body:
            self.check_value(step, body["raise"], scope, ln("raise"))
        if "next" in body:
            self.check_next(step, body["next"], loop, ln("next"))
        if "steps" in body:
            self.walk_steps(body["steps"], scope, ln("steps"), loop=loop, in_loop=in_loop, pdepth=pdepth)
        if "switch" in body:
            self.check_switch(step, body, scope, line, loop=loop, in_loop=in_loop, pdepth=pdepth)
        if "for" in body:
            self.check_for(step, body["for"], scope, ln("for"), pdepth=pdepth, parallel=False)
        if "try" in body:
            self.check_try(step, body, scope, line, loop=loop, in_loop=in_loop, pdepth=pdepth)
        if "parallel" in body:
            self.check_parallel(step, body["parallel"], scope, ln("parallel"), pdepth=pdepth)

    def check_reserved(self, step: str, name: Any, how: str, line: int) -> None:
        if isinstance(name, str) and name in RESERVED_VARIABLES:
            self.add(line, self.sub, step, "reserved-variable",
                     f"{how} assigns to {name!r}, a reserved variable (Google: \"cannot assign to a reserved "
                     f"variable '{name}'\")", name, f"rename the variable (e.g. p{name})")

    def check_next(self, step: str, target: Any, loop: int | None, line: int) -> None:
        if not isinstance(target, str):
            self.add(line, self.sub, step, "jump", f"next must be a step name, not {target!r}")
        elif "${" in target:
            self.add(line, self.sub, step, "expression-not-allowed", "next takes a step name, not an expression",
                     target)
        else:
            self.jumps.append((target, loop, line, step))

    def check_assign(self, step: str, items: Any, scope: Scope, line: int) -> None:
        if not isinstance(items, list) or not items:
            self.add(line, self.sub, step, "step-shape", "assign must be a non-empty list of one-key maps")
            return
        if len(items) > ASSIGN_LIMIT:
            self.add(line, self.sub, step, "limit", f"{len(items)} assignments in one step (limit {ASSIGN_LIMIT})")
        for i, a in enumerate(items):
            al = items.lines[i] if isinstance(items, PList) and i < len(items.lines) else line
            if not isinstance(a, dict) or len(a) != 1:
                self.add(al, self.sub, step, "step-shape", "an assignment must be a one-key map {target: value}",
                         json.dumps(a, default=str)[:160])
                continue
            (target, value), = a.items()
            if not isinstance(target, str):
                self.add(al, self.sub, step, "yaml", f"assignment target {target!r} is not a string")
            else:
                self.check_target(step, target, scope, al)
            self.check_value(step, value, scope, a.lines.get(target, al))

    def check_target(self, step: str, target: str, scope: Scope, line: int) -> None:
        if "${" in target:
            self.add(line, self.sub, step, "expression-not-allowed",
                     "an assignment target is a variable path (a.b[i]), not ${...}", target)
            return
        try:
            p = ExprParser(target)
            node = p.parse_target()
        except ExprSyntaxError as e:
            self.add(line, self.sub, step, "map-literal" if e.rule == "map-literal" else "syntax",
                     f"assignment target: {e} (col {e.pos + 1})", target)
            return
        for pos, msg in p.warnings:
            self.add(line, self.sub, step, "string-escape", f"{msg} (col {pos + 1})", target)
        if len(target) > EXPRESSION_LIMIT:
            self.add(line, self.sub, step, "expression-length", f"target is {len(target)} characters", target)
        self.check_reserved(step, target_root(node).a, "assign", line)
        if node.kind != "name":
            root = target_root(node)
            if not scope.visible(root.a):
                self.add(line, self.sub, step, "undefined-symbol",
                         f"{root.a!r} is updated through a path but is not a variable in this scope", target)
            self.check_node(step, node, scope, line, target, target=True)

    def check_value(self, step: str, v: Any, scope: Scope, line: int) -> None:
        if isinstance(v, str):
            self.check_string(step, v, scope, line)
        elif isinstance(v, dict):
            for k, x in v.items():
                kl = v.klines.get(k, line) if isinstance(v, PDict) else line
                if isinstance(k, str):
                    if "${" in k:
                        self.check_string(step, k, scope, kl)
                else:
                    self.add(kl, self.sub, step, "yaml", f"map key {k!r} is not a string (YAML 1.1 typed it)")
                self.check_value(step, x, scope, v.lines.get(k, line) if isinstance(v, PDict) else line)
        elif isinstance(v, list):
            for i, x in enumerate(v):
                self.check_value(step, x, scope, v.lines[i] if isinstance(v, PList) and i < len(v.lines) else line)

    def check_string(self, step: str, s: str, scope: Scope, line: int) -> Node | None:
        e = expr_text(s)
        if e is None:
            if "${" in s:
                self.add(line, self.sub, step, "embedded-expression",
                         "${...} inside a string that is not one whole expression: Workflows has no interpolation, "
                         "this stays a literal", s, 'make the whole value one expression: ${"a" + x + "b"}')
            return None
        if len(e) > EXPRESSION_LIMIT:
            self.add(line, self.sub, step, "expression-length",
                     f"expression is {len(e)} characters (limit {EXPRESSION_LIMIT})", s,
                     "assign parts to variables in earlier assign items and combine them")
        try:
            p = ExprParser(e)
            node = p.parse()
        except ExprSyntaxError as err:
            ctx = e[max(0, err.pos - 40):err.pos] + " >>>" + e[err.pos:err.pos + 1] + "<<< " + e[err.pos + 1:err.pos + 30]
            self.add(line, self.sub, step, err.rule, f"{err} (col {err.pos + 3} of the string): …{ctx.strip()}…", s,
                     fix_for_map_literal(e) if err.rule == "map-literal" else "")
            if err.rule == "map-literal":
                # Deploy stops at the first error; look behind the map literals for the next ones (the maps
                # read as lists of their keys and values, so the names and calls inside are checked too).
                try:
                    p = ExprParser(maps_as_lists(e))
                    node = p.parse()
                except ExprSyntaxError as err2:
                    self.add(line, self.sub, step, err2.rule, f"{err2} (behind the map literal; col {err2.pos + 3} "
                                                              "of the rewritten string)", s)
                    return None
                self.check_node(step, node, scope, line, s)
            return None
        for pos, msg in p.warnings:
            self.add(line, self.sub, step, "string-escape", f"{msg} (col {pos + 3})", s)
        self.check_node(step, node, scope, line, s)
        return node

    # names, functions and arities in a parsed expression
    def check_node(self, step: str, n: Node, scope: Scope, line: int, text: str, target: bool = False) -> None:
        def visit(n: Node, target_root: bool = False) -> None:
            k = n.kind
            if k == "lit":
                return
            if k == "name":
                if target_root or scope.visible(n.a) or n.a in self.subs:
                    return
                if n.a in STD_MODULES:
                    self.add(line, self.sub, step, "unknown-stdlib-member", f"module {n.a!r} used as a value", text)
                elif scope.root().defined_below(n.a):
                    self.add(line, self.sub, step, "out-of-scope",
                             f"{n.a!r} is only assigned inside a for loop or except block, and does not exist here",
                             text, f"assign {n.a} (e.g. to null) before the loop / try so it lives in the outer scope")
                else:
                    self.add(line, self.sub, step, "undefined-symbol",
                             f"symbol {n.a!r} is neither a variable nor a sub-workflow name", text)
                return
            if k == "attr":
                d = dotted(n)
                if d is not None and not target_root:
                    root = d.split(".")[0]
                    if not scope.visible(root) and root not in self.subs and root in STD_MODULES:
                        if d not in STD_OBJECTS:
                            self.add(line, self.sub, step, "unknown-stdlib-member",
                                     f"{d} is not a documented standard-library object", text)
                        return
                visit(n.a, target_root)
                return
            if k == "index":
                visit(n.a, target_root)
                visit(n.b)
                return
            if k == "list":
                for x in n.a:
                    visit(x)
                return
            if k == "call":
                self.check_func(step, n, scope, line, text)
                for x in n.b:
                    visit(x)
                return
            if k in ("and", "or"):
                visit(n.a)
                visit(n.b)
                return
            if k in ("not", "neg"):
                visit(n.a)
                return
            if k == "bin":
                if n.b.kind == "not" and not n.b.paren:
                    self.add(line, self.sub, step, "not-precedence",
                             f"`not` binds tighter than `{n.a}`: this is (not …) {n.a} …, not not(… {n.a} …)", text,
                             f"write not(A {n.a} B)")
                visit(n.b)
                visit(n.c)

        visit(n, target)

    def check_func(self, step: str, n: Node, scope: Scope, line: int, text: str) -> None:
        fname = dotted(n.a)
        if fname is None:
            return
        argc = len(n.b)
        root = fname.split(".")[0]
        if "." in fname and scope.visible(root):
            self.shadow_calls.add(root)
            if fname not in STDLIB:
                self.add(line, self.sub, step, "unknown-function",
                         f"{fname}(): {root!r} is a variable here and {fname} is no library function", text)
                return
        if fname in STDLIB:
            if fname in BLOCKING:
                self.add(line, self.sub, step, "blocking-in-expression",
                         f"{fname} is a blocking call: call steps only, never inside an expression", text,
                         f"make it a call step (call: {fname}) with a result variable")
                return
            lo, hi = arity(fname)
            if not lo <= argc <= hi:
                want = f"{lo}" if lo == hi else f"{lo} to {hi}"
                self.add(line, self.sub, step, "arity", f"{fname}() takes {want} argument(s), given {argc}", text)
            return
        if fname.startswith("googleapis."):
            self.add(line, self.sub, step, "connector-in-expression",
                     f"{fname}() inside an expression: googleapis.* calls are blocking (expressions#blocking-calls), "
                     "though the Secret Manager page shows accessString used so", text, "use a call step")
            return
        if "." not in fname and fname in self.subs:
            names, required = self.params[fname]
            if not len(required) <= argc <= len(names):
                self.add(line, self.sub, step, "arity",
                         f"subworkflow {fname} takes {len(required)} to {len(names)} positional argument(s), "
                         f"given {argc}", text)
            if fname in self.blocking:
                self.add(line, self.sub, step, "blocking-in-expression",
                         f"subworkflow {fname} has blocking steps and cannot be called inside an expression", text,
                         f"make it a call step (call: {fname})")
            return
        if "." not in fname and scope.visible(fname):
            self.add(line, self.sub, step, "unknown-function", f"{fname}() calls a variable", text)
        elif root in STD_MODULES:
            self.add(line, self.sub, step, "unknown-function", f"{fname}() is not a documented {root} function",
                     text)
        else:
            self.add(line, self.sub, step, "unknown-function",
                     f"{fname}() is neither a standard-library function nor a subworkflow"
                     + (" (Workflows has no bool(); compare instead, e.g. x == true)" if fname == "bool" else ""),
                     text)

    def check_call(self, step: str, body: dict, scope: Scope, line: int) -> None:
        target = body.get("call")
        cl = body.lines.get("call", line)
        al = body.lines.get("args", line)
        res = body.get("result")
        if res is not None and (not isinstance(res, str) or not IDENT_RE.match(res)):
            self.add(body.lines.get("result", line), self.sub, step, "variable-name",
                     f"result {res!r} is not a variable name")
        self.check_reserved(step, res, "result:", body.lines.get("result", line))
        args = body.get("args")
        if args is not None and not isinstance(args, dict):
            self.add(al, self.sub, step, "step-shape", "args must be a map of named arguments", str(args)[:160])
            args = {}
        args = args or {}
        self.check_value(step, args, scope, al)
        if not isinstance(target, str):
            self.add(cl, self.sub, step, "call-target", f"call target {target!r} is not a name")
            return
        if "${" in target:
            self.add(cl, self.sub, step, "expression-not-allowed", "call takes a function name, not an expression",
                     target)
            return
        given = {k for k in args if isinstance(k, str)}

        def check_named(names: list[str], required: set[str], extra_ok: set[str] = set()) -> None:
            extra = given - set(names) - extra_ok
            missing = required - given
            if extra:
                self.add(al, self.sub, step, "call-args", f"{target} has no params {sorted(extra)} "
                                                          f"(params: {names})", target)
            if missing:
                self.add(al, self.sub, step, "call-args", f"{target} needs {sorted(missing)}", target)

        if target in self.subs:
            if target == "main":
                self.add(cl, self.sub, step, "call-target", "main cannot be called as a subworkflow", target)
            check_named(*self.params[target])
        elif target in STDLIB:
            ps = STDLIB[target]
            check_named([p.rstrip("?") for p in ps], {p for p in ps if not p.endswith("?")})
            if target == "sys.log" and not given & {"data", "text", "json"}:
                self.add(al, self.sub, step, "call-args", "sys.log needs one of data, text, json", target)
        elif target in CONNECTORS:
            ps = CONNECTORS[target]
            check_named([p.rstrip("?") for p in ps], {p for p in ps if not p.endswith("?")}, CONNECTOR_COMMON)
        elif target.startswith("googleapis."):
            self.add(cl, self.sub, step, "unverified-connector",
                     f"connector {target} is not in this checker's table: check its page under reference/googleapis",
                     target)
        else:
            self.add(cl, self.sub, step, "call-target",
                     f"call target {target!r} is neither a subworkflow, a standard-library function nor a connector",
                     target)

    def check_switch(self, step: str, body: dict, scope: Scope, line: int, *, loop: int | None, in_loop: bool,
                     pdepth: int) -> None:
        sw = body["switch"]
        sl = body.lines.get("switch", line)
        if not isinstance(sw, list) or not sw:
            self.add(sl, self.sub, step, "step-shape", "switch must be a non-empty list of conditions")
            return
        if len(sw) > SWITCH_LIMIT:
            self.add(sl, self.sub, step, "limit", f"{len(sw)} conditions in one switch (limit {SWITCH_LIMIT})")
        for i, cond in enumerate(sw):
            cl = sw.lines[i] if isinstance(sw, PList) and i < len(sw.lines) else sl
            if not isinstance(cond, dict) or "condition" not in cond:
                self.add(cl, self.sub, step, "step-shape", "a switch entry needs a condition")
                continue
            extra = set(cond) - SWITCH_ENTRY_KEYS
            if extra:
                self.add(cl, self.sub, step, "step-shape", f"switch entry keys {sorted(map(str, extra))}")
            c = cond["condition"]
            ccl = cond.lines.get("condition", cl)
            if isinstance(c, str) and expr_text(c) is not None:
                self.check_string(step, c, scope, ccl)
            elif not isinstance(c, bool):
                self.add(ccl, self.sub, step, "step-shape",
                         f"condition must be ${{...}} or a boolean, not {c!r} (a plain string is a literal)")
            kinds = set(cond) & {"next", "steps", "assign", "return", "raise", "call"}
            if len(kinds - {"next"}) > 1:
                self.add(cl, self.sub, step, "step-shape", f"switch entry mixes {sorted(kinds)}")
            if ("args" in cond or "result" in cond) and "call" not in cond:
                self.add(cl, self.sub, step, "step-shape", "switch entry with args/result but no call")
            if "steps" in cond and "next" in cond:
                self.add(cl, self.sub, step, "switch-fallthrough",
                         "switch entry with both steps and next (order undocumented)")
            if "next" in body and not (set(cond) & {"next", "return", "raise"}):
                self.add(cl, self.sub, step, "switch-fallthrough",
                         "switch entry without next/return/raise in a step that has a next: whether the step's next "
                         "or the following step runs after it is undocumented")
            if "assign" in cond:
                self.check_assign(step, cond["assign"], scope, cond.lines.get("assign", cl))
            if "call" in cond:
                self.check_call(step, cond, scope, cl)
            if "return" in cond:
                self.check_value(step, cond["return"], scope, cond.lines.get("return", cl))
            if "raise" in cond:
                self.check_value(step, cond["raise"], scope, cond.lines.get("raise", cl))
            if "next" in cond:
                self.check_next(step, cond["next"], loop, cond.lines.get("next", cl))
            if "steps" in cond:
                self.walk_steps(cond["steps"], scope, cond.lines.get("steps", cl), loop=loop, in_loop=in_loop,
                                pdepth=pdepth)

    def check_for(self, step: str, f: Any, scope: Scope, line: int, *, pdepth: int, parallel: bool) -> None:
        if not isinstance(f, dict):
            self.add(line, self.sub, step, "step-shape", "for must be a map {value, index?, in | range, steps}")
            return
        allowed = {"value", "index", "in", "range", "steps"}
        if parallel:
            allowed |= {"shared", "concurrency_limit", "exception_policy"}
        extra = set(f) - allowed
        if extra:
            self.add(line, self.sub, step, "step-shape", f"for keys {sorted(map(str, extra))}")
        for k in ("value", "index"):
            if k in f and (not isinstance(f[k], str) or not IDENT_RE.match(f[k])):
                self.add(f.lines.get(k, line), self.sub, step, "variable-name",
                         f"for {k} {f[k]!r} is not a variable name")
            self.check_reserved(step, f.get(k), f"for {k}:", f.lines.get(k, line))
        if "value" not in f:
            self.add(line, self.sub, step, "step-shape", "for needs value")
        if ("in" in f) == ("range" in f):
            self.add(line, self.sub, step, "step-shape", "for needs exactly one of in / range")
        if "in" in f:
            v = f["in"]
            if not isinstance(v, list) and expr_text(v) is None:
                self.add(f.lines.get("in", line), self.sub, step, "step-shape",
                         "for.in must be a list or ${...} evaluating to one", str(v))
            self.check_value(step, v, scope, f.lines.get("in", line))
        if "range" in f:
            r = f["range"]
            rl = f.lines.get("range", line)
            if isinstance(r, list):
                if len(r) != 2:
                    self.add(rl, self.sub, step, "step-shape", "range is a list of two bounds")
            elif expr_text(r) is None:
                self.add(rl, self.sub, step, "step-shape", "range is [a, b] or ${[a, b]}")
            self.check_value(step, r, scope, rl)
        if "steps" not in f:
            self.add(line, self.sub, step, "step-shape", "for needs steps")
            return
        self.walk_steps(f["steps"], self.scope_of.get(id(f), scope), f.lines.get("steps", line), loop=id(f),
                        in_loop=True, pdepth=pdepth)

    def check_try(self, step: str, body: dict, scope: Scope, line: int, *, loop: int | None, in_loop: bool,
                  pdepth: int) -> None:
        t = body["try"]
        tl = body.lines.get("try", line)
        if not isinstance(t, dict):
            self.add(tl, self.sub, step, "step-shape", "try must hold a step body (call ... or steps: [...])")
        else:
            if "for" in t:
                self.add(tl, self.sub, step, "try-for",
                         "for directly under try does not deploy (\"loop step name should not be empty\")", "",
                         "put the for loop in a named step under try: steps:")
            if "retry" in t or "except" in t:
                self.add(tl, self.sub, step, "step-shape", "retry/except belong next to try, not under it")
            self.check_step(step, t, scope, tl, loop=loop, in_loop=in_loop, pdepth=pdepth)
        if "retry" in body:
            self.check_retry(step, body["retry"], scope, body.lines.get("retry", line))
        if "except" in body:
            e = body["except"]
            el = body.lines.get("except", line)
            if not isinstance(e, dict):
                self.add(el, self.sub, step, "step-shape", "except must be a map {as, steps}")
                return
            extra = set(e) - {"as", "steps"}
            if extra:
                self.add(el, self.sub, step, "step-shape", f"except keys {sorted(map(str, extra))}")
            if "as" in e and (not isinstance(e["as"], str) or not IDENT_RE.match(e["as"])):
                self.add(el, self.sub, step, "variable-name", f"except as {e['as']!r} is not a variable name")
            self.check_reserved(step, e.get("as"), "except as:", e.lines.get("as", el))
            if "steps" in e:
                self.walk_steps(e["steps"], self.scope_of.get(id(e), scope), e.lines.get("steps", el), loop=loop,
                                in_loop=in_loop, pdepth=pdepth)
        if "retry" not in body and "except" not in body:
            self.add(line, self.sub, step, "step-shape", "try without retry or except")

    def check_retry(self, step: str, r: Any, scope: Scope, line: int) -> None:
        if isinstance(r, str):
            if expr_text(r) is None:
                self.add(line, self.sub, step, "step-shape", "retry is ${policy} or a map", r)
            self.check_string(step, r, scope, line)
            return
        if not isinstance(r, dict):
            self.add(line, self.sub, step, "step-shape", "retry is ${policy} or {predicate, max_retries, backoff}")
            return
        extra = set(r) - {"predicate", "max_retries", "backoff"}
        if extra:
            self.add(line, self.sub, step, "step-shape", f"retry keys {sorted(map(str, extra))}")
        for k in ("predicate", "max_retries", "backoff"):
            if k not in r:
                self.add(line, self.sub, step, "step-shape", f"retry needs {k}")
        b = r.get("backoff")
        if isinstance(b, dict):
            extra = set(b) - {"initial_delay", "max_delay", "multiplier"}
            if extra:
                self.add(line, self.sub, step, "step-shape", f"backoff keys {sorted(map(str, extra))}")
        pe = expr_text(r.get("predicate"))
        if pe is not None and pe.strip() in self.subs and len(self.params[pe.strip()][0]) != 1:
            self.add(line, self.sub, step, "call-args", f"retry predicate {pe.strip()} must take exactly one param",
                     r["predicate"])
        self.check_value(step, r, scope, line)

    def check_parallel(self, step: str, par: Any, scope: Scope, line: int, *, pdepth: int) -> None:
        if pdepth + 1 > PARALLEL_DEPTH:
            self.add(line, self.sub, step, "limit", f"parallel steps nested deeper than {PARALLEL_DEPTH}")
        if not isinstance(par, dict):
            self.add(line, self.sub, step, "step-shape", "parallel must be a map")
            return
        extra = set(par) - {"shared", "concurrency_limit", "exception_policy", "branches", "for"}
        if extra:
            self.add(line, self.sub, step, "step-shape", f"parallel keys {sorted(map(str, extra))}")
        if ("branches" in par) == ("for" in par):
            self.add(line, self.sub, step, "step-shape", "parallel needs exactly one of branches / for")
        for v in par.get("shared") or []:
            if not scope.visible(v):
                self.add(line, self.sub, step, "undefined-symbol", f"shared variable {v!r} is not assigned before",
                         str(v))
        brs = par.get("branches")
        if isinstance(brs, list):
            if len(brs) > BRANCH_LIMIT:
                self.add(line, self.sub, step, "limit", f"{len(brs)} branches (limit {BRANCH_LIMIT})")
            for br in brs:
                if not isinstance(br, dict) or len(br) != 1:
                    self.add(line, self.sub, step, "step-shape", "a branch is a one-key map {name: {steps}}")
                    continue
                (bn, bb), = br.items()
                if not isinstance(bb, dict) or "steps" not in bb:
                    self.add(line, self.sub, step, "step-shape", f"branch {bn} needs steps")
                    continue
                self.walk_steps(bb["steps"], self.scope_of.get(id(bb), scope), line, loop=id(bb), in_loop=False,
                                pdepth=pdepth + 1)
        if "for" in par:
            self.check_for(step, par["for"], scope, line, pdepth=pdepth + 1, parallel=True)


def iter_steps(steps: Any) -> Iterator[tuple[str, Any]]:
    """(name, body) for every step body under steps, nested ones and try bodies included."""
    if not isinstance(steps, list):
        return
    for item in steps:
        if not isinstance(item, dict) or len(item) != 1:
            continue
        (name, body), = item.items()
        yield from _iter_body(str(name), body)


def _iter_body(name: str, body: Any) -> Iterator[tuple[str, Any]]:
    yield name, body
    if not isinstance(body, dict):
        return
    yield from iter_steps(body.get("steps"))
    if isinstance(body.get("for"), dict):
        yield from iter_steps(body["for"].get("steps"))
    if isinstance(body.get("switch"), list):
        for cond in body["switch"]:
            if isinstance(cond, dict):
                yield from iter_steps(cond.get("steps"))
    if isinstance(body.get("try"), dict):
        yield from _iter_body(name + "/try", body["try"])
    if isinstance(body.get("except"), dict):
        yield from iter_steps(body["except"].get("steps"))
    par = body.get("parallel")
    if isinstance(par, dict):
        for br in par.get("branches") or []:
            if isinstance(br, dict) and len(br) == 1:
                (_, bb), = br.items()
                if isinstance(bb, dict):
                    yield from iter_steps(bb.get("steps"))
        if isinstance(par.get("for"), dict):
            yield from iter_steps(par["for"].get("steps"))


# Words YAML 1.1 reads as booleans when unquoted. Terraform's yamldecode (which turns these files into the deployed
# JSON) is YAML 1.1: a step, variable or value spelled like this silently becomes true / false.
YAML11_BOOL = re.compile(r"^\s*(?:-\s+)?(?:(y|Y|n|N|yes|Yes|YES|no|No|NO|on|On|ON|off|Off|OFF)\s*:|[^#'\"]*?:\s+(y|Y|n|N|"
                         r"yes|Yes|YES|no|No|NO|on|On|ON|off|Off|OFF)\s*(?:#.*)?$)")


def yaml11_booleans(files: list[Path]) -> list[Finding]:
    found = []
    for p in files:
        for i, line in enumerate(Path(p).read_text(encoding="utf-8").splitlines(), 1):
            if not line.lstrip().startswith("#") and YAML11_BOOL.match(line):
                found.append(Finding("", Path(p).name, i, "", "", "yaml11-bool",
                                     "a YAML 1.1 boolean word (Terraform's yamldecode makes it true/false)",
                                     line.strip(), "quote it"))
    return found


def check_source(src: Source, compiled: str | None = None) -> list[Finding]:
    if src.program is None:
        return [Finding(src.name, src.files[0][0], 1, "", "", "yaml", "empty definition", "")]
    return Checker(src, compiled=compiled).run()


def check_definition(name: str, paths: list[Path] | None = None) -> list[Finding]:
    try:
        src = load_source(name, paths)
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        return [Finding(name, f"{name}.yaml", (mark.line + 1) if mark else 1, "", "", "yaml", f"YAML: {e}", "")]
    return check_source(src)


def check_text(text: str, name: str = "test") -> list[Finding]:
    """Check a definition given as YAML text (tests)."""
    return check_source(load_text(name, text))


def check_program(program: Any, name: str = "program") -> list[Finding]:
    """Check an already-loaded program (plain dicts: no line numbers, every finding is at line 1)."""
    return check_source(Source(name, [(f"{name}.yaml", 1)], "", program))


def dedupe(findings: list[Finding]) -> list[Finding]:
    seen, out = set(), []
    for f in findings:
        if f.key() not in seen:
            seen.add(f.key())
            out.append(f)
    return sorted(out, key=lambda f: (f.file, f.line, f.rule, f.message))


def run_all(names: list[str] | None = None) -> list[Finding]:
    names = names or list(DEFINITIONS)
    out = yaml11_booleans(sorted({HERE / f for n in names for f in DEFINITIONS[n]}))
    for n in names:
        out += check_definition(n)
    return dedupe(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", type=Path, help="write {file: [findings]} here")
    ap.add_argument("--warnings", action="store_true", help="also print warnings")
    ap.add_argument("--schema", type=Path, help="workflows-samples jsonschema/workflows.json")
    ns = ap.parse_args()
    findings = run_all()
    if ns.schema:
        import jsonschema
        schema = json.loads(ns.schema.read_text())
        # The schema's first oneOf branch is the subworkflow form; its leaf errors say what is wrong.
        v = jsonschema.Draft7Validator(dict(schema["oneOf"][0], definitions=schema["definitions"]))

        def leaves(err):
            if err.context:
                for c in err.context:
                    yield from leaves(c)
            else:
                yield err

        for name in DEFINITIONS:
            doc = json.loads(compiled_json(load_source(name).program).replace("\\u003c", "<"))
            for err in v.iter_errors(doc):
                for leaf in leaves(err):
                    findings.append(Finding(name, f"{name}.yaml", 1, "/".join(map(str, leaf.absolute_path)), "",
                                            "schema", leaf.message[:300], ""))
        findings = dedupe(findings)
    errors = [f for f in findings if f.severity == "error"]
    warnings = [f for f in findings if f.severity != "error"]
    for f in errors + (warnings if ns.warnings else []):
        print(f)
    if ns.json:
        out: dict[str, list] = {f: [] for n in DEFINITIONS for f in DEFINITIONS[n]}
        for f in findings:
            out.setdefault(f.file, []).append(f.as_dict())
        ns.json.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    sizes = []
    for name in DEFINITIONS:
        try:
            sizes.append(f"{name} {len(compiled_json(load_source(name).program).encode())} B")
        except Exception:  # noqa: BLE001 - the YAML error is a finding already
            pass
    print(f"{'FAIL' if errors else 'ok'}: {len(errors)} error(s), {len(warnings)} warning(s) in "
          f"{', '.join(DEFINITIONS)} (compiled {', '.join(sizes)}; limit {SOURCE_LIMIT} B)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
