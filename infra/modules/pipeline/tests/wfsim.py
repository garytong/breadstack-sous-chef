#!/usr/bin/env python3
"""wfsim — a Cloud Workflows interpreter for the pipeline's dry end-to-end run.

It executes the deployed definitions (workflows/<name>.yaml + lib.yaml, the
same source workflows.tf deploys) step by step, with the HTTP calls, Secret
Manager reads, sleeps and the clock handed to the caller:

    sim = Interpreter(program, http=router, secrets=fn, env={...}, clock=VirtualClock(), on_sleep=hook)
    result = sim.run({"mode": "poll"})

What it implements is the subset the pipeline uses, as strictly as the
documentation reads (a stricter interpreter finds more bugs than a lenient
one):

  steps        assign (dotted / indexed targets), call (+args, result),
               switch (conditions with next / steps / assign / call / raise /
               return), steps, for (in / range, index, break, continue),
               try / retry / except, raise, return, next (a step name, end,
               break, continue); a jump never leaves a for loop
  expressions  ${...} parsed by lint_workflows' parser (Google's documented
               grammar: no map literals, `not` binding tighter than any
               binary operator): literals, list literals, . and [] access,
               and / or (short-circuit, booleans only), not, in, == != < <=
               > >=, + - * / % //, unary -, and the standard library
               functions the definitions call (default, map.*, text.*,
               time.*, sys.*, json.*, list.concat, keys, len, string, int,
               double, get_type)
  deploy check from_files (and the constructor, unless strict=False) refuses
               a definition lint_workflows finds an error in, as a deploy
               would (SimError with file:line)
  values       copied on assignment and on every call (Workflows has value
               semantics); a missing key is a KeyError, a bad operand a
               TypeError, an HTTP status >= 400 an HttpError {code, message,
               body, headers, tags}
  std calls    http.get / post / patch / delete (with the default retry
               policies, or a custom predicate subworkflow), sys.log,
               sys.sleep (virtual time),
               googleapis.secretmanager.v1.projects.secrets.versions.accessString
  responses    header names canonicalized as Workflows returns them
               (Content-Type, X-Goog-Generation: seen in a real execution's
               call log, 2026-10-13); a bytes body is parsed by its
               Content-Type: application/json -> map/list, text/* -> string,
               anything else stays bytes (docs/http-requests); a body the
               router hands over already parsed (or as a str) is kept as is
  limits       https://docs.cloud.google.com/workflows/quotas, enforced:
               100,000 steps per execution (every step run, every retry
               attempt, a subworkflow call = its steps + the calling step;
               /workflows/pricing), 512 KB for all variables of all live
               frames (sized as docs/reference/syntax/variables computes it:
               numbers/booleans 8 bytes, strings/bytes/lists 8 + elements,
               maps 8 + keys + values), 256 KB per string, 2 MB per HTTP
               response: ResourceLimitError, which "cannot be caught and
               causes immediate execution failure" (syntax/error-types), so
               it is not a WfError and no try/except sees it; 20 call levels
               (a RecursionError). sim.steps and sim.peak_memory report usage.

Every call, log line and HTTP request is recorded in sim.trace.
"""
from __future__ import annotations

import copy
import json
import re
import sys
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lint_workflows import (ExprSyntaxError, Node, check_program, check_source, load_source,  # noqa: E402
                            parse_expression)

# ── errors and control flow ─────────────────────────────────────────────────────────────────────────────────────────────


class SimError(Exception):
    """The definition does something Workflows would refuse (or the simulator does not model)."""


class ResourceLimitError(Exception):
    """A Workflows resource limit (steps, variable memory, string or response size). Uncatchable: deliberately not
    a WfError, so no try/except in the definition sees it and the execution fails at once."""

    def __init__(self, message: str):
        super().__init__(message)
        self.value = {"message": message, "tags": ["ResourceLimitError"]}


# https://docs.cloud.google.com/workflows/quotas
STEP_LIMIT = 100_000
MEMORY_LIMIT = 512 * 1024
STRING_LIMIT = 256 * 1024
RESPONSE_LIMIT = 2 * 1024 * 1024
CALL_DEPTH_LIMIT = 20


def wsize(v: Any) -> int:
    """Memory of a value as docs/reference/syntax/variables computes it (raises ResourceLimitError for a string
    over 256 KB)."""
    if v is None or isinstance(v, (bool, int, float)):
        return 8
    if isinstance(v, str):
        n = len(v.encode("utf-8"))
        if n > STRING_LIMIT:
            raise ResourceLimitError(f"a string of {n} bytes (Workflows allows {STRING_LIMIT})")
        return 8 + n
    if isinstance(v, (bytes, bytearray)):
        return 8 + len(v)
    if isinstance(v, list):
        return 8 + sum(wsize(x) for x in v)
    if isinstance(v, dict):
        return 8 + sum(wsize(k) + wsize(x) for k, x in v.items())
    return 8


def canonical_header(name: str) -> str:
    """Go's CanonicalMIMEHeaderKey, which is how a real execution's HttpError.headers spells them
    (Content-Type, X-Guploader-Uploadid, X-Goog-Gcs-Base-Ts in execution 8a0cc8ec's call log)."""
    return "-".join(p[:1].upper() + p[1:].lower() for p in str(name).split("-"))


class WfError(Exception):
    """A Workflows error value in flight (raise, a runtime error, an HttpError)."""

    def __init__(self, value: Any):
        super().__init__(json.dumps(value, default=str)[:500] if not isinstance(value, str) else value[:500])
        self.value = value


def runtime_error(tag: str, message: str) -> WfError:
    return WfError({"message": message, "tags": [tag]})


class _Return(Exception):
    def __init__(self, value: Any):
        self.value = value


class _Jump(Exception):
    def __init__(self, target: str):
        self.target = target


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


# ── the clock ───────────────────────────────────────────────────────────────────────────────────────────────────────────


class VirtualClock:
    """sys.now() starts at the real time and moves only with sys.sleep (or advance)."""

    def __init__(self, start: float | None = None):
        self.t = time.time() if start is None else start

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += float(seconds)


def fmt_time(t: float) -> str:
    dt = datetime.fromtimestamp(t, tz=timezone.utc)
    if abs(t - round(t)) < 1e-9:
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_time(s: str) -> float:
    if not isinstance(s, str):
        raise runtime_error("TypeError", f"time.parse: not a string: {s!r}")
    v = s.strip()
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    m = re.match(r"^(.*T\d\d:\d\d:\d\d)(\.\d+)?(.*)$", v)
    if m and m.group(2) and len(m.group(2)) > 7:  # nanoseconds → microseconds
        v = m.group(1) + m.group(2)[:7] + m.group(3)
    try:
        return datetime.fromisoformat(v).timestamp()
    except ValueError:
        raise runtime_error("ValueError", f"time.parse: not RFC 3339: {s!r}") from None


# ── types ───────────────────────────────────────────────────────────────────────────────────────────────────────────────


def wtype(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "double"
    if isinstance(v, str):
        return "string"
    if isinstance(v, (bytes, bytearray)):
        return "bytes"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "map"
    raise SimError(f"a value of Python type {type(v).__name__} reached the workflow")


def is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def wstring(v: Any) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return "true" if v else "false"
    if is_num(v):
        if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
            return repr(v)
        return str(v)
    if isinstance(v, (bytes, bytearray)):
        return bytes(v).decode("utf-8")
    raise runtime_error("TypeError", f"string(): cannot convert {wtype(v)}")


def to_json(v: Any) -> str:
    def enc(x: Any) -> Any:
        if isinstance(x, (bytes, bytearray)):
            return bytes(x).decode("utf-8", "replace")
        return x
    return json.dumps(v, separators=(",", ":"), ensure_ascii=False, default=enc)


# ── the expression language ─────────────────────────────────────────────────────────────────────────────────────────────

# The grammar is lint_workflows' (one tokenizer and parser for the checker and the simulator, built from
# Google's documented grammar): no map literals inside ${}, `not` binding tighter than any binary operator,
# true/True/TRUE, '...' strings. Node, parse_expression and ExprSyntaxError come from there.


def parse(src: str) -> Node:
    try:
        return parse_expression(src)
    except ExprSyntaxError as e:
        raise SimError(f"Workflows would not deploy this expression ({e.rule}: {e} at col {e.pos + 1}): {src!r}") from None


def dotted(n: Node) -> str | None:
    if n.kind == "name":
        return n.a
    if n.kind == "attr":
        base = dotted(n.a)
        return f"{base}.{n.b}" if base else None
    return None


NAMESPACES = {"map", "text", "time", "sys", "json", "list", "http", "math", "base64", "retry", "events", "uuid", "hash"}

DEFAULT_RETRY = {"predicate": "http.default_retry_predicate", "max_retries": 5,
                 "backoff": {"initial_delay": 1, "max_delay": 60, "multiplier": 1.25}}
DEFAULT_RETRY_NON_IDEMPOTENT = {"predicate": "http.default_retry_predicate_non_idempotent", "max_retries": 5,
                                "backoff": {"initial_delay": 1, "max_delay": 60, "multiplier": 1.25}}
CONSTANTS = {"http.default_retry": DEFAULT_RETRY, "http.default_retry_non_idempotent": DEFAULT_RETRY_NON_IDEMPOTENT}


# ── the scope ───────────────────────────────────────────────────────────────────────────────────────────────────────────


class Scope:
    """A subworkflow's variables; a for-loop iteration is a child (new names die with it, old ones are updated)."""

    def __init__(self, parent: "Scope | None" = None, meter: "Interpreter | None" = None):
        self.vars: dict[str, Any] = {}
        self.parent = parent
        self.meter = meter if meter is not None else (parent.meter if parent is not None else None)
        self.sizes: dict[str, int] = {}
        self.total = 0

    def _put(self, name: str, value: Any) -> None:
        self.vars[name] = value
        self.resize(name)

    def resize(self, name: str) -> None:
        if self.meter is None:
            return
        new = wsize(self.vars[name])
        self.total += new - self.sizes.get(name, 0)
        self.sizes[name] = new
        self.meter.check_memory()

    def touch(self, name: str) -> None:
        """A variable changed in place (an indexed or dotted assignment): size it again where it lives."""
        s: Scope | None = self
        while s is not None:
            if name in s.vars:
                s.resize(name)
                return
            s = s.parent

    def has(self, name: str) -> bool:
        s: Scope | None = self
        while s is not None:
            if name in s.vars:
                return True
            s = s.parent
        return False

    def get(self, name: str) -> Any:
        s: Scope | None = self
        while s is not None:
            if name in s.vars:
                return s.vars[name]
            s = s.parent
        raise runtime_error("KeyError", f"variable {name!r} is not defined")

    def set(self, name: str, value: Any) -> None:
        s: Scope | None = self
        while s is not None:
            if name in s.vars:
                s._put(name, value)
                return
            s = s.parent
        self._put(name, value)

    def local(self, name: str, value: Any) -> None:
        self._put(name, value)


# ── the interpreter ─────────────────────────────────────────────────────────────────────────────────────────────────────

HttpRouter = Callable[[str, str, dict, dict, Any, Any], tuple[int, Any, dict]]


def refuse_undeployable(findings: list, what: str) -> None:
    errors = [f for f in findings if f.severity == "error"]
    if errors:
        head = "\n".join(str(f) for f in errors[:20])
        more = f"\n... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise SimError(f"{what} would not deploy ({len(errors)} error(s); lint_workflows.py lists them all):\n"
                       f"{head}{more}")


@dataclass
class TraceEvent:
    t: float
    kind: str
    detail: dict = field(default_factory=dict)


class Interpreter:
    def __init__(self, program: dict, *, http: HttpRouter, secrets: Callable[[dict], str], env: dict[str, str],
                 clock: VirtualClock, on_sleep: Callable[["Interpreter", float], None] | None = None,
                 max_steps: int = STEP_LIMIT, max_memory: int = MEMORY_LIMIT, log: Callable[[str], None] | None = None,
                 strict: bool = True):
        if "main" not in program:
            raise SimError("no main")
        if strict:
            refuse_undeployable(check_program(program), "the program")
        self.program = program
        self.http = http
        self.secrets = secrets
        self.env = dict(env)
        self.clock = clock
        self.on_sleep = on_sleep
        self.max_steps = max_steps
        self.max_memory = max_memory
        self.steps = 0
        self.peak_memory = 0
        self.max_depth = 0
        self.live: list[Scope] = []  # every frame (subworkflow scope, for-loop iteration) alive now
        self.trace: list[TraceEvent] = []
        self.logf = log or (lambda s: None)
        self.depth = 0
        self._exprs: dict[str, Node] = {}

    # ── entry ──
    @classmethod
    def from_files(cls, *paths: str, strict: bool = True, **kw: Any) -> "Interpreter":
        """The definition as workflows.tf deploys it (the files joined with newlines), refused like a deploy
        would refuse it: every error finding of lint_workflows (with file:line) is a SimError."""
        src = "\n".join(Path(p).read_text(encoding="utf-8") for p in paths)
        if strict:
            refuse_undeployable(check_source(load_source(Path(paths[0]).stem, [Path(p) for p in paths])),
                                " + ".join(Path(p).name for p in paths))
        return cls(yaml.safe_load(src), strict=False, **kw)

    def run(self, args: Any = None) -> Any:
        main = self.program["main"]
        params = main.get("params") or []
        kwargs = {}
        if params:
            kwargs[self._param_name(params[0])] = copy.deepcopy(args)
        return self.call_sub("main", kwargs)

    def count_step(self, where: str) -> None:
        self.steps += 1
        if self.steps > self.max_steps:
            raise ResourceLimitError(f"{where}: more than {self.max_steps} steps in one execution")

    def check_memory(self) -> None:
        used = sum(s.total for s in self.live)
        if used > self.peak_memory:
            self.peak_memory = used
        if used > self.max_memory:
            raise ResourceLimitError(f"variables use {used} bytes (Workflows allows {self.max_memory})")

    def record(self, kind: str, **detail: Any) -> None:
        self.trace.append(TraceEvent(self.clock.now(), kind, detail))

    # ── subworkflows ──
    @staticmethod
    def _param_name(p: Any) -> str:
        return p if isinstance(p, str) else next(iter(p))

    def call_sub(self, name: str, kwargs: dict) -> Any:
        sub = self.program.get(name)
        if not isinstance(sub, dict) or "steps" not in sub:
            raise SimError(f"no subworkflow {name!r}")
        scope = Scope(meter=self)
        self.live.append(scope)
        try:
            return self._call_sub(name, sub, scope, kwargs)
        finally:
            self.live.remove(scope)

    def _call_sub(self, name: str, sub: dict, scope: Scope, kwargs: dict) -> Any:
        declared = []
        for p in sub.get("params") or []:
            if isinstance(p, str):
                declared.append(p)
                if p not in kwargs:
                    raise SimError(f"call {name}: missing required argument {p!r}")
            else:
                (pn, default), = p.items()
                declared.append(pn)
                scope.local(pn, copy.deepcopy(default))
        for k, v in kwargs.items():
            if k not in declared:
                raise SimError(f"call {name}: unknown argument {k!r}")
            scope.local(k, copy.deepcopy(v))
        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)
        try:
            if self.depth > CALL_DEPTH_LIMIT:
                raise runtime_error("RecursionError", f"{name}: more than {CALL_DEPTH_LIMIT} levels of calls")
            self.run_steps(sub["steps"], scope, where=name)
            return None
        except _Return as r:
            return copy.deepcopy(r.value)
        except (_Break, _Continue):
            raise SimError(f"{name}: break/continue outside a for loop")
        except _Jump as j:
            raise SimError(f"{name}: next: {j.target} names no step of the subworkflow") from None
        finally:
            self.depth -= 1

    # ── steps ──
    def run_steps(self, steps: list, scope: Scope, where: str) -> None:
        if not isinstance(steps, list):
            raise SimError(f"{where}: steps must be a list")
        names = []
        for st in steps:
            if not isinstance(st, dict) or len(st) != 1:
                raise SimError(f"{where}: a step must be a one-key map: {st!r}"[:300])
            names.append(next(iter(st)))
        i = 0
        while i < len(steps):
            self.count_step(f"{where}.{names[i]}")
            name = names[i]
            body = steps[i][name]
            try:
                nxt = self.exec_step(name, body, scope, f"{where}.{name}")
            except _Jump as j:
                if j.target in names:
                    i = names.index(j.target)
                    continue
                raise
            if nxt is None:
                i += 1
            elif nxt == "end":
                raise _Return(None)
            elif nxt == "break":
                raise _Break()
            elif nxt == "continue":
                raise _Continue()
            elif nxt in names:
                i = names.index(nxt)
            else:
                raise _Jump(nxt)

    def exec_step(self, name: str, body: Any, scope: Scope, where: str) -> str | None:
        if not isinstance(body, dict):
            raise SimError(f"{where}: step body must be a map")
        known = {"assign", "call", "args", "result", "switch", "steps", "for", "try", "retry", "except", "raise", "return",
                 "next", "parallel"}
        extra = set(body) - known
        if extra:
            raise SimError(f"{where}: unknown step keys {sorted(extra)}")
        if "parallel" in body:
            raise SimError(f"{where}: parallel steps are not simulated")
        if "try" in body:
            return self.exec_try(body, scope, where)
        if "for" in body:
            self.exec_for(body["for"], scope, where)
            return body.get("next")
        if "switch" in body:
            return self.exec_switch(body, scope, where)
        return self.exec_simple(body, scope, where)

    def exec_simple(self, body: dict, scope: Scope, where: str) -> str | None:
        """assign / call / steps / raise / return (+ next): also the body of a switch condition."""
        if "assign" in body:
            self.exec_assign(body["assign"], scope, where)
        if "call" in body:
            res = self.exec_call(body["call"], body.get("args"), scope, where)
            if "result" in body:
                scope.set(body["result"], copy.deepcopy(res))
        elif "args" in body or "result" in body:
            raise SimError(f"{where}: args/result without call")
        if "steps" in body:
            self.run_steps(body["steps"], scope, where)
        if "raise" in body:
            val = self.eval_value(body["raise"], scope, where)
            self.record("raise", where=where, value=val)
            raise WfError(copy.deepcopy(val))
        if "return" in body:
            raise _Return(self.eval_value(body["return"], scope, where))
        return body.get("next")

    def exec_switch(self, body: dict, scope: Scope, where: str) -> str | None:
        conds = body["switch"]
        if not isinstance(conds, list) or not conds:
            raise SimError(f"{where}: switch must be a non-empty list")
        for k, c in enumerate(conds):
            if "condition" not in c:
                raise SimError(f"{where}: switch item {k} has no condition")
            v = self.eval_value(c["condition"], scope, where)
            if not isinstance(v, bool):
                raise runtime_error("TypeError", f"{where}: condition {k} is {wtype(v)}, not boolean")
            if v:
                inner = {kk: vv for kk, vv in c.items() if kk != "condition"}
                bad = set(inner) - {"assign", "call", "args", "result", "steps", "raise", "return", "next"}
                if bad:
                    raise SimError(f"{where}: switch condition {k} has keys {sorted(bad)}")
                if "next" not in inner and "return" not in inner and "raise" not in inner and "next" in body:
                    raise SimError(f"{where}: matched condition {k} has no next while the step has one (ambiguous)")
                return self.exec_simple(inner, scope, f"{where}[{k}]")
        if any(k in body for k in ("assign", "call", "steps", "raise", "return")):
            raise SimError(f"{where}: a switch step carries other step kinds")
        return body.get("next")

    def exec_for(self, spec: dict, scope: Scope, where: str) -> None:
        value = spec.get("value")
        index = spec.get("index")
        if not value:
            raise SimError(f"{where}: for needs value")
        if "in" in spec:
            items = self.eval_value(spec["in"], scope, where)
            if not isinstance(items, list):
                raise runtime_error("TypeError", f"{where}: for in: {wtype(items)}, not list")
        elif "range" in spec:
            rg = self.eval_value(spec["range"], scope, where)
            if not (isinstance(rg, list) and len(rg) == 2 and all(is_num(x) for x in rg)):
                raise runtime_error("TypeError", f"{where}: range must be [begin, end]")
            a, b = rg
            items = []
            x = a
            while x <= b:
                items.append(x)
                x += 1
        else:
            raise SimError(f"{where}: for needs in or range")
        for i, item in enumerate(items):
            child = Scope(scope)
            self.live.append(child)
            try:
                child.local(value, copy.deepcopy(item))
                if index:
                    child.local(index, i)
                self.run_steps(spec["steps"], child, where)
            except _Break:
                break
            except _Continue:
                continue
            except _Jump as j:
                raise SimError(f"{where}: next: {j.target} jumps out of a for loop") from None
            finally:
                self.live.remove(child)

    def exec_try(self, body: dict, scope: Scope, where: str) -> str | None:
        inner = body["try"]
        retry = body.get("retry")
        exc = body.get("except")
        policy = None
        if retry is not None:
            policy = self.eval_value(retry, scope, where) if isinstance(retry, str) else dict(retry)
            if isinstance(policy, dict) and isinstance(policy.get("predicate"), str):
                policy["predicate"] = self._predicate(policy["predicate"], scope, where)
            if isinstance(policy, dict):
                for k in ("max_retries",):
                    if isinstance(policy.get(k), str):
                        policy[k] = self.eval_value(policy[k], scope, where)
        attempt = 0
        delay = None
        while True:
            try:
                if isinstance(inner, dict) and set(inner) <= {"steps"}:
                    self.run_steps(inner["steps"], scope, where + ".try")
                else:
                    nxt = self.exec_simple(inner, scope, where + ".try")
                    if nxt:
                        raise SimError(f"{where}: next inside a try body")
                return body.get("next")
            except WfError as e:
                if policy and attempt < int(policy.get("max_retries", 0)) and self._should_retry(policy, e.value, where):
                    attempt += 1
                    self.count_step(where + ".retry")  # "Each retry attempt counts as a step execution" (pricing)
                    bo = policy.get("backoff") or {}
                    delay = float(bo.get("initial_delay", 1)) if delay is None else min(
                        delay * float(bo.get("multiplier", 2)), float(bo.get("max_delay", 60)))
                    self.record("retry", where=where, attempt=attempt, error=e.value)
                    self._sleep(delay)
                    continue
                if not exc:
                    raise
                name = exc.get("as")
                if name:
                    scope.set(name, copy.deepcopy(e.value))
                self.record("except", where=where, error=e.value)
                self.run_steps(exc["steps"], scope, where + ".except")
                return body.get("next")

    STD_PREDICATES = ("http.default_retry_predicate", "http.default_retry_predicate_non_idempotent")

    def _predicate(self, src: str, scope: Scope, where: str) -> str:
        """A policy's predicate: one of the standard ones, or ${name} of a subworkflow taking the error."""
        s = src.strip()
        inner = s[2:-1].strip() if s.startswith("${") and s.endswith("}") else s
        if inner in self.STD_PREDICATES:
            return inner
        if inner in self.program and inner != "main":
            return "sub:" + inner
        raise SimError(f"{where}: retry predicate {src!r} is neither a standard predicate nor a subworkflow")

    def _should_retry(self, policy: dict, err: Any, where: str = "") -> bool:
        pred = policy.get("predicate")
        if isinstance(pred, str) and pred.startswith("sub:"):
            name = pred[4:]
            params = (self.program[name].get("params") or [])
            if len(params) != 1:
                raise SimError(f"{where}: predicate {name} must take exactly one parameter")
            self.count_step(where + ".predicate")
            out = self.call_sub(name, {self._param_name(params[0]): copy.deepcopy(err)})
            if not isinstance(out, bool):
                raise runtime_error("TypeError", f"{where}: predicate {name} returned {wtype(out)}, not boolean")
            return out
        if not isinstance(err, dict):
            return False
        tags = err.get("tags") or []
        code = err.get("code")
        # stdlib/http/default_retry_predicate: 429, 502, 503, 504, ConnectionError, ConnectionFailedError, TimeoutError
        if pred == "http.default_retry_predicate":
            return code in (429, 502, 503, 504) or any(t in tags for t in ("ConnectionError", "ConnectionFailedError", "TimeoutError"))
        # .../default_retry_predicate_non_idempotent: 429, 503, ConnectionFailedError
        if pred == "http.default_retry_predicate_non_idempotent":
            return code in (429, 503) or "ConnectionFailedError" in tags
        raise SimError(f"retry predicate {pred!r} is not simulated")

    # ── assign ──
    _TARGET = re.compile(r"""\s*(?:(?P<name>[A-Za-z_][A-Za-z0-9_]*)|\.(?P<attr>[A-Za-z_][A-Za-z0-9_]*)|\[(?P<idx>[^\]]+)\])""")

    def exec_assign(self, items: list, scope: Scope, where: str) -> None:
        if not isinstance(items, list):
            raise SimError(f"{where}: assign must be a list")
        if len(items) > 50:
            raise SimError(f"{where}: more than 50 assignments in one step")
        for it in items:
            if not isinstance(it, dict) or len(it) != 1:
                raise SimError(f"{where}: an assignment must be a one-key map")
            (target, expr), = it.items()
            val = copy.deepcopy(self.eval_value(expr, scope, where))
            self.assign_target(str(target), val, scope, where)

    def assign_target(self, target: str, val: Any, scope: Scope, where: str) -> None:
        parts: list[tuple[str, Any]] = []
        pos = 0
        while pos < len(target):
            m = self._TARGET.match(target, pos)
            if not m or m.end() == pos:
                raise SimError(f"{where}: cannot parse assignment target {target!r}")
            pos = m.end()
            if m.group("name") and not parts:
                parts.append(("name", m.group("name")))
            elif m.group("attr"):
                parts.append(("key", m.group("attr")))
            elif m.group("idx"):
                parts.append(("expr", m.group("idx")))
            else:
                raise SimError(f"{where}: bad assignment target {target!r}")
        if parts[0][0] != "name":
            raise SimError(f"{where}: assignment target {target!r} starts without a name")
        if len(parts) == 1:
            scope.set(parts[0][1], val)
            return
        root = scope.get(parts[0][1])
        cur = root
        for kind, p in parts[1:-1]:
            key = p if kind == "key" else self.eval_expr(p, scope, where)
            cur = self._index(cur, key, where)
        kind, p = parts[-1]
        key = p if kind == "key" else self.eval_expr(p, scope, where)
        if isinstance(cur, dict):
            if not isinstance(key, str):
                raise runtime_error("TypeError", f"{where}: map key {key!r} is not a string")
            cur[key] = val
        elif isinstance(cur, list):
            if not is_num(key) or int(key) != key or not 0 <= int(key) < len(cur):
                raise runtime_error("IndexError", f"{where}: list index {key!r} out of range ({len(cur)})")
            cur[int(key)] = val
        else:
            raise runtime_error("TypeError", f"{where}: cannot set {key!r} on a {wtype(cur)}")
        scope.touch(parts[0][1])

    # ── values and expressions ──
    def eval_value(self, v: Any, scope: Scope, where: str) -> Any:
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("${") and s.endswith("}"):
                return self.eval_expr(s[2:-1], scope, where)
            if "${" in v:
                raise SimError(f"{where}: a string with an embedded ${{...}} (Workflows has no interpolation): {v!r}")
            return v
        if isinstance(v, list):
            return [self.eval_value(x, scope, where) for x in v]
        if isinstance(v, dict):
            out = {}
            for k, x in v.items():
                if not isinstance(k, str):
                    raise SimError(f"{where}: a map key that is not a string: {k!r} (YAML 1.1 turned it into {type(k).__name__}?)")
                out[k] = self.eval_value(x, scope, where)
            return out
        if isinstance(v, bool) or v is None or is_num(v):
            return v
        raise SimError(f"{where}: unsupported literal {v!r}")

    def eval_expr(self, src: str, scope: Scope, where: str) -> Any:
        node = self._exprs.get(src)
        if node is None:
            node = parse(src)
            self._exprs[src] = node
        try:
            return self.ev(node, scope, where)
        except WfError as e:
            if isinstance(e.value, dict) and "tags" in e.value and "where" not in e.value and "code" not in e.value:
                e.value = {**e.value, "where": where, "expression": src[:200]}
            raise

    def _index(self, base: Any, key: Any, where: str) -> Any:
        if isinstance(base, dict):
            if not isinstance(key, str):
                raise runtime_error("TypeError", f"{where}: map key must be a string, not {wtype(key)}")
            if key not in base:
                raise runtime_error("KeyError", f"{where}: key {key!r} not found")
            return base[key]
        if isinstance(base, list):
            if not is_num(key) or int(key) != key:
                raise runtime_error("TypeError", f"{where}: list index must be an integer")
            k = int(key)
            if not 0 <= k < len(base):
                raise runtime_error("IndexError", f"{where}: list index {k} out of range ({len(base)})")
            return base[k]
        raise runtime_error("TypeError", f"{where}: cannot index a {wtype(base)} with {key!r}")

    def ev(self, n: Node, scope: Scope, where: str) -> Any:
        k = n.kind
        if k == "lit":
            return n.a
        if k == "name":
            return scope.get(n.a)
        if k == "list":
            return [self.ev(x, scope, where) for x in n.a]
        if k == "attr":
            d = dotted(n)
            if d in CONSTANTS and not scope.has(d.split(".")[0]):
                return copy.deepcopy(CONSTANTS[d])
            base = self.ev(n.a, scope, where)
            return self._index(base, n.b, where)
        if k == "index":
            return self._index(self.ev(n.a, scope, where), self.ev(n.b, scope, where), where)
        if k == "call":
            fname = dotted(n.a)
            if fname is None:
                raise SimError(f"{where}: calling something that is not a function name")
            if "." in fname and fname.split(".")[0] not in NAMESPACES:
                raise SimError(f"{where}: {fname}() is not a standard-library function")
            args = [self.ev(a, scope, where) for a in n.b]
            return self.func(fname, args, where)
        if k == "and" or k == "or":
            left = self.ev(n.a, scope, where)
            if not isinstance(left, bool):
                raise runtime_error("TypeError", f"{where}: {k} on a {wtype(left)}")
            if (k == "and" and not left) or (k == "or" and left):
                return left
            right = self.ev(n.b, scope, where)
            if not isinstance(right, bool):
                raise runtime_error("TypeError", f"{where}: {k} on a {wtype(right)}")
            return right
        if k == "not":
            v = self.ev(n.a, scope, where)
            if not isinstance(v, bool):
                raise runtime_error("TypeError", f"{where}: not on a {wtype(v)}")
            return not v
        if k == "neg":
            v = self.ev(n.a, scope, where)
            if not is_num(v):
                raise runtime_error("TypeError", f"{where}: unary - on a {wtype(v)}")
            return -v
        if k == "bin":
            return self.binop(n.a, self.ev(n.b, scope, where), self.ev(n.c, scope, where), where)
        raise SimError(f"node {k}")

    def binop(self, op: str, a: Any, b: Any, where: str) -> Any:
        if op == "==":
            return self._eq(a, b)
        if op == "!=":
            return not self._eq(a, b)
        if op == "in":
            if isinstance(b, list):
                return any(self._eq(a, x) for x in b)
            if isinstance(b, dict):
                if not isinstance(a, str):
                    raise runtime_error("TypeError", f"{where}: `in` a map with a {wtype(a)} key")
                return a in b
            raise runtime_error("TypeError", f"{where}: `in` a {wtype(b)}")
        if op in ("<", "<=", ">", ">="):
            if not ((is_num(a) and is_num(b)) or (isinstance(a, str) and isinstance(b, str))):
                raise runtime_error("TypeError", f"{where}: {wtype(a)} {op} {wtype(b)}")
            return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]
        if op == "+":
            if is_num(a) and is_num(b):
                return a + b
            if isinstance(a, str) and isinstance(b, str):
                return a + b
            raise runtime_error("TypeError", f"{where}: {wtype(a)} + {wtype(b)}")
        if not (is_num(a) and is_num(b)):
            raise runtime_error("TypeError", f"{where}: {wtype(a)} {op} {wtype(b)}")
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            if b == 0:
                raise runtime_error("ZeroDivisionError", f"{where}: division by zero")
            return float(a) / float(b)
        if op == "//":
            if b == 0:
                raise runtime_error("ZeroDivisionError", f"{where}: division by zero")
            return a // b
        if op == "%":
            if b == 0:
                raise runtime_error("ZeroDivisionError", f"{where}: modulo by zero")
            return a % b
        raise SimError(op)

    @staticmethod
    def _eq(a: Any, b: Any) -> bool:
        if isinstance(a, bool) or isinstance(b, bool):
            return type(a) is type(b) and a == b
        if is_num(a) and is_num(b):
            return a == b
        return type(a) is type(b) and a == b

    # ── the standard library (expression functions) ──
    def func(self, name: str, args: list, where: str) -> Any:
        def need(n: int, m: int | None = None) -> None:
            if not (n <= len(args) <= (m if m is not None else n)):
                raise runtime_error("TypeError", f"{where}: {name}() takes {n}{'' if m in (None, n) else f'-{m}'} arguments, got {len(args)}")

        def string_arg(i: int) -> str:
            v = args[i]
            if not isinstance(v, str):
                raise runtime_error("TypeError", f"{where}: {name}() argument {i + 1} is {wtype(v)}, not string")
            return v

        if name == "default":
            need(2)
            return args[1] if args[0] is None else args[0]
        if name == "map.get":
            need(2)
            m, key = args
            if not isinstance(m, dict):
                raise runtime_error("TypeError", f"{where}: map.get on a {wtype(m)}")
            keys = key if isinstance(key, list) else [key]
            cur: Any = m
            for kk in keys:
                if not isinstance(cur, dict) or kk not in cur:
                    return None
                cur = cur[kk]
            return copy.deepcopy(cur)
        if name == "map.merge":
            need(2)
            if not isinstance(args[0], dict) or not isinstance(args[1], dict):
                raise runtime_error("TypeError", f"{where}: map.merge on {wtype(args[0])}, {wtype(args[1])}")
            return {**copy.deepcopy(args[0]), **copy.deepcopy(args[1])}
        if name == "map.delete":
            need(2)
            m = copy.deepcopy(args[0])
            m.pop(args[1], None)
            return m
        if name == "keys":
            need(1)
            if not isinstance(args[0], dict):
                raise runtime_error("TypeError", f"{where}: keys() of a {wtype(args[0])}")
            return list(args[0].keys())
        if name == "len":
            need(1)
            if isinstance(args[0], (str, list, dict, bytes)):
                return len(args[0])
            raise runtime_error("TypeError", f"{where}: len() of a {wtype(args[0])}")
        if name == "string":
            need(1)
            return wstring(args[0])
        if name == "int":
            need(1)
            v = args[0]
            if isinstance(v, bool):
                raise runtime_error("TypeError", f"{where}: int() of a boolean")
            if is_num(v):
                return int(v)
            if isinstance(v, str):
                try:
                    return int(v)
                except ValueError:
                    raise runtime_error("ValueError", f"{where}: int({v!r})") from None
            raise runtime_error("TypeError", f"{where}: int() of a {wtype(v)}")
        if name == "double":
            need(1)
            try:
                return float(args[0])
            except (TypeError, ValueError):
                raise runtime_error("ValueError", f"{where}: double({args[0]!r})") from None
        if name == "get_type":
            need(1)
            return wtype(args[0])
        if name == "list.concat":
            need(2)
            if not isinstance(args[0], list):
                raise runtime_error("TypeError", f"{where}: list.concat on a {wtype(args[0])}")
            return copy.deepcopy(args[0]) + [copy.deepcopy(args[1])]
        if name == "text.substring":
            need(3)
            s = string_arg(0)
            a, b = int(args[1]), int(args[2])
            return s[max(a, 0):max(min(b, len(s)), 0)]
        if name == "text.replace_all":
            need(3)
            return string_arg(0).replace(string_arg(1), string_arg(2))
        if name == "text.replace_all_regex":
            need(3)
            return re.sub(string_arg(1), string_arg(2).replace("\\", "\\\\"), string_arg(0))
        if name == "text.match_regex":
            need(2)
            return re.search(string_arg(1), string_arg(0)) is not None
        if name == "text.find_all":
            need(2)
            s, sub = string_arg(0), string_arg(1)
            return [{"index": m.start(), "match": sub} for m in re.finditer(re.escape(sub), s)] if sub else []
        if name == "text.split":
            need(2)
            return string_arg(0).split(string_arg(1))
        if name == "text.url_encode":
            need(1)
            # "Letters, digits, and the characters .-_ are not escaped" (stdlib/text/url_encode): "~" is
            return urllib.parse.quote(string_arg(0), safe="-_.").replace("~", "%7E")  # quote() never escapes ~
        if name == "text.to_lower":
            need(1)
            return string_arg(0).lower()
        if name == "text.decode":
            need(1, 2)
            v = args[0]
            return bytes(v).decode("utf-8") if isinstance(v, (bytes, bytearray)) else string_arg(0)
        if name == "time.format":
            need(1, 2)
            if not is_num(args[0]):
                raise runtime_error("TypeError", f"{where}: time.format of a {wtype(args[0])}")
            return fmt_time(float(args[0]))
        if name == "time.parse":
            need(1)
            return parse_time(args[0])
        if name == "sys.now":
            need(0)
            return self.clock.now()
        if name == "sys.get_env":
            need(1)
            return self.env.get(string_arg(0))
        if name == "json.encode_to_string":
            need(1, 2)
            return to_json(args[0])
        if name == "json.encode":
            need(1, 2)
            return to_json(args[0]).encode("utf-8")
        if name == "json.decode":
            need(1)
            v = args[0]
            if isinstance(v, (bytes, bytearray)):
                v = bytes(v).decode("utf-8")
            if not isinstance(v, str):
                raise runtime_error("TypeError", f"{where}: json.decode of a {wtype(v)}")
            def no_constant(c: str) -> Any:  # NaN / Infinity are not JSON; Python's json would accept them
                raise ValueError(f"{c} is not valid JSON")
            try:
                return json.loads(v, parse_constant=no_constant)
            except ValueError as e:
                raise runtime_error("ValueError", f"{where}: json.decode: {e}") from None
        if name == "math.floor":
            need(1)
            import math
            return math.floor(args[0])
        raise SimError(f"{where}: function {name}() is not simulated")

    # ── std calls ──
    @staticmethod
    def response_body(body: Any, headers: dict, where: str) -> Any:
        """What a call's result.body (or HttpError.body) holds: a router's bytes as Workflows parses them by
        Content-Type; anything else as the router gave it."""
        if isinstance(body, (bytes, bytearray)):
            raw = bytes(body)
            if len(raw) > RESPONSE_LIMIT:
                raise ResourceLimitError(f"{where}: an HTTP response of {len(raw)} bytes (Workflows allows {RESPONSE_LIMIT})")
            ct = headers.get("Content-Type", "").split(";")[0].strip().lower()
            if ct == "application/json" or ct.endswith("+json"):
                try:
                    return json.loads(raw, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
                except ValueError:
                    return raw
            if ct.startswith("text/"):
                return raw.decode("utf-8", "replace")
            return raw
        return body

    def _sleep(self, seconds: float) -> None:
        self.clock.advance(seconds)
        if self.on_sleep:
            self.on_sleep(self, seconds)

    def exec_call(self, fn: str, args_spec: Any, scope: Scope, where: str) -> Any:
        args = self.eval_value(args_spec or {}, scope, where)
        if not isinstance(args, dict):
            raise SimError(f"{where}: args must be a map")
        if fn in self.program and fn != "main":
            self.record("call", where=where, fn=fn)
            return self.call_sub(fn, args)
        if fn == "sys.log":
            text = args.get("text", args.get("data"))
            self.record("log", where=where, severity=args.get("severity", "INFO"), text=text)
            self.logf(f"[{fmt_time(self.clock.now())}] {args.get('severity', 'INFO')}: {text}")
            return None
        if fn == "sys.sleep":
            s = args.get("seconds")
            if not is_num(s) or s < 0:
                raise runtime_error("TypeError", f"{where}: sys.sleep seconds {s!r}")
            self.record("sleep", where=where, seconds=s)
            self._sleep(float(s))
            return None
        if fn in ("http.get", "http.post", "http.patch", "http.delete", "http.put", "http.request"):
            method = fn.split(".")[1].upper() if fn != "http.request" else str(args.get("method", "GET")).upper()
            bad = set(args) - {"url", "method", "query", "headers", "body", "auth", "timeout", "private_service_name"}
            if bad:
                raise SimError(f"{where}: {fn} args {sorted(bad)}")
            url = args.get("url")
            if not isinstance(url, str):
                raise runtime_error("TypeError", f"{where}: {fn} url is {wtype(url)}")
            query = args.get("query") or {}
            if not isinstance(query, dict):
                raise runtime_error("TypeError", f"{where}: query must be a map")
            for qk, qv in query.items():
                if isinstance(qv, (dict, list)):
                    raise runtime_error("TypeError", f"{where}: query {qk} is a {wtype(qv)}")
            timeout = args.get("timeout")
            if timeout is not None and (not is_num(timeout) or timeout > 1800):
                raise SimError(f"{where}: http timeout {timeout!r} (Workflows allows at most 1800 s)")
            auth = args.get("auth")
            code, body, headers = self.http(method, url, {k: wstring(v) for k, v in query.items()},
                                            dict(args.get("headers") or {}), copy.deepcopy(args.get("body")), auth)
            headers = {canonical_header(k): str(v) for k, v in (headers or {}).items()}
            body = self.response_body(body, headers, where)
            self.record("http", where=where, method=method, url=url, query=query, code=code,
                        auth=(auth or {}).get("type") if isinstance(auth, dict) else None)
            if code >= 400:
                raise WfError({"code": code, "message": f"HTTP server responded with error code {code}", "body": body,
                               "headers": headers, "tags": ["HttpError"]})
            return {"code": code, "body": body, "headers": headers}
        if fn == "googleapis.secretmanager.v1.projects.secrets.versions.accessString":
            self.record("secret", where=where, secret=args.get("secret_id"))
            return self.secrets(args)
        raise SimError(f"{where}: call {fn} is not simulated")
