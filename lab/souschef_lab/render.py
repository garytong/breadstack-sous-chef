"""render — the bytes the service puts in front of the model, rebuilt in
Python byte for byte (docs/LAB.md §4, "Prompt parity").

The ONE place the lab renders a user block or a tool_result's content:
collect.py (the browser emulator's logged `messages`) and dataset.py
(`messages_from_events`, rows rebuilt from a trace) both import from here,
so a training row never carries bytes the service never sends.

What the service does (read-only references):

* `service/internal/prompt/build.go` ContextBlock — the turn's first user
  text block is ``"<context>\\n" + json.Compact(raw) + "\\n</context>\\n" +
  strings.TrimSpace(text)``; a missing / ``null`` context gives the bare
  text. ``json.Compact`` strips whitespace from the RAW bytes the browser
  sent and keeps their key order (it never re-encodes, never HTML-escapes),
  so the compact form of what the collector sends is what the model reads —
  the collector serialises its request body with the same encoder and the
  same options as ``compact_json`` (see ``collect.Client``).
  On a route with `fields_context` on, `prompt.UserText` puts the turn's
  candidate-fields block (`internal/fields`, computed for the lab by
  fields.py through the service's own CLI) between `</context>\n` and the
  text, followed by one newline.
* `service/internal/loop/pending.go` browserResult — a browser op's
  tool_result is Go ``json.Marshal`` of ``map[string]any{"ok": true,
  "written"?, "warnings"?, "hidden"?, "sectionId"?, "slug"?}`` (map keys SORTED;
  ``slug`` — the new page's final slug — only for a ``create_page``;
  empty lists omitted; ``hidden`` = the browser's ``[{key, by, byValue}]``
  next to ``W_HIDDEN:<key>`` — shared contract 4 — each entry a sorted map,
  entries without a string key/by dropped, a missing byValue ``null``,
  `loop/hidden.go` parseHidden); a failure is the ``api.ToolError`` struct ``{code, message,
  key?}`` with ``is_error``; a ``pick_asset`` answer is the ``api.Asset``
  struct ``{public_url, alt?, srcset?}`` (``srcset`` is a RawMessage,
  passed through compacted), or ``{"code":"E_ASSET",…}`` with ``is_error``
  when no asset came back. Go's encoder is compact, writes UTF-8 as is,
  and escapes ``<``, ``>``, ``&`` (HTML-safe) and U+2028/U+2029.

Verified against ``fixtures/prompt/real/<capture>/model-requests.jsonl`` (the real
binary's own POST /v1/messages bodies) by ``tests/test_render.py``.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

SERVER_TOOLS = frozenset(("lookup_fields", "describe_target", "report_unsupported"))
PICK_ASSET = "pick_asset"
CREATE_PAGE = "create_page"
E_ASSET_NO_CHOICE = ("E_ASSET", "no asset was chosen")
E_OP_NO_DETAILS = ("E_OP", "the browser reported a failure without details")


# --- Go encoding/json in Python ---------------------------------------------------------------------------------------

def compact_json(value: Any) -> str:
    """``json.Compact`` of what Python would send: no whitespace, insertion
    order kept, UTF-8 as is, nothing HTML-escaped. This is also how
    ``collect.Client`` serialises the request body, so the context bytes
    the service compacts are these."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _go_escape(s: str) -> str:
    """Go's HTML-safe escaping: only inside strings can these appear in
    JSON text, so a global replace is exact."""
    return (s.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def go_marshal(value: Any, *, sort_keys: bool = True) -> str:
    """Go ``json.Marshal``: compact, HTML-escaped; map keys sorted
    (``sort_keys=True``, a ``map[string]any``) or in struct order
    (``sort_keys=False``, a struct or a RawMessage passed through).
    Numbers: Go prints an integral float as ``1`` where Python prints
    ``1.0``; the service never marshals a float the lab produces (paths,
    warnings, ids and URLs are strings)."""
    return _go_escape(json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=sort_keys))


# --- Go's strings.TrimSpace ---------------------------------------------------------------------------------------------

# unicode.IsSpace: Go's White_Space set. Python's str.strip() also trims U+001C–U+001F; Go does not.
_GO_SPACE = frozenset("\t\n\v\f\r \u0085            "
                      "     　")


def go_trim_space(s: str) -> str:
    """Go's ``strings.TrimSpace``: what the service trims the merchant's text
    and the candidate-fields block with (prompt.UserText, loop.candidateFields).
    ``str.strip()`` would also cut U+001C–U+001F, which Go keeps."""
    a, b = 0, len(s)
    while a < b and s[a] in _GO_SPACE:
        a += 1
    while b > a and s[b - 1] in _GO_SPACE:
        b -= 1
    return s[a:b]


# --- the user block -----------------------------------------------------------------------------------------------------

def context_block(context: dict | None, text: str, candidates: str = "") -> str:
    """prompt.UserText (prompt.ContextBlock when ``candidates`` is empty):
    the context compacted inside <context> tags on their own lines, then —
    on a route with ``fields_context`` on — the turn's candidate-fields block
    (fields.py; trimmed, on its own lines), then the trimmed merchant text.
    ``None`` = no context (the service accepts an absent/null context; the
    block then opens the message); ``{}`` is still an object and is rendered
    as one, exactly as the collector sends it."""
    text = go_trim_space(text or "")
    candidates = go_trim_space(candidates or "")
    if candidates:
        candidates += "\n"
    if context is None:
        return candidates + text
    return "<context>\n" + compact_json(context) + "\n</context>\n" + candidates + text


def user_message(context: dict | None, text: str, candidates: str = "") -> dict:
    """The turn's user message as an Anthropic block list."""
    return {"role": "user", "content": [{"type": "text", "text": context_block(context, text, candidates)}]}


def turn_candidates(row: dict | None) -> str:
    """The candidate-fields block a logged turn was SENT with, as the
    collector recorded it (``row.candidate_fields``; collect on a route
    whose ``fields_context`` the service says is on): "" for a row without one — an Anthropic route,
    a turn nothing matched, a log from before the block."""
    return str((row or {}).get("candidate_fields") or "")


def turn_context(case: dict | None) -> dict:
    """The context object the collector SENDS for a case and the one every
    rebuild renders: ``case.context``, or ``{}`` when the case has none. The
    lab never sends a null context (prompt.ContextBlock renders null as the
    bare text but an empty object as ``<context>\\n{}\\n</context>``), so the
    object on the wire, in the logged row (``row.context``) and in the
    rebuilt user block is one and the same."""
    ctx = (case or {}).get("context")
    return ctx if isinstance(ctx, dict) else {}


def turn_inputs(row: dict, case: dict | None = None) -> tuple[dict | None, str]:
    """``(context, text)`` a logged row's turn was asked with: the row's own
    ``context``/``text`` when the collector recorded them (the very objects
    it sent; a trace row's ``context`` may be ``None``), else the case's
    (``turn_context``). Never ``or``-chained: an empty context is an object
    the service rendered as one."""
    case = case or {}
    context = row["context"] if "context" in row else turn_context(case)
    text = row["text"] if "text" in row and row["text"] is not None else case.get("request")
    return context, str(text or "")


# --- tool results -------------------------------------------------------------------------------------------------------

def tool_result_content(name: str, item: dict) -> tuple[str, bool]:
    """loop.browserResult for one posted tool-results item
    (``{toolCallId, ok, error?, warnings?, written?, hidden?, sectionId?, slug?, asset?}``):
    the content string the model reads and whether it is an error. ``slug``
    (create_page's final page slug) is forwarded for ``create_page`` only."""
    if not item.get("ok"):
        err = item.get("error")
        if not isinstance(err, dict) or not err:
            code, message = E_OP_NO_DETAILS
            err = {"code": code, "message": message}
        out = {"code": str(err.get("code") or ""), "message": str(err.get("message") or "")}
        if err.get("key"):
            out["key"] = str(err["key"])
        return go_marshal(out, sort_keys=False), True
    if name == PICK_ASSET:
        asset = item.get("asset")
        if not isinstance(asset, dict) or not asset.get("public_url"):
            code, message = E_ASSET_NO_CHOICE
            return go_marshal({"code": code, "message": message}, sort_keys=False), True
        out = {"public_url": str(asset["public_url"])}
        if asset.get("alt"):
            out["alt"] = str(asset["alt"])
        if asset.get("srcset") is not None:  # a JSON null is normalised to absent (api.Asset.Validate)
            out["srcset"] = asset["srcset"]
        return go_marshal(out, sort_keys=False), False
    out: dict[str, Any] = {"ok": True}
    if item.get("written"):
        out["written"] = [str(w) for w in item["written"]]
    if item.get("warnings"):
        out["warnings"] = [str(w) for w in item["warnings"]]
    hidden = wire_hidden(item.get("hidden"))
    if hidden:
        out["hidden"] = hidden
    if item.get("sectionId"):
        out["sectionId"] = str(item["sectionId"])
    if name == CREATE_PAGE and item.get("slug"):
        out["slug"] = str(item["slug"])  # browserResult: `slug` only for create_page (api.ToolResult.Slug)
    return go_marshal(out, sort_keys=True), False


def wire_hidden(hidden: Any) -> list[dict]:
    """The ``hidden`` member as the service reads it (loop/hidden.go
    parseHidden) and forwards it to the model: a list of ``{key, by,
    byValue}``; anything else is dropped, entry by entry (a string key and
    by are required, a missing byValue is null). Also the shape the
    browser posts (launcher client/turns.js wireHidden): the engine's own
    ``rule``/``gate`` stay out."""
    if not isinstance(hidden, list):
        return []
    out = []
    for h in hidden:
        if not isinstance(h, dict):
            continue
        key, by = h.get("key"), h.get("by")
        if not isinstance(key, str) or not key or not isinstance(by, str) or not by:
            continue
        out.append({"key": key, "by": by, "byValue": h.get("byValue")})
    return out


def tool_result_block(name: str, item: dict) -> dict:
    """One Anthropic ``tool_result`` block (``is_error`` only when true)."""
    content, is_error = tool_result_content(name, item)
    block = {"type": "tool_result", "tool_use_id": str(item.get("toolCallId") or ""), "content": content}
    if is_error:
        block["is_error"] = True
    return block


def pick_asset_call(suspended: dict) -> dict:
    """The `pick_asset` tool_use the model emitted, rebuilt from the
    `turn.suspended` payload: the service streams no `tool.call` for it
    (loop/turn.go: "the turn.suspended event carries the call"), but the
    event copies the call's `purpose` and `asset_type` (pending.go
    suspendedEvent). The tool schema (fixtures/prompt/tools.json, `strict`)
    REQUIRES both `purpose` and `asset_type`, so the model always sent both
    and both are rebuilt — `assetType` defaults to "image" in the event
    exactly as the schema's only enum member. Keys sorted, as the service
    stores the model's input (Go re-marshals the map) and as the captured
    history shows it (fixtures/prompt/real/pick_asset.*)."""
    inp: dict[str, Any] = {"asset_type": str(suspended.get("assetType") or "image"),
                           "purpose": str(suspended.get("purpose") or "")}
    return {"type": "tool_use", "id": str(suspended.get("toolCallId") or ""), "name": PICK_ASSET, "input": inp}


def tool_results_message(items: Iterable[dict], names: dict[str, str]) -> dict:
    """The user message answering one batch: every result in ONE message, in
    call order (model.ToolResultsTurn). ``names`` maps tool call id → tool
    name (``pick_asset`` renders differently)."""
    blocks = [tool_result_block(names.get(str(it.get("toolCallId")), ""), it) for it in items]
    return {"role": "user", "content": blocks}


# --- a whole turn from its events ----------------------------------------------------------------------------------------

def messages_from_events(events: list[dict], context: dict | None, text: str, candidates: str = "") -> list[dict]:
    """Anthropic-shaped messages rebuilt from a turn's events (LAB.md §4):
    user = the context block (with ``candidates``, the turn's
    candidate-fields block, when it had one); assistant = joined ``text.delta`` + one
    ``tool_use`` per ``tool.call`` of the stream; one user message of
    ``tool_result`` blocks per posted batch (``tool.result`` events, the
    posted item rendered as the service renders it). Nothing is fabricated:
    a tool the service answered itself (``report_unsupported``, the server
    tools) has no result in the trace and gets none here."""
    from .verify import events_of  # lazy: verify imports nothing from here, but keep the import graph one-way

    msgs: list[dict] = [user_message(context, text, candidates)]
    blocks: list[dict] = []
    text_parts: list[str] = []
    results: list[dict] = []
    names: dict[str, str] = {}

    def flush_assistant() -> None:
        nonlocal blocks, text_parts
        content = []
        joined = "".join(text_parts)
        if joined:
            content.append({"type": "text", "text": joined})
        content += blocks
        if content:
            msgs.append({"role": "assistant", "content": content})
        blocks, text_parts = [], []

    def flush_results() -> None:
        nonlocal results
        if results:
            msgs.append(tool_results_message(results, names))
            results = []

    for e in events_of(events):
        n = str(e.get("event") or e.get("type") or "")
        d = e.get("data") or {}
        if n == "text.delta":
            flush_results()
            text_parts.append(str(d.get("text") or ""))
        elif n == "tool.call":
            flush_results()
            tid = str(d.get("id") or "")
            names[tid] = str(d.get("name") or "")
            blocks.append({"type": "tool_use", "id": tid, "name": names[tid],
                           "input": d.get("input") if isinstance(d.get("input"), dict) else {}})
        elif n == "turn.suspended":
            flush_results()
            call = pick_asset_call(d)
            names[call["id"]] = PICK_ASSET
            blocks.append(call)
            flush_assistant()
        elif n in ("turn.waiting", "turn.done", "turn.error"):
            flush_assistant()
        elif n == "tool.result":
            results.append({k: v for k, v in d.items() if k != "for_seq"})
    flush_assistant()
    flush_results()
    return msgs


# --- comparing with the wire ----------------------------------------------------------------------------------------------

def flatten(messages: Iterable[dict]) -> list[tuple]:
    """A message list reduced to what the model reads, in order: ``(role,
    "text", text)``, ``(role, "tool_use", id, name, canonical input)``,
    ``(role, "tool_result", tool_use_id, is_error, content text)``. Accepts
    both the lab's block shape (content a string, ``is_error`` only when
    true) and the SDK's wire shape (content a list of text blocks,
    ``is_error`` always present), so a rebuilt turn can be compared with a
    captured POST /v1/messages body."""
    out: list[tuple] = []
    for m in messages:
        role = str(m.get("role") or "")
        content = m.get("content")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": str(content or "")}]
        for b in blocks:
            t = b.get("type")
            if t == "text":
                out.append((role, "text", str(b.get("text") or "")))
            elif t == "tool_use":
                out.append((role, "tool_use", str(b.get("id") or ""), str(b.get("name") or ""),
                            json.dumps(b.get("input") or {}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))))
            elif t == "tool_result":
                c = b.get("content")
                if isinstance(c, list):
                    c = "".join(str(x.get("text") or "") for x in c if isinstance(x, dict) and x.get("type") == "text")
                out.append((role, "tool_result", str(b.get("tool_use_id") or ""), bool(b.get("is_error")), str(c or "")))
    return out
