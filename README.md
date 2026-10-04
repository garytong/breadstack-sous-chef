# Sous Chef service

**The service decides, the browser applies.** Sous Chef is a Go service that runs the agent loop for the Pantry site editor. It talks to a model in the Anthropic Messages API shape, turns the model's tool calls into events, and streams them to the browser. The browser applies data ops locally with the generic ops engine already built in cookie-cutter (`launcher/client/src/sousChef/`), posts the results back, and the loop continues. The service never holds site state.

This README is the implementation contract. Numbers cited as FACTS were measured on `service/testdata/manifest.snapshot.json` (1,248,567 bytes pretty, 758,693 compact; 19 sections, 12 globalPanels, 17 pages, 11 details, 2,094 fields, 69 lists).

Layout: `service/` (Go module `github.com/breadstack/sous-chef`), `infra/` (Terraform), `lab/` (Phase 5 placeholder), `docker-compose.yml` (local Postgres), `Makefile`, `.env.example`. Only the user commits.

---

## 1. HTTP API

The launcher reverse-proxies `/api/sous-chef/*` → `/v1/*`, strips incoming `Authorization`, `X-User-Email` and `Cookie`, then sets `Authorization: Bearer <service key>`, `X-User-Email` (plain-text email from its `breadstack_session` cookie → attribution only), `X-Launcher-Build`, `X-Manifest-Hash` (FACTS 1 §3). No JWT verification here (user decision).

**Auth.** Bearer key checked with `crypto/subtle.ConstantTimeCompare` against each key in `SERVICE_KEYS` (comma-separated; rotation = two keys live). 401 on miss. `/healthz`, `/readyz`, `/version` are open.

| Route | Behaviour |
|---|---|
| `PUT /v1/manifests/{hash}` | Idempotent register. Body = manifest JSON (≤ 8 MiB); server recomputes the hash (§1.1), mismatch → 400 `hash_mismatch`. Returns `{hash, fields, lists, sections, globalPanels, pages, details, generated_from, diff}`; `diff` = `{prev_hash, added_fields[], removed_fields[], added_lists[], removed_lists[]}` vs the previous manifest of the same build line (`X-Launcher-Build` starting `v` = prod, else int), or `null`. 201 first time, 200 after. |
| `POST /v1/manifests` | Same as `PUT`, but the **server computes the hash**: body = manifest JSON (≤ 8 MiB), response = the identical `{hash, fields, …, diff}` body, 201/200 by the same rule. For a launcher that cannot reproduce §1.1 itself; the returned `hash` is what it sends as `X-Manifest-Hash` from then on. Unparsable or invalid manifest → 400 `bad_request`. (`registry.ComputeAndRegister`, the `api.ComputeRegistrar` the handler prefers; `manifest.HashBytes` + `Register` is its fallback.) |
| `GET /v1/manifests/{hash}` | `{hash, generated_from, counts, first_seen, index}`; `?body=1` adds the raw manifest. |
| `POST /v1/turns` | Start a turn (§1.2); streams SSE. `siteId`, `storeId`, `manifestHash`, `buildId` are all required (400 `bad_request` when missing — `storeId` gates trace opt-out, so a turn without one is refused rather than silently traced). `conversationId` is optional: omitted, the turn continues the site's open conversation (created when there is none); named, it must exist, belong to `siteId` and be open (400 `bad_request`: "no such conversation" — also for another site's — or "conversation has ended"). A named conversation is subject to the same 40-message rotation as the open one (§3, §4): when it already holds 40 messages the service **ends it and starts a fresh one**, the turn lands there, and `turn.done` carries `conversationRotated:true` with the new `conversationId` — the caller must not assume the turn ran on the conversation it named. |
| `POST /v1/turns/{turnId}/tool-results` | Continue the **same** turn. Body `{siteId?, storeId?, results:[{toolCallId, ok, error?:{code,message,key?}, warnings?, written?:[paths], sectionId?, asset?:{public_url, alt?, srcset?}}]}`. `siteId`/`storeId` are the proxy's tenant assertions (§11): each one is verified whenever present — the turn's conversation must belong to it or the answer is 404 `turn_not_found` (same as for no such turn — existence is not revealed); a body with neither is accepted and logged once per turn (a `siteId`-less body is noted once per turn either way). Returns the next SSE stream. 409 `turn_not_waiting` (with `status`) unless status is `waiting_tools`/`suspended` — `status:"superseded"` means a newer turn started for the site, or the conversation was ended (§4), and the browser drops these results silently; 404 after retention. |
| `GET /v1/conversations/{siteId}` | `{conversationId, storeId, messages:[{seq,role,content}], turns:[...]}` — last 200 messages, `?before=seq`. |
| `DELETE /v1/conversations/{siteId}` | Supersedes the site's live turns (§4: their late tool-results are 409 `status:"superseded"`, a running one ends `turn.error turn_expired`), then sets `ended_at`; the next turn starts fresh. Rows stay for retention. |
| `POST /v1/feedback` | `{turnId, siteId?, storeId?, kept_paths?:[], undone?:bool, saved?:bool, note?}` merged into `traces.feedback` (minus the ids). 204. `siteId`/`storeId` as on tool-results: each is verified whenever present; a mismatch, or either id with an unknown `turnId`, is 404 `turn_not_found`; a body with neither is accepted and logged once per turn. |
| `PUT`/`DELETE /v1/stores/{storeId}/trace-optout` | Trace opt-out flag. |
| `GET /healthz`, `/readyz`, `/version` | process up; Postgres pings + default route configured (else 503); `{build_id, go, routes}`. |

**Body limits.** Manifest bodies (`PUT`/`POST /v1/manifests`) ≤ 8 MiB. Every other JSON body (`/v1/turns`, `tool-results`, `/v1/feedback`) ≤ **256 KB** (`MaxBodyBytes`, 262,144 bytes); larger → 413. The opt-out `PUT`/`DELETE` take no body (whatever is sent is ignored). `/v1/feedback` additionally caps the merged `traces.feedback` document per turn at 16 KB.

Errors are `{error, message, ...}`: `unauthorized`, `bad_request`, `unknown_manifest` (409, carries `hash`; the proxy uploads then retries), `hash_mismatch`, `turn_not_found`, `turn_not_waiting`, `model_unavailable` (503), `busy` (503), `store_unavailable` (503: storage failed while a Start was deciding whether its row was still its own — nothing of the turn is stored, retry; §4 "Storage failures"). Opt-out only disables trace storage, never fails a turn; the opt-out lookup fails **closed** (a DB error while reading `store_optout` skips the trace and logs).

### 1.1 Manifest hash

`hash = hex(sha256(canonical(manifest minus generated_from and coverage)))`; canonical = sorted keys, no whitespace, Go `encoding/json` number formatting. Rationale (FACTS 1 §2): `generated_from.at` changes every build, which would mint a new hash when nothing else changed. `generated_from` is stored beside the hash. Every turn carries `X-Manifest-Hash`; unknown → 409 `unknown_manifest`.

**Who computes it.** Not the browser: JavaScript cannot reproduce the canonical form (`JSON.stringify` orders integer-like keys first and leaves U+2028/U+2029 raw). The launcher has two options: `POST /v1/manifests` and take `hash` from the response, or — since the launcher server is Go — compute it locally with exactly this recipe (`manifest.CanonicalHash` in `internal/manifest/canonical.go` is the same code, tested against the fast path):

```go
// hash of a manifest document, README §1.1
func manifestHash(data []byte) (string, error) {
	var doc map[string]any
	if err := json.Unmarshal(data, &doc); err != nil { // numbers → float64
		return "", err
	}
	delete(doc, "generated_from")
	delete(doc, "coverage")
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false) // keep < > & raw; encoding/json already sorts map keys
	if err := enc.Encode(doc); err != nil {
		return "", err
	}
	sum := sha256.Sum256(bytes.TrimSuffix(buf.Bytes(), []byte("\n"))) // Encoder appends one '\n'
	return hex.EncodeToString(sum[:]), nil
}
```

Test vector: `service/testdata/manifest.snapshot.json` (1,248,567 bytes) → `727d151b5d6b04cd68b73199ab0b3bdf061263f9623f4f6eda00a79137c06f38` (`TestCanonicalHashRecipeMatchesHashBytes`, `TestRegisterComputed_SameResultAsPut`; also cross-checked against Python `json.dumps(sort_keys=True, separators=(',',':'), ensure_ascii=False)`). The hash covers the whole document minus the two members, including members the service does not understand. A Go version older than 1.22 escapes `\b`/`\f` as `\u0008`/`\u000c` and would disagree — the launcher must build with ≥ 1.22 (the service uses 1.24).

### 1.2 Turn request and SSE events

```json
{"conversationId":"opt — omitted = site's open conversation, or a new one",
 "siteId":"req","storeId":"req","manifestHash":"req","buildId":"req",
 "context":{"page":"home","template":"cozy","layout":"default",
   "sections":[{"id":"hero-1","type":"hero","title":"Hero","enabled":true}],
   "categories":[{"slug":"cookies","name":"Cookies"}],
   "customPages":[{"slug":"about","title":"About"}],
   "nav":[{"label":"Home","href":"/"}],
   "locations":[{"id":"loc-1","name":"Main St"}]},
 "text":"make the hero title bigger","model":"opt route name, allowlisted"}
```

`Accept: text/event-stream` (default) streams `event:`/`data:` lines; `Accept: application/json` returns `{turnId, events:[...]}` instead. Every event except `heartbeat` carries `turnId` and `seq`; `seq` is per turn and continues across the turn's streams (a `tool-results` continuation picks up where the previous stream stopped, so `tool.call` ids and trace `seq` never collide).

| event | data |
|---|---|
| `text.delta` | `{text}` |
| `tool.call` | `{id, name, input}` |
| `turn.suspended` | `{reason:"pick_asset", toolCallId, purpose, assetType}` |
| `turn.waiting` | `{toolCallIds:[...]}`; stream closes until tool-results arrive |
| `turn.done` | `{usage:{input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens, calls}, stop:"end_turn"\|"max_calls"\|"max_tokens"\|"refusal"}` |
| `turn.error` | `{code, message}` |
| `heartbeat` | `{}` every 15 s while nothing else is being sent (model call in flight); the SSE writer emits it, so it has **no** `turnId`/`seq` |

Within one stream: `text.delta`×N, then `tool.call`×N + `turn.waiting`, or `turn.suspended`, or `turn.done`/`turn.error`. The browser applies the whole batch and posts one tool-results body.

**`tool.call` → browser op.** The mapping is the identity: `name` **is** the engine op (`set_fields`, `edit_list`, `reset`, `restore_default`, `set_page_layout`, `reset_pinned_section`), `input` **is** the op's fields exactly as the engine's `validate.js` expects them (`target`, `values`, `assets`, `list`, `action`, `index`, `item`, `order`, `page`, `sectionId`, …), and `target` is passed through as-is (the host only fills a missing `page` from `ctx.currentPage`). The browser runs `applyOps([{op: name, ...input}])` and answers each `id` once in `results[].toolCallId`. `report_unsupported` is also surfaced as a `tool.call` for display only — the service has already answered it, the browser must **not** post a result for it. `pick_asset` is never applied by the engine: it surfaces as `turn.suspended` and is answered with `asset`.

**`pick_asset` in a batch.** The model is instructed to call `pick_asset` alone. If a response still carries several `pick_asset` calls, only the **first** suspends the turn (`turn.suspended.toolCallId`, `pending_tool_calls.suspended_id`); every further `pick_asset` in that batch is answered server-side with `tool_result is_error:true` "only one pick_asset per batch: call pick_asset alone, in its own response, after the other edits are confirmed" (`E_OP`) and is not emitted to the browser. Data ops in the same batch are still emitted as `tool.call` and are owed results alongside the asset in the one `tool-results` body.

## 2. Tools exposed to the model

Generated per manifest hash by `internal/toolgen`, cached, in the fixed order below (the tool list is part of the cached prefix). Data-op schemas are `strict: true`, `additionalProperties:false`; `tool_choice` is always `auto` (forced `any`/`tool` is rejected by Claude Sonnet 5.5 / Opus 5.5).

**Strict-schema budget.** The Messages API caps, per request and summed over **every** tool sent with `strict:true` (plus any JSON output schema), the **optional parameters at 24** — each property not listed in `required`, counted through nested objects — and **union-typed parameters at 16** (`anyOf`, `type:[..,"null"]`). Over either limit every call is a 400 `invalid_request_error` ("Schemas contains too many optional parameters (N) … (limit: 24)") before any token is billed; the fake-model e2e does not enforce it, which is how int launched at 29/24 on 2026-10-04. The `target` object costs 4 optionals per strict tool that embeds it, so `lookup_fields` (11) and `describe_target` (6) — all-optional server queries that `Dispatch` validates anyway, answering `E_OP`/`E_TARGET`/`E_LIST` — are `strict: false` like `set_fields`/`edit_list` (open maps); the six strict tools spend 10 of the 24 and 0 of the 16 (12 until `keys` became required on `reset`/`restore_default`). `TestStrictOptionalParameterBudget` counts both the way the API does (it reproduces the 29) and fails under 4 optionals of headroom. `minItems` is the one array constraint strict mode accepts, and only as 0 or 1 — `keys` uses `minItems: 1`.

**`reset` / `restore_default` require `keys`.** The engine's keyless form acts on every field of the target (`validate.js` `validateResetLike`, `op.keys === undefined`: a section loses every setting, a page its options including `published`). The first int collection (300 cases, Claude Sonnet) showed the model reaching for that shape by default — 289 cases cleared a whole page/panel/section, followed by "I made a mistake. I called reset without any keys…" — so the clear-all form is no longer offered: `keys` is required (`minItems: 1`), the descriptions say exactly which keys are touched and that clearing a whole target is not available ("list every key you mean"), and the loop refuses a `reset`/`restore_default` tool call whose `keys` is missing, empty or not all non-blank strings with an `is_error` `E_OP` ("reset needs keys …") answered to the model and **no `tool.call` emitted** (`toolgen.ValidateBrowserOp`, called from the loop's dispatch — belt and braces over the schema). Making optional members required-with-`null` instead was rejected: nullable `target` members would cost 4 union types per embedding (16 for the four target-bearing tools = the whole union budget) and the browser's `resolveSectionTarget` requires `page` to be a string (a `null` page is `E_TARGET` unless the host fill treats it like absent), whereas `storeId`/`panel` already accept `null` (`targets.js`, `!= null`).

| tool | runs | input |
|---|---|---|
| `set_fields` | browser | `{target, values:{key:val}, assets?:{key:{alt?,srcset?,mobile?}}}` |
| `edit_list` | browser | `{target, list, action: add\|remove\|reorder\|update\|duplicate, index?: int\|[int,int], item?, order?, sectionId?}`; `order` = the COMPLETE new order (every current row once): indexes for plain lists, row ids for id-keyed lists (see below) |
| `reset` | browser | `{target, keys}` (keys required, ≥1) → writes `unset_value` of exactly those keys; keyless = `E_OP`, not emitted |
| `restore_default` | browser | `{target, keys}` (keys required, ≥1) → writes `default` of exactly those keys; keyless = `E_OP`, not emitted |
| `set_page_layout` | browser | `{page, layout}` |
| `reset_pinned_section` | browser | `{page, sectionId}` |
| `pick_asset` | suspends | `{purpose, asset_type:"image"}`; resumes with the URL in the tool result |
| `lookup_fields` | server | `{target?, container?, query?, keys?, list?, limit=40, help?}` (not strict: all-optional, see budget) |
| `describe_target` | server | `{target?, container?}` (not strict, see budget) |
| `report_unsupported` | server | `{reason, manual_hint?}`; also emitted as `tool.call` so the UI can show it |

`target` is the engine's union (FACTS 1 §1): `{kind:"section", page, sectionId, storeId?}` | `{kind:"global", panel?}` | `{kind:"branding"}` | `{kind:"page", page}`. Descriptions say `page` may be omitted for the viewed page (the host fills it from `ctx.currentPage`).

**`edit_list.order` spells out the engine's two reorder rules** (engine README rule 7, `validate.js` reorder / `orderIdsError`). int-run2 (300 cases, Claude Sonnet) had exactly three `E_ORDER` refusals, two shapes: a *partial* order — `[1, 0]` for a 3-row `footerCustomColumns[].links`, "order must be a permutation of 0..2" — and *indexes on an id-keyed list* — `[1, 0]` on the products page's `filterOrder`, "unknown id(s) 1, 0; order must be a permutation of the list's ids (category, brand, price, …)". The description now says: list EVERY current row exactly once (a 3-row list needs 3 entries; move one row by writing the full new order, e.g. `[1, 0, 2]`); plain lists take 0-based indexes; **id-keyed lists take the row ids, never indexes** — a list the manifest marks with `orders` (`filterOrder`, `headerNavOrder`, `footerNavOrder`, `paymentMethodsOrder`, `cardChipOrder`, `sortOptionsOrder`, …; `describe_target` / `lookup_fields` return the flag as `orders`) and a page's `sections` list (section ids); an `E_ORDER` reply names the valid ids / row count, so the fix is to resend the full order. The service holds no site state, so `describe_target` tells the model *which* lists are id-keyed, not their current rows — those come from the turn context (`sections`) or the engine's refusal. `edit_list` is `strict: false`, so the longer text costs nothing on the strict budget (still 10/24 optional, 0/16 union; `TestEditListOrderDescription`). The lab's frozen copy of the tool list is `lab/fixtures/prompt/tools.json` (hand-synced from `prompt.SDKTools` wire bytes; the real captures under `lab/fixtures/prompt/real/` must be re-run with `lab/tests/capture_real.py` to carry the new bytes, LAB.md §9).

**Compact server-tool outputs.** `lookup_fields` returns per field `key, type, label, enum(+labels when ≤12), min/max/step, default, unset_value, if/hide gates, ro (derived_from), ref, keyed_by, companions, list{key, ops, orders, item_ref}`; never `source` or `path`, `help` only on request. Matching: key prefix, camelCase stem, label substring, or explicit `keys`. Capped at `limit` and 12 KB — a slim field averages 110 bytes, an unfiltered perksCards reaches 22.9 KB (FACTS 1 §2). `details.<Panel>` lookups strip the `item.` prefix. `describe_target` returns the container's lists (`ops`/`orders`/`item_ref`), key stems, `mirror`, `presets`, and the pinned-section id for page targets. The model never sees the full manifest: it gets the index (§3) and calls `lookup_fields` before writing unfamiliar keys.

## 3. Prompt

Render order: `tools` → `system[0]` → `system[1]` → messages.

- **`system[0]` behavioural rules** — frozen, `cache_control: {type:"ephemeral"}`. Salvaged from the old branch (FACTS 2 §9), tool names replaced: preamble (edit only this site's configuration via tools; never claim to write code, run commands, reach external systems); stay on topic; smallest set of calls, then a brief plain-language confirmation; changes go to a live preview the merchant saves — never "permanent" or "published"; `pick_asset` and wait for the URL; PAGE CONTEXT (omit `page` for the viewed page, pass one only when named, never assume home); `report_unsupported` instead of guessing, no fabricated mailto; never invent keys, pages or slugs. Engine rules added: one structural `edit_list` (add/remove/reorder/duplicate) per call; `hero`/`heroBanner` top-level keys mirror `slides[0]` — edit the slide via `edit_list update` on `slides` index 0; after `pick_asset`, URL in `values` **and** `alt`/`srcset` in `assets`; dotted keys for keyed fields (`locationOverrides.<locId>.<f>`, `categories.<slug>.<f>`); `derived_from` fields are read-only; `lookup_fields` before any key not already confirmed in this conversation; two or more matching controls → ask in one sentence; a failed op (`E_*`) is explained and retried at most once. `record_knowledge` is dropped (no knowledge base in v1).
- **`system[1]` manifest index** — per hash, `cache_control: {type:"ephemeral"}`. One line per container (`s:heroBanner f=2 lists=slides[add|remove] slides[].bullets[…]->HeroBulletDetailsPanel pages=home mirror=slides[0]`), key stems per container, target/op grammar, error codes, pinned-ids table. Measured ≈ 3k tokens (4,038 + 3,922 bytes + fixed text); the full key list (35,926 bytes ≈ 9–10k tokens) is excluded (FACTS 1 §2). `reviews`/`testimonials` (byte-identical), `hero`/`heroBanner`, `blog`/`blogSection` collapse to alias lines.
- **Per-turn context** — first text block of the turn's user message: `<context>` JSON (§1.2) then the merchant text. It sits after the cached prefix. Not a mid-conversation `system` message, because messages-compatible backends don't accept that role.
- **History** — append-only, stored in `messages`, replayed verbatim: every `tool_use`/`tool_result` **and every `thinking`/`redacted_thinking` block** (the assistant `content` array is stored exactly as the model returned it — the API requires preserved thinking back on the next call of the same route; a `messages-compatible` route with `thinking` disabled — the default, §5 — drops the stored `thinking`/`redacted_thinking` blocks from the history it re-sends, at send time, not at rest, so the rendered history matches the SFT rows' empty think blocks; `thinking:"enabled"` replays them). Never edited or compacted in v1; at 40 messages the service rotates to a new conversation (`conversationRotated:true` on the turn). `thinking` text is never written to **traces** (thermometer rule) — `messages` is the only table that holds it, and `GET /v1/conversations` returns it; the proxy decides whether to show it.


## 4. Agent loop (`internal/loop`)

From thermometer `chat.go` (FACTS 2 §1); `maxCalls = 8` per turn.

1. Request = prefix(hash) + stored history + this turn's messages.
2. Call the route (§5) under `context.WithTimeout(60s)`; heartbeat every 15 s.
3. Stream text as `text.delta`; collect `tool_use` blocks; parse inputs with `encoding/json` only.
4. Answer server tools inline — including a `reset`/`restore_default` without `keys`, refused `is_error` `E_OP` by `toolgen.ValidateBrowserOp` and never emitted (§2). All server-side → append results, go to 2.
5. Else persist `turns.pending_tool_calls` with status `waiting_tools` (or `suspended` for `pick_asset`, which the model is told to call alone) — status-guarded, **before** anything is streamed — then emit `tool.call` per browser op, `turn.waiting`/`turn.suspended`, and close the stream. A row that is no longer `running` at that write (expired, superseded) streams no `tool.call`: the stored batch is answered `is_error` and the turn ends `turn.error`.
6. `POST tool-results`: every pending id must be answered (missing → `is_error` "no result returned"); append one user message of `tool_result` blocks (`pick_asset` → `{public_url, alt, srcset}`); go to 2 on the same turn. In-memory continuation on the same instance, else rebuilt from Postgres.
7. Stops: `end_turn` → `turn.done`; `max_tokens` → text + "(cut short)"; `refusal` → fixed message, `stop_details.category` logged; 8th call still wanting tools → nothing emitted, `stop:"max_calls"` with a fixed message (thermometer `noConclusion`).
8. TTL 10 min from `started`; a reaper marks stale turns `expired`. The loop re-reads its row before **every** model call and ends `turn.error turn_expired` when the row is no longer `running` (expired, or superseded below) — so a TTL always stops the loop, also when the reaper's pass found nothing dangling to answer. The same exit covers a row that stops being `running` **during** a model call: every message a turn writes is a compare-and-set on the conversation's tail (its own last message), and `SetTurnWaiting` and `FinishTurn` are status-guarded — a turn whose append conflicts stores nothing of that reply (no `tool.call` is emitted for a batch that was never stored), a turn whose status guard fails after its batch was stored answers that batch itself (compare-and-set on the tail) without streaming it, and both end `turn.error turn_expired` — never `turn.done` on a row storage says did not complete. The `message` names the cause (expired, or superseded); the trace's `outcome.status` is the row's terminal status. `MAX_CONCURRENT_TURNS` (default 8) → 503 `busy`.

**One live turn per site.** `POST /v1/turns` does its storage work as **one transaction serialized per site** (`store.StartTurn`: `pg_advisory_xact_lock` on the site id): it resolves the conversation (the named one, else the site's open one, created when none; **rotated** here when it holds 40 messages — a *named* conversation included: it is ended, the turn lands on the fresh one, `turn.done` says `conversationRotated:true`), inserts its turn row, and marks **every other live turn of the site** — `running`, `waiting_tools` or `suspended`, on **any** of the site's conversations (the open one, the one the rotation just closed, one a `DELETE` ended) — **`superseded`** (ended, pending doc cleared; each row returned once, to the Start that closed it; its trace is finished `superseded` and its parked entry dropped on whichever instance holds it). It then answers the dangling batch on its own conversation in storage — browser ids `is_error` "unanswered: the user sent a new message before the browser reported this call's result; the page's current state is in this message's context block", server tools with the result they already had — before its own user text (the owner of a dangling tail is found by matching its `tool_use` ids against the superseded pending docs). `DELETE /v1/conversations/{siteId}` runs under the same lock (`store.EndSiteConversation`): it supersedes **all** the site's live turns, then ends the open conversation. Typing while the asset picker is open, giving up on a batch, or typing while the previous stream is still open therefore starts cleanly instead of racing the old turn for the tail. **The launcher must treat a 409 `turn_not_waiting` with `status:"superseded"` on tool-results as "superseded": drop the pending results silently, no retry, no error to the user** — the newer turn already owns the conversation and the model has been told those calls went unanswered.

*Deterministic survivor.* Starts for a site are serialized by the database; **the newest committed Start supersedes everything else.** Two Starts racing on two instances commit one after another, and the second one's supersede sees the first one's row and closes it — whatever order the requests arrived in, and whatever the instances' clocks say (`turns.started` is the instance's clock and plays no part; an instance 2 s behind supersedes a turn started "after" it). Rotation sharing the lock means two Starts at the 40-message boundary never collide on the open-conversation index: one rotates, the other lands on the fresh conversation. A `DELETE` racing a Start either ends the conversation first (the Start then opens a fresh one) or second (the Start's turn is superseded by it — also when that Start had a dangling batch of the turn it superseded to repair on the conversation the `DELETE` has just ended: the repair is refused by the ended conversation, the Start finds its row superseded and ends `turn.error turn_expired` like any lost Start, never a failed request; measured 40/40 rounds on two instances, `TestFleet_StartRacesDeleteWithWaitingTurn`). A Start that finds its own row no longer `running` right before it stores its user text (the check that precedes the append — a newer Start committed between this one's transaction and its history read) stores nothing and ends `turn.error turn_expired` at its first row check, without a model call; its user text is not in the conversation. The user-text append itself is a compare-and-set on the tail the Start read, retried (up to 3 reads) when a message landed in between, so the request a turn sends is exactly the stored history, never built over a message it did not see.

*Late writes of a superseded turn.* A turn that is `running` when it is superseded (its model call is in flight on some instance) makes no further call: its loop ends at its next row check, or — when the reply lands first — at the compare-and-set append of that reply, which fails because the newer turn's user text now sits on the tail that this turn last wrote. Nothing of the stale reply is stored and no `tool.call` is streamed for it (the page must not apply ops of a turn the user has moved on from); the stream ends `turn.error turn_expired` with a message naming the supersede, and the trace's `outcome.status` is `superseded`. Measured orderings, all stored as `u(A) u(B) a(B)`: `[u(A) u(B) a(A) a(B)]` (A's batch lands first, is refused by the compare-and-set), `[u(A) u(B) a(B) a(A)]` (A's batch lands behind B's reply), a text-only late reply (streamed as `text.delta`, then `turn.error`, not stored), and the newer turn **waiting** on its own batch when the older one's lands (`u(A) u(B) a(opB)`, then B's results continue B normally). When the newer turn lives on **another** conversation (the Start rotated this one away, or a `DELETE` ended it and a fresh conversation followed), the superseded turn's tail is still its own text, so the compare-and-set alone would pass — an **ended conversation refuses every append** (`store.ErrConversationEnded`, an `ErrConflict`), so the reply is not stored there either and the same `turn.error` follows. The only case a superseded turn still answers its own batch is when the batch was stored **before** the supersede: the row goes `waiting_tools` **before** any `tool.call` is streamed, so when that status guard fails (`SetTurnWaiting` finds the row superseded or expired) the stored `tool_use` blocks are answered `is_error` and the stream ends `turn.error` with **no `tool.call`** — `u(A) a(opA) u(answered) u(B) …`, no dangle either way, and the page never sees opA. The launcher should still not send a new turn while a stream is open — `tool.call`s the superseded stream had already emitted may have been applied, and their results will be dropped. A turn whose tail moves while its row is **still** `running` (a foreign write between its row check and its append — not a supersede) ends `turn.error` with a message naming the lost tail; its row and trace say `error`.

**No dangling `tool_use`.** Every persisted assistant `tool_use` block is followed, before the next model request, by a `tool_result` with its id — the Messages API rejects the whole history otherwise (400 until the 40-message rotation). The paths that could leave one open each append one user message of `tool_result is_error:true` per unanswered id: `max_calls` (the 8th response's tool calls), TTL expiry (the reaper, on whichever instance reaps, from the stored `pending_tool_calls`), an `end_turn` response that still carries `tool_use` blocks, and a failed continuation after resume. Every message the loop writes — user text, assistant turn, results, repair — is a compare-and-set on the conversation's tail (`store.AppendMessageAfter`; the loop has no unconditional append): when the reaper and the loop race on the same batch, exactly one answer lands and the loser ends the turn (`turn.error` `turn_expired`, or 409 `turn_not_waiting` to a continuation) instead of answering the ids twice, and a superseded turn's late reply never lands on top of the newer turn's messages. When a new turn finds the stored history dangling anyway (a crash between two writes), it persists the `is_error` repair before its own user message, so later turns replay cleanly too, and logs a **WARNING** — a write path failed to close its own tail — unless the turn that left the tail was closed by someone else: a reaper on another instance that expired it between this Start's transaction and its history read (its closing append still landing; the Start answers the batch with the reaper's reason, and whichever append lands second finds the tail closed), or a Start that superseded it first. That is the expected outcome of the race and is logged as information (`TestFleet_StartRacesReaperNoWarning`: 40 rounds, no WARNING). *Who left the tail* is a heuristic, not a lookup: the conversation's newest other turn by `turns.started` (`tailOwner`). No stored pending doc can identify it — every terminal transition (finish, expire, supersede) NULLs `pending_tool_calls`, and the Start has already superseded every live turn of the site before it reads the history — and `started` is the writing instance's clock, so an instance skewed by more than the gap between two turns can get the owner wrong (a tail left by a turn whose own write path failed, attributed to a neighbouring expired/superseded row: INFO and that actor's reason instead of WARNING). The repair itself is identical either way; only the log level and the `tool_result` text differ. A skew-proof owner needs a monotonic column on `turns` (none exists: uuid ids, instance-clock `started`), not added for a log-precision issue. The reaper answers only ids of the expired turn's own stored `pending_tool_calls`; a turn that expired while `running` (no doc) has the tail attributed to it only when it is the conversation's newest turn (`ExpiredTurn.HasNewer`) — a younger turn's `waiting_tools` batch on the same conversation is never closed by an older turn's expiry. The reaper also evicts in-memory parked turns whose row is no longer `waiting_tools`/`suspended` (completed, superseded or expired on another instance) or is waiting on a batch they do not know (continued and parked again on another instance — a continuation that lands here rebuilds from the row), and ages them out by TTL on its own.

**Storage failures.** The loop never guesses a row's status. Where it re-reads its row to name an exit — after a status-guarded write (`SetTurnWaiting`, `FinishTurn`) found the row no longer `running`, or after a compare-and-set append lost the tail — a read that **fails** is a storage blip, not a lost row: a Start answers **503 `store_unavailable`** (its row is finished `error` when storage takes the write, nothing of the turn is in the conversation, no model call was made; the launcher retries), and a running turn ends **`turn.error store_unavailable`** with no further model call (row finished `error` where it is still `running`, trace `outcome.status` `error`); a batch that turn had stored is left open for the next Start's repair, which attributes it by the row's real status once storage is back, rather than being answered now with a guessed reason. Only a row that is genuinely gone (`store.ErrNotFound` — rows are retained, so academic) reads as `expired`. The pre-call row check keeps its documented leniency (a failed read is logged and the call proceeds; the next write fails on its own), because the turn's history is stored at that point. Before this rule a failed re-read read as `expired`, so a blip during a Start handed out a turn whose user text was never stored; if the database was back by its first row check it made a model call on that unpersisted history and died at its compare-and-set (`store_blip_test.go`: one failed `GetTurn` at each of the six consulted sites).

Browser-reported failures reach the model as `tool_result` with `is_error:true` and the engine's `{code,message,key}`. Raw model ops (`traces.ops`) and browser outcomes (`traces.events`) are stored separately so training never learns a correction as model output. In `traces.events`, a browser result is a `tool.result` event whose `seq` continues the turn's stream seq (never colliding with `tool.call`/`text.delta` seqs) and whose `data` is the posted result item plus `data.for_seq` = the `seq` of the `tool.call` it answers.

## 5. Models (`internal/model`)

Only the Anthropic Messages API shape is spoken. Client = `github.com/anthropics/anthropic-sdk-go`, as in thermometer `chat.go:199` / `sourdough.go:293`: `WithAPIKey` + `WithMaxRetries(2)` for `anthropic`; `WithBaseURL` + `WithMaxRetries(0)` + a signing middleware for `messages-compatible`. Hand-written wire structs exist only for traces.

`MODEL_ROUTES` (JSON): name → `{kind: anthropic|messages-compatible, url?, auth: api-key|id-token|none, model, max_tokens, effort?, stream?, thinking?, temperature?, top_p?}`. Default:

```json
{"default":{"kind":"anthropic","auth":"api-key","model":"claude-sonnet-5-5","max_tokens":8192,"effort":"medium"},
 "opus":{"kind":"anthropic","auth":"api-key","model":"claude-opus-5-5","max_tokens":8192,"effort":"medium"}}
```

A request's `model` must be in `MODEL_ROUTE_ALLOWLIST` (default `default,opus`). Model ids per the claude-api skill (2026-10-01): current Sonnet `claude-sonnet-5-5` (user decision: Sonnet-class default), current Opus `claude-opus-5-5`; no date suffixes. Request rules for both: `thinking` omitted (adaptive), `output_config.effort` from the route (set explicitly; model defaults differ), `tool_choice: auto`, no prefill, `strict:true` tools, append-only history (preserved thinking), `betas:["server-side-fallback-2026-07-01"]` + `fallbacks:"default"`. `stream` defaults to true on `anthropic` routes (model SSE → `text.delta`), false on `messages-compatible` routes, which also get thermometer's `plainRequest` (strip `output_config`, `cache_control`, betas, fallbacks) plus the route's **thinking / sampling fields**, `messages-compatible` only (an `anthropic` route rejects them at config load — Claude runs adaptive thinking, which fixes temperature and disallows `top_p`):

| field | values | default | on the wire |
|---|---|---|---|
| `thinking` | `disabled` \| `enabled` | `disabled` | `"thinking":{"type":"disabled"}` — Ollama's `anthropic.go` (v0.34.4, `FromMessagesRequest`) maps `type:"disabled"` → `think=false`, so a qwen3.5-family renderer emits the **empty** `<think>\n\n</think>` block every SFT row was trained with (default `think=nil` → renderer default **true** → the prompt ends in an open `<think>`, and the adapter scores below base: 2026-10-04 parity review). Disabled also **drops `thinking`/`redacted_thinking` blocks from the re-sent history** (§3). `enabled` sends `{"type":"enabled","budget_tokens":<max_tokens/2>}` (Ollama reads only `type`; the budget keeps the shape valid for a Claude-shaped server) and replays the blocks. |
| `temperature` | `0..2` | unset (server default) | `"temperature":<v>` only when set |
| `top_p` | `0..1` | unset (server default) | `"top_p":<v>` only when set |

`auth: id-token` = thermometer `idtoken.go` (metadata server, audience = route URL, cached until `exp-5m`, sent as `X-Serverless-Authorization`). 408/429/5xx/deadline → 503 `model_unavailable`.

## 6. Storage

Postgres via `pgx/v5`, hand-written SQL, `service/migrations/NNNN_name.{up,down}.sql` (golang-migrate), applied by `MIGRATE_AT_BOOT=on` or `sous-chef -migrate`.

| table | columns |
|---|---|
| `manifests` | `hash text pk, body jsonb, index jsonb, generated_from jsonb, build_line text, counts jsonb, first_seen timestamptz` |
| `conversations` | `id uuid pk, site_id, store_id, created, updated, ended_at null`; index `(site_id) where ended_at is null`; an ended conversation refuses every append |
| `messages` | `conversation_id fk, seq int, role, content jsonb, created`; pk `(conversation_id, seq)` |
| `turns` | `id uuid pk, conversation_id fk, status (running\|waiting_tools\|suspended\|done\|error\|expired\|superseded), manifest_hash, build_id, model, user_email, started, ended, usage jsonb, pending_tool_calls jsonb`; `superseded` = closed by a newer `POST /v1/turns` for the site or by `DELETE /v1/conversations` (§4). Both run as one transaction under `pg_advisory_xact_lock(1, hashtext(site_id))`, which serializes a site's Starts and DELETEs (`store.StartTurn`, `store.EndSiteConversation`) |
| `traces` | `turn_id pk, site_id, store_id, context jsonb, events jsonb, ops jsonb, feedback jsonb, created`; index `(created)` |
| `store_optout` | `store_id pk, created, by_email` |
| `usage_rows` | `id bigserial, turn_id, model, token counts (4), latency_ms, status, stop_reason, request_id, occurred_at, shipped bool` |

Traces are kept (user decision) unless the store is in `store_optout`. `sous-chef -retention` (daily job) deletes `traces` older than `TRACE_RETENTION_DAYS` (90), **ends** conversations whose `updated` is older than 90 days (inactivity cutoff — a conversation nobody closed or rotated must not keep merchant text forever), then deletes ended conversations/messages/turns older than 90 days, shipped `usage_rows` older than 30 days. `traces.feedback` is capped at 16 KB per turn by `POST /v1/feedback`. One `usage_rows` row per model call (thermometer `recordUsage`, `context.WithoutCancel`). When `THERMOMETER_API_URL`/`THERMOMETER_API_KEY` are set, a shipper POSTs batches to `/api/ingest/usage` with `X-API-Key` in the exact 16-key shape of FACTS 2 §5 (`source:"sous-chef"`, `operation:"chat"|"tool_turn"`, wire names `cache_creation_input_tokens`/`cache_read_input_tokens`); 202 = shipped.

## 7. Configuration

| env | default | meaning |
|---|---|---|
| `PORT` | `8090` | listen port |
| `DATABASE_URL` | required | pgx DSN; Cloud Run uses the `/cloudsql/<conn>` socket |
| `MIGRATE_AT_BOOT` | `off` | `on` migrates before listening |
| `SERVICE_KEYS` | required | comma-separated bearer keys |
| `ANTHROPIC_API_KEY` | — | required for `anthropic` routes |
| `ANTHROPIC_BASE_URL` | — | endpoint override (e2e points it at the fake) |
| `MODEL_ROUTES` | §5 JSON | route table |
| `DEFAULT_ROUTE` / `MODEL_ROUTE_ALLOWLIST` | `default` / `default,opus` | route selection |
| `MODEL_CALL_TIMEOUT` | `60s` | per model call |
| `TURN_TTL` | `10m` | waiting/suspended lifetime |
| `MAX_CONCURRENT_TURNS` | `8` | per instance |
| `TRACE_RETENTION_DAYS` | `90` | also the conversation inactivity cutoff (`store.DefaultRetention.InactivityDays`) |
| `THERMOMETER_API_URL` / `THERMOMETER_API_KEY` | — | usage shipping, off when unset |
| `BUILD_ID` | `dev` | `-ldflags -X main.buildID=` |

Not configurable by env (compiled-in constants): the 256 KB non-manifest body cap (`api.DefaultMaxBodyBytes`, 413 above it), the 8 MiB manifest cap, the 16 KB `traces.feedback` cap (`store.MaxFeedbackBytes`), `maxCalls = 8`, the 40-message rotation.

## 8. Package layout (`service/`)

```
cmd/sous-chef/main.go   -migrate | -retention | serve (default)
internal/api            ServeMux router, SSE writer, JSON fallback, errors
internal/auth           bearer check, X-User-Email
internal/manifest       canonical JSON, hash, index, lookup/describe
internal/toolgen        tool definitions per hash, server-tool dispatch
internal/prompt         system[0] text, system[1] renderer, context block, toolgen → SDK tool params
internal/registry       PUT/POST/GET manifests, hash check or hash computation, build-line diff, compiled per-hash prefix cache
internal/loop           turn state machine, resume (in-memory or rebuilt from Postgres), TTL reaper,
                        conversation/feedback/opt-out adapters
internal/model          routes, SDK client, plainRequest, id-token
internal/store          pgx pool, migrations, repositories
internal/trace          trace builder (thinking stripped), retention
internal/usage          usage rows, thermometer shipper
internal/config         env parsing
migrations/             0001_init.up.sql …
testdata/               manifest.snapshot.json (from cookie-cutter), fake-model scripts
```

Go 1.22+ (`go 1.24` in go.mod). Dependencies: `net/http`, `github.com/jackc/pgx/v5`, `github.com/anthropics/anthropic-sdk-go`, `github.com/golang-migrate/migrate/v4` — nothing else.

## 9. Terraform (`infra/`)

```
infra/modules/sous-chef/   apis run sql secrets iam registry job variables outputs (.tf)
infra/envs/{int,prod}/     main.tf, backend.tf (gcs, prefix sous-chef/<env>), terraform.tfvars.example
```

Shapes from thermometer (FACTS 2 §8). Cloud Run v2: ingress internal + load balancer, **CPU only, min 0 / max 10, concurrency 20**, timeout 600 s (SSE), port 8090, probe `/healthz`, `cloud_sql_instance` volume, secrets via `secret_key_ref` `latest` (optional ones through `dynamic "env"`). Cloud SQL `POSTGRES_16`, `db-f1-micro` (int) / `db-custom-1-3840` (prod), backups on, `deletion_protection` var, `random_password`, `DATABASE_URL` secret built from password + `connection_name`. Secrets: `ANTHROPIC_API_KEY`, `SERVICE_KEYS`, `DATABASE_URL`, optional `THERMOMETER_API_KEY`. Artifact Registry repo `sous-chef`. SA `sous-chef-run` with `roles/cloudsql.client` + `secretAccessor` per secret; launcher SA gets `roles/run.invoker`. Jobs `sous-chef-migrate` (`max_retries=0`) and `sous-chef-retention` (Cloud Scheduler 03:00 UTC). Must pass `terraform fmt -check` and `terraform validate` (terraform is at `/usr/bin/terraform`); never `apply` from automation.

## 10. Test plan

- **Unit, every package**: hash stable under key order and `generated_from` churn; index ≤ 5k tokens (3.0 bytes/token); `lookup_fields` matching, cap, `item.` stripping; toolgen schema validity and order; prefix byte-stability; loop state machine against a scripted model (all-server batch, mixed batch, pick_asset, max_calls, max_tokens, refusal, TTL expiry, missing result; the supersede: site-wide, running turns, the late batch of a superseded turn never stored over the newer turn's — `supersede_site_test.go`, `supersede_race_test.go` — and Starts racing inside each other's StartTurn/user-text windows, three nested, three concurrent ×25, with a clock-skewed second instance, or against a `DELETE` — also one that lands while the Start must repair the dangling batch of the turn it superseded (`TestStart_SupersedingWaitingTurnRacesConversationsEnd`: the Start ends `turn.error`, never fails); a transient `GetTurn` failure at each site where the loop re-reads its row status — three in Start → 503 `store_unavailable`, three in Run → `turn.error store_unavailable`, no model call either way (`store_blip_test.go`) — leaving exactly the last committed one live; a Start whose dangling tail belongs to a turn the reaper expired first repairing it without a WARNING (`TestStart_DanglingTailOfExpiredTurnIsNotAWarning`); the rotated-away late batch not stored; no `tool.call` when the row is lost before `SetTurnWaiting`; a lost tail on a still-running row ending `error` — `start_serialized_test.go`); auth + rotation; config; plainRequest; SSE writer + JSON fallback; store repositories against Postgres when `TEST_DATABASE_URL` is set (else skipped), including `StartTurn` (conversation resolve/rotate/create, supersede-all-but-self, named-conversation errors), `EndSiteConversation`, the ended-conversation append refusal, and two `store.Store` instances on one database racing `StartTurn` (60 rounds, never both live, `STORE_RACE_ROUNDS` to soak), with a 2 s clock skew, at the rotation boundary, and against `EndSiteConversation`.
- **E2E (`service/e2e_test.go`)**: in-process server against `TEST_DATABASE_URL` (docker-compose Postgres), an `httptest` fake model speaking `/v1/messages` (non-streaming and SSE, scripted), and `testdata/manifest.snapshot.json` as the launcher manifest. Asserts: unknown hash → 409 `unknown_manifest`; `PUT` → 201 with `fields=2094 lists=69`; turn → `tool.call set_fields` + `turn.waiting`; tool-results → `text.delta` + `turn.done`; `pick_asset` → `turn.suspended`, resume with `asset` → `set_fields` with the URL in `values` and alt/srcset in `assets`; identical `tools`+`system` bytes on every model call; one `usage_rows` row per call; `traces` row present, absent for an opted-out store; `Accept: application/json` returns the same events as an array; TTL expiry through the real store; tool-results/feedback with the right `siteId` → ok, wrong `siteId`/`storeId` → 404; a Start while the previous turn is suspended supersedes it and its stale Continue → 409 `status:"superseded"`; the same across conversations (`TestEndToEnd_SupersedeAcrossConversations`): a suspended turn on the conversation the rotation just closed, a waiting turn on a conversation `DELETE` ended (superseded by the `DELETE` itself), and a turn still in the model when the next Start lands (it ends `turn.error turn_expired` with nothing stored and no `tool.call`, its Continue is 409 superseded, and the following turn replays with no WARNING); the newer turn waiting on its own batch when the superseded one's late batch lands (`TestEndToEnd_LateBatchDoesNotBuryWaitingNewerTurn`: the late batch is not stored, the newer turn's own tool-results are 200 and it finishes `done`, no WARNING afterwards); the late batch of a turn whose conversation was rotated away not stored into the ended conversation and not streamed (`TestEndToEnd_LateBatchOnRotatedConversationNotStored`). **Fleet (`service/e2e_fleet_test.go`)**: two `store.Store` + `loop.Runner` instances on one Postgres — racing Starts never leave two live turns (`TestFleet_RacingStartsNeverBothLive`, 60 rounds, `E2E_RACE_ROUNDS` to soak), the survivor is continued on the other instance and completes (`TestFleet_RacingStartsSurvivorCompletes`), an instance 2 s behind still supersedes the older waiting turn (`TestFleet_SkewedInstanceSupersedesOlderTurn`), two Starts at the rotation boundary never fail (`TestFleet_ConcurrentRotation`), a Start racing a `DELETE` with a waiting turn's dangling tail on the conversation never fails and ends `turn.error turn_expired` when superseded (`TestFleet_StartRacesDeleteWithWaitingTurn`, 40 rounds), a Start racing the reaper on another instance over an 11-minute-old waiting turn completes with no WARNING (`TestFleet_StartRacesReaperNoWarning`, three instances, 40 rounds).
- `make test` = `go vet` + `go test -race ./...`; `make e2e` starts Postgres and runs the e2e tag.

## 11. Deployment contract (what the launcher proxy must enforce)

The service authenticates the **caller** (service key) and nothing else: it has no notion of which user may touch which site, store or turn, and every route acts on whatever ids the body or path names. Tenant scoping is therefore the launcher proxy's job, and the service is only safe behind a proxy that does all of the following before forwarding:

| Route | Proxy must check |
|---|---|
| `POST /v1/turns` | the session user may edit `siteId`; `storeId` is the store that site belongs to (not taken from the browser); `conversationId`, when sent, belongs to that site (or drop it and let the service pick the site's open conversation). |
| `POST /v1/turns/{turnId}/tool-results` | `turnId` was issued to this user's site in this session (the proxy keeps the `turnId → siteId` map it saw on the `POST /v1/turns` stream; a stranger's `turnId` is 404 at the proxy, never forwarded). The proxy also **sets `siteId` and `storeId` in the body** from its own session (never from the browser); the service verifies them against the turn's conversation and answers 404 `turn_not_found` on a mismatch — the second line of defence. |
| `GET`/`DELETE /v1/conversations/{siteId}` | the user may edit `siteId`. `GET` returns full merchant text and `thinking` blocks; `DELETE` ends the conversation. |
| `POST /v1/feedback` | `turnId` belongs to one of the user's sites (same map as tool-results); the proxy sets `siteId`/`storeId` in the body the same way and the service verifies them (404 on mismatch). |
| `PUT`/`DELETE /v1/stores/{storeId}/trace-optout` | the user administers `storeId`. This is a per-store privacy switch: expose it only to store owners/admins, never to every editor. |
| `PUT`/`POST /v1/manifests`, `GET /v1/manifests/{hash}` | launcher-internal (the launcher build registers its own manifest); never reachable from the browser. |
| `/healthz`, `/readyz`, `/version` | not proxied. |

Also: strip incoming `Authorization`, `X-User-Email`, `X-Launcher-Build`, `X-Manifest-Hash`, `Cookie` from the browser request and set them from the proxy's own session/build; never expose the service URL or key to the browser; enforce its own body limit at or below the service's 256 KB so the service's 413 is never the first line of defence. Apart from the `siteId`/`storeId` verification on tool-results and feedback (which only bites when the proxy sends them), none of this is implemented in the service — a direct caller holding the key can read or end any site's conversation, and a tool-results body without `siteId` is accepted for any turn.

## Gaps and non-goals (v1)

No `apply_preset` op (engine has none; `ProductCardSection.presets` exist, FACTS 1). No reverse proxy in the launcher yet (FACTS 1 §3) — the launcher's half of this contract, §11. No knowledge base. No history compaction. In-house model work is Phase 5 (`lab/`).
