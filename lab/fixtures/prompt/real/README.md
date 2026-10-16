# fixtures/prompt/real — the real service binary's model requests, captured

One directory per captured turn, each the REAL `sous-chef` binary (built from `service/cmd/sous-chef`, `BUILD_ID lab-e2e`)
talking to `tests/fake_model.py` over a `messages-compatible` route (`auth: none`, `stream: false`), with the lab's own
collector (`souschef_lab.collect.run_case`) as the browser and the real engine (`headless.mjs`) applying the ops.
No API spend; nothing leaves 127.0.0.1. Captured 2026-10-01 by `tests/capture_real.py`. Re-captured 2026-10-10 (`--from fixtures/prompt/real`) after the wave-2 service changes: new system[0]/system[1]/tools, the fields-context block re-ranked; everything else byte-identical (docs/LAB.md §10 "Deferred", prompt-parity note). Re-captured once more the same day after the verify pass's rule/tool wording and the gallery alias: only `system` and `tools` moved. Re-captured 2026-10-15 (`--from`) after create_page and the post-demo font / vague-request rules: new system[0], `create_page` in tools, and — on the lab's cc-snapshot-r4 engine — the warnings capture's gated result carries the engine's `hidden` member; verdicts unchanged. Two create_page captures added 2026-10-16 (`capture_real.create_page_plan`, the 160ef91 binary, the launcher 1d4193e2 engine snapshot). All seven re-captured the same day (`--from`, same binary and engine) after the collector's `sectionId` fix: a tool result's `sectionId` is now the browser's echo of the CALL's own target section (client/turns.js `toToolResult`: `op.target.sectionId || op.sectionId`, ok results only), never the engine's new-section id — so the set_field / pick_asset results gain `"sectionId":"contentBlock-1"` / `"gallery"`, the page-target section adds lose theirs; every other byte, every verdict and every call count unchanged.

```
<capture>/
  case.json             the LAB.md §2 case the turn was asked
  steps.json            the fake model's scripted replies (tests/fake_model.py step shape)
  collect-real.jsonl    the collector row: context, text, messages, events (incl. tool.result), engine_results, verdict
  model-requests.jsonl  every POST /v1/messages body the service sent, in order ({path, headers, body, stream, raw_len})
```

| capture | tool_result shape (loop.browserResult) | model calls | verdict |
|---|---|---|---|
| `ok.set_field.contentBlock-1.title` | `{"ok":true,"sectionId":"contentBlock-1","written":[…]}` (the call's own section, echoed) | 2 | pass |
| `error.set_field.contentBlock-1.tagline` | `{"code":"E_FIELD","message":…,"key":"showButton"}` with `is_error`, then the retry's ok | 3 | invalid_op (the scripted bad op) |
| `warnings-sectionid.list_add.home.sections.loyaltyForm` | `{"ok":true,"written":[…]}` (a page-target section add: no sectionId) and `{"hidden":[{"by":"style",…}],"ok":true,"sectionId":"featuredCategories","warnings":["W_HIDDEN:tileBorder"],"written":[…]}` — two results in ONE user message | 2 | extra_writes (the scripted gated write) |
| `fields-context.set_field.contentBlock-1.title` | the `ok.` turn again on route `lab-fake-fields` (`fields_context: on`): the user turn carries the `<candidate_fields>` block between `</context>` and the text; `plan.json` names the route; the row logs the block as `candidate_fields` (captured 2026-10-04) | 2 | pass |
| `pick_asset.image.gallery.bgImage` | `turn.suspended` → the bare `api.Asset` `{"public_url":…,"srcset":{…}}`, then the set_fields' ok | 3 | pass |
| `create_page.slug.deals` | `{"ok":true,"slug":"deals","written":["customPages[0]","pageVisibility.deals.inHeaderNav"]}` — `slug` forwarded for create_page only — and the same batch's section add on the new slug (captured 2026-10-16 on the 160ef91 binary with the launcher 1d4193e2 engine) | 2 | extra_writes (the case expects no diff; the bytes are the point) |
| `create_page.refused-stop.faq` | `{"code":"E_SLUG",…,"key":"slug"}` (`faq` is a built-in page) and the same batch's follow-up on `faq` stopped by the browser: `{"code":"E_TARGET","message":"page \"faq\" was not created (create_page was refused earlier in this reply), so this op was not applied"}` (2026-10-16, same binary and engine) | 2 | invalid_op |

`tests/test_render.py` asserts, for every capture: the row's `messages` == `render.messages_from_events(row.events)` ==
the wire, request after request (each request is the prefix of the turn up to that model reply); `system[0]`,
`system[1]` and `tools` equal `../system0.rules.txt`, `../system1.index.txt`, `../tools.json` byte for byte on every call
(`../README.json` has the hashes); `dataset.to_hf` reads those bytes. Shape-specific assertions pin each row of the table.

`../../parity/served.json` is Ollama 0.34.4's OWN prompt for every `model-requests.jsonl` body here, rendered by
`lab/parity/` (a Go module pinning github.com/ollama/ollama v0.34.4; `make -C lab parity-write`); `tests/test_serving.py`
holds `serving.render_prompt` (the lab's serving-parity render, LAB.md §5) to it byte for byte, request after request.
Re-capturing here means re-writing that fixture too (`make -C lab parity-write`), and so does a new Ollama version.

## Recipe (re-capture when the service's prompt or result rendering changes)

```sh
cd lab
docker compose -f ../docker-compose.yml up -d postgres           # or: make -C .. db-up (Postgres 16 on 5433)
.venv/bin/python tests/capture_real.py --default-plan data/e2e/cases.jsonl --out fixtures/prompt/real
#   --from fixtures/prompt/real --out <dir>   re-runs the stored case.json + steps.json instead of the default plan
#   --only <capture name>                     run one capture of the plan (repeatable)
#   --binary <path>                           skip the `go build`; --db-url, --engine, --manifest, --port, --model-port
```

What the script does: `go build` the binary into a temp dir; start `FakeModel` (free port) and the binary
(`DATABASE_URL` = `…/souschef_lab`, `MIGRATE_AT_BOOT=on`, `SERVICE_KEYS=test-key`, `MODEL_ROUTES` = `lab-fake` with
`fields_context: off` — a `messages-compatible` route defaults to on since 2026-10-04, and the four tool_result captures
are the block-less bytes — and `lab-fake-fields` with it on, `BUILD_ID=lab-e2e`); per capture, append the steps, `POST /v1/manifests` (`fixtures/manifest.snapshot.json`),
`run_case` (which first `DELETE`s the site's open conversation — a turn without `conversationId` continues it, README §1.1,
and the lab's `siteId` is `lab-<case id>` for every sample), then write the four files. The default plan takes the cases
from `data/e2e/cases.jsonl` (+ `fixtures/traces/` for the two `set_field.contentBlock-1.*` cases); the `pick_asset` plan
scripts the `set_fields` with the asset the collector's fake picker answers (`collect.pick_asset_for` →
`state.fixture_asset_for`, keyed by the case id alone — the same function the generator used for the case's expected
ops, so the scripted write reproduces `expected.diff`), and refuses a case whose expected asset is not that pick: such a
case predates the shared function (the 2026-10-01 `cases.jsonl` expected `paper-texture` for
`image/gallery/gallery/bgImage/pick`; since 377cefa the pick for that id is `hero-field`) — regenerate it
(`souschef_lab.cases --kinds image --no-measure`, then `cases remeasure` on the row) before capturing. The
`pick_asset` capture was re-captured that way 2026-10-04 (verdict `pass`; the call id `toolu_lab_1`). Wall ≈ 30 s for
the four (one warm engine boot per batch).

The pre-renderer capture of 2026-10-01 (one `set_field` turn, bytes from before `render.py`) was removed: its
`collect-real.jsonl` carried `messages` the service never sends (`<context>{…}</context>` without inner newlines,
Python-spaced tool results), which `dataset.to_hf` used to prefer over the rebuild. `to_hf` now always rebuilds from
events and rejects such a row `messages_mismatch` (LAB.md §5).
