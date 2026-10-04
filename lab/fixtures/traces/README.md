# fixtures/traces — scripted turns for the fake service and the verifier goldens

One file per case, `<case id>.json`, hand-written 2026-10-01:

```
{schema: 1, note, at, buildId,
 case:            a LAB.md §2 case row (site_state, context, request, expected:{ops, diff}, holdout, manifest_hash, …)
 streams:         [[event…], [event…]]  — streams[0] is what POST /v1/turns returns, streams[k] what the k-th
                  POST /v1/turns/{id}/tool-results returns; events are {event, data, turnId, seq} (README §1.2)
 tool_results:    [body…]  — the tool-results body the browser emulator must post after streams[k] (the sink asserts it)
 tools:           the tool definitions the turn saw ([] until the service exposes them per hash)
 expected_verdict, expected_reasons}
```

| file | verdict | why |
|---|---|---|
| `set_field.contentBlock-1.title` | `pass` | exactly the expected `set_fields` |
| `set_enum.productCarousel-1.productCount` | `extra_writes` | also writes `contentAlign` (the replay also warns `W_HIDDEN:productCount`: the gate is closed in the state) |
| `multi.home.contentBlock-1-title+productCarousel-1-productCount` | `missing_writes` | does the first of two edits |
| `set_range.contentBlock-1.paddingY` | `wrong_value` | 64 where 48 was asked |
| `set_field.contentBlock-1.tagline` | `invalid_op` | writes an unknown key (`E_FIELD showButton`); the browser reports `ok:false` |
| `clarify.home.contentBlock.title` | `clarify_ok` | two `contentBlock` sections; asks which, names both; no ops |

`souschef_lab.verify.trace_from_fixture(fx)` turns a file into the collector row collect.py would have logged
(streams flattened, one `tool.result` event per posted item with `for_seq`).

## Schema 2 — real collector rows (int-run1 warm-up #2, 2026-10-04)

```
{schema: 2, note, at, buildId, source, manifest,
 case:  the case as the 2026-10-04 generator makes it (set_fields + assets; pick variant = state.fixture_asset_for),
        remeasured with the engine named in `engine_fingerprint` against `manifest` (lab/data/int-run1/manifest.json,
        canonical hash 907fc297… — NOT the fixture snapshot, 727d151b…)
 row:   the collector row verbatim (LAB.md §4) as logged from the live int service, route `default`; its stored
        `verdict`/`reasons`/`actual_diff` are verify-1's live verdict — the finding — and the tests assert verify-2 disagrees
 expected_verdict, expected_reasons, expected_info}
```

| file | verify-1 (stored in `row`) | verify-2 | why |
|---|---|---|---|
| `image.heroBanner.backgroundImage` | `missing_writes` + `extra_writes` | `pass`, info `companions` | the model's `set_fields` + `assets` wrote slide-0 + flat mirror incl. `SrcSet.800/1600`; the old case expected `edit_list update` without assets (alt `''`, srcset `null`) |
| `image.featuredCategories.taglineImage.pick` | `missing_writes` + `extra_writes` | `pass`, info `pick_substituted`, `companions` | the model wrote the asset the picker returned (`product-jar`); the case expects the generator's (`paper-texture` under the new picker rule) → substituted and re-measured |

`engine/recorded-images.json` (same `{state, ctx, runs}` shape, measured on the int-run1 case state with the int manifest):
`F1_hero_image_set_fields` (the F1 model ops = the new case's expected ops), `F2_taglineImage_picked_product_jar` (the F2
model ops = `substitute_picked(expected ops)`), `F2_taglineImage_generator_texture`, `W_hero_image_wrong_asset` (the logo
instead → `wrong_value`), `M_hero_headline_instead_of_image` (→ `missing_writes`), `P_hero_image_no_assets` (URL only →
alt `''` / srcset `null`), `L_hero_image_edit_list_with_assets` (≡ `F1_hero_image_set_fields`, leaf for leaf). The tests
merge its runs into a `RecordedEngine` over `recorded.json`; the real-engine test for these two fixtures runs only when
`lab/data/int-run1/manifest.json` exists.

Every `expected.diff` and every `written`/`warnings`/`error` in `tool_results` was **measured** by running the ops
through `launcher/client/src/sousChef/ops/headless.mjs` (cookie-cutter `sous-chef-v2`, `ops/__fixtures__/manifest.snapshot.json`,
ctx `{templateId:editorial, cdnUrl:https://cdn.int.breadstack.io, galleryFolder:gal, now:1700000000000}`). The raw
outputs are in `engine/recorded.json` (`{state, ctx, runs:{key:{ops, output}}}`), which `verify.RecordedEngine` replays
so the tests run without node. Recorded runs beyond the six cases: both hero forms (`H1_hero_flat`, `H2_hero_list` —
identical diffs), an image with srcset companions (`I_image`), `reset` (`J_reset`), `edit_list add` (`K_bullets_add`),
`restore_default` (`N_restore_title`).

Measured boot of `headless.mjs` on this machine: 7.4–10.5 s cold, 3.2 s once Vite's cache is warm.
`usage` token counts in the scripted `turn.done` events are placeholders (0), not measurements.
