# lab/ — Sous Chef in-house model kit

The contract is **[`docs/LAB.md`](../docs/LAB.md)**; this directory implements it.

The same code runs in GCP as the drift check and automated retraining
pipeline (int): one entry, `python -m souschef_lab.pipeline <cmd>`, in the lab
runtime image. Runbooks: [`docs/PIPELINE.md`](../docs/PIPELINE.md) (turning it
on), [`docs/DRIFT.md`](../docs/DRIFT.md), [`docs/RETRAIN.md`](../docs/RETRAIN.md)
(every stage, automated and by hand); the lab's contract for it is
`docs/LAB.md` §5c.

```
make -C lab drift-dry          # the drift check end to end on the fake service (no network)
make -C lab retrain-dry        # a retrain round end to end on the fake service with a fake author
make -C lab pipeline-selftest  # what the lab image's selftest checks, on this machine
make -C lab engine-vendor SHA=<cookie-cutter sha> CC=<checkout> UPLOAD=1   # an engine snapshot for GCP
```

```
make -C lab venv test          # lab/.venv (stdlib runtime + pytest), run the suite
make -C lab test-engine        # same, but the browser-engine tests must run (CI)
make -C lab cases              # cases.jsonl with engine-measured expected.diff
```

Built so far (docs/LAB.md §1–§2, engine side of §3):

| module | what |
|---|---|
| `souschef_lab/manifest.py` | manifest loader, README §1.1 hash, field/list iterators, writable-target enumeration |
| `souschef_lab/state.py` | `EditorState` builder (basic template `editorial`, `DEFAULT_DEFAULT_HOME_SECTIONS` rows seeded from the probe fixtures and manifest defaults), turn-context derivation, gate opening |
| `souschef_lab/engine.py` | subprocess wrapper around cookie-cutter's `headless.mjs` (stdin or file mode, one boot per call, cached) |
| `souschef_lab/cases.py` | deterministic case generator (14 kinds, holdout split) + `measure()` through the engine |

The engine lives in the cookie-cutter checkout: set `COOKIE_CUTTER_DIR` (default: `../../breadstack-cookie-cutter`) or `SOUS_CHEF_ENGINE` to the `headless.mjs` path. Tests that need it skip when it is absent unless `SOUS_CHEF_LAB_REQUIRE_ENGINE=1`.

Data directories (`lab/data/`, `lab/runs/`) and `lab/.venv/` are git-ignored. Fixtures under `lab/fixtures/` are copies of read-only sources with their origin recorded in a `_source` member; `manifest.snapshot.json` is a symlink to `service/testdata/`. `fixtures/parity/served.json` is generated: Ollama 0.34.4's own render of the real captures (`make -C lab parity-write`, through `lab/parity/`, a Go module pinning ollama v0.34.4) — the serving-parity fixture `tests/test_serving.py` holds `souschef_lab/serving.py` to (docs/LAB.md §5); re-record it when Ollama's renderer version changes.
