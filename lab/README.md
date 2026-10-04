# lab/ — Sous Chef in-house model kit

The contract is **[`docs/LAB.md`](../docs/LAB.md)**; this directory implements it.

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

Data directories (`lab/data/`, `lab/runs/`) and `lab/.venv/` are git-ignored. Fixtures under `lab/fixtures/` are copies of read-only sources with their origin recorded in a `_source` member; `manifest.snapshot.json` is a symlink to `service/testdata/`.
