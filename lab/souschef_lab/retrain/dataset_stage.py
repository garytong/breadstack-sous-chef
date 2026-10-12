"""dataset_stage — the training set and its gates (design §3.4): job `souschef-dataset`.

r4dataset.build on the round's cases and every collection log, on the deployed
manifest, `--section-cap-scope F0,F1`. Gates (each fails the stage):

  1. render parity   parity_gate.py: three conversations through the deployed binary,
                     every model call byte-equal to r4dataset.student_source
  2. stats           block_drift_rows == 0, preview_claim_kept == 0, and no kept
                     `where` row without report_unsupported's `where`
  3. row checks      tools/run-r4.sh's: rows in both splits, follow-ups > 0 (full mode),
                     no reply mentioning the preview, no row with block drift
  4. train plan      train.py's dry plan on the split: rows, tokens, updates, 0 over 32,768

Outputs: runs/<id>/dataset/{dataset/{train,val,rejects}.jsonl, stats.json,
dataset-manifest.json, manifest.json} (the layout train.py and pipeline-train.sh
read) and runs/<id>/model/context.tgz (the lab image's lab/model-image, the build
context Cloud Build uses — no repository access needed there).
"""
from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path
from typing import Any, Callable

from .. import LAB_DIR
from .common import ConfigError, Store, notification, read_jsonl, run_prefix, write_result

TRAIN_FLAGS = ["--fields-context", "on", "--eval-steps", "150", "--val-max-rows", "150", "--checkpoint-steps", "50"]


def where_missing(rows: list[dict]) -> int:
    """Kept `where` rows whose answer has no report_unsupported call carrying `where`."""
    n = 0
    for r in rows:
        if (r.get("meta") or {}).get("mode") != "where":
            continue
        ok = False
        for m in r.get("messages") or []:
            for c in m.get("tool_calls") or []:
                fn = c.get("function") or {}
                if fn.get("name") == "report_unsupported":
                    try:
                        ok = ok or bool(json.loads(fn.get("arguments") or "{}").get("where"))
                    except ValueError:
                        pass
        n += 0 if ok else 1
    return n


def row_checks(data: Path, *, need_followups: bool) -> dict:
    """tools/run-r4.sh's row checks, as a dict of problems (empty = pass)."""
    from ..r4score import mentions_preview
    out: dict[str, Any] = {"splits": {}, "problems": []}
    for split in ("train", "val"):
        rows = read_jsonl(data / f"{split}.jsonl")
        block = sum(1 for r in rows if (r.get("meta") or {}).get("fields_context"))
        follow = sum(1 for r in rows if (r.get("meta") or {}).get("history_assistant_turns"))
        drift = sum(1 for r in rows if (r.get("meta") or {}).get("block_drift"))
        preview = sum(1 for r in rows for m in r.get("messages") or [] if m.get("role") == "assistant"
                      and mentions_preview(m.get("content")))
        out["splits"][split] = {"rows": len(rows), "with_block": block, "followups": follow, "drift": drift,
                                "preview": preview, "where_missing": where_missing(rows)}
        if not rows:
            out["problems"].append(f"{split}: no rows")
        if preview:
            out["problems"].append(f"{split}: {preview} replies mention the preview")
        if drift:
            out["problems"].append(f"{split}: {drift} rows with block drift")
        if out["splits"][split]["where_missing"]:
            out["problems"].append(f"{split}: {out['splits'][split]['where_missing']} where rows without `where`")
    if need_followups and not out["splits"]["train"]["followups"]:
        out["problems"].append("train: no follow-up rows")
    return out


def train_plan(data: Path, name: str) -> dict:
    """train.py's dry plan (no torch): rows, tokens, updates, over-length."""
    from .. import train as T
    args = T.parse_args(["--data", str(data), "--name", name, *TRAIN_FLAGS, "--dry-run"])
    tr, va = T.load_split(args.data, "train"), T.load_split(args.data, "val")
    problems = T.check_rows(tr, "train") + T.check_rows(va, "val")
    p = T.plan(args, tr, va)
    mismatch = T.check_fields_context(p["fields_context"], args.data)
    over = int((p.get("rows") or {}).get("over_length_estimate") or 0)
    return {"problems": problems + ([mismatch] if mismatch else []) + ([f"{over} rows over max_seq"] if over else []),
            "updates": p.get("updates"), "rows": p.get("rows"), "tokens_estimate": p.get("tokens_estimate")}


def ensure_val(data: Path) -> int:
    """SMOKE's handful of rows can leave val empty after the group split: move the train rows of one group there."""
    tr, va = read_jsonl(data / "train.jsonl"), read_jsonl(data / "val.jsonl")
    if va or len(tr) < 2:
        return 0
    group = sorted({(r.get("meta") or {}).get("group") or r["id"] for r in tr})[0]
    moved = [r for r in tr if ((r.get("meta") or {}).get("group") or r["id"]) == group]
    if len(moved) == len(tr):
        moved = tr[:1]
    keep = [r for r in tr if r not in moved]
    for name, rows in (("train.jsonl", keep), ("val.jsonl", moved)):
        (data / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return len(moved)


def model_context_tgz(dest: Path) -> Path:
    """A gzip tar of lab/model-image (Dockerfile, entrypoint, model-context.sh): the Cloud Build source."""
    src = LAB_DIR / "model-image"
    if not (src / "Dockerfile").is_file():
        raise ConfigError(f"{src}/Dockerfile is missing")
    dest.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in sorted(src.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                tar.add(p, arcname=f"model-image/{p.relative_to(src)}")
    dest.write_bytes(buf.getvalue())
    return dest


def run(store: Store, run_id: str, *, mode: str, workdir: Path, parity: str = "required", engine_fp: str | None = None,
        binary: str | None = None, caps: bool | None = None, progress: Callable[[str], None] = print) -> dict:
    """parity: required (GCP) | auto (run when the binary is there) | skip (DRY without a binary).
    caps: r4dataset's section and family caps (tuned for thousands of rows) — off for SMOKE by default."""
    from .. import r4dataset
    from . import parity_gate
    pre = run_prefix(run_id)
    smoke = mode == "smoke"
    local = Path(workdir)
    cases, manifest = local / "cases.jsonl", local / "manifest.json"
    if not store.download(f"{pre}/prepare/cases.jsonl", cases) or not store.download(f"{pre}/prepare/manifest.json", manifest):
        raise ConfigError("the prepare stage's cases.jsonl / manifest.json are missing")
    col = store.read_json(f"{pre}/result-collect.json") or {}
    if col.get("status") != "ok":
        raise ConfigError(f"the collect stage has not finished ok (status {col.get('status')})")
    logs_dir = local / "logs"
    store.sync_down(col["logs_prefix"], logs_dir)
    logs = [logs_dir / n for n in col.get("logs") or [] if (logs_dir / n).is_file()]
    if not logs:
        raise ConfigError(f"no collection logs under {col['logs_prefix']}")
    why = None
    if store.list("cases/why"):
        why = local / "why"
        store.sync_down("cases/why", why)
    if engine_fp is None and os.environ.get("ENGINE"):
        from ..engine import engine_fingerprint
        engine_fp = (engine_fingerprint(os.environ["ENGINE"]) or {}).get("fingerprint")
    out = local / "data"
    progress(f"dataset: building from {len(logs)} log(s)")
    caps = (not smoke) if caps is None else caps
    st = r4dataset.build(cases, logs, manifest, out, why=why, val_pct=20 if smoke else 8, section_cap_scope=("F0", "F1"),
                         section_cap=0.05 if caps else 0.0, family_max_share=0.40 if caps else 0.0,
                         engine_fp=engine_fp, progress=progress)
    if not caps:
        st["val_moved"] = ensure_val(out / "dataset")
    gates: dict[str, Any] = {}
    problems: list[str] = []
    # 1. render parity
    exe = parity_gate.service_binary(binary)
    if parity == "skip" or (parity == "auto" and exe is None):
        gates["parity"] = {"skipped": True}
    else:
        if exe is None:
            raise ConfigError("the parity gate needs the deployed service binary (SOUS_CHEF_BIN / /opt/sous-chef/bin/sous-chef)")
        gates["parity"] = parity_gate.run(manifest, local, binary=exe, engine_path=os.environ.get("ENGINE"))
        if not gates["parity"]["ok"]:
            problems.append("render parity: " + "; ".join(gates["parity"]["failures"][:5] or ["no call checked"]))
    # 2. stats
    if st.get("block_drift_rows"):
        problems.append(f"stats: block_drift_rows {st['block_drift_rows']}")
    if st.get("preview_claim_kept"):
        problems.append(f"stats: preview_claim_kept {st['preview_claim_kept']}")
    # 3. row checks
    gates["rows"] = row_checks(out / "dataset", need_followups=not smoke)
    problems += gates["rows"]["problems"]
    # 4. the train plan
    gates["train_plan"] = train_plan(out / "dataset", f"{run_id}-sft")
    problems += gates["train_plan"]["problems"]
    (out / "gates.json").write_text(json.dumps(gates, indent=1), encoding="utf-8")
    store.sync_up(out, f"{pre}/dataset")
    ctx = model_context_tgz(local / "context.tgz")
    store.upload(ctx, f"{pre}/model/context.tgz")
    metrics = {"rows": st.get("rows"), "train": st.get("train"), "val": st.get("val"),
               "followups": st.get("multi_turn_rows"), "by_family": st.get("by_family"),
               "rejects_first_reason": st.get("rejects_first_reason"), "block_drift_rows": st.get("block_drift_rows"),
               "preview_claim_kept": st.get("preview_claim_kept"), "updates": gates["train_plan"].get("updates"),
               "tokens_train": st.get("tokens_train"), "parity": {k: v for k, v in gates["parity"].items() if k != "failures"}}
    if problems:
        note = notification("stage_failed", "error", "Sous Chef pipeline failed: dataset (gate)",
                            "The training set failed its gates: " + "; ".join(problems[:6]) +
                            f". Nothing was trained. Artifacts: {pre}/dataset/ · Playbook: docs/RETRAIN.md#r3-dataset",
                            {"stage": "dataset", "reason": "gate", "problems": problems}, dedup_key=f"{run_id}:dataset")
        write_result(store, run_id, "dataset", "gate_failed", metrics=metrics, notification_=note)
        raise ConfigError("dataset gates: " + "; ".join(problems[:6]), reason="dataset_gate")
    return write_result(store, run_id, "dataset", "ok", metrics=metrics,
                        artifacts=[f"{pre}/dataset/dataset/train.jsonl", f"{pre}/dataset/dataset/stats.json",
                                   f"{pre}/model/context.tgz"])
