# Training the Sous Chef student on the workstation (RTX 5090)

How to run the lab's SFT → DPO → export → exam on this machine once a run's
dataset exists (`lab/data/int-run2/dataset/` is the next one; run 2 is still
being collected and nothing here reads it). Every number below was measured
on 2026-10-03/04 on this workstation or read from the thermometer kit's own
`run.json` files; nothing was trained for this document — the GPU was busy with
the run-2 headroom eval, so only CPU installs and `--dry-run`s were run.

## 0. What the machine has (measured)

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32,607 MiB, driver 615.71.09 (CUDA 13 capable; PyTorch brings its own CUDA 13.0 runtime — the `/usr/local/cuda-12.*` toolkits on disk are irrelevant and there is no `nvcc` on PATH) |
| RAM / CPU | 125 GB, 16 cores |
| Disk | `/home` 1.9 TB, **132 GB free** (93 % used) at the time of writing |
| Python | system `python3` = 3.14.7 (`lab/.venv`, the stdlib runtime); **Python 3.12.14** is installed by uv at `~/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu` (uv 0.12.19 at `~/.local/bin/uv`). No conda. |
| Base weights | `unsloth/Qwen3.5-9B` already in `~/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B` (snapshot `005429ce…`, 19 GB, 4 safetensors shards) — no download needed |
| Lab Ollama | `~/.local/opt/ollama-0.34.4/bin/ollama serve`, **0.34.4**, `OLLAMA_HOST=127.0.0.1:11435`, `OLLAMA_MODELS=~/.ollama-lab/models` (173 GB), `OLLAMA_CONTEXT_LENGTH=32768`, `OLLAMA_KEEP_ALIVE=30m`; has `qwen3.5:9b` and `qwen3.5:9b-ctx64k`. `/usr/local/bin/ollama` is the **0.5.7** client, which refuses a Modelfile with `RENDERER` — never use it for the export. |

## 1. The training environment: `lab/.venv-train`

### Where the thermometer's environment is

The thermometer kit that trained the 9B here lives at
`/home/gtong/projects/breadstack-thermometer/lab/in-house-model/.venv`
(uv-managed, prompt `thermometer-lab`, **Python 3.12.14**, 5.3 GB). Its
installed set, from `uv pip freeze` on 2026-10-03:

```
torch==2.12.1+cu130   torchvision==0.27.1+cu130   triton==3.7.1   torchao==0.18.0
unsloth==2026.9.11    unsloth-zoo==2026.9.7       transformers==5.5.0
trl==0.24.0           peft==0.21.0                datasets==4.3.0   accelerate==1.15.0
bitsandbytes==0.50.2  xformers==0.0.35            cut-cross-entropy==25.1.1
flash-linear-attention==0.5.2  (fla-core==0.5.2)   huggingface-hub==1.33.0
```

That venv imports torch / trl / peft / transformers / datasets / fla cleanly
with the GPU hidden (`CUDA_VISIBLE_DEVICES=`), 6.3 s. It can be used directly:

```sh
make -C lab train ... TRAINPY=/home/gtong/projects/breadstack-thermometer/lab/in-house-model/.venv/bin/python
```

(`-m souschef_lab.…` resolves from `lab/` as the working directory, so nothing
has to be installed into that venv.) But it is another project's environment
and a plain `uv sync` there removes its train group, so the lab has its own.

### `make -C lab train-env` (done 2026-10-04T05:39Z)

```sh
make -C lab train-env
```

does, verbatim:

```sh
test -x .venv-train/bin/python || uv venv --python 3.12 .venv-train
uv pip install --python .venv-train/bin/python \
    --index https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match \
    -c train-constraints.txt -e '.[train]'
CUDA_VISIBLE_DEVICES= .venv-train/bin/python -c 'import torch, trl, peft, transformers, fla; print(...)'
```

- `lab/train-constraints.txt` is the thermometer venv's `uv pip freeze` (108
  pins), so `.venv-train` is the measured set, not a fresh resolution.
  `lab/pyproject.toml`'s `[train]` extra gained `torchvision`, `xformers` and
  `flash-linear-attention==0.5.2` to match it.
- Torch comes from PyTorch's cu130 index: PyPI's default `torch==2.12.1` wheel
  is not the `+cu130` build the 5090 was measured with.
- **Measured:** `uv venv` 0.01 s; the install resolved 105 packages in 2.3 s and
  installed them in **4.4 s wall** because every wheel but two was already in
  `~/.cache/uv` (11 GB; the thermometer's install) — only `xformers` (5.3 MiB)
  and `torchao` (3.2 MiB) were downloaded. `du` reports 5.2 GB for
  `.venv-train`, but uv hard-links from its cache: `df` on `/home` showed
  132 GB free before and after. From a cold cache expect the thermometer's
  figure: 87 packages, 5.1 GB download, 49 s.
- Verified in `.venv-train` (CPU only):
  `python 3.12.14 · torch 2.12.1+cu130 (cuda build 13.0) · trl 0.24.0 · peft 0.21.0 · transformers 5.5.0 · datasets 4.3.0 · accelerate 1.15.0 · bitsandbytes 0.50.2 · xformers 0.0.35 · fla 0.5.2 · triton 3.7.1 · unsloth 2026.9.11 · unsloth_zoo 2026.9.7 · cut_cross_entropy 25.1.1 · torchao 0.18.0+cu130`.
- `import unsloth` **cannot be checked without the GPU**: `unsloth_zoo/device_type.py:389`
  raises `NotImplementedError: Unsloth cannot find any torch accelerator? You
  need a GPU.` when `CUDA_VISIBLE_DEVICES` is empty (both venvs, same
  version). `make -C lab train-gpu-check` runs `import unsloth, trl, peft` with
  the GPU visible; it opens a CUDA context on the card, so run it only when
  Ollama is idle (not run for this document).
- The `.venv-train/bin/souschef-lab` entry point and `.venv-train/bin/hf` are
  installed; `.venv-train/` and `unsloth_compiled_cache/` (Unsloth writes it in
  the working directory) are in `lab/.gitignore`.

### Compatibility notes (measured, not assumed)

- **Qwen3.5 support.** transformers 5.5.0 ships `models/qwen3_5` (and
  `qwen3_5_moe`); the thermometer trained `unsloth/Qwen3.5-9B` with exactly
  this unsloth 2026.9.11 / transformers 5.5.0 / trl 0.24.0 set twice
  (`thermometer-9b-lab1`, `thermometer-9b-lab2`) on this card.
- **Gated DeltaNet in the 9B.** The base's `config.json`: `model_type qwen3_5`
  (`Qwen3_5ForConditionalGeneration`, text part `qwen3_5_text`), 32 layers =
  **24 `linear_attention` (Gated DeltaNet) + 8 `full_attention`**
  (`full_attention_interval 4`, `linear_conv_kernel_dim 4`), hidden 4096,
  bf16, 262,144 max positions. So yes, the 9B has the same Gated DeltaNet
  layers as the 27B. It does **not need** flash-linear-attention to train:
  `thermometer-9b-lab2` (310 min) ran before fla was in the thermometer's train
  group (the round-1 notes: "The Gated DeltaNet layers run on the torch
  fallback"; the focaccia notes: without the package Unsloth injects its
  vendored fla 0.5.1 delta-rule kernels and a plain `Conv1d`). With
  `flash-linear-attention==0.5.2` installed, transformers'
  `is_flash_linear_attention_available()` is `torch.cuda.is_available() and
  fla >= 0.2.2` — True on the GPU, False in the CPU-only check above — and
  `modeling_qwen3_5.py` then imports `fla.modules.FusedRMSNormGated`. Its
  speed effect on the 5090 is **unmeasured** (the thermometer only used it on
  the cloud VM).
- **causal-conv1d** is not installed and cannot be built here: it needs a CUDA
  13.0 `nvcc` and a host compiler CUDA 13 accepts; this workstation has gcc 16
  and no nvcc (the thermometer's `g4-kernels` group is for its VM only).
  transformers falls back to a plain `Conv1d` (correct, a little slower).

## 2. The GPU must be free: Ollama

At the time of writing `nvidia-smi` showed 9,885 MiB used: the lab Ollama's
`llama-server` holding `qwen3.5:9b-ctx64k` (9,302 MiB; `ollama ps` says 7.5 GiB
at ctx 65536) for the run-2 headroom eval, plus Chrome (90 MiB) and nautilus
(42 MiB). SFT of the 9B at `max_seq_length 32768` peaked at **25.8 GiB reserved
by PyTorch** (`thermometer-9b-lab1`, 28,803 MiB for the whole card) and
**26.8 GiB** (`thermometer-9b-lab2`, longest row 32,454 tokens). All figures
in this section are GiB (`peak_vram_gb` in the thermometer's `run.json` is
`torch.cuda.max_memory_reserved() / 2**30`; the card is 32,607 MiB = 31.8 GiB).
26.8 + 9.1 GiB does not fit in 31.8 GiB, so:

- **Do not start any GPU job while an eval or collection is running on
  `127.0.0.1:11435`.** Check with `curl -s http://127.0.0.1:11435/api/ps` (an
  empty `models` list = idle) and `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`.
- When the eval is finished, the model stays loaded for `OLLAMA_KEEP_ALIVE=30m`.
  Unload it at once with
  `curl -s http://127.0.0.1:11435/api/generate -d '{"model":"qwen3.5:9b-ctx64k","keep_alive":0}'`
  (repeat for whatever `ollama ps` lists). The Ollama server itself holds no
  GPU memory when idle and can stay up.
- The thermometer's "out of memory at the start" note: it usually means
  something else holds the GPU.

## 3. SFT

```sh
cd /home/gtong/projects/breadstack-sous-chef
make -C lab train-env                                   # once; idempotent (6.7 s on the second run)
make -C lab train-dry DATA=data/int-run2 LABEL=run2     # the three plans, no torch, no GPU
make -C lab train     DATA=data/int-run2 LABEL=run2     # the real run (adapter → lab/runs/adapters/souschef-run2-sft)
```

`make train` runs
`.venv-train/bin/python -m souschef_lab.train --data data/int-run2/dataset --name souschef-run2-sft`
with the recipe: `unsloth/Qwen3.5-9B`, bf16 base + fp32 LoRA (`load_in_16bit`,
no 4-bit), **r 16 / α 16 / dropout 0** on attention and MLP modules,
**lr 2e-4**, **1 epoch**, batch 1 × grad-accum 4 (effective 4), adamw_8bit,
linear schedule, warmup 0.03, weight decay 0.01, **max_seq_length 32768**,
tiled MLP on (auto above 24,576 tokens), seed 3407, no thinking, validation
loss before the first update, every `--eval-steps 50` and after the last.
Flags override any of these (`--epochs`, `--eval-steps`, `--save-steps N` for
checkpoints under `runs/adapters/<name>/checkpoints`, `--val-max-rows`,
`--max-steps 4 --name smoke` for a smoke run); `--config file.yaml|json` too.

**Run-1 dry run (measured, 2026-10-04):** `make -C lab train-dry DATA=data/int-run1 LABEL=run1`
→ SFT plan: rows `{"train": 57, "val": 10, "val_scored": 10, "followups": 0, "over_length_estimate": 0}`,
`tokens_estimate {"median": 15194, "max": 15532, "total": 861725, "chars_per_token": 3.0}`,
**15 updates**; "Dry run: the data is well-formed." 0.12 s for all three plans.

### Expected wall time

The thermometer's measured 9B runs on this card, from their `run.json`:

| run | rows | train tokens | updates | tokens/s | train time | wall | peak reserved |
|---|---|---|---|---|---|---|---|
| `thermometer-9b-lab1` (2026-09-26) | 116 (+15 val) | 1,669,142 | 29 | 1,656 | 1,007.7 s | **19.1 min** | 25.8 GiB |
| `thermometer-9b-lab2` (2026-09-29) | 1,701 (+56 val) | 30,392,741 | 426 | 1,672 | 18,175 s | **310.3 min** | 26.8 GiB |

Both at `max_seq_length 32768`, tiled MLP, r16/α16, lr 2e-4, 1 epoch, bf16,
without flash-linear-attention. The row rate is 310.3 min / 1,701 rows =
**10.9 s per row**; the token rate **~1,670 tokens/s**. Fixed overhead:
lab1's wall minus its train time is 2.3 min (model load, first validation);
the first update took 83–100 s while PyTorch compiled.

Sous Chef rows are a little shorter than the thermometer's (median 15,194
tokens vs the thermometer's ~17,900 average), so token-scaling is the better
estimate and row-scaling the upper bound. At run 1's median row:

| training rows | tokens (≈ rows × 15,194) | token-scaled | row-scaled |
|---|---|---|---|
| 57 (run 1) | 0.9 M | 9 min | 10 min |
| 200 | 3.0 M | 30 min | 36 min |
| 500 | 7.6 M | 76 min | 91 min |
| 1,000 | 15.2 M | 151 min | 182 min |
| 1,701 | 25.8 M | 258 min | 310 min |

Add ~3 min of load + first validation, plus one validation pass over the val
set per `--eval-steps`. The dry run's `tokens_estimate.total` for run 2 is the
number to divide by 1,670. These estimates assume the same kernels as lab2
(no fla); with fla 0.5.2 present the speed is unmeasured — the first run-2 SFT
will measure it (`run.json` → `train_tokens_per_second`).

### VRAM

bf16 base (≈18 GB of weights) + fp32 r16 LoRA + 32k-token activations with
tiled MLP: **25.8–26.8 GiB reserved** on the two thermometer runs, 28.1 GiB
(28,803 MiB) for the whole card including the desktop. Fits the 5090's
31.8 GiB only with Ollama
unloaded. If it does not fit: `--max-seq-length 24576` (drops the longest
rows; run 1's longest estimated row is 15,532 tokens, so nothing would drop).
The thermometer's Trainer-evaluation OOM (full fp32 logits, 14.5 GiB for one
15k row) does not apply: the lab's validation uses the fused training loss.

### Outputs

`lab/runs/adapters/souschef-run2-sft/`: `adapter/` (the LoRA + tokenizer),
`run.json` (settings, counts, losses, `train_tokens_per_second`,
`peak_vram_gb`, `base_eval_loss` → `last_eval_loss`, versions), `loss.jsonl`
(one line per update), `sample_rendered.txt` (the first training text as the
model sees it), `checkpoints/` when `--save-steps` > 0. `lab/runs/` is
git-ignored. Unsloth also writes `lab/unsloth_compiled_cache/` (ignored).

**Rows are rendered through the serving parity layer** (`souschef_lab/serving.py`,
LAB.md §5 "Serving parity"), never the base's chat template: `sample_rendered.txt`
is the text Ollama 0.34.4's `qwen3.5` renderer puts in front of the served
student with thinking disabled — its tools block (Go struct order, dropped
`additionalProperties`/`strict`, `<` escapes, spaced JSON), its echo of
every tool call (sorted keys, compact JSON) and the empty `<think>` block that
opens each assistant turn — and `run.json` records it as `render`. The parity
fixture `lab/fixtures/parity/served.json` (Ollama's own render of the real
captures, `make -C lab parity-write` through `lab/parity/`, a Go module
pinning ollama v0.34.4) is what `tests/test_serving.py` holds that render to.
**Re-capture the parity fixture when Ollama's renderer version changes**
(a bump in `lab/model-image`, `parity/go.mod`, the export's `RENDERER`/`PARSER`,
or how the service builds a request), then rebuild the dataset before training.
A row whose call Ollama would echo as another call is dropped and counted
(`run.json` → `rows.dropped_unservable`); `--thinking` is refused.

## 4. DPO

```sh
make -C lab dpo DATA=data/int-run2 LABEL=run2           # pairs: data/int-run2/pairs, SFT: runs/adapters/souschef-run2-sft/adapter
```

Recipe: sigmoid DPO, **β 0.1, lr 5e-6**, 1 epoch, batch 1 × grad-accum 16,
warmup 0.1, adamw_8bit, `max_length` = the SFT run's `max_seq_length`
(32,768), continued from the SFT adapter; the reference is the frozen SFT
model's log-probs, computed once into `runs/adapters/<name>-dpo/ref_logps.jsonl`
(`ref_model=None`, so one model in memory). Run 1 has **2 pairs → 1 update**
(dry run: `rows {"train": 2, "val": 0}`, `side_median 778` tokens); the dry
run also notes the SFT adapter must exist first ("has no adapter_config.json
yet … the run would stop here"). **No DPO run has been timed on this card**
(the thermometer's DPO runs were on its VM; no `run.json` of one exists in its
`runs/adapters/`), so there is no wall-time figure to scale from. Skip DPO when
`pairs/stats.json` is as thin as run 1's: `passk` says "SFT then DPO" only
when pass@4 − pass@1 ≥ 5 points.

## 5. Export and the Ollama tag

```sh
export OLLAMA_HOST=127.0.0.1:11435
export LAB_OLLAMA_BIN=$HOME/.local/opt/ollama-0.34.4/bin/ollama
make -C lab export DRY=1 LABEL=run2 ADAPTER=runs/adapters/souschef-run2-sft/adapter   # the plan + a Modelfile preview
make -C lab export        LABEL=run2 ADAPTER=runs/adapters/souschef-run2-sft/adapter CREATE=1
```

(`ADAPTER` defaults to the DPO adapter, `runs/adapters/souschef-run2-dpo/adapter`;
name it explicitly when exporting the SFT one.) The export

1. runs `ollama show qwen3.5:9b --modelfile` on the 0.34.4 client (it refuses
   the 0.5.7 one: "the ollama client … is 0.5.7 but the server at
   OLLAMA_HOST=… is 0.34.4"). The base's lines, read on 2026-10-04:
   `TEMPLATE {{ .Prompt }}`, `RENDERER qwen3.5`, `PARSER qwen3.5`,
   `PARAMETER presence_penalty 1.5 / temperature 1 / top_k 20 / top_p 0.95`;
2. `FastModel.from_pretrained(adapter, load_in_16bit=True)` — merges the LoRA
   into the bf16 base **on the GPU** (so the GPU must be free here too);
3. `save_pretrained_gguf(… 'q4_k_m')` and `(… 'q8_0')` → `runs/exports/souschef-run2/souschef-run2-q4_k_m.gguf` and `-q8_0.gguf`;
4. writes `Modelfile.q4_k_m` / `Modelfile.q8_0` (`FROM ./<gguf>`, the base's
   lines, `PARAMETER num_ctx 65536`) and, with `CREATE=1`,
   `ollama create souschef-run2:q4_k_m -f Modelfile.q4_k_m` and the q8_0 tag.

**Measured by the thermometer** (same recipe, q4_k_m only): 3 min 25 s end to
end; the q4_k_m GGUF is 5.4 GB; the intermediate 16-bit merge folder
(`runs/exports/<name>/gguf`) is 18 GB and can be deleted once the tag exists;
the created model takes about 7,500 MiB of GPU memory at 64k when served.
The q8_0 GGUF is not measured (arithmetic, not a measurement: one byte per
weight for the 9.2 B exported parameters ≈ 9–10 GB). With 132 GB free,
one export fits; delete the merge folder between runs. Check the tag with
`OLLAMA_HOST=127.0.0.1:11435 $LAB_OLLAMA_BIN show souschef-run2:q4_k_m` —
capabilities must include `tools`, else the student cannot call a single op.

## 6. The exam: a new route on the local service

The service already has a workstation profile for Ollama-backed routes,
`lab/headroom.env` (`PORT=8091`, Postgres on `:5433`, `SERVICE_KEYS`, no
Anthropic route, `MODEL_ROUTES` with `qwen9b`/`qwen27b`/`gemma31b` →
`http://127.0.0.1:11435`, `stream:false`, `max_tokens 4096`,
`MODEL_CALL_TIMEOUT=180s`, `MAX_CONCURRENT_TURNS=2`). Add the student:

1. In `lab/headroom.env`, add to `MODEL_ROUTES`
   `"souschef-run2":{"kind":"messages-compatible","url":"http://127.0.0.1:11435","auth":"none","model":"souschef-run2:q4_k_m","max_tokens":4096,"stream":false}`
   and `souschef-run2` to `MODEL_ROUTE_ALLOWLIST` (a request's `model` must be
   allowlisted; `DEFAULT_ROUTE` can stay `qwen9b`). A second route for
   `souschef-run2:q8_0` decides the quant.
2. Start Postgres and the service, detached (an exam of 66 cases × 8 outlasts
   a 2-hour agent slot — one run-1 eval was lost that way):
   ```sh
   docker compose up -d --wait postgres
   make build
   set -a; . lab/headroom.env; set +a
   setsid nohup ./service/bin/sous-chef >> /tmp/souschef-headroom.log 2>&1 < /dev/null &
   ```
3. Run the held-out exam from the lab venv (the stdlib one is enough):
   ```sh
   export COOKIE_CUTTER_DIR=/home/gtong/projects/breadstack-cookie-cutter
   export ENGINE=$COOKIE_CUTTER_DIR/launcher/client/src/sousChef/ops/headless.mjs SOUS_CHEF_ENGINE=$ENGINE
   lab/.venv/bin/souschef-lab eval --url http://127.0.0.1:8091 --key "$SERVICE_KEYS" \
       --route souschef-run2 --label souschef-run2-q4 --samples 8 \
       --cases lab/data/int-run2/cases.jsonl --manifest lab/data/int-run2/manifest.json --engine "$ENGINE"
   ```
   or `make -C lab eval ROUTE=souschef-run2 LABEL=souschef-run2-q4 DATA=data/int-run2 MANIFEST=data/int-run2/manifest.json SOUS_CHEF_URL=http://127.0.0.1:8091 SOUS_CHEF_KEY="$SERVICE_KEYS"`
   (one sample per case; `--samples` needs the CLI). The run lands in
   `lab/runs/eval-souschef-run2-q4-<UTC>.json`; compare with the base's with
   `souschef-lab eval --compare <base run> <student run>` and
   `souschef-lab passk <runs…>`.

**Measured baseline (run 1, 2026-10-04):** `qwen3.5:9b-ctx64k` through this
exact wiring, 66 exam cases × 8 samples, took 03:47–05:32 UTC (**105 min**),
p50 8.9 s / p95 20.8 s per turn, **165/528 pass (31.2 pts)** on rules
`verify-2` — the number the student has to beat. The first exam answer loads
the model (~7.5 GiB at 64k); keep the GPU free of training while it runs.

## 7. Order of operations, in one place

1. Wait for run 2's collection and headroom eval to finish; `curl …/api/ps`
   shows no models (or unload with `keep_alive: 0`).
2. `make -C lab train-env` (already built; 6.7 s to re-verify) and
   `make -C lab train-gpu-check` (first use of the GPU: `import unsloth`).
3. `make -C lab train-dry DATA=data/int-run2 LABEL=run2` — read
   `tokens_estimate.total`, divide by 1,670 for the minutes.
4. `make -C lab train DATA=data/int-run2 LABEL=run2`; optional first
   `make -C lab train DATA=data/int-run2 LABEL=run2-smoke` with
   `--max-steps 4` via `souschef-lab train … --max-steps 4` (the thermometer's
   smoke took 4 min 36 s).
5. `make -C lab dpo …` only if the pairs justify it.
6. `OLLAMA_HOST=127.0.0.1:11435 LAB_OLLAMA_BIN=~/.local/opt/ollama-0.34.4/bin/ollama make -C lab export LABEL=run2 ADAPTER=… CREATE=1`.
7. Route + service + `souschef-lab eval` as in §6; compare with the run-2
   `qwen9b` baseline in `lab/data/int-run2/`.
