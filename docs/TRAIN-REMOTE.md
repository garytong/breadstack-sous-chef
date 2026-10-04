# Training the Sous Chef student on a GCE GPU VM (train-remote)

The workstation recipe (docs/TRAIN-WORKSTATION.md) run on a rented GPU in
`breadstack-sous-chef-int`, so the 5090 stays with the collection and the lab
Ollama on `:11435` is never touched. The recipe is **unchanged**: `unsloth/Qwen3.5-9B`,
bf16 base + fp32 LoRA r16/α16, lr 2e-4, 1 epoch, batch 1 × grad-accum 4,
`max_seq_length 32768` (tiled MLP on), seed 3407 — what lab1/lab2 measured at
25.8–26.8 GiB reserved. Only the machine changes.

Decisions (the 2026-10-04 plan; nothing here was run on a VM yet, every
number is the workstation/thermometer measurement or a ratio from it):

| | |
|---|---|
| Machine | **`a2-highgpu-1g` (1× A100 40 GB), us-central1, on-demand** — the one no-quota-request GPU that fits the recipe as measured (26.8 GiB < 37.5 GiB usable); int already holds `NVIDIA_A100_GPUS 16` there. A2 exists only in `us-central1-a/b/c`, not Toronto — training data leaves Canada (int data; the thermometer got the same approval for Iowa). |
| Switch | `a3-highgpu-1g` (1× H100 80 GB) **Spot** for the short run only: fastest, but preemptible and `train.py` has no `--resume` (adapter-latest only, no optimizer state) — a preempted run restarts from zero. |
| Image | DLVM `common-cu129-ubuntu-2404-nvidia-580`: the only image proven to run the pinned wheel set (`torch 2.12.1+cu130` needs driver R580+). |
| Infra | Terraform root `infra/envs/train` (ported from the thermometer's `infra/training/`): own VPC, IAP-only SSH, Cloud NAT + Private Google Access, bucket `gs://breadstack-sous-chef-int-training`, SA `sous-chef-train`, `max_run_duration` 12 h hard stop, notify-only budget. `terraform apply` is the user's; the agent plans only. |
| Export | On the same VM right after training, with `--base-modelfile lab/base-modelfiles/qwen3.5-9b.modelfile` — the base's lines as recorded in TRAIN-WORKSTATION.md §5, committed so no Ollama is ever asked. q4_k_m only (the 16 GiB Cloud Run container cannot mmap a ~9–10 GB q8_0 comfortably). |
| Cost | A100 on-demand ≈ US$3.67/h; H100 Spot ≈ US$6.57/h; disk ≈ US$0.08/h; bucket negligible. The VM bills from `terraform apply` to the stop — take it down the same session. |

## 0. Before the VM: what leaves the workstation (once)

Three things go to the bucket; none of them a secret, none of them the
collection in progress (`lab/data/int-run3` is not touched).

```sh
cd /home/gtong/projects/breadstack-sous-chef
BUCKET=gs://breadstack-sous-chef-int-training

# 1. The base weights — 18 GiB, so later VMs copy in-region instead of 19 GB
#    through Cloud NAT per run. The local cache's snapshot files are symlinks
#    into ~/.cache/huggingface/hub/blobs/<xx>/ (hf_xet); `make base-tar`
#    follows them (-h) and skips the model's own blobs/ dir, so the tar holds
#    each shard once and `tar -x -C $HF_HOME` restores a cache transformers
#    reads offline (refs/main → snapshots/005429ce…). Upload bandwidth here is
#    unmeasured; stream it rather than spending 18 GiB of the 132 GB free:
make -C lab base-tar OUT=- | gcloud storage cp - $BUCKET/base/unsloth-qwen3.5-9b-hf.tar
#    (or OUT=/path/x.tar, then gcloud storage cp.) Check: `gcloud storage ls -l $BUCKET/base/`
#    should show ≈ 19.3 GB.

# 2. The dataset — 15 MB. trainkit has no gs:// support, so plain files.
gcloud storage rsync --recursive lab/data/merged-r12 $BUCKET/data/merged-r12

# 3. The kit — lab/ at HEAD, code only (lab/data and lab/runs are untracked).
git archive --format=tar HEAD lab | gcloud storage cp - $BUCKET/kit/lab-$(git rev-parse --short HEAD).tar
```

`make -C lab train DRY=1 DATA=data/merged-r12 LABEL=r1` first (not `train-dry`:
that one also plans the DPO and stops on a dataset without `pairs/`). It prints
the rows and `tokens_estimate.total`, the number every time estimate below
divides. Run 1 (2026-10-04): rows `{"train": 162, "val": 25, …, "over_length_estimate": 0}`,
tokens `{"median": 15263, "max": 15833, "total": 2485120}`, **41 updates**.

## 1. VM up (the user; billed from here)

```sh
cd infra/envs/train && cp terraform.tfvars.example terraform.tfvars   # machine_type / zone are the knobs
terraform init && terraform plan                 # the agent stops here
terraform apply                                  # the user (or make -C lab train-vm-up once that target lands)
gcloud compute ssh --tunnel-through-iap --zone us-central1-a <vm>   # IAP only; no public SSH
```

The tfvars machine switch is the one knob: `a2-highgpu-1g` on-demand (default)
or `a3-highgpu-1g` Spot with stop-on-preempt; `variables.tf` refuses any zone
but `us-central1-a/b/c`. Zone a → b → c on an A100 stockout (the thermometer
saw them): down, change `zone`, up — it replaces the VM and its disk. Enable `iap.googleapis.com` and
`billingbudgets.googleapis.com` in int first — neither is on today.

## 2. On the VM, once per disk (≈ 30 min, most of it waiting on the tar)

The startup script (the thermometer's, ported) installs uv, sets
`HF_HOME=/work/hf` for login shells and pulls `base/unsloth-qwen3.5-9b-hf.tar`
into it in the background (`journalctl -u <unit> -f` until "ready"). If that
fetch is not wired yet, do it by hand — the same command:

```sh
export BUCKET=gs://breadstack-sous-chef-int-training HF_HOME=/work/hf
sudo install -d -m 1777 /work/hf
gcloud storage cat $BUCKET/base/unsloth-qwen3.5-9b-hf.tar | tar -x -C $HF_HOME     # 18 GiB, in-region
ls $HF_HOME/hub/models--unsloth--Qwen3.5-9B/snapshots/*/ | grep -c safetensors      # 4 shards + index
export HF_HUB_OFFLINE=1                     # read the cache; never ask huggingface.co through NAT

mkdir -p /work/$USER && cd /work/$USER
gcloud storage cat $BUCKET/kit/lab-<sha>.tar | tar -x && cd lab
sudo apt-get install -y g++ python3.12-dev git cmake libcurl4-openssl-dev     # the DLVM has none; unsloth's
                                                                             # GGUF writer builds llama.cpp (§5)
make train-env                              # uv venv 3.12 + the [train] extra, torch from the cu130 index,
                                            # pinned by train-constraints.txt: 5.1 GB, ~49 s measured cold
make train-gpu-check                        # `import unsloth` + the card: expect "NVIDIA A100-SXM4-40GB … bf16 True"
                                            # — this has never been run on a GPU (the 5090 was busy); a failure
                                            # here is the driver/wheel pin (R580+), not the data
gcloud storage rsync --recursive $BUCKET/data/merged-r12 data/merged-r12
make train DRY=1 DATA=data/merged-r12 LABEL=r1   # same rows/tokens as on the workstation
tmux new -s train                           # a dropped IAP tunnel must not kill the run
```

`make train-env` wants `uv` on PATH (the startup script's `/usr/local/bin/uv`;
else `curl -LsSf https://astral.sh/uv/0.12.19/install.sh | sh`). The lab's own
`PYTHONPATH`/engine variables are irrelevant here — nothing on the VM runs the
browser engine or the collection; `make test` would skip those tests.

## 3. The smoke: `--max-steps 4` (≈ 5–8 min on the A100, ≈ US$0.50)

`train.py` has `--max-steps N` ("stop after this many updates; overrides
`--epochs`"); the Makefile passes it as `MAX_STEPS`. The smoke runs the base
validation loss, 4 updates (16 rows at effective batch 4) and the final
validation, then writes the normal `run.json`.

```sh
make train DATA=data/merged-r12 LABEL=r1-smoke MAX_STEPS=4
python3 -c 'import json; r=json.load(open("runs/adapters/souschef-r1-smoke-sft/run.json")); \
  print({k: r.get(k) for k in ("base_eval_loss","last_eval_loss","train_tokens_per_second","peak_vram_gb")})'
```

Read three numbers before spending on the full run:

- **loss** — `base_eval_loss` well above 3, or not falling by the fourth
  update, means the render or the load is wrong: read `sample_rendered.txt`
  beside `run.json`. Stop. The text is the serving parity render (LAB.md §5:
  `souschef_lab/serving.py`, the bytes Ollama 0.34.4's `qwen3.5` renderer
  shows the served student with thinking disabled — not the base's chat
  template); **rows are rendered through the serving parity layer** on the VM
  too, from the kit's own Python (no Go needed there). **Re-capture the parity
  fixture** (`lab/fixtures/parity/served.json`, `make -C lab parity-write` on
  the workstation, through `lab/parity/`) **when Ollama's renderer version
  changes** — `lab/model-image`, `parity/go.mod`, the export's `RENDERER` /
  `PARSER` — and push the kit again before training on it.
- **speed** — `train_tokens_per_second`. The 5090 does ~1,670 tok/s on this
  recipe; the A100-40 is **unmeasured** and the table below assumes 0.5–0.65 of
  that. Re-quote: `tokens_estimate.total / tok_per_s / 60` minutes, plus ~3 min
  load + first validation, plus one validation pass per `--eval-steps 50`.
- **memory** — `peak_vram_gb` ≥ 36 on the 40 GB card: `--max-seq-length 24576`
  (run 1's longest row is 15,833 tokens, nothing drops). Do **not** go to 16,384:
  80 tokens of headroom over the longest row, and rows over the cap are dropped
  silently.

The thermometer's smoke of the same recipe on the 5090 took 4 min 36 s.

## 4. The full run

```sh
make train DATA=data/merged-r12 LABEL=r1          # → runs/adapters/souschef-r1-sft/{adapter,run.json,loss.jsonl,sample_rendered.txt}
gcloud storage rsync --recursive runs/adapters/souschef-r1-sft $BUCKET/adapters/souschef-r1-sft   # right away, before the export
```

Rsync the adapter out as soon as it exists: an H100 Spot preemption or the
12 h `max_run_duration` stop loses whatever is only on the VM's disk.
`--save-steps N` writes checkpoints under `runs/adapters/<name>/checkpoints`
for a long run, but there is nothing to resume them with — they are for
inspection, not recovery.

## 5. Export: GGUF q4_k_m on the same VM (≈ 5 min)

```sh
make export LABEL=r1 ADAPTER=runs/adapters/souschef-r1-sft/adapter QUANTS=q4_k_m \
    BASE_MODELFILE=base-modelfiles/qwen3.5-9b.modelfile
# → runs/exports/souschef-r1/{souschef-r1-q4_k_m.gguf (5.4 GB), Modelfile.q4_k_m, export.json, gguf/ (18 GB merge)}
gcloud storage rsync --recursive --exclude 'gguf/.*' runs/exports/souschef-r1 $BUCKET/exports/souschef-r1
```

- `--base-modelfile` replaces `ollama show qwen3.5:9b --modelfile`: no Ollama
  client, no `OLLAMA_HOST`, nothing on the VM talks to the lab server. The
  committed file holds exactly the lines read on 2026-10-04 (`TEMPLATE {{ .Prompt }}`,
  `RENDERER qwen3.5`, `PARSER qwen3.5`, `PARAMETER presence_penalty 1.5 /
  temperature 1 / top_k 20 / top_p 0.95`); `tests/test_train_dry.py` checks the
  export reads them. Re-record it when the base tag is re-pulled.
- Without `CREATE=1` the export writes the GGUF and `Modelfile.q4_k_m` and
  stops — `ollama create` happens later, inside the serving image build
  (plan §2: `lab/model-image/`, model name `sous-chef`, `FROM ./<gguf>`,
  `num_ctx 65536`). Do not install Ollama on the VM for it.
- The merge runs on the GPU (`FastModel.from_pretrained(adapter, load_in_16bit=True)`),
  so export before the VM goes down, after training. Unsloth's `save_pretrained_gguf`
  clones and builds llama.cpp on first use: that is the `git cmake g++ libcurl4-openssl-dev`
  in §2 and GitHub reach through Cloud NAT. If the build fails,
  `--method merged` writes the 16-bit model and prints the two llama.cpp
  commands to run by hand; the Modelfile step is the same.
- Measured on the 5090 (thermometer, q4_k_m): 3 min 25 s end to end, 5.4 GB
  GGUF, 18 GB intermediate under `gguf/` — leave that on the VM. The A100 is
  unmeasured (unsloth's merge has a per-tensor CPU fallback; the a2 has 85 GB RAM).

## 6. VM down (the user)

```sh
gcloud storage ls -l $BUCKET/adapters/souschef-r1-sft/ $BUCKET/exports/souschef-r1/   # both complete?
cd infra/envs/train && terraform destroy          # or `gcloud compute instances stop …` to keep the disk (≈ US$0.08/h)
```

Then on the workstation: `gcloud storage rsync --recursive $BUCKET/exports/souschef-r1 lab/runs/exports/souschef-r1`
(5.4 GB) for the image build, and the adapter into `lab/runs/adapters/` for
the record. The bucket's copies stay for the lifecycle rule.

## 7. Time and cost

Tokens from the dry run; the 5090 reference is lab1/lab2's 1,670 tok/s; the
A100-40 column assumes 0.5–0.65 of that (**unmeasured** — the smoke replaces it);
the H100 column the thermometer's ratio. Costs include ~30 min of bootstrap.

| Run | Tokens | 5090 ref | A100-40 est. | A100 cost @ US$3.67/h | H100 Spot @ US$6.57/h |
|---|---|---|---|---|---|
| run 1, `merged-r12` (162 + 25 val) | 2.49 M | 25 min | 45–60 min | ≈ US$5–6 | 20–30 min, ≈ US$5 |
| ~900 rows (after run 3) | ~14 M | 2.3 h | 3–5 h | ≈ US$13–20 | 1–1.5 h, ≈ US$10 (preemptible, no resume) |

Plus the smoke (≈ US$0.50), the export (~5 min on the GPU), disk at
~US$0.08/h, and ~US$0.15 to read the 18 GiB tar from the bucket on each new
disk. Estimates are ±2× until the first A100 smoke writes `train_tokens_per_second`.

## 8. Fallback ladder

1. **A100 stockout in `us-central1-a`** → the same root in `-b`, then `-c`
   (the thermometer met stockouts; a plain re-apply with the zone changed).
2. **No A100 anywhere** → `a3-highgpu-1g` H100 **Spot** via the tfvars switch;
   run 1 only (~25 min exposure). Rsync the adapter out the moment training
   ends; a preemption mid-run means starting over, so do not put the 900-row
   run on Spot.
3. **Neither** → `g2-standard-8` (L4 24 GB) in Toronto with the **4-bit port**
   (`load_in_4bit=True`, five lines from the thermometer's train.py), `--tiled-mlp on`
   and a smoke first. Different numerics from lab1/lab2, bf16 at 32k does not fit
   (22–23 GiB estimated vs 22.5 usable), and exporting a 4-bit-trained adapter
   through `export.py` is unverified. Last resort, not a plan.

Not on the ladder: Cloud Run Jobs GPU (1 h task cap, L4 only, driver unknown,
in-memory filesystem vs the 18 GiB cache), A100-80 (quota 0), G4 RTX PRO 6000
(stockouts, no Flex grant).

## 9. What can go wrong (ranked)

1. `make train-gpu-check` fails to import unsloth — the driver/wheel pin. Only
   the `…nvidia-580` DLVM image is proven; do not switch images to fix it.
2. The A100's tok/s is half the guess → the 900-row run approaches the 12 h
   `max_run_duration`. Re-set it from the smoke; a hard stop loses the run.
3. llama.cpp will not build on the VM (toolchain, or GitHub unreachable through
   NAT) → `--method merged` + hand conversion; the adapter is already in the bucket.
4. `base-tar` was made with the wrong cache → `grep -c safetensors` in §2 shows
   fewer than 4 shards; the tar must come from `make base-tar`, which follows
   the xet symlinks (a plain `tar` of the directory is 8.6 MB of dangling links).
5. Forgetting the VM: on-demand A100 is US$88/day. The 12 h `max_run_duration`
   is the backstop, not the plan.
