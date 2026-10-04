# infra/envs/train — the Sous Chef training VM (one GPU, us-central1)

A Terraform root of its own for training the 9B student on a Google Compute
Engine GPU VM: the measured workstation recipe (`docs/TRAIN-WORKSTATION.md`:
bf16 LoRA r16, `max_seq_length 32768`, tiled MLP) unchanged, on a card with
room for it. Ported from `breadstack-thermometer/infra/training` (the Focaccia
VM) with the shapes the 9B needs. The lab's `make -C lab train-…` targets wrap
every command below; `docs/TRAIN-REMOTE.md` is the run-by-run narrative.

**Never applied by an agent.** A person runs `train-vm-apply`, `train-vm-up`
and `train-vm-down` from their own login. The VM bills by the hour from
`train-vm-up` to `train-vm-down` (or to its own hard stop).

## What it makes (`terraform plan`, 2026-10-04: 17 resources, 18 with the VM)

| | | |
|---|---|---|
| APIs | `compute`, `storage` (already on), `iap`, `billingbudgets` (not on yet) | `apis.tf`; never disabled by a destroy |
| Network | VPC `sous-chef-train`, subnet `10.20.0.0/24` with Private Google Access, Cloud Router + NAT, firewall: SSH in from IAP's range only, tcp 80/443 out, everything else out denied | `network.tf`; the VM has no external address |
| Identity | SA `sous-chef-train`: objectAdmin on the bucket (authoritative binding with `operators`), log + metric writer, nothing else | `iam.tf` |
| Bucket | `gs://breadstack-sous-chef-int-training` (US-CENTRAL1): uniform access, public access prevented, no versioning / soft delete; `data/ kit/ runs/` expire after 30 days, `base/` after 180 | `storage.tf` |
| VM | `sous-chef-train`, off by default (`training_vm_enabled`): **`a2-highgpu-1g` (1 × A100 40 GB) ON DEMAND** in `us-central1-a`, 200 GB SSD PD, DLVM image `common-cu129-ubuntu-2404-nvidia-580`, OS Login, Shielded (vTPM + integrity), `max_run_duration` **12 h**, STOP at the hard stop / preemption; switch: **`a3-highgpu-1g` (1 × H100 80 GB) SPOT**, Hyperdisk Balanced | `vm.tf`, `terraform.tfvars` |
| Budget | notify-only, `purpose=sous-chef-train` spend since `budget_start_date`, 50/90/100 % e-mails to the billing account's admins; **stops nothing** | `budget.tf` |

Why these shapes (the plan of 2026-10-04): the recipe peaks at 26.8 GiB on the
5090, so an L4 (24 GB, the only GPU Cloud Run or Toronto offers) does not hold
it and the A100 40 GB does; int already has 16 on-demand A100s of quota
(`NVIDIA_A100_GPUS`, us-central1) and 64 preemptible H100s; on demand because
`train.py` has no `--resume` (a preempted run is lost); the H100 Spot is the
fast fallback for a ~1 h run. A2 Standard and A3 High exist in us-central1
a/b/c only — **the int collection data leaves Canada for the run**: the user
approves that before the first `train-vm-up`.

## Before the first run (once, your login)

```sh
gcloud auth login && gcloud auth application-default login       # terraform uses ADC, make uses gcloud
cd infra/envs/train && cp terraform.tfvars.example terraform.tfvars  # check operators, machine_type, zone, budget
make -C lab train-vm-init          # terraform init against gs://breadstack-sous-chef-int-sous-chef-tfstate, prefix sous-chef/train
make -C lab train-vm-plan          # read-only: 17 to add
make -C lab train-vm-apply         # bucket, SA, network, NAT, budget, APIs — no VM yet, nothing bills by the hour
```

The budget needs `billing.budgets.create` on the billing account
(`01DF7D-F1F664-EC2F0A`); without it, `budget_enabled = false` and make one in
the console. Project Owner covers the rest (network, IAM, bucket, IAP, OS Login).

Then what the VM needs in the bucket (none of it a secret; `lab/data/int-run3`,
the collection in progress, is never touched):

```sh
make -C lab train-remote-push DATA=data/merged-r12
```

pushes three things: `data/merged-r12/` (15 MB, `gcloud storage rsync`), the
lab's **working tree** as `kit/lab.tar.gz` (code, fixtures, `pyproject.toml`,
`train-constraints.txt`; no data, venvs or runs — `KIT-VERSION` inside records
HEAD and whether it was dirty), and the base weights as
`base/unsloth-qwen3.5-9b-hf.tar` through `make -C lab base-tar OUT=-` (18 GiB
streamed from `~/.cache/huggingface`, skipped when the object is already there;
`BASE=force` re-sends, `BASE=skip` never does). Upstream bandwidth from the
workstation is unmeasured: the base upload is the long step, once.

## Up → ssh → train → export → pull → down

```sh
make -C lab train-vm-up            # targeted apply: the VM only. BILLED from here: ~US$3.72/h (A100 on demand, disk incl.)
make -C lab train-vm-ssh           # IAP tunnel, OS Login, then `sudo -iu souschef` (SSH_USER=me for your own shell)
```

On the VM the startup script has started the bootstrap (`sous-chef-bootstrap`,
a transient systemd unit; `sous-chef-train-status` shows where it is,
`journalctl -u sous-chef-bootstrap -f` follows it): uv 0.12.19, the kit from
`kit/lab.tar.gz` into `/work/sous-chef/lab`, `make train-env`
(`lab/.venv-train`, the pinned cu130 set, ~1 min from the index), the base from
`base/unsloth-qwen3.5-9b-hf.tar` into `/work/hf` (in-region, checked shard by
shard), then `make train-gpu-check` (`import unsloth` on the card — never yet
run on a GPU; a failure here is the driver/wheel pin, not the data) →
`/work/gpu-check.txt`. About 10–30 min. Everything under `/work` belongs to the
`souschef` user. Then, as `souschef`:

```sh
sous-chef-data-pull merged-r12                       # gs://…/data/merged-r12 → lab/data/merged-r12
cd $SOUS_CHEF_LAB                                    # /work/sous-chef/lab
tmux new -s train                                    # a dropped IAP tunnel must not kill the run
make train DRY=1 DATA=data/merged-r12 LABEL=r1       # the plan: rows, tokens, updates (41 for merged-r12)
make train MAX_STEPS=4 DATA=data/merged-r12 LABEL=r1-smoke   # the smoke: fit + tokens/s on this card (unmeasured for the 9B)
make train DATA=data/merged-r12 LABEL=r1             # the run → runs/adapters/souschef-r1-sft (run.json has tokens/s, peak VRAM)
make export NAME=souschef-r1 ADAPTER=runs/adapters/souschef-r1-sft/adapter \
     BASE_MODELFILE=base-modelfiles/qwen3.5-9b.modelfile QUANTS=q4_k_m        # GGUF on this GPU; the Modelfile from the
                                                                              # committed file, never from an Ollama
sous-chef-runs-push souschef-r1-sft; sous-chef-runs-push souschef-r1          # adapters + exports → gs://…/runs/ (gguf/ merge dir excluded)
```

Back on the workstation:

```sh
make -C lab train-remote-pull NAME=souschef-r1-sft   # → lab/runs/adapters/souschef-r1-sft
make -C lab train-remote-pull NAME=souschef-r1       # → lab/runs/exports/souschef-r1 (the q4_k_m GGUF + Modelfile.q4_k_m)
make -C lab train-vm-down                            # delete the VM and its disk; the bucket, network and budget stay
make -C lab train-vm-status                          # no VM, no disk; the bucket by folder
```

`train-vm-down` is the end of every session: a stopped VM (hard stop,
preemption) still bills its disk (~US$0.05/h) and `train-vm-up` starts it again
with `/work` intact. `make -C lab train-vm-ssh` works only while it runs.

## The machine switch (`terraform.tfvars`)

```hcl
machine_type       = "a2-highgpu-1g"   # default: A100 40 GB, on demand, ~US$3.67/h + disk
zone               = "us-central1-a"   # a stockout → -b or -c (train-vm-down first: a new VM, its disk lost)
provisioning_model = "STANDARD"
# machine_type       = "a3-highgpu-1g" # fallback: H100 80 GB, SPOT only, ~US$6.57/h + disk; preemptible, no resume
# provisioning_model = "SPOT"
```

Every plan with a Spot VM prints the no-resume warning (`check
"spot_has_no_resume"`); `a3-highgpu-1g` with `STANDARD` is refused (Google
offers its 1-GPU shape as Spot or Flex-start only). Flex-start is not
available through provider 6.x; a Flex-start VM would be made with `gcloud
compute instances create … --provisioning-model=FLEX_START` mirroring `vm.tf`
field by field, as the thermometer's runbook shows.

## Cost (list prices; the A100's 9B speed is unmeasured until the smoke)

| | per hour | ~190 rows (2.5–2.9 M tokens) | ~900 rows (~14 M tokens) |
|---|---|---|---|
| `a2-highgpu-1g` on demand | US$3.67 + 0.05 disk | ~45–60 min, **~US$5–6** incl. ~30 min bootstrap | ~3–5 h, **~US$13–20** |
| `a3-highgpu-1g` Spot | US$6.57 + 0.07 disk | ~20–30 min, ~US$5 | ~1–1.5 h, ~US$10 (preemption risk) |

Plus ~5 min of export on the GPU (the 5090 measured 3 m 25 s), the bucket
(cents), NAT (the uv install ~5 GB, ~US$0.25 per fresh disk). The 5090
reference is 1,670 tokens/s; the A100 estimate is a ratio, ±2×. The budget
(200 in the account's currency by default) notifies; the hard stop
(`max_run_hours = 12`) is what actually stops a forgotten VM.

## Quota (int, us-central1, read 2026-10-04, all usage 0)

`NVIDIA_A100_GPUS` 16 · `PREEMPTIBLE_NVIDIA_A100_GPUS` 64 · `A2_CPUS` 192 ·
`PREEMPTIBLE_NVIDIA_H100_GPUS` 64 (Cloud Quotas API) · `CPUS` 3000 ·
`SSD_TOTAL_GB` 40,960. Nothing to request for either shape. Capacity is
another matter: the thermometer saw Spot stockouts in us-central1; on an
A100 stockout in zone a, try b then c.

## Checks (offline, no project, no credentials)

```sh
cd infra/envs/train
terraform fmt -check -recursive
terraform init -backend=false && terraform validate
terraform test -var-file=terraform.tfvars.example    # 10 runs: shapes, zones, Spot-only A3, prices, bucket posture
bash -n startup.sh
```

A read-only `terraform plan` against int from a scratch copy with a local
backend (2026-10-04): `Plan: 17 to add` with the VM off, 18 with it on.

## Files

`versions.tf` (backend gcs, prefix `sous-chef/train`) · `providers.tf` (region,
labels, names) · `apis.tf` · `iam.tf` · `network.tf` · `storage.tf` · `vm.tf`
(shapes, the instance, the Spot check) · `budget.tf` · `variables.tf` ·
`outputs.tf` · `startup.sh` (the VM's boot: env, `souschef` user, apt tools, uv,
helpers `sous-chef-kit-pull` / `-data-pull` / `-runs-push` / `-train-status`,
`sous-chef-check-base` / `-fetch-base`, the `sous-chef-bootstrap` unit) ·
`terraform.tfvars.example` · `tests/vm.tftest.hcl`.
