# Sous Chef lab targets (docs/LAB.md §1). Run from lab/ or with `make -C lab`.
# Nothing here commits, posts or spends API budget: lab-collect / lab-eval
# talk to whatever SOUS_CHEF_URL names (the tests use the fake service).
#
#   make -C lab venv test              # create lab/.venv, run pytest
#   make -C lab cases                  # generate + measure cases into lab/data
#   make -C lab train-env train-dry    # lab/.venv-train ([train] extra) + the three --dry-run plans
#   ENGINE=... COOKIE_CUTTER_DIR=...   # where the browser engine lives
#
# Training (docs/TRAIN-WORKSTATION.md): train / dpo / export run with TRAINPY,
# which is lab/.venv-train's python when `make train-env` has built it and
# lab/.venv's otherwise (good enough for DRY=1 — a dry run imports no torch).

PY            ?= python3
VENV          := .venv
BIN           := $(VENV)/bin
PYTEST        := $(BIN)/pytest
LABPY         := $(BIN)/python

# The training environment: Python 3.12 via uv, the [train] extra pinned by
# train-constraints.txt (the thermometer venv's measured set), torch from
# PyTorch's cu130 index. TRAINPY=/path/to/python overrides (e.g. the
# thermometer kit's own venv, ../../breadstack-thermometer/lab/in-house-model/.venv/bin/python).
TRAIN_VENV        := .venv-train
TRAINPY           ?= $(if $(wildcard $(TRAIN_VENV)/bin/python),$(TRAIN_VENV)/bin/python,$(LABPY))
UV                ?= uv
TRAIN_PYTHON      ?= 3.12
TORCH_INDEX       ?= https://download.pytorch.org/whl/cu130
TRAIN_CONSTRAINTS ?= train-constraints.txt

COOKIE_CUTTER_DIR ?= $(abspath ../../breadstack-cookie-cutter)
ENGINE            ?= $(COOKIE_CUTTER_DIR)/launcher/client/src/sousChef/ops/headless.mjs
MANIFEST          ?= fixtures/manifest.snapshot.json
DATA              ?= data
SEED              ?= 0
ROUTE             ?= claude
N                 ?= 1
WORKERS           ?= 4
LABEL             ?= dev
NAME              ?= souschef-$(LABEL)
SFT               ?= runs/adapters/$(NAME)-sft/adapter
ADAPTER           ?= runs/adapters/$(NAME)-dpo/adapter
SOUS_CHEF_URL     ?= http://127.0.0.1:8080
SOUS_CHEF_KEY     ?= test-key

# Remote training (docs/TRAIN-REMOTE.md): the base weights go to the VM as one
# tar of the local Hugging Face cache (18 GiB), laid out like the thermometer's
# base/<slug>-hf.tar so `tar -x -C $HF_HOME` restores it. This cache's snapshot
# files are symlinks into hub/blobs/<xx>/ (hf_xet), so the tar dereferences them
# (-h) and skips the model's own blobs/ dir, which would hold the data twice.
# BASE_MODELFILE=base-modelfiles/qwen3.5-9b.modelfile is the committed stand-in
# for `ollama show qwen3.5:9b --modelfile` (no Ollama needed).
HF_HOME           ?= $(HOME)/.cache/huggingface
BASE_HF_DIR       ?= hub/models--unsloth--Qwen3.5-9B

# The model image (lab/model-image/README.md): Ollama 0.34.4 + the exported
# q4_k_m GGUF, for the Cloud Run GPU service in infra/modules/model-gpu.
# MODEL_EXPORT is export.py's output (runs/exports/$(NAME)); model-stage
# hard-links its one GGUF and a rewritten Modelfile into MODEL_STAGE, the
# only thing the build sees. The repository is Terraform's (gpu_model_repo
# output, immutable tags): MODEL_TAG must be new for every build that is
# pushed, and gpu_model_image_tag in infra/envs/int follows it.
QUANT        ?= q4_k_m
MODEL_EXPORT ?= runs/exports/$(NAME)
MODEL_STAGE  ?= runs/model-image/$(NAME)-$(QUANT)
MODEL_TAG    ?= r1
# Sampling for the SERVED student (the export's Modelfile carries the base's
# chat defaults: temperature 1, presence_penalty 1.5 — tuned for prose, not
# for re-emitting manifest keys verbatim). Empty = keep the export's lines.
MODEL_TEMPERATURE      ?=
MODEL_PRESENCE_PENALTY ?=
GCP_PROJECT  ?= breadstack-sous-chef-int
MODEL_AR_HOST = us-east4-docker.pkg.dev
MODEL_REPO    = $(MODEL_AR_HOST)/$(GCP_PROJECT)/sous-chef-model
MODEL_IMAGE   = $(MODEL_REPO)/model:$(MODEL_TAG)

export COOKIE_CUTTER_DIR ENGINE SOUS_CHEF_ENGINE=$(ENGINE)

.PHONY: help venv test test-engine cases cases-dry cases-remeasure engine-fingerprint collect dataset pairs eval passk retrain-check parity parity-write train-env train-dry train-gpu-check train dpo export model-stage model-build model-push train-vm-init train-vm-plan train-vm-apply train-vm-up train-vm-ssh train-vm-down train-vm-status train-remote-push train-remote-pull fake-service fake-model clean

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

venv: ## create lab/.venv with pytest (stdlib runtime; [train] is opt-in)
	test -x $(LABPY) || $(PY) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip
	$(BIN)/pip install -q -e '.[dev]'

test: ## pytest (engine tests skip without node + ENGINE)
	$(PYTEST)

test-engine: ## pytest with the engine tests REQUIRED (CI: cookie-cutter checked out alongside)
	SOUS_CHEF_LAB_REQUIRE_ENGINE=1 $(PYTEST)

cases: ## generate cases → $(DATA)/cases.jsonl (+ cases-rejects.jsonl) with measured expected.diff (SAMPLE=n, WORKERS=4)
	mkdir -p $(DATA)
	$(LABPY) -m souschef_lab.cases --manifest $(MANIFEST) --seed $(SEED) --out $(DATA)/cases.jsonl --engine $(ENGINE) --workers $(WORKERS) $(if $(SAMPLE),--sample $(SAMPLE),)

cases-dry: ## generate cases without the engine (expected.diff left empty)
	mkdir -p $(DATA)
	$(LABPY) -m souschef_lab.cases --manifest $(MANIFEST) --seed $(SEED) --out $(DATA)/cases.jsonl --no-measure

cases-remeasure: ## refresh expected.diff + engine_fingerprint of $(CASES) in place with the current engine (CASES=$(DATA)/cases.jsonl)
	$(LABPY) -m souschef_lab.cases remeasure --cases $(or $(CASES),$(DATA)/cases.jsonl) --manifest $(MANIFEST) --engine $(ENGINE) --workers $(WORKERS)

engine-fingerprint: ## print the engine's identity (sha256 over the static import closure of headless.mjs + cookie-cutter HEAD)
	$(LABPY) -m souschef_lab.engine --engine $(ENGINE) --fingerprint

# ── later phases (docs/LAB.md §4-§8); the modules land with their own tests ──
collect: ## collect N samples per case from ROUTE via SOUS_CHEF_URL / SOUS_CHEF_KEY
	$(LABPY) -m souschef_lab.collect --cases $(DATA)/cases.jsonl --route $(ROUTE) --samples $(N) --out $(DATA)/collect-$(LABEL).jsonl --engine $(ENGINE) --url $(SOUS_CHEF_URL) --key '$(SOUS_CHEF_KEY)' --manifest $(MANIFEST)
dataset: ## build train/val/exam from the collected logs (LOGS=space-separated jsonl files)
	$(LABPY) -m souschef_lab.dataset $(foreach f,$(LOGS),--log $(f)) --cases $(DATA)/cases.jsonl --out $(DATA)/dataset $(if $(BEST_OF_N),--best-of-n,)
pairs: ## DPO pairs from the collected logs (LOGS=…)
	$(LABPY) -m souschef_lab.pairs $(foreach f,$(LOGS),--log $(f)) --cases $(DATA)/cases.jsonl --out $(DATA)/pairs
eval: ## held-out exam on ROUTE → runs/eval-$(LABEL)-<UTC>.json
	$(LABPY) -m souschef_lab.eval --route $(ROUTE) --label $(LABEL) --cases $(DATA)/cases.jsonl --engine $(ENGINE) --url $(SOUS_CHEF_URL) --key '$(SOUS_CHEF_KEY)' --manifest $(MANIFEST)
passk: ## pass@k over several eval runs (RUNS=space-separated runs/eval-*.json)
	$(LABPY) -m souschef_lab.passk $(RUNS)
retrain-check: ## manifest drift / real-kept pass / age triggers (exit 0/1/2)
	$(LABPY) -m souschef_lab.retrain_check --manifest $(MANIFEST)
parity: ## serving parity: re-render fixtures/prompt/real through parity/ (Ollama 0.34.4's own code; needs go) and check fixtures/parity/served.json
	$(LABPY) -m souschef_lab.serving
parity-write: ## re-record fixtures/parity/served.json (after an Ollama bump in lab/model-image + parity/go.mod, or a new capture)
	$(LABPY) -m souschef_lab.serving --write
train-env: ## create lab/.venv-train (Python $(TRAIN_PYTHON) via uv) with the [train] extra pinned to the thermometer's measured set — a CPU install (5.2 GB; 4 s from a warm uv cache, several GB of downloads from a cold one)
	test -x $(TRAIN_VENV)/bin/python || $(UV) venv --python $(TRAIN_PYTHON) $(TRAIN_VENV)
	$(UV) pip install --python $(TRAIN_VENV)/bin/python --index $(TORCH_INDEX) --index-strategy unsafe-best-match -c $(TRAIN_CONSTRAINTS) -e '.[train]'
	CUDA_VISIBLE_DEVICES= $(TRAIN_VENV)/bin/python -W ignore -c 'import torch, trl, peft, transformers, fla; print("train-env ok: python", __import__("sys").version.split()[0], "torch", torch.__version__, "trl", trl.__version__, "peft", peft.__version__, "transformers", transformers.__version__, "fla", fla.__version__)'
train-dry: ## the three --dry-run plans (SFT, DPO, export) against $(DATA) with TRAINPY; imports no torch, touches no GPU
	$(TRAINPY) -m souschef_lab.train --data $(DATA)/dataset --name $(NAME)-sft --dry-run
	$(TRAINPY) -m souschef_lab.dpo --pairs $(DATA)/pairs --sft-adapter $(SFT) --name $(NAME)-dpo --dry-run
	$(TRAINPY) -m souschef_lab.export --adapter $(ADAPTER) --name $(NAME) $(if $(BASE_MODELFILE),--base-modelfile $(BASE_MODELFILE),) --dry-run
train-gpu-check: ## `import unsloth, trl, peft` + the GPU's name (unsloth refuses to import without a GPU; this opens a CUDA context — run it only when Ollama is idle)
	$(TRAINPY) -W ignore -c 'import unsloth, trl, peft, torch; print("unsloth", unsloth.__version__, "trl", trl.__version__, "peft", peft.__version__, "gpu", torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported())'
train: ## SFT with TRAINPY (make train-env first); DRY=1 for --dry-run; NAME=run name (adapter → runs/adapters/$(NAME)-sft); MAX_STEPS=4 for a smoke run
	$(TRAINPY) -m souschef_lab.train --data $(DATA)/dataset --name $(NAME)-sft $(if $(MAX_STEPS),--max-steps $(MAX_STEPS),) $(if $(DRY),--dry-run,)
dpo: ## DPO against the frozen SFT adapter (TRAINPY); SFT=adapter dir, NAME=run name
	$(TRAINPY) -m souschef_lab.dpo --pairs $(DATA)/pairs --sft-adapter $(SFT) --name $(NAME)-dpo $(if $(DRY),--dry-run,)
export: ## GGUF q4_k_m / q8_0 + Ollama Modelfile (num_ctx 65536); ADAPTER=adapter dir, NAME=Ollama model name; QUANTS=q4_k_m for one; BASE_MODELFILE=file instead of `ollama show`; CREATE=1 runs `ollama create` (OLLAMA_HOST + LAB_OLLAMA_BIN from the environment)
	$(TRAINPY) -m souschef_lab.export --adapter $(ADAPTER) --name $(NAME) $(if $(QUANTS),--quants $(QUANTS),) $(if $(BASE_MODELFILE),--base-modelfile $(BASE_MODELFILE),) $(if $(CREATE),--create,) $(if $(DRY),--dry-run,)
# ── the model image (lab/model-image/README.md) ──
# The staged Modelfile is export.py's Modelfile.$(QUANT) with its FROM line
# pointing at the staged GGUF and exactly one `PARAMETER num_ctx 65536`
# (export.py's own num_ctx line is dropped and the 64k one appended, so a
# differently exported Modelfile still comes out right). The GGUF is a hard
# link when the staging dir is on the export's filesystem (runs/ is), a
# copy otherwise — never a symlink, which the build would send as a link.
model-stage: ## stage the one $(QUANT) GGUF + its Modelfile (FROM ./<gguf>, num_ctx 65536) from MODEL_EXPORT into MODEL_STAGE (hard links; the dir is emptied first) and check it
	@[ -f '$(MODEL_EXPORT)/Modelfile.$(QUANT)' ] || { echo "model-stage: no $(MODEL_EXPORT)/Modelfile.$(QUANT) — MODEL_EXPORT must be an export.py output (make export NAME=…), QUANT one of its quants"; exit 1; }
	@gguf=$$(sed -n -E 's/^FROM[[:space:]]+\.\///p' '$(MODEL_EXPORT)/Modelfile.$(QUANT)' | sed -E 's/[[:space:]]+$$//'); \
	  [ -n "$$gguf" ] || { echo "model-stage: $(MODEL_EXPORT)/Modelfile.$(QUANT) has no FROM ./<gguf> line"; exit 1; }; \
	  [ -f "$(MODEL_EXPORT)/$$gguf" ] || { echo "model-stage: $(MODEL_EXPORT)/Modelfile.$(QUANT) names $$gguf, which is not in $(MODEL_EXPORT)"; exit 1; }; \
	  rm -rf '$(MODEL_STAGE)' && mkdir -p '$(MODEL_STAGE)'; \
	  ln "$(MODEL_EXPORT)/$$gguf" "$(MODEL_STAGE)/$$gguf" 2>/dev/null || { echo "model-stage: hard link failed (another filesystem?); copying $$gguf"; cp "$(MODEL_EXPORT)/$$gguf" "$(MODEL_STAGE)/$$gguf"; }; \
	  { grep -Ev '^PARAMETER[[:space:]]+num_ctx([[:space:]]|$$)' '$(MODEL_EXPORT)/Modelfile.$(QUANT)' \
	      | { [ -z '$(MODEL_TEMPERATURE)' ] && cat || grep -Ev '^PARAMETER[[:space:]]+temperature([[:space:]]|$$)'; } \
	      | { [ -z '$(MODEL_PRESENCE_PENALTY)' ] && cat || grep -Ev '^PARAMETER[[:space:]]+presence_penalty([[:space:]]|$$)'; } \
	      | sed -E "s#^FROM[[:space:]].*#FROM ./$$gguf#"; \
	    [ -z '$(MODEL_TEMPERATURE)' ] || echo 'PARAMETER temperature $(MODEL_TEMPERATURE)'; \
	    [ -z '$(MODEL_PRESENCE_PENALTY)' ] || echo 'PARAMETER presence_penalty $(MODEL_PRESENCE_PENALTY)'; \
	    echo 'PARAMETER num_ctx 65536'; } >'$(MODEL_STAGE)/Modelfile'; \
	  sh model-image/model-context.sh '$(MODEL_STAGE)' >/dev/null
model-build: ## build the model image (Ollama 0.34.4 + the staged GGUF) for linux/amd64 from MODEL_STAGE as $(MODEL_IMAGE); CPU only, nothing started or pushed
	@dir=$$(sh model-image/model-context.sh '$(MODEL_STAGE)') || exit 1; \
	  echo "model-build: building $(MODEL_IMAGE)"; \
	  docker build --platform linux/amd64 --build-context model="$$dir" -f model-image/Dockerfile -t '$(MODEL_IMAGE)' model-image || exit 1; \
	  docker image inspect -f '{{.Size}}' '$(MODEL_IMAGE)' | \
	    awk '{ printf "model-build: built $(MODEL_IMAGE), %.1f GB on this disk; make model-push MODEL_TAG=$(MODEL_TAG) shows how to send it\n", $$1 / 1e9 }'
model-push: ## print the commands that push $(MODEL_IMAGE) to the private repository (nothing is run: the repository comes from the first-half apply, infra/README.md Phase 5)
	@echo "# $(MODEL_IMAGE): the repository is created by infra/envs/int with gpu_model_enabled = true (a person's apply);"
	@echo "# its tags are immutable, so a tag that is already there needs a new MODEL_TAG (make model-build MODEL_TAG=<tag>)."
	@echo "gcloud auth configure-docker $(MODEL_AR_HOST) --quiet"
	@echo "docker push '$(MODEL_IMAGE)'"
	@echo "# then: gpu_model_image_tag = \"$(MODEL_TAG)\" and gpu_model_serve = true in infra/envs/int, the second-half apply."

base-tar: ## tar the cached base weights (HF_HOME/BASE_HF_DIR, 18 GiB, symlinks followed) into OUT=<path>.tar for the training VM's bucket (docs/TRAIN-REMOTE.md §2); OUT=- streams to stdout
	@test -n "$(OUT)" || { echo "base-tar: pass OUT=<path>.tar (or OUT=- to stream, e.g. into 'gcloud storage cp - gs://…/base/unsloth-qwen3.5-9b-hf.tar')"; exit 2; }
	@test -d $(HF_HOME)/$(BASE_HF_DIR)/snapshots || { echo "base-tar: $(HF_HOME)/$(BASE_HF_DIR) has no snapshots/ — is the base cached here?"; exit 2; }
	tar -h -C $(HF_HOME) -cf $(OUT) --exclude='$(BASE_HF_DIR)/blobs' --exclude='*.lock' --exclude='.no_exist' $(BASE_HF_DIR)

# ── remote training: a GPU VM in us-central1 (infra/envs/train; its README.md is the runbook) ──
# A Terraform root of its own (state prefix sous-chef/train in the int state
# bucket), so no service apply can touch the VM and no training apply can
# touch the service. Everything that changes anything (train-vm-apply, -up,
# -down) runs on YOUR login: Application Default Credentials for terraform
# (gcloud auth application-default login), your gcloud login for the rest.
# Nothing here is run by an agent: a person types train-vm-up and -down.
#
# The VM exists only between train-vm-up and train-vm-down; -plan and -apply
# pass whether it exists now (TRAIN_VM_READ), so a routine apply never
# creates or deletes it. Data moves only through train-remote-push (a data
# set, the kit, the base tar) and train-remote-pull (adapters, exports); the
# VM's startup script copies no dataset. The kit tar follows symlinks (-h):
# fixtures/manifest.snapshot.json points into service/testdata, which the VM
# does not have.
TRAIN_ROOT         := $(abspath ../infra/envs/train)
TRAIN_STATE_BUCKET ?= $(GCP_PROJECT)-sous-chef-tfstate
TRAIN_BUCKET       ?= gs://$(GCP_PROJECT)-training
TRAIN_VM           ?= sous-chef-train
TRAIN_VARS          = -var "project_id=$(GCP_PROJECT)"
TRAIN_KIT          ?= $(TRAIN_BUCKET)/kit/lab.tar.gz
BASE_MODEL         ?= unsloth/Qwen3.5-9B
BASE_SLUG           = $(shell echo '$(BASE_MODEL)' | tr '/' '-' | tr '[:upper:]' '[:lower:]')
BASE_TAR_OBJ        = $(TRAIN_BUCKET)/base/$(BASE_SLUG)-hf.tar
# The gcloud filter for Terraform's VM and nothing else (an RE anchored at both ends).
TRAIN_VM_FILTER     = name~^$(TRAIN_VM)$$
# Whether the VM exists now, into $$on. A read that fails refuses: an apply
# that cannot see the VM would plan to delete it.
TRAIN_VM_READ = vm=$$(gcloud compute instances list --project $(GCP_PROJECT) --filter='$(TRAIN_VM_FILTER)' --format='value(zone.basename(),status)' 2>/dev/null) || \
	  { echo "train: could not list the project's VMs (gcloud auth login?): refusing, since a plan that cannot see $(TRAIN_VM) would delete it"; exit 1; }; \
	  if [ -n "$$vm" ]; then on=true; echo "train: $(TRAIN_VM) exists ($$vm): kept as it is"; else on=false; echo "train: no $(TRAIN_VM) VM: none planned (make train-vm-up makes one)"; fi
# The zone of the VM, into $$zone, or stop.
TRAIN_VM_ZONE = zone=$$(gcloud compute instances list --project $(GCP_PROJECT) --filter='$(TRAIN_VM_FILTER)' --format='value(zone.basename())' 2>/dev/null); \
	  test -n "$$zone" || { echo "train: no $(TRAIN_VM) VM (make train-vm-up), or gcloud cannot list the project's VMs on this login"; exit 1; }
# -init must have run against the state bucket, and terraform.tfvars must exist.
TRAIN_GUARD = test -f $(TRAIN_ROOT)/terraform.tfvars || { echo "train: no $(TRAIN_ROOT)/terraform.tfvars — cp terraform.tfvars.example terraform.tfvars there and check it"; exit 1; }; \
	  grep -q '"bucket": "$(TRAIN_STATE_BUCKET)"' $(TRAIN_ROOT)/.terraform/terraform.tfstate 2>/dev/null || { echo "train: $(TRAIN_ROOT) is not initialised against gs://$(TRAIN_STATE_BUCKET): make train-vm-init"; exit 1; }

train-vm-init: ## remote training: terraform init infra/envs/train against the int state bucket, prefix sous-chef/train (creates nothing)
	cd $(TRAIN_ROOT) && terraform init -reconfigure -backend-config="bucket=$(TRAIN_STATE_BUCKET)"
train-vm-plan: ## remote training: plan infra/envs/train (read-only), the VM kept as it is
	@$(TRAIN_GUARD); $(TRAIN_VM_READ); cd $(TRAIN_ROOT) && terraform plan -lock=false $(TRAIN_VARS) -var training_vm_enabled=$$on
train-vm-apply: ## remote training, YOUR login: the bucket, the VM's account, the network (IAP SSH in, Cloud NAT out), the budget and the APIs — the VM kept as it is (none until train-vm-up). Nothing here bills by the hour but the VM
	@$(TRAIN_GUARD); $(TRAIN_VM_READ); cd $(TRAIN_ROOT) && terraform apply $(TRAIN_VARS) -var training_vm_enabled=$$on
train-vm-up: ## remote training, YOUR login: create the GPU VM as infra/envs/train/terraform.tfvars shapes it (a targeted apply), or start it again after a stop. BILLED while it runs: ~US$3.72/h (a2-highgpu-1g on demand, disk included), ~US$6.64/h (a3-highgpu-1g Spot); stops by itself after max_run_hours
	@$(TRAIN_GUARD); gcloud storage buckets describe '$(TRAIN_BUCKET)' --format='value(name)' >/dev/null 2>&1 || \
	  { echo "train: $(TRAIN_BUCKET) is not there (or not readable on your login): make train-vm-apply first"; exit 1; }
	cd $(TRAIN_ROOT) && terraform apply -target=google_compute_instance.train $(TRAIN_VARS) -var training_vm_enabled=true
	@st=$$(gcloud compute instances list --project $(GCP_PROJECT) --filter='$(TRAIN_VM_FILTER)' --format='value(status,zone.basename())' 2>/dev/null); \
	  case "$$st" in \
	    TERMINATED*|STOPPED*) z=$${st##*	}; echo "train: $(TRAIN_VM) is stopped (a preemption, or its max run time): starting it in $$z"; \
	      gcloud compute instances start $(TRAIN_VM) --project $(GCP_PROJECT) --zone "$$z" || \
	        { echo "train: no start — usually no capacity in $$z just now: try again later, or set another zone in $(TRAIN_ROOT)/terraform.tfvars and make train-vm-down train-vm-up (a new VM: its disk is lost; the bucket is not)"; exit 1; };; \
	    "") echo "train: $(TRAIN_VM) is not there after the apply"; exit 1;; \
	  esac
	@echo "train: UP — $(TRAIN_VM) bills by the hour until it stops by itself (max_run_hours) or make train-vm-down deletes it with its disk. make train-vm-ssh (the bootstrap runs ~10-30 min: sous-chef-train-status on the VM); make train-vm-status"
train-vm-ssh: ## remote training: SSH to the VM through IAP with your OS Login, into the souschef account that owns the lab (SSH_USER=me for your own shell; SSH_ARGS='-- -L 8888:localhost:8888' passes more)
	@$(TRAIN_VM_ZONE); gcloud compute ssh $(TRAIN_VM) --project $(GCP_PROJECT) --zone "$$zone" --tunnel-through-iap $(if $(filter me,$(SSH_USER)),$(SSH_ARGS),$(or $(SSH_ARGS),-- -t sudo -iu souschef))
train-vm-down: ## remote training, YOUR login: delete the VM and its disk (a targeted apply; the bucket, network and budget stay). The end of every run: pull the adapters first (train-remote-pull)
	@$(TRAIN_GUARD); cd $(TRAIN_ROOT) && terraform apply -target=google_compute_instance.train $(TRAIN_VARS) -var training_vm_enabled=false
	@left=$$(gcloud compute disks list --project $(GCP_PROJECT) --filter='labels.purpose=sous-chef-train' --format='value(name,zone.basename())' 2>/dev/null); \
	  test -z "$$left" && echo "train: DOWN — no VM, no training disk" || echo "train: WARNING — disks labelled purpose=sous-chef-train remain: $$left (gcloud compute disks delete <name> --zone <zone>)"
train-vm-status: ## remote training: the VM (state, shape, when it stops by itself), its disks and the bucket by folder. Read-only, your gcloud login
	@gcloud compute instances list --project $(GCP_PROJECT) --filter='$(TRAIN_VM_FILTER)' \
	  --format='value(name,zone.basename(),status,machineType.basename(),scheduling.provisioningModel,resourceStatus.scheduling.terminationTimestamp)' 2>/dev/null | \
	  awk 'BEGIN { n = 0 } { n++; printf "vm:        %s in %s: %s, %s, %s%s\n", $$1, $$2, $$3, $$4, $$5, ($$6 ? ", stops by itself at " $$6 : "") } END { if (n == 0) print "vm:        none (make train-vm-up)" }'
	@gcloud compute disks list --project $(GCP_PROJECT) --filter='labels.purpose=sous-chef-train' --format='value(name,zone.basename(),sizeGb,type.basename(),status)' 2>/dev/null | \
	  awk 'BEGIN { n = 0 } { n++; printf "disk:      %s in %s, %s GB %s, %s\n", $$1, $$2, $$3, $$4, $$5 } END { if (n == 0) print "disk:      none labelled purpose=sous-chef-train" }'
	@if gcloud storage buckets describe '$(TRAIN_BUCKET)' --format='value(name)' >/dev/null 2>&1; then \
	  for p in data kit base runs/adapters runs/exports; do \
	    n=$$(gcloud storage ls '$(TRAIN_BUCKET)'/$$p/ 2>/dev/null | wc -l); printf 'bucket:    %-14s %s entries\n' "$$p/" "$$n"; done; \
	  else echo "bucket:    $(TRAIN_BUCKET) not there (make train-vm-apply) or not readable on this login"; fi
train-remote-push: ## remote training: DATA=data/<set> → $(TRAIN_BUCKET)/data/<set>/ (rsync); the lab's working tree → kit/lab.tar.gz (code, fixtures, pins — no data, venvs or runs); the base weights (make base-tar, 18 GiB) → base/, skipped when already there (BASE=force re-sends, BASE=skip never does)
	@set=$$(basename '$(DATA)'); case "$$set" in data|.|/|'') echo "train-remote-push: DATA must name one set, e.g. DATA=data/merged-r12 (not the whole data/ folder)"; exit 1;; esac; \
	  test -d '$(DATA)/dataset' || { echo "train-remote-push: $(DATA)/dataset is not there (make dataset first)"; exit 1; }; \
	  echo "train-remote-push: $(DATA) ($$(du -sh '$(DATA)' | cut -f1)) → $(TRAIN_BUCKET)/data/$$set/"; \
	  gcloud storage rsync --recursive --exclude='^engine-cache/.*' '$(DATA)' '$(TRAIN_BUCKET)'/data/$$set || exit 1
	@sha=$$(git rev-parse --short HEAD 2>/dev/null || echo nogit); dirty=$$(test -z "$$(git status --porcelain -- . 2>/dev/null)" || echo "+uncommitted"); \
	  echo "$$sha$$dirty $$(date -u +%FT%TZ)" > KIT-VERSION; \
	  tar -h -C .. -czf - --exclude=lab/.venv --exclude=lab/.venv-train --exclude=lab/data --exclude=lab/runs --exclude=lab/unsloth_compiled_cache --exclude='__pycache__' --exclude=.pytest_cache --exclude='*.egg-info' lab | \
	    gcloud storage cp - '$(TRAIN_KIT)' && echo "train-remote-push: the lab's working tree ($$sha$$dirty) → $(TRAIN_KIT); on the VM: sous-chef-kit-pull (the bootstrap does it at boot)"; \
	  rm -f KIT-VERSION
	@case '$(BASE)' in skip) echo "train-remote-push: base tar skipped (BASE=skip)";; \
	  *) if [ '$(BASE)' != force ] && gcloud storage objects describe '$(BASE_TAR_OBJ)' --format='value(size)' >/dev/null 2>&1; then \
	       echo "train-remote-push: $(BASE_TAR_OBJ) is there already ($$(gcloud storage objects describe '$(BASE_TAR_OBJ)' --format='value(size)') bytes): kept (BASE=force re-sends)"; \
	     else echo "train-remote-push: $(BASE_MODEL) → $(BASE_TAR_OBJ) (18 GiB streamed from $(HF_HOME)/$(BASE_HF_DIR); minutes to an hour, your upstream bandwidth)"; \
	       $(MAKE) --no-print-directory base-tar OUT=- | gcloud storage cp - '$(BASE_TAR_OBJ)' || exit 1; fi;; esac
train-remote-pull: ## remote training: NAME=<run> brings $(TRAIN_BUCKET)/runs/adapters/<run> and runs/exports/<run> (whichever exist) into runs/adapters|exports/<run>
	@echo '$(NAME)' | grep -Eq '^[A-Za-z0-9][A-Za-z0-9._-]*$$' || { echo "train-remote-pull: NAME must be one folder name, got '$(NAME)' — gcloud storage ls $(TRAIN_BUCKET)/runs/adapters/ lists them"; exit 1; }
	@n=0; for kind in adapters exports; do \
	  gcloud storage ls '$(TRAIN_BUCKET)'/runs/$$kind/$(NAME)/ >/dev/null 2>&1 || continue; \
	  mkdir -p runs/$$kind/$(NAME) && gcloud storage rsync --recursive '$(TRAIN_BUCKET)'/runs/$$kind/$(NAME) runs/$$kind/$(NAME) && \
	    echo "train-remote-pull: runs/$$kind/$(NAME) ($$(du -sh runs/$$kind/$(NAME) | cut -f1))" && n=$$((n + 1)); done; \
	  test $$n -gt 0 || { echo "train-remote-pull: nothing named $(NAME) under $(TRAIN_BUCKET)/runs/adapters or runs/exports (on the VM: sous-chef-runs-push $(NAME))"; exit 1; }

fake-service: ## run the tests' fake Sous Chef service on 127.0.0.1:8089 (SCRIPTS=dir of <case id>.json)
	$(LABPY) -m tests.fake_service --port 8089 --manifest $(MANIFEST) --key '$(SOUS_CHEF_KEY)' $(if $(SCRIPTS),--scripts $(SCRIPTS),)
fake-model: ## run the tests' fake /v1/messages model on 127.0.0.1:8091 (STEPS=steps.json; for the real service binary)
	$(LABPY) -m tests.fake_model --port 8091 $(if $(STEPS),--steps $(STEPS),)

clean: ## remove caches (keeps data/ and runs/)
	rm -rf .pytest_cache souschef_lab/__pycache__ tests/__pycache__ $(DATA)/engine-cache
