# Sous Chef lab targets (docs/LAB.md §1). Run from lab/ or with `make -C lab`.
# Nothing here commits, posts or spends API budget: lab-collect / lab-eval
# talk to whatever SOUS_CHEF_URL names (the tests use the fake service).
#
#   make -C lab venv test              # create lab/.venv, run pytest
#   make -C lab cases                  # generate + measure cases into lab/data
#   ENGINE=... COOKIE_CUTTER_DIR=...   # where the browser engine lives

PY            ?= python3
VENV          := .venv
BIN           := $(VENV)/bin
PYTEST        := $(BIN)/pytest
LABPY         := $(BIN)/python

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

export COOKIE_CUTTER_DIR ENGINE SOUS_CHEF_ENGINE=$(ENGINE)

.PHONY: help venv test test-engine cases cases-dry cases-remeasure engine-fingerprint collect dataset pairs eval passk retrain-check train dpo export fake-service fake-model clean

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
train: ## SFT (needs [train]); DRY=1 for --dry-run; NAME=run name (adapter → runs/adapters/$(NAME)-sft)
	$(LABPY) -m souschef_lab.train --data $(DATA)/dataset --name $(NAME)-sft $(if $(DRY),--dry-run,)
dpo: ## DPO against the frozen SFT adapter (needs [train]); SFT=adapter dir, NAME=run name
	$(LABPY) -m souschef_lab.dpo --pairs $(DATA)/pairs --sft-adapter $(SFT) --name $(NAME)-dpo $(if $(DRY),--dry-run,)
export: ## GGUF q4_k_m / q8_0 + Ollama Modelfile (num_ctx 65536); ADAPTER=adapter dir, NAME=Ollama model name
	$(LABPY) -m souschef_lab.export --adapter $(ADAPTER) --name $(NAME) $(if $(BASE_MODELFILE),--base-modelfile $(BASE_MODELFILE),) $(if $(DRY),--dry-run,)
fake-service: ## run the tests' fake Sous Chef service on 127.0.0.1:8089 (SCRIPTS=dir of <case id>.json)
	$(LABPY) -m tests.fake_service --port 8089 --manifest $(MANIFEST) --key '$(SOUS_CHEF_KEY)' $(if $(SCRIPTS),--scripts $(SCRIPTS),)
fake-model: ## run the tests' fake /v1/messages model on 127.0.0.1:8091 (STEPS=steps.json; for the real service binary)
	$(LABPY) -m tests.fake_model --port 8091 $(if $(STEPS),--steps $(STEPS),)

clean: ## remove caches (keeps data/ and runs/)
	rm -rf .pytest_cache souschef_lab/__pycache__ tests/__pycache__ $(DATA)/engine-cache
