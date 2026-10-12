# Sous Chef — repo-level targets (README §7, §10). The Go module lives in
# service/; Terraform in infra/. Nothing here commits or applies.

SERVICE_DIR := service
BIN         := $(SERVICE_DIR)/bin/sous-chef
BUILD_ID    ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
GOFLAGS     ?=
LDFLAGS     := -X main.buildID=$(BUILD_ID)

# Container image. IMAGE is the local tag; docker-push needs AR_REPO (the
# Terraform `artifact_repo` output, e.g. northamerica-northeast2-docker.pkg.dev/breadstack-sous-chef-int/sous-chef).
IMAGE       ?= sous-chef:$(BUILD_ID)
AR_REPO     ?=
# The docker-compose network (project name = this directory) and the DSN a
# container on it uses to reach Postgres.
COMPOSE_NET := breadstack-sous-chef_default
DOCKER_DSN  := postgres://souschef:souschef@postgres:5432/souschef?sslmode=disable
SMOKE_PORT  ?= 18090

.PHONY: all build test vet fmt run e2e tidy db-up db-down db-logs clean tf-check help \
        docker-build docker-run docker-push image-smoke

all: build

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

build: ## compile service/bin/sous-chef with the build id baked in
	cd $(SERVICE_DIR) && CGO_ENABLED=0 go build $(GOFLAGS) -ldflags "$(LDFLAGS)" -o bin/sous-chef ./cmd/sous-chef

vet: ## go vet
	cd $(SERVICE_DIR) && go vet ./...

fmt: ## gofmt check (fails on unformatted files)
	@cd $(SERVICE_DIR) && out=$$(gofmt -l .); if [ -n "$$out" ]; then echo "unformatted:"; echo "$$out"; exit 1; fi

test: vet ## unit tests with the race detector (README §10: make test = go vet + go test -race)
	cd $(SERVICE_DIR) && go test -race ./...

e2e: db-up ## end-to-end tests against the docker-compose Postgres
	cd $(SERVICE_DIR) && TEST_DATABASE_URL=$${TEST_DATABASE_URL:-postgres://souschef:souschef@localhost:5433/souschef?sslmode=disable} go test -race -tags e2e -run . ./...

run: build ## run the server locally; reads .env when present
	@set -a; [ -f .env ] && . ./.env; set +a; ./$(BIN)

tidy: ## go mod tidy
	cd $(SERVICE_DIR) && go mod tidy

db-up: ## start local Postgres 16 (port 5433)
	docker compose up -d --wait postgres

db-down: ## stop local Postgres and drop its volume
	docker compose down -v

db-logs: ## tail Postgres logs
	docker compose logs -f postgres

tf-check: ## terraform fmt -check + validate for every env (never apply)
	@test -d infra && terraform -chdir=infra fmt -check -recursive || true
	@for d in infra/envs/*/; do [ -d "$$d" ] && (cd "$$d" && terraform init -backend=false -input=false >/dev/null && terraform validate) || true; done

clean: ## remove build output
	rm -rf $(SERVICE_DIR)/bin $(SERVICE_DIR)/coverage.out $(SERVICE_DIR)/coverage.html

# --- container image (service/Dockerfile; docs/DEPLOY-INT.md) ---------------

docker-build: ## build the service image as $(IMAGE) (linux/amd64, BUILD_ID baked in)
	docker build --platform linux/amd64 -f $(SERVICE_DIR)/Dockerfile --build-arg BUILD_ID=$(BUILD_ID) -t $(IMAGE) $(SERVICE_DIR)

docker-run: db-up ## run $(IMAGE) against the compose Postgres: -migrate, then serve on :8090 (keys from .env or dev placeholders)
	docker run --rm --network $(COMPOSE_NET) -e DATABASE_URL='$(DOCKER_DSN)' $(IMAGE) -migrate
	@set -a; [ -f .env ] && . ./.env; set +a; \
	  docker run --rm --network $(COMPOSE_NET) -p 8090:8090 \
	    -e DATABASE_URL='$(DOCKER_DSN)' \
	    -e SERVICE_KEYS="$${SERVICE_KEYS:-local-dev-key}" \
	    -e ANTHROPIC_API_KEY="$${ANTHROPIC_API_KEY:-local-dev-not-a-key}" \
	    -e ANTHROPIC_BASE_URL="$${ANTHROPIC_BASE_URL:-}" \
	    -e MODEL_ROUTES="$${MODEL_ROUTES:-}" \
	    $(IMAGE)

image-smoke: db-up ## migrate + serve $(IMAGE) on :$(SMOKE_PORT) with placeholder keys, curl /healthz /readyz /version, stop
	@docker rm -f sous-chef-smoke >/dev/null 2>&1 || true
	docker run --rm --network $(COMPOSE_NET) -e DATABASE_URL='$(DOCKER_DSN)' $(IMAGE) -migrate
	docker run -d --name sous-chef-smoke --network $(COMPOSE_NET) -p $(SMOKE_PORT):8090 \
	  -e DATABASE_URL='$(DOCKER_DSN)' -e SERVICE_KEYS=smoke-key -e ANTHROPIC_API_KEY=smoke-not-a-key $(IMAGE) >/dev/null
	@rc=0; \
	  for i in $$(seq 1 40); do curl -sf localhost:$(SMOKE_PORT)/healthz >/dev/null 2>&1 && break; sleep 0.5; done; \
	  for p in healthz readyz version; do \
	    code=$$(curl -s -o /tmp/sous-chef-smoke.$$p -w '%{http_code}' localhost:$(SMOKE_PORT)/$$p); \
	    printf '  /%-8s %s %s\n' $$p "$$code" "$$(cat /tmp/sous-chef-smoke.$$p)"; \
	    [ "$$code" = 200 ] || rc=1; \
	  done; \
	  code=$$(curl -s -o /dev/null -w '%{http_code}' localhost:$(SMOKE_PORT)/v1/manifests/x); \
	  printf '  /v1 without key: %s (want 401)\n' "$$code"; [ "$$code" = 401 ] || rc=1; \
	  [ $$rc -eq 0 ] || docker logs sous-chef-smoke; \
	  docker rm -f sous-chef-smoke >/dev/null; \
	  [ $$rc -eq 0 ] && echo "image-smoke: ok ($(IMAGE))" || { echo "image-smoke: FAILED"; exit 1; }

docker-push: ## tag $(IMAGE) as $(AR_REPO)/sous-chef:$(BUILD_ID) and push (needs AR_REPO and `gcloud auth configure-docker <region>-docker.pkg.dev`)
	@test -n "$(AR_REPO)" || { echo "AR_REPO is required (terraform -chdir=infra/envs/int output -raw artifact_repo)"; exit 1; }
	docker tag $(IMAGE) $(AR_REPO)/sous-chef:$(BUILD_ID)
	docker push $(AR_REPO)/sous-chef:$(BUILD_ID)
	@echo "pushed $(AR_REPO)/sous-chef:$(BUILD_ID)"

# --- Sous Chef pipeline (infra/modules/pipeline; docs/RETRAIN.md, docs/DRIFT.md) ---
# Every target here either builds in Cloud Build (lab-image, trainer-image),
# changes production (model-rollout) or starts a workflow (pipeline-*): none
# runs on its own, each is a person's command.
GCP_PROJECT      ?= breadstack-sous-chef-int
CB_PROJECT       ?= prod-devops-core-project
PIPELINE_REGION  ?= us-central1
MODEL_REGION     ?= us-east4
MODEL_REPO       ?= us-east4-docker.pkg.dev/$(GCP_PROJECT)/sous-chef-model/model
TAG              ?=
MODE             ?= check
RESUME           ?=
RUN              ?=
RETRAIN_FULL_DATA   = {"mode":"full","trigger":{"drift_run":null,"reason":"manual"}}
RETRAIN_RESUME_DATA = {"resume_from":"$(RESUME)","run_id":"$(RUN)"}

.PHONY: lab-image trainer-image trainer-tag model-rollout pipeline-smoke pipeline-drift pipeline-retrain pipeline-lint pipeline-tf-test

lab-image: ## build + push sous-chef-lab:$(TAG) in Cloud Build ($(CB_PROJECT)) from THIS checkout — it must be the commit the service tag was built from
	@test -n "$(TAG)" || { echo "TAG=<service tag> is required (the int service's image tag)"; exit 2; }
	gcloud builds submit --project $(CB_PROJECT) --config images/lab-runtime/cloudbuild.yaml \
	  --ignore-file=images/images.gcloudignore --substitutions=_TAG=$(TAG),_PROJECT_ID=$(GCP_PROJECT) .

trainer-image: ## build sous-chef-trainer:t-<content hash> when absent, then tag it $(TAG) (Cloud Build, $(CB_PROJECT))
	@test -n "$(TAG)" || { echo "TAG=<service tag> is required"; exit 2; }
	gcloud builds submit --project $(CB_PROJECT) --config images/trainer/cloudbuild.yaml \
	  --ignore-file=images/images.gcloudignore --substitutions=_TAG=$(TAG),_PROJECT_ID=$(GCP_PROJECT) .

trainer-tag: ## print this checkout's trainer content tag (t-<hash12>)
	@bash images/trainer/tag.sh

# TAG may be a tag (r4) or a digest (sha256:…, what the pipeline's rollout.json names since it rolls out by digest)
MODEL_REF = $(if $(filter sha256:%,$(TAG)),$(MODEL_REPO)@$(TAG),$(MODEL_REPO):$(TAG))

model-rollout: ## roll sous-chef-model (production, int) to model:$(TAG) or model@$(TAG) (sha256:…) — rollouts by hand and ROLLBACKS (rollout.json names the old one)
	@test -n "$(TAG)" || { echo "TAG=<model image tag or sha256:digest> is required, e.g. TAG=r4 or TAG=sha256:…"; exit 2; }
	gcloud run services update sous-chef-model --project $(GCP_PROJECT) --region $(MODEL_REGION) \
	  --image $(MODEL_REF)
	@echo "model-rollout: sous-chef-model -> $(MODEL_REF); drift judges the model state/model.json (or its history) names for this digest (calibrating it if it has no baseline)"

# `gcloud workflows execute` starts an execution and returns its name at once (`run` would block until it ends:
# up to the decision for a retrain). Follow one with `gcloud workflows executions describe <name>` or the console.
pipeline-smoke: ## start a SMOKE retrain round (2–6 h with the GPU wait, ~US$2 of API, ~1 GPU-hour; dry rollout)
	gcloud workflows execute souschef-retrain --project $(GCP_PROJECT) --location $(PIPELINE_REGION) --data='{"mode":"smoke"}'

pipeline-drift: ## run the drift workflow once: MODE=poll|check|weekly|calibrate|night (check/weekly/calibrate force a run)
	gcloud workflows execute souschef-drift --project $(GCP_PROJECT) --location $(PIPELINE_REGION) \
	  --data='{"mode":"$(MODE)","force":$(if $(filter poll night,$(MODE)),false,true)}'

pipeline-retrain: ## start a full retrain round by hand (RESUME=<stage> RUN=<run id> resumes a failed round)
	@test -z "$(RESUME)" || test -n "$(RUN)" || { echo "RESUME=<stage> needs RUN=<run id>"; exit 2; }
	gcloud workflows execute souschef-retrain --project $(GCP_PROJECT) --location $(PIPELINE_REGION) \
	  --data='$(if $(RESUME),$(RETRAIN_RESUME_DATA),$(RETRAIN_FULL_DATA))'

pipeline-lint: ## offline lint of the two workflow definitions (+ SCHEMA=<workflows-samples jsonschema/workflows.json>)
	python3 infra/modules/pipeline/tests/lint_workflows.py $(if $(SCHEMA),--schema $(SCHEMA),)

pipeline-tf-test: ## terraform test for the pipeline module and the int root (mock providers; nothing is read or applied)
	cd infra/modules/pipeline && terraform init -backend=false -input=false >/dev/null && terraform test
	cd infra/envs/int && terraform init -backend=false -input=false >/dev/null && terraform test
