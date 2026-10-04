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
