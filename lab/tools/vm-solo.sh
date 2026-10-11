#!/usr/bin/env bash
# vm-solo.sh — the student ALONE on the training VM (r4 PLAN §2.6, EVAL.md): Ollama 0.34.4 serving an exported
# GGUF, a local Postgres, and the Sous Chef service binary of the deployed commit with ONE route, `student`
# (messages-compatible, Ollama on 127.0.0.1, fields_context on, thinking disabled — the defaults — and NO fallback,
# so no Anthropic key and no API spend). The exam runs on the workstation through an IAP tunnel to :8091.
#
# On the VM (an operator who can sudo; the lab belongs to souschef):
#   sudo bash /work/sous-chef/lab/tools/vm-solo.sh setup          # once: Postgres (docker or apt), Ollama (docker or tarball)
#   sudo -iu souschef bash /work/sous-chef/lab/tools/vm-solo.sh models r3 r4   # stage + `ollama create` each export
#   sudo -iu souschef bash /work/sous-chef/lab/tools/vm-solo.sh serve r4       # (re)start the service, student → r4
#   sudo -iu souschef bash /work/sous-chef/lab/tools/vm-solo.sh status
#   sudo -iu souschef bash /work/sous-chef/lab/tools/vm-solo.sh stop
#
# Serving parity with the Cloud Run model image (lab/model-image/Dockerfile): the same Ollama tag (0.34.4; the
# docker path pins the same digest), OLLAMA_NUM_PARALLEL=1, OLLAMA_CONTEXT_LENGTH=65536, OLLAMA_KEEP_ALIVE=-1, and
# the Modelfile `make model-stage MODEL_TEMPERATURE=0.3 MODEL_PRESENCE_PENALTY=0` writes (r3's serving settings).
# The service key is a fixed LOCAL key (lab-solo-key, like lab/headroom.env's): the service is reachable only
# through the IAP tunnel (the VM's firewall lets in IAP SSH and nothing else) and holds no real data.
set -euo pipefail

LAB=${SOUS_CHEF_LAB:-/work/sous-chef/lab}
WORK=${SOUS_CHEF_TRAIN_WORK:-/work}
SOLO=$WORK/solo
BUCKET=${SOUS_CHEF_TRAIN_BUCKET:-gs://breadstack-sous-chef-int-training}
OLLAMA_VERSION=0.34.4
OLLAMA_IMAGE=ollama/ollama:0.34.4@sha256:8262851b2846b87c649eddf3e76beb270c52f4d1bc94559f47efde16b0841551
SERVICE_OBJ=${SERVICE_OBJ:-$BUCKET/kit/sous-chef-f79753b-linux-amd64}
SERVICE_SHA256=${SERVICE_SHA256:-}   # COLLECT/TRAIN.md: the sha256 printed when the binary was built; checked when set
SERVICE_KEY=lab-solo-key
PG_PORT=5433
SVC_PORT=8091
OLLAMA_PORT=11434

die() { echo "vm-solo: $*" >&2; exit 2; }
say() { echo "vm-solo: $*"; }
have() { command -v "$1" >/dev/null 2>&1; }
docker_gpu() { have docker && have nvidia-ctk && docker info >/dev/null 2>&1; }

setup() {
  [ "$(id -u)" = 0 ] || die "setup needs root (sudo bash $0 setup)"
  install -d -m 0755 -o souschef -g souschef "$SOLO" "$WORK/ollama" "$LAB/runs" "$LAB/runs/model-image"
  if have docker; then usermod -aG docker souschef; fi   # `docker exec` from souschef's next login (sudo -iu souschef)
  # --- Postgres 16 on 127.0.0.1:5433, user/db souschef ---
  if docker_gpu || have docker; then
    if ! docker ps --format '{{.Names}}' | grep -qx solo-postgres; then
      docker rm -f solo-postgres >/dev/null 2>&1 || true
      docker run -d --name solo-postgres --restart unless-stopped -p 127.0.0.1:$PG_PORT:5432 \
        -e POSTGRES_USER=souschef -e POSTGRES_PASSWORD=souschef -e POSTGRES_DB=souschef_solo postgres:16 >/dev/null
    fi
    say "postgres: docker container solo-postgres on 127.0.0.1:$PG_PORT"
  else
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq postgresql >/dev/null
    conf=$(ls /etc/postgresql/*/main/postgresql.conf | head -1)
    sed -i "s/^#\?port = .*/port = $PG_PORT/; s/^#\?listen_addresses = .*/listen_addresses = '127.0.0.1'/" "$conf"
    systemctl restart postgresql
    runuser -u postgres -- psql -p $PG_PORT -tc "SELECT 1 FROM pg_roles WHERE rolname='souschef'" | grep -q 1 || \
      runuser -u postgres -- psql -p $PG_PORT -c "CREATE ROLE souschef LOGIN PASSWORD 'souschef'"
    runuser -u postgres -- psql -p $PG_PORT -tc "SELECT 1 FROM pg_database WHERE datname='souschef_solo'" | grep -q 1 || \
      runuser -u postgres -- psql -p $PG_PORT -c "CREATE DATABASE souschef_solo OWNER souschef"
    say "postgres: apt package on 127.0.0.1:$PG_PORT"
  fi
  # --- Ollama 0.34.4 on 127.0.0.1:11434, as the Cloud Run image runs it ---
  # The docker GPU container when docker + nvidia-ctk are there AND it actually starts with the GPU (a runtime that
  # is installed but not configured fails `docker run --gpus all`); otherwise the release tarball of the same version.
  if docker_gpu && docker_ollama; then
    echo docker > "$SOLO/ollama-mode"
    say "ollama: docker $OLLAMA_IMAGE (GPU) on 127.0.0.1:$OLLAMA_PORT"
  else
    tarball_ollama
    echo tarball > "$SOLO/ollama-mode"
    chown souschef:souschef "$SOLO/ollama-mode"
    say "ollama: release tarball in $WORK/ollama/dist (started by 'serve' as souschef)"
  fi
  chown -R souschef:souschef "$SOLO"
}

docker_ollama() {
  # 0 when the solo-ollama container runs and sees the GPU; on any failure the container is removed and 1 returned
  if ! docker ps --format '{{.Names}}' | grep -qx solo-ollama; then
    docker rm -f solo-ollama >/dev/null 2>&1 || true
    docker run -d --name solo-ollama --restart unless-stopped --gpus all -p 127.0.0.1:$OLLAMA_PORT:11434 \
      -e OLLAMA_KEEP_ALIVE=-1 -e OLLAMA_NUM_PARALLEL=1 -e OLLAMA_CONTEXT_LENGTH=65536 -e OLLAMA_NO_CLOUD=1 \
      -v "$WORK/ollama:/root/.ollama" -v "$LAB/runs/model-image:/stage:ro" "$OLLAMA_IMAGE" >/dev/null 2>"$SOLO/docker-run.err" \
      || { say "docker run --gpus all failed ($(tail -1 "$SOLO/docker-run.err")) — falling back to the tarball"; return 1; }
  fi
  for _ in $(seq 30); do curl -fsS "http://127.0.0.1:$OLLAMA_PORT/api/version" >/dev/null 2>&1 && break; sleep 2; done
  if ! curl -fsS "http://127.0.0.1:$OLLAMA_PORT/api/version" >/dev/null 2>&1 || ! docker exec solo-ollama nvidia-smi -L >/dev/null 2>&1; then
    say "the solo-ollama container is not answering or sees no GPU — removing it, falling back to the tarball"
    docker rm -f solo-ollama >/dev/null 2>&1 || true
    return 1
  fi
  return 0
}

tarball_ollama() {
  # The release tarball (the same version); the asset name changed across releases, so try both.
  local d="$WORK/ollama/dist" base tmp
  if [ ! -x "$d/bin/ollama" ]; then
    install -d -m 0755 -o souschef -g souschef "$d"
    base=https://github.com/ollama/ollama/releases/download/v$OLLAMA_VERSION
    tmp=$(mktemp -d)
    if curl -fsSL -o "$tmp/o.tar.zst" "$base/ollama-linux-amd64.tar.zst"; then
      have zstd || { apt-get update -qq && apt-get install -y -qq zstd >/dev/null; }
      tar --use-compress-program=unzstd -xf "$tmp/o.tar.zst" -C "$d"
    else
      curl -fsSL -o "$tmp/o.tgz" "$base/ollama-linux-amd64.tgz" || die "cannot download Ollama $OLLAMA_VERSION"
      tar -xzf "$tmp/o.tgz" -C "$d"
    fi
    rm -rf "$tmp"
    chown -R souschef:souschef "$d"
  fi
  "$d/bin/ollama" --version 2>/dev/null | grep -q "$OLLAMA_VERSION" || say "WARNING: $("$d/bin/ollama" --version 2>&1 | tail -1) — expected $OLLAMA_VERSION"
}

ollama_cmd() {
  if [ "$(cat "$SOLO/ollama-mode" 2>/dev/null)" = docker ]; then docker exec solo-ollama ollama "$@"
  else OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT "$WORK/ollama/dist/bin/ollama" "$@"; fi
}

ollama_up() {
  if [ "$(cat "$SOLO/ollama-mode" 2>/dev/null)" = tarball ] && ! curl -fsS "http://127.0.0.1:$OLLAMA_PORT/api/version" >/dev/null 2>&1; then
    OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT OLLAMA_MODELS=$WORK/ollama/models OLLAMA_KEEP_ALIVE=-1 OLLAMA_NUM_PARALLEL=1 \
      OLLAMA_CONTEXT_LENGTH=65536 OLLAMA_NO_CLOUD=1 setsid nohup "$WORK/ollama/dist/bin/ollama" serve \
      >>"$SOLO/ollama.log" 2>&1 </dev/null &
  fi
  for _ in $(seq 60); do curl -fsS "http://127.0.0.1:$OLLAMA_PORT/api/version" >/dev/null 2>&1 && return 0; sleep 2; done
  die "ollama is not answering on 127.0.0.1:$OLLAMA_PORT ($SOLO/ollama.log)"
}

models() {
  [ $# -ge 1 ] || die "models NAME… (each an export under runs/exports/NAME, or in the bucket's runs/exports/NAME)"
  ollama_up
  cd "$LAB"
  for name in "$@"; do
    if [ ! -f "runs/exports/$name/Modelfile.q4_k_m" ]; then
      say "$name: pulling $BUCKET/runs/exports/$name"
      mkdir -p "runs/exports/$name" && gcloud storage rsync --recursive "$BUCKET/runs/exports/$name" "runs/exports/$name"
    fi
    make --no-print-directory model-stage NAME="$name" MODEL_EXPORT="runs/exports/$name" \
      MODEL_STAGE="runs/model-image/$name-q4_k_m" MODEL_TEMPERATURE=0.3 MODEL_PRESENCE_PENALTY=0
    if [ "$(cat "$SOLO/ollama-mode")" = docker ]; then
      ollama_cmd create "$name" -f "/stage/$name-q4_k_m/Modelfile"
    else
      (cd "runs/model-image/$name-q4_k_m" && ollama_cmd create "$name" -f Modelfile)
    fi
    ollama_cmd show "$name" --parameters | grep -E 'temperature|presence_penalty|num_ctx|top_k|top_p' | sed "s/^/vm-solo: $name: /"
  done
}

service_bin() {
  local b=$SOLO/sous-chef
  if [ ! -x "$b" ]; then
    gcloud storage cp "$SERVICE_OBJ" "$b.tmp" >/dev/null && chmod 0755 "$b.tmp" && mv "$b.tmp" "$b" || die "no service binary at $SERVICE_OBJ (TRAIN.md: the main session uploads it)"
  fi
  if [ -n "$SERVICE_SHA256" ]; then
    echo "$SERVICE_SHA256  $b" | sha256sum -c --quiet - || die "$b is not the f79753b build (sha256 differs)"
  fi
  echo "$b"
}

serve() {
  local model=${1:-}
  [ -n "$model" ] || die "serve MODEL (an Ollama model created by 'models', e.g. r4)"
  ollama_up
  ollama_cmd list | awk 'NR > 1 { print $1 }' | grep -Eq "^$model(:latest)?$" || die "no Ollama model $model (run: models $model)"
  local bin; bin=$(service_bin)
  stop_service
  local routes
  routes=$(printf '{"student":{"kind":"messages-compatible","url":"http://127.0.0.1:%s","auth":"none","model":"%s","max_tokens":4096,"stream":false,"report_as":"student-%s"}}' "$OLLAMA_PORT" "$model" "$model")
  ( export PORT=$SVC_PORT DATABASE_URL="postgres://souschef:souschef@127.0.0.1:$PG_PORT/souschef_solo?sslmode=disable" \
      MIGRATE_AT_BOOT=on SERVICE_KEYS=$SERVICE_KEY BUILD_ID="f79753b-solo-$model" DEFAULT_ROUTE=student \
      MODEL_ROUTE_ALLOWLIST=student MODEL_ROUTES="$routes" MODEL_CALL_TIMEOUT=180s TURN_TTL=15m \
      MAX_CONCURRENT_TURNS=4 TRACE_RETENTION_DAYS=30
    setsid nohup "$bin" >>"$SOLO/service-$model.log" 2>&1 </dev/null & echo $! > "$SOLO/service.pid" )
  for _ in $(seq 60); do curl -fsS "http://127.0.0.1:$SVC_PORT/healthz" >/dev/null 2>&1 && break; sleep 1; done
  curl -fsS "http://127.0.0.1:$SVC_PORT/version" | tee "$SOLO/version-$model.json"; echo
  grep -q '"student"' "$SOLO/version-$model.json" || die "the service is not up with the student route ($SOLO/service-$model.log)"
  # Warm the model (the first call loads it to the GPU): one short generation.
  curl -fsS "http://127.0.0.1:$OLLAMA_PORT/api/generate" -d "{\"model\":\"$model\",\"prompt\":\"ok\",\"stream\":false,\"options\":{\"num_predict\":1}}" >/dev/null && \
    say "student → $model, service on 127.0.0.1:$SVC_PORT (key $SERVICE_KEY, local only). Workstation: EVAL.md"
}

stop_service() {
  if [ -f "$SOLO/service.pid" ] && kill -0 "$(cat "$SOLO/service.pid")" 2>/dev/null; then
    kill "$(cat "$SOLO/service.pid")"; sleep 2
  fi
  rm -f "$SOLO/service.pid"
}

status() {
  echo "ollama:  $(curl -fsS http://127.0.0.1:$OLLAMA_PORT/api/version 2>/dev/null || echo down) ($(cat "$SOLO/ollama-mode" 2>/dev/null || echo 'not set up'))"
  ollama_cmd ps 2>/dev/null | sed 's/^/         /' || true
  echo "service: $(curl -fsS http://127.0.0.1:$SVC_PORT/version 2>/dev/null || echo down)"
  echo "gpu:     $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo '?')"
}

cmd=${1:-}; shift || true
case "$cmd" in
  setup) setup ;;
  models) models "$@" ;;
  serve) serve "$@" ;;
  stop) stop_service; say "service stopped (Ollama and Postgres keep running; train-vm-down removes everything)" ;;
  status) status ;;
  *) die "usage: setup | models NAME… | serve MODEL | stop | status (header of $0)" ;;
esac
