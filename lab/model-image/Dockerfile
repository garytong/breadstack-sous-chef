# The in-house model's image (lab/model-image/README.md): Ollama serving the
# lab's trained 9B student on the Anthropic-compatible POST /v1/messages, for
# the private Cloud Run GPU service in infra/modules/model-gpu (the
# `inhouse` route, kind messages-compatible, auth id-token). A copy of the
# thermometer's lab/sourdough/Dockerfile with the model named sous-chef, as
# infra/README.md's route example expects.
#
# Built by `make -C lab model-build`, never from the repo root. The build
# context is this directory (the entrypoint alone, per .dockerignore); the
# model arrives through the named context "model", the staging directory
# `make model-stage` wrote and model-context.sh has checked holds exactly
# one GGUF and its Modelfile. The weights are gitignored (lab/runs/), never
# committed, and pushed only to the project's private repository
# (make model-push prints the command).
#
# Two stages, so the image holds the weights once: the first imports the
# GGUF into an Ollama model store (`ollama create` copies it into a blob),
# the second takes that store alone. One stage would ship the COPY layer and
# the blob, ~5.4 GB twice.
#
# The base is the Ollama the lab's exams ran on, 0.34.4 (docs/
# TRAIN-WORKSTATION.md: the lab Ollama on :11435), pinned by tag and by its
# index digest. It has no curl: the waits and the preload use the ollama
# CLI. Cloud Run runs linux/amd64, which the build's --platform picks from
# the index (make model-build passes it).
ARG OLLAMA_IMAGE=ollama/ollama:0.34.4@sha256:8262851b2846b87c649eddf3e76beb270c52f4d1bc94559f47efde16b0841551

FROM ${OLLAMA_IMAGE} AS build
ENV OLLAMA_MODELS=/models OLLAMA_NO_CLOUD=1
COPY --from=model . /src/
# A throwaway server for the import. The model must keep the lab's 64k
# context (PARAMETER num_ctx 65536, which make model-stage writes): Ollama
# cuts a longer prompt silently, and the editor's system prompt plus a few
# big tool results outgrow 32k.
RUN ollama serve >/tmp/serve.log 2>&1 & server=$!; \
    i=0; until ollama list >/dev/null 2>&1; do \
      i=$((i + 1)); [ "$i" -le 60 ] || { cat /tmp/serve.log; exit 1; }; sleep 1; \
    done; \
    ollama create sous-chef -f /src/Modelfile || { cat /tmp/serve.log; exit 1; }; \
    ollama show sous-chef --parameters | grep -Eq '^num_ctx[[:space:]]+65536$' || \
      { echo "sous-chef: the created model lacks num_ctx 65536:"; ollama show sous-chef --modelfile; exit 1; }; \
    ollama list; \
    kill "$server"; wait "$server" || true

FROM ${OLLAMA_IMAGE}
# - OLLAMA_HOST: Cloud Run sends requests to 8080 (the module's
#   container_port); Ollama reads OLLAMA_HOST, not PORT.
# - OLLAMA_KEEP_ALIVE=-1: the model stays loaded for the instance's life.
# - OLLAMA_CONTEXT_LENGTH: the Modelfile's num_ctx wins anyway.
# - OLLAMA_NUM_PARALLEL=1: one answer at a time, the rest queue inside
#   Ollama. 0.34.4 forces 1 for this architecture (qwen35) whatever is set;
#   the module runs at most one instance.
# - OLLAMA_NO_CLOUD=1: no ollama.com remote models or web search, so no
#   model name can ever send a question out of the container.
# Left at their defaults, as in the lab's exams: flash attention and the KV
# cache type. Left off: OLLAMA_DEBUG and OLLAMA_DEBUG_LOG_REQUESTS, which
# would write request bodies (site content in the tool results) to disk.
ENV OLLAMA_HOST=0.0.0.0:8080 \
    OLLAMA_MODELS=/models \
    OLLAMA_KEEP_ALIVE=-1 \
    OLLAMA_CONTEXT_LENGTH=65536 \
    OLLAMA_NUM_PARALLEL=1 \
    OLLAMA_NO_CLOUD=1 \
    SOUS_CHEF_MODEL=sous-chef
COPY --from=build /models /models
COPY --chmod=0755 entrypoint.sh /usr/local/bin/sous-chef-model-entrypoint
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/sous-chef-model-entrypoint"]
