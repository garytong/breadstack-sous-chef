# syntax=docker/dockerfile:1.7
#
# sous-chef-mx: the Mistral arm's exam/eval image (docs/MISTRAL.md §(f), §(g)). One container per Vertex job runs
# Ollama 0.34.4 (the model image's exact binary, one `ollama serve` per GPU) and the Sous Chef service + lab harness
# from the deployed lab image, so the prompt is the one int serves:
#
#   <ne2>-docker.pkg.dev/breadstack-sous-chef-int/sous-chef/sous-chef-mx:<branch sha>
#
# Never the pipeline's images: built only by images/mx/cloudbuild.yaml, which pushes this one name and nothing
# else. Built from the REPOSITORY ROOT (ignore file: Dockerfile.dockerignore beside this file):
#
#   docker build --platform linux/amd64 -f images/mx/Dockerfile -t sous-chef-mx:dev .
#
# Layers:
#   base     sous-chef-lab:28491da (the deployed service tag), pinned by digest: Python 3.14 + the lab venv,
#            Postgres 16, node 22, the three service binaries built from service/ at 28491da, gcloud
#   ollama   /usr/bin/ollama + /usr/lib/ollama (llama-server b11081 + the CUDA 12/13 backends) from
#            ollama/ollama:0.34.4, the digest lab/model-image/Dockerfile pins; plus the three runtime libraries
#            Ollama's own image installs (libgomp1 for the CPU backend's OpenMP, libopenblas0, libvulkan1)
#   lab      the branch's lab/ over the image's (the lab is byte-identical to 28491da except the Mistral files;
#            the editable install already points at /opt/lab)
#   env      what the Ollama image and the model image set and a bare COPY loses: NVIDIA_*, LD_LIBRARY_PATH,
#            OLLAMA_NO_CLOUD / NUM_PARALLEL / KEEP_ALIVE / CONTEXT_LENGTH. No OLLAMA_DEBUG*: request bodies never
#            go to disk.
#
# No model is baked in: each job pulls its staged GGUF + Modelfile from the bucket and runs `ollama create`
# (the build account cannot read the sous-chef-model repository, and one image serves every arm).

ARG LAB_IMAGE=northamerica-northeast2-docker.pkg.dev/breadstack-sous-chef-int/sous-chef/sous-chef-lab:28491da@sha256:aa450fa3b804b242801754d09187540f7289c0c0056d7a45a555ab141790b4c9
ARG OLLAMA_IMAGE=ollama/ollama:0.34.4@sha256:8262851b2846b87c649eddf3e76beb270c52f4d1bc94559f47efde16b0841551

FROM ${OLLAMA_IMAGE} AS ollama

FROM ${LAB_IMAGE}
USER root
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends libgomp1 libopenblas0 libvulkan1; \
    rm -rf /var/lib/apt/lists/*
COPY --from=ollama /usr/bin/ollama /usr/bin/ollama
COPY --from=ollama /usr/lib/ollama /usr/lib/ollama
COPY --chown=10001:10001 lab/ /opt/lab/
# The image must hold what the runner needs before it is pushed: Ollama 0.34.4 that links (no missing library on
# the CPU side; the CUDA backends need the host driver, which Vertex mounts at run time), the lab imports, the
# Mistral runner and both base Modelfiles. `selftest --no-gpu` checks the rest (service binaries, node, gcloud).
RUN set -eux; \
    test -x /usr/lib/ollama/llama-server; \
    ! ldd /usr/bin/ollama /usr/lib/ollama/llama-server | grep -F 'not found'; \
    ollama --version 2>&1 | grep -F 0.34.4; \
    test -f /opt/lab/tools/mxarm.py; test -f /opt/lab/tools/mx_gguf.py; test -f /opt/lab/tools/mx_latency.py; \
    test -f /opt/lab/base-modelfiles/qwen3.5-9b.modelfile; \
    test -f /opt/lab/base-modelfiles/ministral-3-8b.modelfile; \
    PYTHONDONTWRITEBYTECODE=1 /opt/lab/.venv/bin/python /opt/lab/tools/mxarm.py selftest --no-gpu; \
    rm -rf /home/souschef/.config/gcloud
# ^ the selftest runs gcloud as root, which left a root-owned ~/.config/gcloud: every gcloud call of the 10001 user
#   then failed (2026-10-17: all three exam jobs exited 1 at their first write). CLOUDSDK_CONFIG below moves it to /tmp.

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    OLLAMA_NO_CLOUD=1 \
    OLLAMA_NUM_PARALLEL=1 \
    OLLAMA_KEEP_ALIVE=-1 \
    OLLAMA_CONTEXT_LENGTH=65536 \
    MX_WORKDIR=/tmp/mx \
    CLOUDSDK_CONFIG=/tmp/mx/gcloud

WORKDIR /opt/lab
USER 10001:10001
# Vertex passes args: exam … | eval … | gate … | report … | selftest (tools/mxarm.py)
ENTRYPOINT ["/opt/lab/.venv/bin/python", "/opt/lab/tools/mxarm.py"]
CMD ["selftest"]
