# syntax=docker/dockerfile:1.7
#
# The trainer image (DESIGN §6.4): Vertex AI custom training runs it for the
# pipeline's SFT + GGUF export + model staging (lab/tools/pipeline-train.sh),
# r4's recipe on Qwen3.5-9B unchanged.
#
#   <ne2>-docker.pkg.dev/<project>/sous-chef/sous-chef-trainer:t-<hash12>
#   ... and an alias tag per service tag (sous-chef-trainer:<service tag>),
#   which is what the retrain workflow resolves.
#
# t-<hash12> = images/trainer/tag.sh: sha256 over every file this image
# copies (the lab without tests, data, runs or venvs, and images/trainer/),
# so a trainer image never runs other code than its tag says. Built by CI
# only when that tag is absent; otherwise CI only adds the alias tag.
#
# Built from the REPOSITORY ROOT (ignore file: Dockerfile.dockerignore):
#
#   docker build --platform linux/amd64 -f images/trainer/Dockerfile -t sous-chef-trainer:dev .
#
# Parts:
#   base       CUDA 13.0 devel (Triton JIT compiles kernels: headers and a C
#              compiler) on Ubuntu 24.04, plus cuda-compat-13-0: torch's
#              cu130 wheels need driver >= 580, and the entrypoint puts the
#              forward-compatibility libcuda first when Vertex's host driver
#              is older (H100 / A100 are datacenter GPUs, which it supports)
#   python     uv (pinned) -> Python 3.12 (the measured training env,
#              lab/Makefile TRAIN_PYTHON) in /opt/lab/.venv-train with the
#              [train] extra pinned by lab/train-constraints.txt, torch from
#              the cu130 index: exactly `make -C lab train-env`
#   llama.cpp  the unslothai fork at b11160-mix-a6922cc (the build r4's export
#              used, ~/.unsloth/llama.cpp on the workstation), CPU only:
#              llama-quantize + the converter scripts. Unsloth finds it through
#              UNSLOTH_LLAMA_CPP_PATH and never clones or compiles at run time
#              (UNSLOTH_LLAMA_CPP_OFFLINE=1, UNSLOTH_AUTO_INSTALL=0)
#   lab        /opt/lab (no tests), the trainer's entrypoint and self-test
#   tools      google-cloud-cli (gcloud storage), make (make model-stage)

ARG CUDA_IMAGE=nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.19

FROM ${UV_IMAGE} AS uv

# --- llama.cpp: quantizer + converter, pinned by commit -------------------------------
FROM ${CUDA_IMAGE} AS llama
ARG LLAMA_CPP_REPO=https://github.com/unslothai/llama.cpp
ARG LLAMA_CPP_TAG=b11160-mix-a6922cc
ARG LLAMA_CPP_COMMIT=119fa9dc43cd4d63e89a7b5d11f9378b84c52822
ENV DEBIAN_FRONTEND=noninteractive
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates git cmake build-essential; \
    rm -rf /var/lib/apt/lists/*
RUN set -eux; \
    git init -q /opt/llama.cpp; cd /opt/llama.cpp; \
    git remote add origin "${LLAMA_CPP_REPO}"; \
    git fetch -q --depth 1 origin "refs/tags/${LLAMA_CPP_TAG}"; \
    git checkout -q --detach FETCH_HEAD; \
    test "$(git rev-parse HEAD)" = "${LLAMA_CPP_COMMIT}"; \
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF \
      -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF; \
    cmake --build build -j"$(nproc)" --target llama-quantize; \
    find build -mindepth 1 -maxdepth 1 ! -name bin -exec rm -rf {} +; \
    ln -s build/bin/llama-quantize llama-quantize; \
    ./llama-quantize --help 2>&1 | grep -qi usage; \
    printf '{"source":"sous-chef trainer image","repo":"unslothai/llama.cpp","tag":"%s","commit":"%s"}\n' "${LLAMA_CPP_TAG}" "${LLAMA_CPP_COMMIT}" > SOUS_CHEF_LLAMA_CPP.json; \
    rm -rf .git

# --- runtime ------------------------------------------------------------------------------
FROM ${CUDA_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl gnupg git make build-essential tar zstd procps cuda-compat-13-0; \
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" > /etc/apt/sources.list.d/google-cloud-sdk.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends google-cloud-cli; \
    rm -rf /var/lib/apt/lists/*; \
    test -d /usr/local/cuda/compat

COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy UV_PYTHON_PREFERENCE=only-managed \
    TORCH_INDEX=https://download.pytorch.org/whl/cu130

# The training environment first (the heavy, rarely changing layer): the
# [train] extra's requirements from pyproject.toml under the constraints,
# then the lab itself without dependencies.
COPY lab/pyproject.toml lab/train-constraints.txt /opt/lab/
RUN --mount=type=cache,target=/root/.cache/uv set -eux; \
    uv python install 3.12; \
    uv venv --python 3.12 /opt/lab/.venv-train; \
    uv pip install --python /opt/lab/.venv-train/bin/python --index "${TORCH_INDEX}" --index-strategy unsafe-best-match \
      -c /opt/lab/train-constraints.txt -r /opt/lab/pyproject.toml --extra train; \
    CUDA_VISIBLE_DEVICES= /opt/lab/.venv-train/bin/python -W ignore -c 'import torch, trl, peft, transformers, fla; print("train-env ok: torch", torch.__version__, "cuda", torch.version.cuda, "trl", trl.__version__, "peft", peft.__version__, "transformers", transformers.__version__)'

COPY --from=llama /opt/llama.cpp /opt/llama.cpp
COPY lab/ /opt/lab/
COPY images/trainer/entrypoint.sh images/trainer/selftest.py images/trainer/tag.sh /opt/trainer/
RUN set -eux; \
    uv pip install --python /opt/lab/.venv-train/bin/python --no-deps -e /opt/lab; \
    /opt/lab/.venv-train/bin/python /opt/llama.cpp/convert_hf_to_gguf.py --help >/dev/null; \
    /opt/lab/.venv-train/bin/python /opt/trainer/selftest.py --no-gpu; \
    chmod 0755 /opt/trainer/entrypoint.sh; \
    mkdir -p /work/hf

ENV PATH=/opt/lab/.venv-train/bin:$PATH \
    SOUS_CHEF_LAB=/opt/lab \
    HF_HOME=/work/hf \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    UNSLOTH_LLAMA_CPP_PATH=/opt/llama.cpp \
    UNSLOTH_LLAMA_CPP_SCRIPTS_DIR=/opt/llama.cpp \
    UNSLOTH_LLAMA_CPP_OFFLINE=1 \
    UNSLOTH_AUTO_INSTALL=0 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/lab
# Vertex passes args [<run id>, <full|smoke>]; `selftest` checks the GPU
# environment alone (entrypoint.sh).
ENTRYPOINT ["/opt/trainer/entrypoint.sh"]
