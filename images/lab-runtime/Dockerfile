# syntax=docker/dockerfile:1.7
#
# The lab runtime image (DESIGN §6.1): every pipeline stage that runs lab
# code is a Cloud Run job on this image (infra/modules/pipeline/jobs.tf).
# One image per deployed service tag, so the lab renders exactly the prompt
# the deployed service serves:
#
#   <ne2>-docker.pkg.dev/<project>/sous-chef/sous-chef-lab:<service tag>
#
# Built from the REPOSITORY ROOT (context = repo root; the ignore file is
# Dockerfile.dockerignore beside this file, which BuildKit reads for a
# Dockerfile at this path):
#
#   docker build --platform linux/amd64 -f images/lab-runtime/Dockerfile \
#     --build-arg BUILD_ID=<service tag> -t sous-chef-lab:<service tag> .
#
# CI builds it after the tests (cloudbuild.yaml step lab-image); a manual
# service build needs `make lab-image TAG=<tag>` (images/lab-runtime/
# cloudbuild.yaml).
#
# Contents: the lab (no data, runs or venvs) with Python 3.14 in
# /opt/lab/.venv (.[dev,author]); node 22 for the browser engine (the
# engine snapshot itself is NOT here: souschef_lab.enginebundle fetches the
# vendored snapshot from the pipeline bucket at start); Postgres 16 server
# binaries for the local-service harness; the three service binaries built
# from the same service/ tree and toolchain as the service image, with
# BUILD_ID baked in; the service sources (the lab hashes the retrieval and
# prefix sources); google-cloud-cli for `gcloud storage`. No secret is used
# or stored at build time.

ARG GO_IMAGE=golang:1.24-bookworm
ARG NODE_IMAGE=node:22-bookworm-slim
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.19
ARG BASE_IMAGE=debian:bookworm-slim

FROM ${UV_IMAGE} AS uv
FROM ${NODE_IMAGE} AS node

# --- the service binaries (service/Dockerfile's flags) --------------------------------
FROM ${GO_IMAGE} AS go
WORKDIR /src
ENV GOTOOLCHAIN=local CGO_ENABLED=0 GOOS=linux GOARCH=amd64
COPY service/go.mod service/go.sum ./
RUN go mod download
COPY service/ ./
ARG BUILD_ID=dev
RUN set -eu; mkdir -p /out; \
    for c in sous-chef sous-chef-fields sous-chef-prefix; do \
      go build -trimpath -ldflags "-s -w -X main.buildID=${BUILD_ID}" -o "/out/$c" "./cmd/$c"; \
    done; \
    printf '%s\n' "${BUILD_ID}" > /out/BUILD_ID

# --- runtime ------------------------------------------------------------------------------
FROM ${BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8

# Postgres 16 from PGDG (bookworm ships 15) with no cluster created at
# install; google-cloud-cli from Google's repository; git (the engine's
# fingerprint asks it, harmlessly), zstd (the engine snapshot), procps.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl gnupg git zstd tar procps make; \
    install -d /usr/share/postgresql-common/pgdg /etc/postgresql-common; \
    curl -fsSL -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc https://www.postgresql.org/media/keys/ACCC4CF8.asc; \
    echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt bookworm-pgdg main" > /etc/apt/sources.list.d/pgdg.list; \
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" > /etc/apt/sources.list.d/google-cloud-sdk.list; \
    echo 'create_main_cluster = false' > /etc/postgresql-common/createcluster.conf; \
    apt-get update; \
    apt-get install -y --no-install-recommends postgresql-16 google-cloud-cli; \
    apt-get purge -y gnupg; apt-get autoremove -y; \
    rm -rf /var/lib/apt/lists/*; \
    /usr/lib/postgresql/16/bin/pg_ctl --version

# node 22 (the lab needs >= 20) and npm, from the official image.
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm && node --version

# The runtime user: uid 10001, owns the lab (stages write under it) and its
# home (gcloud's config, the lab cache).
RUN useradd --uid 10001 --user-group --create-home --home-dir /home/souschef --shell /bin/bash souschef

# The service binaries and sources. The sources sit at /opt/service, beside
# /opt/lab as in the repository: the lab's default service dir is
# <lab>/../service, and lab/fixtures/manifest.snapshot.json is a relative
# symlink into ../../service/testdata. /opt/sous-chef/service points there.
COPY --from=go /out/sous-chef /out/sous-chef-fields /out/sous-chef-prefix /opt/sous-chef/bin/
COPY --from=go /out/BUILD_ID /opt/sous-chef/BUILD_ID
COPY service/ /opt/service/
RUN ln -s /opt/service /opt/sous-chef/service

# Python 3.14 (uv, pinned) and the lab with its [dev,author] extras. The
# lab's core is stdlib-only; [author] is the Anthropic SDK (case authoring).
COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_NO_CACHE=1 UV_PYTHON_PREFERENCE=only-managed
COPY lab/ /opt/lab/
RUN set -eux; \
    uv python install 3.14; \
    uv venv --python 3.14 /opt/lab/.venv; \
    uv pip install --python /opt/lab/.venv/bin/python -e '/opt/lab[dev,author]'; \
    /opt/lab/.venv/bin/python -c 'import souschef_lab, sys; print("souschef_lab on", sys.version.split()[0])'; \
    test -f /opt/lab/fixtures/manifest.snapshot.json; \
    chown -R souschef:souschef /opt/lab

ENV PATH=/opt/lab/.venv/bin:/usr/lib/postgresql/16/bin:$PATH \
    SOUS_CHEF_BIN=/opt/sous-chef/bin/sous-chef \
    SOUS_CHEF_FIELDS_BIN=/opt/sous-chef/bin/sous-chef-fields \
    SOUS_CHEF_PREFIX_BIN=/opt/sous-chef/bin/sous-chef-prefix \
    SOUS_CHEF_SERVICE_DIR=/opt/sous-chef/service \
    SOUS_CHEF_BUILD_ID_FILE=/opt/sous-chef/BUILD_ID \
    PG_BIN=/usr/lib/postgresql/16/bin \
    SOUS_CHEF_LAB_CACHE=/tmp/lab-cache \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/souschef

WORKDIR /opt/lab
USER 10001:10001
ENTRYPOINT ["/opt/lab/.venv/bin/python", "-m", "souschef_lab.pipeline"]
CMD ["selftest"]
