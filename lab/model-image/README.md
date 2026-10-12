# lab/model-image — the in-house model's serving image

Ollama 0.34.4 plus the lab's exported student (one q4_k_m GGUF), as the
Cloud Run GPU service `sous-chef-model` runs it
(`infra/modules/model-gpu`, infra/README.md "Phase 5"). The service speaks
the Anthropic-compatible `POST /v1/messages`; sous-chef reaches it through
a `messages-compatible` route with `auth: id-token` and `model: sous-chef`,
which is the model's name inside the image. A copy of the thermometer's
`lab/sourdough/{Dockerfile,entrypoint.sh,model-context.sh}` with that name;
the serving behaviour is the one the thermometer verified.

Nothing here touches the lab Ollama on `:11435`, the local service on
`:8091`, or a GPU: the build imports the GGUF with a throwaway CPU Ollama
inside the build stage. Nothing here pushes or applies either —
`make model-push` prints the commands.

The drift pipeline builds the same image without a workstation: Cloud Build
in the int project runs `model-context.sh` and this `Dockerfile` on a round's
staged GGUF (`infra/modules/pipeline/cloudbuild-model.yaml`, tag
`c-<run_id>`; docs/RETRAIN.md §R5). The build context it uses is the lab
image's copy of this directory, so a change here reaches the pipeline with
the next service build.

## Files

| File | Role |
|---|---|
| `Dockerfile` | Two stages on `ollama/ollama:0.34.4` (pinned by index digest). Stage 1 `ollama create sous-chef -f /src/Modelfile` from the named build context `model`, then asserts `num_ctx 65536`; stage 2 copies only the resulting model store, so the weights are in the image once (~5.4 GB for q4_k_m). Env: `OLLAMA_HOST=0.0.0.0:8080`, `OLLAMA_KEEP_ALIVE=-1`, `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_NO_CLOUD=1`, `SOUS_CHEF_MODEL=sous-chef`. |
| `entrypoint.sh` | `ollama serve` as PID 1's child, plus a preload (`ollama run sous-chef ""`) as soon as the server answers, so a warm instance (`gpu_model_min_instances = 1`) holds the model in the GPU before the first question. Logs `sous-chef: sous-chef loaded in N s`. |
| `model-context.sh` | Checks the staging directory the build is handed: exactly `Modelfile` + the one GGUF its single `FROM ./<file>.gguf` names, no `ADAPTER`, no symlink, `GGUF` magic, `PARAMETER num_ctx 65536`. Prints the resolved path. |
| `.dockerignore` | The build context is this directory and sends only `entrypoint.sh`; the weights come through `--build-context model=…`. |

## Build

export.py (`make -C lab export`) writes `lab/runs/exports/<name>/` with the
q4_k_m **and** q8_0 GGUFs, `Modelfile.q4_k_m`, `Modelfile.q8_0` and
`export.json` side by side. The image must hold exactly one GGUF and one
`Modelfile`, so a staging step comes first:

```sh
make -C lab model-stage NAME=souschef-9b-r1          # runs/exports/souschef-9b-r1 → runs/model-image/souschef-9b-r1-q4_k_m/
make -C lab model-build NAME=souschef-9b-r1 MODEL_TAG=r1
make -C lab model-push  MODEL_TAG=r1                 # prints the push; run it by hand
```

- `model-stage` empties `MODEL_STAGE` (default `runs/model-image/$(NAME)-$(QUANT)`),
  hard-links the `QUANT` GGUF (default `q4_k_m`; a copy if `runs/` is on
  another filesystem than the export, never a symlink) and writes `Modelfile`
  = export.py's `Modelfile.$(QUANT)` with `FROM ./<gguf>` and exactly one
  `PARAMETER num_ctx 65536` (export.py's own num_ctx line is dropped and the
  64k one appended). Then runs `model-context.sh` on the result.
  `MODEL_EXPORT=<dir>` for an export elsewhere.
- `model-build` runs `model-context.sh` again and
  `docker build --platform linux/amd64 --build-context model=<stage> -f model-image/Dockerfile -t <image> model-image`.
  CPU only; the thermometer's equivalent took a few minutes and ~11 GB of
  disk, pushing ~9.3 GB. The image is
  `us-east4-docker.pkg.dev/$(GCP_PROJECT)/sous-chef-model/model:$(MODEL_TAG)`
  (`GCP_PROJECT` defaults to `breadstack-sous-chef-int`; the repository is
  the module's `sous-chef-model`, created by the first-half apply with
  `gpu_model_enabled = true`).
- `model-push` only prints `gcloud auth configure-docker` + `docker push`.
  The repository's tags are immutable: a new build that is pushed needs a new
  `MODEL_TAG`, and `gpu_model_image_tag` in `infra/envs/int` follows it for
  the second-half apply (`gpu_model_serve = true`).

Use q4_k_m: the service's container is 16 GiB and the q8_0 GGUF (~9–10 GB)
leaves no room beside Ollama's mmap; q4_k_m at 64k context measured ~7.5 GiB
of VRAM on the lab's GPU, which fits the L4.

## Checks

- `make -C lab test` runs `tests/test_model_image.py`: `model-stage` on a
  fake export (a few bytes with the `GGUF` magic) — the staging layout, the
  hard link, the rewritten Modelfile, the quant pick, the copy fallback;
  every refusal in `model-context.sh`; `make -n model-build` (amd64, the
  model context, the image name); `model-push` prints and runs nothing; the
  Dockerfile's pins and env.
- Dockerfile lint, no daemon work beyond resolving the base's metadata:
  `cd lab && docker build --check --platform linux/amd64 --build-context model=<any staged dir> -f model-image/Dockerfile model-image`
  (2026-10-04: "Check complete, no warnings found"). `sh -n` on both scripts
  is part of the pytest; shellcheck is not installed here.
- The real build cannot be dry-run with a fake GGUF (`ollama create` reads
  the weights), so the manual check once an export exists is the build itself
  followed by a CPU smoke on this machine, on a free port and never on
  `:11435`:
  `docker run --rm -p 18080:8080 <image>` then
  `curl -s localhost:18080/api/tags` should list `sous-chef:latest`, and the
  log shows `sous-chef: sous-chef loaded in N s` (CPU load of a 9B q4 takes
  minutes; it is only a wiring check — answers come from the L4 on Cloud Run).
