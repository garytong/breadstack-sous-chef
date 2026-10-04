# Deploying sous-chef to int — runbook

Ordered commands for the first deploy of `breadstack-sous-chef-int`, then
what every later roll-out looks like (a push to `main`). A person runs every
`terraform apply` and every `gcloud` command that changes something; Cloud
Build (`cloudbuild.yaml`) only builds, pushes, migrates and rolls the image.

Ownership, in one line: **Terraform owns the project, database, secrets, IAM,
the Run service and both jobs; CI owns which image they run.** The module's
`lifecycle { ignore_changes }` on the image (and on the labels/annotations
gcloud stamps) is what lets both exist without fighting — see
`infra/README.md` "Deploy".

Status of this runbook (2026-10-02): written against the Terraform in
`infra/` and the verified pantry pipeline; **no step below has been run
against GCP yet** (the int project does not exist). Anything marked *shape*
is the expected command, not a measured one.

```sh
export PROJECT=breadstack-sous-chef-int
export REGION=northamerica-northeast2
export AR_REPO=$REGION-docker.pkg.dev/$PROJECT/sous-chef   # = terraform output artifact_repo
gcloud config set account gary.tong@breadstack.com
```

## 1. Project + state bucket (once)

Create the project, link billing, then (from `infra/README.md` "Bootstrap"):

```sh
gcloud storage buckets create gs://$PROJECT-sous-chef-tfstate --project $PROJECT \
  --location $REGION --uniform-bucket-level-access
gcloud storage buckets update gs://$PROJECT-sous-chef-tfstate --versioning --public-access-prevention
gcloud storage buckets add-iam-policy-binding gs://$PROJECT-sous-chef-tfstate \
  --member user:gary.tong@breadstack.com --role roles/storage.objectAdmin
```

## 2. Terraform init + tfvars

```sh
cd infra/envs/int
terraform init -backend-config="bucket=$PROJECT-sous-chef-tfstate"
cp terraform.tfvars.example terraform.tfvars
```

The example already carries the int decisions; check them:

| tfvar | int value | why |
|---|---|---|
| `ingress` | `INGRESS_TRAFFIC_ALL` | the launcher runs on a GKE cluster in another project (section 10) with no internal LB to this one |
| `public_invoker` | `true` | the launcher sends only `Authorization: Bearer <service key>`; without this Cloud Run's IAM check answers 403 (section 10) |
| `ci_service_account` | `52346985251@cloudbuild.gserviceaccount.com` | Cloud Build's default SA in `prod-devops-core-project`, where the pantry triggers live |
| `launcher_service_account` | `""` | no launcher GSA exists; irrelevant while `public_invoker` is true |
| `image` | `.../sous-chef/sous-chef:dev` | placeholder — every apply passes `-var=image=` explicitly |

## 3. Foundations (targeted apply; Cloud SQL takes 5–10 min)

Everything except the service and the retention job. The CI bindings are in
the list so Cloud Build can push the first image in step 5.

```sh
terraform apply -var=image=$AR_REPO/sous-chef:bootstrap \
  -target=module.sous_chef.google_project_service.services \
  -target=module.sous_chef.google_artifact_registry_repository.sous_chef \
  -target=module.sous_chef.google_sql_database_instance.sous_chef \
  -target=module.sous_chef.google_sql_database.sous_chef \
  -target=module.sous_chef.google_sql_user.sous_chef \
  -target=module.sous_chef.google_secret_manager_secret.user \
  -target=module.sous_chef.google_secret_manager_secret_version.database_url \
  -target=module.sous_chef.google_service_account.run \
  -target=module.sous_chef.google_service_account.scheduler \
  -target=module.sous_chef.google_artifact_registry_repository_iam_member.ci_writer \
  -target=module.sous_chef.google_project_iam_member.ci_run_developer \
  -target=module.sous_chef.google_project_iam_member.ci_service_usage \
  -target=module.sous_chef.google_service_account_iam_member.ci_acts_as_run
```

## 4. Secrets (the two Terraform creates empty)

```sh
# one service key; the launcher gets the same value in step 9
openssl rand -hex 32 | tr -d '\n' > /tmp/sous-chef-int.key
gcloud secrets versions add sous-chef-service-keys --project $PROJECT --data-file=/tmp/sous-chef-int.key

# the Anthropic key (paste, Enter, Ctrl-D; or --data-file=)
gcloud secrets versions add sous-chef-anthropic-api-key --project $PROJECT --data-file=-
```

Two keys separated by a comma = rotation (`SERVICE_KEYS` is comma-separated).
Never add a version to `sous-chef-database-url`; Terraform owns it.

## 5. First image (push only — the jobs and service do not exist yet)

**Commit first.** Every command below tags the image with
`git rev-parse --short HEAD`; on a repo with no commits that prints nothing
and you end up with an invalid `…/sous-chef:` reference. So:

```sh
cd ../../..   # repo root
git add -A && git commit -m "sous-chef service v1"
export TAG=$(git rev-parse --short HEAD)
```

Either through Cloud Build in the devops project, with the same SA the
triggers will use (`_DEPLOY=false` skips the migrate/deploy steps):

```sh
gcloud builds submit --project prod-devops-core-project --region $REGION \
  --config cloudbuild.yaml \
  --substitutions=_PROJECT_ID=$PROJECT,_DEPLOY=false,_VERSION=$TAG .
```

or from this machine (Docker 29 is installed; the image is 18.9 MB):

```sh
gcloud auth configure-docker $REGION-docker.pkg.dev --quiet
make docker-build image-smoke IMAGE=sous-chef:$TAG      # migrates + serves it against compose Postgres, curls /healthz /readyz /version
make docker-push AR_REPO=$AR_REPO IMAGE=sous-chef:$TAG  # pushes $AR_REPO/sous-chef:$TAG
```

Either way: `export IMAGE=$AR_REPO/sous-chef:$TAG` (the same tag the push used).

## 6. Migrate job: create it with the image, run it

```sh
cd infra/envs/int
terraform apply -var=image=$IMAGE -target=module.sous_chef.google_cloud_run_v2_job.migrate
gcloud run jobs execute sous-chef-migrate --project $PROJECT --region $REGION --wait
```

Expected log line: `sous-chef: schema at version 5` (five embedded migrations
as of 2026-10-02). `max_retries = 0`: a failure stops here.

## 7. Everything else

```sh
terraform apply -var=image=$IMAGE
terraform output
```

Creates the service (ready only because steps 4 and 6 happened), the
retention job + scheduler and, with `public_invoker = true`, the `allUsers`
invoker binding. Keep `run_url` and `artifact_repo` from the output.

## 8. Verify the service

```sh
URL=$(terraform output -raw run_url)
curl -s $URL/readyz             # {"checks":{"db":"ok","default_route":"ok"},"status":"ready"}
# /healthz is NOT reachable from outside: Google's front end answers its own
# 404 page for that path on *.run.app (measured 2026-10-03). The container's
# startup probe still uses it internally; use /readyz or /version externally.
curl -s $URL/version            # {"build_id":"<sha>","go":"go1.24.x","routes":["default","opus"]}

KEY=$(cat /tmp/sous-chef-int.key)
curl -s -o /dev/null -w '%{http_code}\n' $URL/v1/manifests/x                                   # 401: no key
curl -s -w '\n%{http_code}\n' -X POST $URL/v1/manifests \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  --data-binary @../../../service/testdata/manifest.snapshot.json                                 # 201 {"hash":...,"fields":2094,"lists":69}; 200 if that hash was already registered
```

The snapshot is the launcher's manifest as of the last capture; the launcher
posts its own current one at boot (`POST /v1/manifests`, step 9), so a 201
here proves the path, not the hash the launcher will use.

If `public_invoker` were false (prod shape), every curl additionally needs
`-H "X-Serverless-Authorization: Bearer $(gcloud auth print-identity-token)"`
— `Authorization` already carries the service key, and Cloud Run honours the
ID token in that second header.

## 9. Cloud Build triggers (once) — after this, a push to `main` deploys itself

1. Push this repo to Bitbucket under `advesagroup` (it has no remote today)
   and mirror it into `prod-devops-core-project` the way `cookie-cutter` is
   (1st-gen connection; the mirrored repo appears as
   `bitbucket_advesagroup_<repo>`).
2. Two triggers, same shape as `int-pantry` / `prod-pantry` (region
   `northamerica-northeast2`, build config `cloudbuild.yaml`, default SA) —
   *shape*, create in the console or:
   ```sh
   gcloud builds triggers create cloud-source-repositories --project prod-devops-core-project --region $REGION \
     --name int-sous-chef --repo bitbucket_advesagroup_sous-chef --branch-pattern '^main$' \
     --build-config cloudbuild.yaml --substitutions _PROJECT_ID=breadstack-sous-chef-int
   gcloud builds triggers create cloud-source-repositories --project prod-devops-core-project --region $REGION \
     --name prod-sous-chef --repo bitbucket_advesagroup_sous-chef --tag-pattern '^v[0-9]+\.[0-9]+\.[0-9]+$' \
     --build-config cloudbuild.yaml --substitutions _PROJECT_ID=breadstack-sous-chef-prod
   ```
   The prod trigger MUST set `_PROJECT_ID`; the config refuses a tag build
   whose target ends in `-int`.
3. Slack needs nothing: the default SA already reads
   `pantry-cloudbuild-slack-bot-token` for the pantry build. Failures go to
   `#pantry-int-alert` (branch) / `#pantry-prod-alert` (tag); int success is
   silent, prod success posts.

What a `main` build does: `go vet` + `go test` → `docker build` with
`BUILD_ID=<sha>` → push `$AR_REPO/sous-chef:<sha>` → `gcloud run jobs update
sous-chef-migrate --image` + `execute --wait` → `gcloud run services update
sous-chef --image` + `jobs update sous-chef-retention --image` → check the
newest revision is serving and `/version` reports `<sha>`.

After the first CI deploy, `terraform plan` in `envs/int` must be empty. If
it is not, whatever gcloud stamped belongs in the `ignore_changes` lists in
`infra/modules/sous-chef/{run,job}.tf`.

## 10. Reachability: what the launcher needs

Measured facts behind this:

- The launcher is deployed by ArgoCD from `gitops/int/pantry/launcher.yaml`
  in Bitbucket `advesagroup/wordpress` (per cookie-cutter's `cloudbuild.yaml`).
  Which cluster that is was NOT confirmed: a read-only `kubectl get deploy -A`
  found no launcher on `int-breadstack` (`int-breadstack-services`) nor on
  `prd-breadstack`, so it most likely runs on the WordPress/ArgoCD cluster.
  Find the cluster, project and namespace in the gitops repo before
  "Wiring it" below.
- Whichever cluster it is, it is another project and VPC from
  `breadstack-sous-chef-int`, so `INGRESS_TRAFFIC_INTERNAL*` would not see
  its traffic as internal; only `INGRESS_TRAFFIC_ALL` gives a path
  (inference, not tested live).
- The launcher sends `Authorization: Bearer <service key>` and nothing else:
  no Google ID token (`launcher/server/souschef_proxy.go` `TODO(iam)`), no
  Google auth libraries, and it strips `X-Serverless-Authorization`.
- Cloud Run's IAM layer therefore answers **403** to every launcher call
  unless `allUsers` holds `roles/run.invoker` — which is what
  `public_invoker = true` grants.

**int (recommended, what the tfvars example does):** `INGRESS_TRAFFIC_ALL` +
`public_invoker = true`. The launcher needs exactly two values, nothing else:

| launcher env | value |
|---|---|
| `SOUS_CHEF_URL` | the `run_url` output, no trailing slash |
| `SOUS_CHEF_SERVICE_KEY` | the key from step 4 |

`SOUS_CHEF_JSON_FALLBACK` stays unset. Safety: every `/v1/*` route requires
the key (the smoke test above shows the 401), and the three unauthenticated
routes expose only health, readiness and the build id. This is the same
posture as the thermometer's public ingest URL.

**Wiring it** (read from the gitops repo `advesagroup/wordpress` on
2026-10-03; clone it with the `bitbucket-breadstack` SSH alias):

The launcher runs on GKE cluster `the-oven` (project
`prod-breadstack-services`, region `northamerica-northeast2`; int and prod
share the cluster, separated by namespace `pantry-int` / `pantry` and node
pool). `gitops/int/pantry/launcher.yaml` carries ONE plain env var (`PORT`);
every other launcher setting comes from `envFrom.secretRef: pantry-launcher-int`,
a Secret that External Secrets materialises from **Google Secret Manager
secret `pantry-launcher-int` in project `prod-breadstack-services`** — a
single JSON object whose top-level keys become env vars (`dataFrom.extract`).
ESO refreshes every 5 min and `reloader.stakater.com/auto` restarts the pods
when the Secret changes. No gitops change is needed; `launcher.yaml` has only
ever had its image tag edited.

1. Add the two keys to the JSON and publish a new version (a person has to
   run this: the agent classifier blocks writes to that secret):
   ```sh
   gcloud secrets versions access latest --secret pantry-launcher-int --project prod-breadstack-services > /tmp/pl.json
   python3 - <<'EOF'
   import json; p='/tmp/pl.json'; d=json.load(open(p))
   d['SOUS_CHEF_URL']='https://sous-chef-<hash>-<region>.a.run.app'   # run_url output, no trailing slash
   d['SOUS_CHEF_SERVICE_KEY']=open('/tmp/sous-chef-int.key').read().strip()
   json.dump(d, open(p,'w'), indent=2)
   EOF
   gcloud secrets versions add pantry-launcher-int --project prod-breadstack-services --data-file=/tmp/pl.json
   shred -u /tmp/pl.json
   ```
   Within ~5 min ESO rolls the Secret and reloader restarts `pantry-launcher`
   in `pantry-int`. (Prod later: the same on secret `pantry-launcher`.)
2. Nothing to commit in gitops. If you prefer `SOUS_CHEF_URL` visible in git,
   the OTel precedent (`gitops/int/pantry/site.yaml`, 2026-06-01) is a plain
   `env:` entry next to `PORT` — but the key stays in GSM either way.
3. Verify from the launcher's side: its boot log prints a `[Sous Chef]`
   registration line, `GET https://<launcher>/api/sous-chef/meta` answers
   `{"enabled":true,...}`, and on the service
   `gcloud run services logs read sous-chef --project $PROJECT --region $REGION --limit 20`
   shows `POST /v1/manifests 201`.

**prod needs more** (not built; `public_invoker = false` and
`INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER` in its example), one of:

- *IAM-gated over the public URL*: `ingress = INGRESS_TRAFFIC_ALL`, launcher
  code that mints an ID token (audience = `SOUS_CHEF_URL`) from the GKE
  metadata server via Workload Identity and sends it as
  `X-Serverless-Authorization`, plus `roles/run.invoker` for the launcher's
  identity — either a GSA bound to its KSA (`launcher_service_account`), or
  the KSA principal directly
  (`principal://iam.googleapis.com/projects/<gke project number>/locations/global/workloadIdentityPools/<project>.svc.id.goog/subject/ns/<ns>/sa/<ksa>`,
  which the module's `serviceAccount:`-only variable does not accept yet).
- *Network-gated*: an internal ALB with a serverless NEG in a VPC the
  launcher can reach (Shared VPC attach or PSC), keeping internal ingress;
  the module has no LB resources today (`infra/README.md` "Known gaps").

Until one is chosen, the prod example's `launcher_service_account` stays
empty.

## 11. Day two

- **Roll out**: push to `main`. Nothing else.
- **Roll back**: `gcloud run services update sous-chef --project $PROJECT --region $REGION --image $AR_REPO/sous-chef:<previous sha>` (or
  `update-traffic --to-revisions <rev>=100`). Migrations are forward-only;
  roll back the image only if the schema still fits it.
- **Infra change**: edit `infra/`, `terraform plan`, `terraform apply
  -var=image=$(gcloud run services describe sous-chef --project $PROJECT --region $REGION --format='value(spec.template.spec.containers[0].image)')`.
  The value only matters if the service is being (re)created; any pushed
  image works otherwise.
- **Re-mint a revision from Terraform** (a revision that never became ready):
  add `-var=deploy_nonce=$(date -u +%Y%m%dT%H%M%SZ)`.
- **Rotate the service key**: add a version to `sous-chef-service-keys` with
  `new,old`, roll (push or `deploy_nonce`), move the launcher to `new`, then
  a version with `new` alone and roll again.
- **Retention job**: nightly 03:00 UTC from Cloud Scheduler; run it by hand
  with `gcloud run jobs execute sous-chef-retention --project $PROJECT --region $REGION --wait`.

## 12. In-house model on int (Phase 5, `infra/modules/model-gpu`)

The trained Sous Chef (Qwen3.5-9B LoRA, exported to a q4_k_m GGUF) served by
Ollama from a private Cloud Run **L4** service in `us-east4` (Cloud Run has
no GPU region in Canada; the Toronto service calls it cross-region, as the
thermometer does), reachable only by `sous-chef-run` with a Google ID token.
The sous-chef service gets one extra route, `inhouse`; **`default_route`
stays `default`**, so nobody's turns move to the model unless the request
says `"model":"inhouse"` (the launcher drawer does that from
`?sousChefModel=inhouse`). Shape and numbers come from the thermometer's
Sourdough (`breadstack-thermometer/lab/sourdough/README.md`,
`infra/sourdough.tf`). Status (2026-10-04): the Terraform is written and
planned read-only; **nothing below has been applied** (no `sous-chef-model`
repository, SA or service exists in int yet).

What the tfvars mean (`infra/envs/int/terraform.tfvars`; the example carries
the same lines, both halves off):

| tfvar | off | on | effect |
|---|---|---|---|
| `gpu_model_enabled` | `false` | `true` | **half 1**: Artifact Registry `sous-chef-model` (us-east4, immutable tags) + roleless SA `sous-chef-model-run`. Nothing else changes: the sous-chef service is not rolled |
| `gpu_model_serve` | `false` | `true` | **half 2**: the Cloud Run GPU service `sous-chef-model` (1 x L4, 4 vCPU / 16Gi, max 1 instance, 300 s timeout, ingress all, authoritative `run.invoker` = `sous-chef-run`) **and** a new sous-chef revision with `MODEL_ROUTES` (+`inhouse`), `MODEL_ROUTE_ALLOWLIST=default,opus,inhouse`, `MODEL_CALL_TIMEOUT=180s` |
| `gpu_model_image_tag` | `"r1"` | | the tag the service runs; a new export = a new tag (tags are immutable) |
| `gpu_model_min_instances` | `0` | `1` | asleep (scale to zero) / awake (model loaded, billed every second) |
| `gpu_model_route` | default | | `{name="inhouse", model="sous-chef", max_tokens=4096, call_timeout="180s"}`; the url is the module output, auth is `id-token` |

Plans measured 2026-10-04 (`terraform plan -lock=false`, read-only): both
flags off → `No changes`; `-var=gpu_model_enabled=true` → exactly 2 creates
(repository, SA) and the `gpu_model_repo` output; both on → 4 creates
(repository, SA, service, invoker binding) + 1 in-place update of
`sous-chef` (the three env vars above; `MODEL_ROUTES` "known after apply"
because it carries the new service's URL).

### Half 1: repository + SA, then the image

```sh
cd infra/envs/int
sed -i 's/^gpu_model_enabled .*/gpu_model_enabled       = true/' terraform.tfvars
terraform plan  -var=image=$IMAGE          # 2 to add, 0 to change
terraform apply -var=image=$IMAGE          # a person: repository + SA creation
terraform output gpu_model_repo            # us-east4-docker.pkg.dev/breadstack-sous-chef-int/sous-chef-model
```

Build the image on the workstation (CPU only, no GPU involved; the lab
Ollama on :11435 is never touched) from the export's q4_k_m GGUF and push
it — `make -C lab model-stage model-build NAME=<run> QUANT=q4_k_m MODEL_TAG=r1`,
then the commands `make -C lab model-push` prints (`gcloud auth
configure-docker us-east4-docker.pkg.dev`, `docker push`). About 9.3 GB goes
up; the repository stores it for ~US$0.90/month per tag. The Ollama model
inside the image is named `sous-chef` (`lab/model-image/Dockerfile`), which
is what the route's `model` says.

### Half 2: the GPU service and the route

```sh
sed -i 's/^gpu_model_serve .*/gpu_model_serve         = true/' terraform.tfvars
terraform plan  -var=image=$IMAGE          # 4 to add, 1 to change (sous-chef env)
terraform apply -var=image=$IMAGE
terraform output gpu_model_url
```

If the apply fails on the GPU service with a quota error, int's Cloud Run L4
quota (`NvidiaL4GpuAllocNoZonalRedundancyPerProjectRegion`, us-east4) is 0:
request it in the console (the thermometer's went through as 3; its README
section "Quota" has the form), or move `gpu_model_region = "us-central1"`
and re-plan (the repository stays in us-east4; Cloud Run imports the image).
The service starts with `gpu_model_min_instances` from the file: `0` means
it scales to zero straight away and costs nothing until the first request.

Verify, read-only:

```sh
URL=$(terraform output -raw run_url)
curl -s $URL/version                       # routes now ["default","inhouse","opus"]
curl -s $URL/readyz                        # default_route still "ok" (Claude)
gcloud run services describe sous-chef-model --project $PROJECT --region us-east4 \
  --format='value(status.url,spec.template.spec.containers[0].image)'
```

A direct smoke test of the model service with your own login (Owner holds
`run.routes.invoke`; this wakes the GPU for ~11 billed minutes):

```sh
MODEL_URL=$(terraform output -raw gpu_model_url)
curl -s -H "Authorization: Bearer $(gcloud auth print-identity-token)" $MODEL_URL/api/ps
gcloud run services logs read sous-chef-model --project $PROJECT --region us-east4 --limit 20   # "sous-chef loaded in N s", CUDA not library=cpu
```

Then one turn through the launcher with `?sousChefModel=inhouse` on the int
pantry (the launcher's half: cookie-cutter `launcher/client/src/sousChef/useSousChef.js`,
deployed to `pantry-int`; its `SOUS_CHEF_URL`/key are the section 10 wiring).
A 403 class in the model service's logs means the ID token's audience or the
invoker binding is wrong; a `400 bad_request "model: …"` from sous-chef
means `inhouse` is not in the allowlist the revision runs with.

### Warm and asleep (the only cost switch)

| state | tfvar | what it costs (us-east4 list, 2026-09, from the thermometer) |
|---|---|---|
| asleep | `gpu_model_min_instances = 0` | $0 idle. First request after a quiet spell wakes the GPU: ~1 min, first full answer 45–75 s; the instance then idles **10 min** after the last request, so one wake is about **US$0.19**, a 30-min demo ~$0.50–0.70 |
| awake | `gpu_model_min_instances = 1` | model loaded, billed every second: **~US$1.05/hour, ~$25/day, ~$764/month** |

Flip it with a targeted apply of the model service alone (nothing else is
planned, and the sous-chef service is not rolled):

```sh
# warm, ~15 min before a demo
terraform apply -var=image=$IMAGE -var=gpu_model_min_instances=1 \
  -target='module.model_gpu[0].google_cloud_run_v2_service.model[0]'
# sleep, right after
terraform apply -var=image=$IMAGE -var=gpu_model_min_instances=0 \
  -target='module.model_gpu[0].google_cloud_run_v2_service.model[0]'
```

Keep the file's value equal to the state you want to persist: every later
full apply passes the file's `gpu_model_min_instances`, so a model warmed by
`-var` goes back to sleep on the next plain apply (the thermometer reads the
live value first — `make sourdough-warm/-sleep`; sous-chef has no Makefile
for this yet, so read the plan). Per-answer cost is in the GPU time only:
Ollama answers are **$0.00** on the Anthropic bill. Leaving it awake over a
weekend is ~$75.

### Turning it off

`gpu_model_serve = false` + apply: removes the service and the binding, and
the sous-chef revision goes back to `MODEL_CALL_TIMEOUT=60s`, the default
route table and `default,opus` (a turn that still says `inhouse` gets `400
bad_request`). `gpu_model_enabled = false` + apply then removes the SA and
the repository **with its images** (push again after). There is no reason to
retire the route while the service stays: with `default_route = "default"`
an unused route costs nothing beyond the sleeping service.

### Rolling a new model

Export → `make -C lab model-stage model-build MODEL_TAG=r2` → push →
`gpu_model_image_tag = "r2"` → apply (one new revision of the model
service; the route is unchanged). Immutable tags mean `r1` can never be
overwritten; roll back by setting the tag back.
