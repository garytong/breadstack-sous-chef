# sous-chef infrastructure (Terraform)

Google Cloud, one project per environment. Shapes copied from
`breadstack-thermometer/infra` (Cloud Run v2, Cloud SQL, Secret Manager,
Cloud Run jobs, the Sourdough GPU service), adapted to the sous-chef
contract in the root `README.md` (section 9).

Never `terraform apply` from automation or from an agent. A person runs
every apply from their own login.

## Layout

```
infra/
  modules/sous-chef/     the service: apis, registry, sql, secrets, iam, run, job, outputs
  modules/model-gpu/     Phase 5 stub: private Ollama GPU service, IAM-only (behind a flag)
  envs/int/              root module for int   (backend prefix sous-chef/int)
  envs/prod/             root module for prod  (backend prefix sous-chef/prod)
  envs/*/terraform.tfvars.example   copy to terraform.tfvars (gitignored) and fill in
```

### What the module creates

| Resource | Name | Notes |
|---|---|---|
| Cloud Run v2 service | `sous-chef` | CPU only, min 0 / max 4, concurrency 20, timeout 600 s (SSE), port 8090, probe `/healthz`, ingress internal + load balancer (int example uses ALL; IAM still gates) |
| Cloud Run v2 job | `sous-chef-migrate` | image with `-migrate`, `max_retries = 0`, run by the deploy before the service rolls |
| Cloud Run v2 job | `sous-chef-retention` | image with `-retention`; Cloud Scheduler `sous-chef-retention` at 03:00 UTC daily |
| Cloud SQL | `sous-chef` | POSTGRES_16, `db-f1-micro` (int) / `db-custom-1-3840` (prod), PD-SSD, backups + PITR, deletion protection; `ssl_mode = ENCRYPTED_ONLY` always; **prod example: private IP only** (`sql_private_ip = true` + `sql_private_network`/`sql_private_subnetwork`, Service Networking peering + direct VPC egress); **int example: public IPv4 + SSL**, no authorized networks, reached only via the Cloud SQL connector volume |
| Cloud SQL db / user | `sous_chef` / `sous_chef` | password from `random_password` |
| Secret Manager | `sous-chef-database-url` | value owned by Terraform (pgx DSN over `/cloudsql/<conn>`) |
| Secret Manager | `sous-chef-anthropic-api-key`, `sous-chef-service-keys`, `sous-chef-thermometer-api-key` | created EMPTY; see "Secrets to populate" |
| Artifact Registry | `sous-chef` | Docker, in `region` |
| Service account | `sous-chef-run` | `roles/cloudsql.client`; `roles/secretmanager.secretAccessor` on each secret above |
| Service account | `sous-chef-scheduler` | `roles/run.invoker` on the retention job only |
| IAM | launcher SA → `roles/run.invoker` on the service | `launcher_service_account` tfvar; empty = no binding yet (placeholder) |
| IAM | `allUsers` → `roles/run.invoker` on the service | `public_invoker` tfvar (int example: true; prod: false). The launcher sends no ID token, so without it Cloud Run answers 403; the service key stays the gate |
| IAM | Cloud Build SA → `artifactregistry.writer` (repo), `run.developer` + `serviceusage.serviceUsageConsumer` (project), `iam.serviceAccountUser` on `sous-chef-run` | `ci_service_account` tfvar; what `cloudbuild.yaml` needs to push, migrate and roll |

Env on the service (README section 7): `ENV`, `MIGRATE_AT_BOOT=off`,
`TRACE_RETENTION_DAYS`, `DEPLOY_NONCE`, `MODEL_ROUTES`, `DEFAULT_ROUTE`,
`MODEL_ROUTE_ALLOWLIST`, `MODEL_CALL_TIMEOUT`, `TURN_TTL`,
`MAX_CONCURRENT_TURNS`, `THERMOMETER_API_URL`; secret-backed `DATABASE_URL`,
`SERVICE_KEYS`, `ANTHROPIC_API_KEY`, and `THERMOMETER_API_KEY` only when
`thermometer_api_url` is set (two-halves gate: Cloud Run refuses to start a
revision that references a secret with no version). `PORT` is injected by
Cloud Run from `container_port`.

## Bootstrap (once per project, a person's login)

1. Create the project and link billing.
2. Create the state bucket (backends cannot interpolate variables):
   ```
   gcloud storage buckets create gs://<project>-sous-chef-tfstate \
     --project <project> --location northamerica-northeast2 \
     --uniform-bucket-level-access
   gcloud storage buckets update gs://<project>-sous-chef-tfstate --versioning
   ```
   **State-bucket IAM.** The state holds the Cloud SQL password
   (`random_password.db`) and the `sous-chef-database-url` secret payload in
   plaintext, so treat the bucket as a secret store:
   - Do not grant any project-wide `roles/storage.*` role that reaches it;
     uniform bucket-level access (set above) disables per-object ACLs, so
     IAM on the bucket is the only path in.
   - Grant `roles/storage.objectAdmin` on the bucket only to the people who
     run `terraform apply` (their own Google logins). Nobody else — no CI
     service account, no agent, no deployer SA — gets read on it:
     ```
     gcloud storage buckets add-iam-policy-binding gs://<project>-sous-chef-tfstate \
       --member user:<person>@breadstack.com --role roles/storage.objectAdmin
     ```
     `objectAdmin` (not `objectCreator`/`objectViewer`) because the gcs
     backend needs create, read, overwrite and delete of the `.tflock` object.
   - Leave `roles/storage.admin` to the project Owner for bucket-level
     changes (retention, versioning); it is not needed day to day.
   - Enforce public-access prevention on the bucket (unless an org policy
     already does):
     `gcloud storage buckets update gs://<project>-sous-chef-tfstate --public-access-prevention`.
   - Audit who holds it with
     `gcloud storage buckets get-iam-policy gs://<project>-sous-chef-tfstate`
     whenever a person leaves the apply rotation.
3. `cd infra/envs/<env>` and
   `terraform init -backend-config="bucket=<project>-sous-chef-tfstate"`.
4. `cp terraform.tfvars.example terraform.tfvars` and set `project_id`
   (and `launcher_service_account` once the launcher's SA is known).

## Apply order

The first apply of a fresh project fails on purpose at the Run service
unless the two user secrets already have a version, so the order is:

1. **Foundations** (APIs, registry, SQL, secrets, IAM, jobs) — a targeted apply
   that leaves the service out:
   ```
   terraform apply -var=image=<any> \
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
   (Cloud SQL takes 5–10 minutes. The four `ci_*` targets are no-ops when
   `ci_service_account` is empty.)
2. **Populate the secrets** (below).
3. **Build and push the image** to the `artifact_repo` output
   (`<region>-docker.pkg.dev/<project>/sous-chef/sous-chef:<tag>`): a Cloud
   Build run with `_DEPLOY=false`, or `make docker-build docker-push
   AR_REPO=<artifact_repo>` — see "Deploy".
4. **Migrate**: `terraform apply -var=image=<uri> -target=module.sous_chef.google_cloud_run_v2_job.migrate`
   then `gcloud run jobs execute sous-chef-migrate --region <region> --wait`.
5. **Everything**: `terraform apply -var=image=<uri>` — the service, the
   retention job, the scheduler, the invoker bindings.
6. Give the launcher the `run_url` output and one of the `SERVICE_KEYS`
   values; set `launcher_service_account` and apply again if it was empty.

Every later roll-out is a push to `main` (int) or a `v*` tag (prod): Cloud
Build does 3 → 4 → 5 with `gcloud run` (see "Deploy"). Terraform is only
re-applied for infrastructure changes. A revision that never became ready
(pending migration, version-less secret) is already in state, so re-applying
the same template is a no-op: pass `-var=deploy_nonce=$(date +%s)` to mint a
fresh revision.

## Deploy

`cloudbuild.yaml` at the repo root, triggered like cookie-cutter's pantry
pipeline from `prod-devops-core-project` (push to `main` → int, tag
`v[0-9]+.[0-9]+.[0-9]+` → prod with `_PROJECT_ID=breadstack-sous-chef-prod`
on the trigger). Steps: `go vet`/`go test` → `docker build
-f service/Dockerfile --build-arg BUILD_ID=<sha|tag>` → push to
`artifact_repo` → `gcloud run jobs update sous-chef-migrate --image` +
`execute --wait` → `gcloud run services update sous-chef --image` + `jobs
update sous-chef-retention --image` → assert the newest revision is serving
and `/version` reports the build id. Slack alerts reuse pantry's bot token
(fetched inside the build, so a missing secret only silences them).
`_DEPLOY=false` builds and pushes without touching Run (the first image,
before the jobs and service exist). The ordered first-time commands are in
`docs/DEPLOY-INT.md`.

**Who owns the image.** Terraform creates the service and both jobs with
`var.image`, then stops caring: `run.tf` and `job.tf` carry
`lifecycle { ignore_changes = [<container image>, template[0].labels,
template[0].annotations, client, client_version] }`. Without it the next
`terraform apply -var=image=<older>` would revert what CI deployed, and
every CI deploy would show as drift (gcloud also stamps a
`client.knative.dev/nonce` template label and its client name/version).
Consequences:

- `var.image` matters only when a resource is created or recreated; pass any
  pushed image on later applies.
- After the first CI deploy, `terraform plan` must be empty. Anything else
  gcloud stamps belongs in the ignore lists.
- A Terraform-driven roll-out (the thermometer's model) is still possible
  with `-var=deploy_nonce=...`, but the image stays whatever CI last set.

The alternative — CI builds and pushes only, Terraform rolls the image — was
rejected because it keeps a manual `terraform apply` in every deploy;
Terraform owns every other field of the template either way.

**CI identity.** `ci_service_account` grants the Cloud Build SA
`artifactregistry.writer` on the repository, `run.developer` and
`serviceusage.serviceUsageConsumer` on the project, and
`iam.serviceAccountUser` on `sous-chef-run` (to deploy a revision that runs
as it). `run.developer` rather than `run.admin`: CI cannot change IAM on the
service. The state bucket is never granted to it.

**Invoker.** `public_invoker = true` (int example) adds `allUsers` →
`roles/run.invoker`: the launcher sends only the service key, so Cloud Run's
own IAM check would otherwise 403 every call. prod keeps it false; what prod
needs instead is in `docs/DEPLOY-INT.md` section 10.

## Secrets to populate (by hand, before step 5)

Secret Manager rejects empty payloads, so Terraform creates these with no
version. Add one with `gcloud secrets versions add <name> --data-file=-`:

| Secret | Env var | Value |
|---|---|---|
| `sous-chef-anthropic-api-key` | `ANTHROPIC_API_KEY` | an Anthropic API key (Claude is the day-one model) |
| `sous-chef-service-keys` | `SERVICE_KEYS` | comma-separated random keys, e.g. `openssl rand -hex 32`; two live keys = rotation. The launcher sends one as `Authorization: Bearer <key>` |
| `sous-chef-thermometer-api-key` | `THERMOMETER_API_KEY` | only if `thermometer_api_url` is set: a `tm_` ingest key from the thermometer |

`sous-chef-database-url` is written by Terraform; never edit it by hand (a
re-apply would overwrite it, and the password lives in state).

Both int and prod need their own values; the Anthropic key may be shared,
the service keys must not be.

## Destroy

```
terraform apply -var=image=<uri> -var=sql_deletion_protection=false \
  -target=module.sous_chef.google_sql_database_instance.sous_chef
terraform destroy -var=image=<uri>
```

`disable_on_destroy = false` on the APIs: a destroy never switches a project
service off.

## Phase 5: the GPU model stub (`modules/model-gpu`)

Off by default (`gpu_model_enabled = false`: nothing is created). The two
halves, as for the thermometer's Sourdough:

1. `gpu_model_enabled = true` → the private repository `sous-chef-model`
   (us-east4, immutable tags) and the roleless SA `sous-chef-model-run`.
   Needs a person's login (repository + SA creation).
2. Push the Ollama + weights image as `<gpu_model_repo output>/model:<tag>`,
   then `gpu_model_serve = true` → the Cloud Run GPU service
   `sous-chef-model` (one L4, 4 vCPU / 16Gi, max 1 instance,
   `gpu_model_min_instances` 0 or 1) with an **authoritative**
   `roles/run.invoker` binding for `sous-chef-run` alone. No shared secret:
   sous-chef sends a Google ID token with the service URL as audience
   (`auth: id-token` route).
3. Add a route to `model_routes`, e.g.
   `{"inhouse":{"kind":"messages-compatible","url":"<gpu_model_url>","auth":"id-token","model":"sous-chef","max_tokens":4096,"stream":false}}`,
   and extend `model_route_allowlist`.

Cost reference from the thermometer (us-east4 list price, 2026-09): about
US$1.05/hour while an instance runs, billed until ~10 minutes after the last
request; `min_instances = 1` is ~$25/day.

## Verified on this machine (2026-10-01; re-run 2026-10-02 after the Deploy/CI changes)

- `terraform fmt -recursive -check` over `infra/`: clean.
- `terraform init -backend=false && terraform validate` in `envs/int` and
  `envs/prod`: `Success! The configuration is valid.` Both modules
  (`sous_chef`, `model_gpu`) were loaded; providers resolved to
  `hashicorp/google 6.50.0`, `hashicorp/random 3.9.1`.
- Terraform v1.16.3. No plan or apply has been run; no GCP project exists yet.

## Known gaps

- No load balancer resources: `INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER` in
  prod assumes the launcher reaches the service through an internal LB or
  from the same VPC/project. If the launcher stays in another project without
  an LB, use `INGRESS_TRAFFIC_ALL` (as int's example does) — IAM
  (`launcher_service_account`) and the service key still gate every call.
- No deployer service account / CI key for Terraform (the thermometer's
  `deployer.tf`); a person applies. CI (`ci_service_account`) can only push
  images and roll Run, not apply.
- `launcher_service_account` only accepts `serviceAccount:` members; a GKE
  Workload Identity KSA principal (`principal://...`) would need a second
  variable (docs/DEPLOY-INT.md section 10).
- `run_max_instances` defaults to 4 per the task; root `README.md` section 9
  says 10 — the task text is the later decision.
