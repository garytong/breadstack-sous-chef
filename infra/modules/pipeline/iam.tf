# Service accounts and grants (DESIGN §5). Resource-scoped wherever GCP has
# resource-level IAM; project-level only for Vertex custom jobs, Cloud Build
# builds, Workflows invocation and log writing, which have none.
#
#   souschef-workflow   both workflows: runs and re-points the jobs, patches
#                       the eval / candidate / production model services'
#                       image, starts Vertex jobs and Cloud Builds as their
#                       own accounts, reads the thermometer key, the bucket
#   souschef-drift      jobs souschef-drift, souschef-eval: the service key,
#                       invoker on the three GPU services, the bucket
#   souschef-author     job souschef-prepare: the service key and the
#                       Anthropic key (case authoring), the bucket
#   souschef-prep       jobs souschef-collect / -dataset: the service key,
#                       the bucket
#   souschef-trainer    Vertex custom jobs: write runs/<id>/train/ only, read
#                       runs/ and base/ — nothing else, no secret
#   souschef-builder    Cloud Build: push to the model repository, read runs/
#   souschef-scheduler  Cloud Scheduler: start a workflow execution
#
# The jobs never hold the thermometer key; only the author account holds the
# Anthropic key; the trainer and builder hold no secret at all.
#
# The bucket (IAM conditions, uniform bucket-level access): the workflow
# account alone may write infra/ (what the workflows act on: the thermometer
# URL, the switches, the model repository, the build spec — Terraform writes
# it as the person applying) and config/ (the thresholds override, by hand).
# The job accounts read everything and write everything else; the trainer,
# which runs the most third-party code, writes only runs/*/train/.

locals {
  service_accounts = {
    workflow  = { id = "souschef-workflow", name = "Sous Chef pipeline workflows (drift, retrain)" }
    drift     = { id = "souschef-drift", name = "Sous Chef pipeline drift and eval jobs" }
    author    = { id = "souschef-author", name = "Sous Chef pipeline prepare job (case authoring)" }
    prep      = { id = "souschef-prep", name = "Sous Chef pipeline collect / dataset jobs" }
    trainer   = { id = "souschef-trainer", name = "Sous Chef pipeline Vertex training (bucket only)" }
    builder   = { id = "souschef-builder", name = "Sous Chef pipeline model image Cloud Build" }
    scheduler = { id = "souschef-scheduler", name = "Sous Chef pipeline Cloud Scheduler trigger" }
  }

  # Emails from the configured account ids (known at plan time, and still a
  # dependency on the accounts).
  sa_email = { for k, v in google_service_account.sa : k => "${v.account_id}@${var.project_id}.iam.gserviceaccount.com" }
  member   = { for k, e in local.sa_email : k => "serviceAccount:${e}" }

  # Accounts the workflow deploys or launches work as (actAs).
  acts_as = { for k in ["trainer", "builder", "drift", "prep", "author"] : k => "projects/${var.project_id}/serviceAccounts/${local.sa_email[k]}" }

  # IAM condition expressions over the pipeline bucket's objects.
  obj = "projects/_/buckets/${local.bucket_name}/objects/"
  bucket_conditions = {
    # every object but infra/, config/ and wf/ (and the bucket itself: listing)
    jobs_write = "!resource.name.startsWith(\"${local.obj}infra/\") && !resource.name.startsWith(\"${local.obj}config/\") && !resource.name.startsWith(\"${local.obj}wf/\")"
    # runs/<id>/train/... only
    trainer_write = "resource.name.extract(\"${local.obj}runs/{run}/train/\") != \"\""
    # runs/ and base/, and the bucket (listing)
    trainer_read = "resource.name.startsWith(\"${local.obj}runs/\") || resource.name.startsWith(\"${local.obj}base/\") || resource.name == \"projects/_/buckets/${local.bucket_name}\""
    # runs/ (the build context and the staged model), and the bucket (listing)
    builder_read = "resource.name.startsWith(\"${local.obj}runs/\") || resource.name == \"projects/_/buckets/${local.bucket_name}\""
  }
}

resource "google_service_account" "sa" {
  for_each     = local.service_accounts
  project      = var.project_id
  account_id   = each.value.id
  display_name = each.value.name
}

# --- project-level (no resource-level IAM exists for these) ---------------

resource "google_project_iam_member" "workflow" {
  for_each = toset([
    "roles/aiplatform.user",          # Vertex custom jobs
    "roles/cloudbuild.builds.editor", # the model image build
    "roles/workflows.invoker",        # drift starts a retrain execution
    "roles/logging.logWriter",
  ])
  project = var.project_id
  role    = each.value
  member  = local.member["workflow"]
}

resource "google_project_iam_member" "log_writer" {
  for_each = toset(["drift", "prep", "author", "trainer", "builder"])
  project  = var.project_id
  role     = "roles/logging.logWriter"
  member   = local.member[each.value]
}

resource "google_project_iam_member" "scheduler_invoker" {
  project = var.project_id
  role    = "roles/workflows.invoker"
  member  = local.member["scheduler"]
}

# --- actAs: the workflow starts work as these accounts --------------------

resource "google_service_account_iam_member" "workflow_acts_as" {
  for_each           = local.acts_as
  service_account_id = each.value
  role               = "roles/iam.serviceAccountUser"
  member             = local.member["workflow"]
}

# The GPU services (eval, candidate, production) run as sous-chef-model-run;
# patching their image deploys a revision as it.
resource "google_service_account_iam_member" "workflow_acts_as_model" {
  service_account_id = "projects/${var.project_id}/serviceAccounts/${var.model_service_account}"
  role               = "roles/iam.serviceAccountUser"
  member             = local.member["workflow"]
}

# --- the bucket -------------------------------------------------------------

# The workflow account: the whole bucket (it is the orchestrator, and the only
# account besides Terraform's that may change infra/).
resource "google_storage_bucket_iam_member" "admin" {
  bucket = google_storage_bucket.pipeline.name
  role   = "roles/storage.objectAdmin"
  member = local.member["workflow"]
}

# The job accounts: read everything; write all but infra/, config/ and wf/
# (the workflow's own records: its outbox, which it posts with the pipeline
# key, and each round's decide.json and rollout.json).
resource "google_storage_bucket_iam_member" "jobs_read" {
  for_each = toset(["drift", "prep", "author"])
  bucket   = google_storage_bucket.pipeline.name
  role     = "roles/storage.objectViewer"
  member   = local.member[each.value]
}

resource "google_storage_bucket_iam_member" "jobs_write" {
  for_each = toset(["drift", "prep", "author"])
  bucket   = google_storage_bucket.pipeline.name
  role     = "roles/storage.objectAdmin"
  member   = local.member[each.value]

  condition {
    title       = "not-infra-config-or-wf"
    description = "Every object but infra/ (the workflows' configuration and build spec), config/ (the thresholds override) and wf/ (the workflow's outbox and decision records)"
    expression  = local.bucket_conditions.jobs_write
  }
}

# The trainer: read runs/ and base/, write runs/<id>/train/ only.
resource "google_storage_bucket_iam_member" "trainer_read" {
  bucket = google_storage_bucket.pipeline.name
  role   = "roles/storage.objectViewer"
  member = local.member["trainer"]

  condition {
    title       = "runs-and-base"
    description = "runs/ (the dataset, checkpoints) and base/ (the base weights), and listing"
    expression  = local.bucket_conditions.trainer_read
  }
}

resource "google_storage_bucket_iam_member" "trainer_write" {
  bucket = google_storage_bucket.pipeline.name
  role   = "roles/storage.objectAdmin"
  member = local.member["trainer"]

  condition {
    title       = "train-outputs"
    description = "runs/<run id>/train/ only (progress, checkpoints, adapter, export, stage, DONE)"
    expression  = local.bucket_conditions.trainer_write
  }
}

resource "google_storage_bucket_iam_member" "builder_viewer" {
  bucket = google_storage_bucket.pipeline.name
  role   = "roles/storage.objectViewer"
  member = local.member["builder"]

  condition {
    title       = "runs"
    description = "runs/ (the build context and the staged model), and listing"
    expression  = local.bucket_conditions.builder_read
  }
}

# --- secrets ------------------------------------------------------------------

resource "google_secret_manager_secret_iam_member" "service_key" {
  for_each  = toset(["drift", "prep", "author"])
  project   = var.project_id
  secret_id = google_secret_manager_secret.pipeline["service_key"].secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = local.member[each.value]
}

resource "google_secret_manager_secret_iam_member" "thermometer_key" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.pipeline["thermometer"].secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = local.member["workflow"]
}

# An added member on the existing secret (modules/sous-chef owns it and its
# other member, sous-chef-run): the author account only (the prepare job), not
# collect or dataset.
resource "google_secret_manager_secret_iam_member" "anthropic" {
  project   = var.project_id
  secret_id = var.anthropic_secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = local.member["author"]
}

# --- Cloud Run: jobs ----------------------------------------------------------

# run.developer: re-point a job's image to the run's lab image (jobs.patch);
# jobsExecutorWithOverrides: run it with per-stage args and env.
resource "google_cloud_run_v2_job_iam_member" "workflow" {
  for_each = {
    for pair in setproduct(keys(local.job_names), ["roles/run.developer", "roles/run.jobsExecutorWithOverrides"]) :
    "${pair[0]}:${pair[1]}" => { job = pair[0], role = pair[1] }
  }
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_job.job[each.value.job].name
  role     = each.value.role
  member   = local.member["workflow"]
}

# --- Cloud Run: services ------------------------------------------------------

# The int Sous Chef service: read its image tag (the lab and trainer images
# are resolved from it per run, DESIGN §6.3).
resource "google_cloud_run_v2_service_iam_member" "workflow_reads_int" {
  project  = var.project_id
  location = var.region
  name     = var.int_service_name
  role     = "roles/run.viewer"
  member   = local.member["workflow"]
}

# The GPU services: patch the image (+ the souschef/candidate annotation).
# IAM cannot narrow run.developer to the image field: the workflow is the
# only code path that patches, and its patch changes the image and the
# annotation only. On the production service this is the rollout.
resource "google_cloud_run_v2_service_iam_member" "workflow_develops_gpu" {
  for_each = {
    eval = google_cloud_run_v2_service.gpu["eval"].name
    cand = google_cloud_run_v2_service.gpu["cand"].name
    prod = var.prod_model_service_name
  }
  project  = var.project_id
  location = var.gpu_region
  name     = each.value
  role     = "roles/run.developer"
  member   = local.member["workflow"]
}

# Invoker on the eval and candidate services: authoritative, the pipeline's
# runner accounts only (the production service's binding is modules/
# model-gpu's, which takes these two through extra_invokers).
resource "google_cloud_run_v2_service_iam_binding" "gpu_invoker" {
  for_each = google_cloud_run_v2_service.gpu
  project  = var.project_id
  location = var.gpu_region
  name     = each.value.name
  role     = "roles/run.invoker"
  members  = [local.member["drift"], local.member["workflow"]]
}

# --- Artifact Registry --------------------------------------------------------

# The workflow reads tags in the service repository (sous-chef-lab:<tag>,
# sous-chef-trainer:<tag> must exist before a run starts).
resource "google_artifact_registry_repository_iam_member" "workflow_reads_lab" {
  project    = var.project_id
  location   = var.region
  repository = var.lab_repository_id
  role       = "roles/artifactregistry.reader"
  member     = local.member["workflow"]
}

# The model repository: Cloud Build pushes model:c-<run id>; the workflow
# reads tags (the production image's digest) and adds live-<run id> to an
# image it rolls out (the cleanup policy's keep rule, modules/model-gpu).
resource "google_artifact_registry_repository_iam_member" "model_writer" {
  for_each   = toset(["builder", "workflow"])
  project    = var.project_id
  location   = var.model_repo_location
  repository = var.model_repository_id
  role       = "roles/artifactregistry.writer"
  member     = local.member[each.value]
}
