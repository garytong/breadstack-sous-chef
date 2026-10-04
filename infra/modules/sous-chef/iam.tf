# Service accounts and roles. Shape copied from
# breadstack-thermometer/infra/iam.tf.
#
# sous-chef-run        the Run service and both jobs: Cloud SQL client +
#                      secretAccessor on each secret it reads, nothing else.
# sous-chef-scheduler  Cloud Scheduler's identity for the nightly retention
#                      job: run.invoker on that job alone.
# launcher SA          (external, placeholder variable) run.invoker on the
#                      service: the pantry launcher's reverse proxy is the
#                      only caller; the service key is the second barrier.

resource "google_service_account" "run" {
  project      = var.project_id
  account_id   = "sous-chef-run"
  display_name = "sous-chef Cloud Run runtime (${var.env})"

  depends_on = [google_project_service.services]
}

resource "google_project_iam_member" "run_cloudsql" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.run.email}"
}

resource "google_secret_manager_secret_iam_member" "run_user_secrets" {
  for_each  = google_secret_manager_secret.user
  project   = var.project_id
  secret_id = each.value.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.run.email}"
}

resource "google_secret_manager_secret_iam_member" "run_database_url" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.database_url.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.run.email}"
}

# The launcher may invoke the service. No allUsers: unlike the thermometer's
# public ingest URL, nothing but the launcher's proxy should ever reach this.
resource "google_cloud_run_v2_service_iam_member" "launcher_invoker" {
  count = var.launcher_service_account != "" ? 1 : 0

  project  = var.project_id
  name     = google_cloud_run_v2_service.sous_chef.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "serviceAccount:${var.launcher_service_account}"
}

# Cloud Scheduler -> retention job.
resource "google_service_account" "scheduler" {
  project      = var.project_id
  account_id   = "sous-chef-scheduler"
  display_name = "sous-chef Cloud Scheduler trigger for the retention job (${var.env})"

  depends_on = [google_project_service.services]
}

resource "google_cloud_run_v2_job_iam_member" "scheduler_runs_retention" {
  project  = var.project_id
  name     = google_cloud_run_v2_job.retention.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.scheduler.email}"
}

# Cloud Build (cloudbuild.yaml) pushes the image, re-points the jobs, executes
# the migrate job and rolls the service. The trigger lives in another project
# (prod-devops-core-project, like pantry's), so its default SA is granted here.
# run.developer rather than run.admin: it cannot change IAM on the service.
resource "google_artifact_registry_repository_iam_member" "ci_writer" {
  count = var.ci_service_account != "" ? 1 : 0

  project    = var.project_id
  location   = var.region
  repository = google_artifact_registry_repository.sous_chef.repository_id
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${var.ci_service_account}"
}

resource "google_project_iam_member" "ci_run_developer" {
  count = var.ci_service_account != "" ? 1 : 0

  project = var.project_id
  role    = "roles/run.developer"
  member  = "serviceAccount:${var.ci_service_account}"
}

# Calling the Run API with --project <this project> from a build in another
# project: the thermometer's deployer needed this role for the same reason.
resource "google_project_iam_member" "ci_service_usage" {
  count = var.ci_service_account != "" ? 1 : 0

  project = var.project_id
  role    = "roles/serviceusage.serviceUsageConsumer"
  member  = "serviceAccount:${var.ci_service_account}"
}

resource "google_service_account_iam_member" "ci_acts_as_run" {
  count = var.ci_service_account != "" ? 1 : 0

  service_account_id = google_service_account.run.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${var.ci_service_account}"
}
