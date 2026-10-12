# The APIs the pipeline adds (measured 2026-10-10: none of the first four is
# on in int). run, secretmanager, artifactregistry, cloudscheduler and iam
# are modules/sous-chef's; storage and monitoring are on already and listed
# so a fresh project works. disable_on_destroy = false: a destroy of the
# pipeline never switches an API off under another workload.
resource "google_project_service" "pipeline" {
  for_each = toset([
    "aiplatform.googleapis.com",
    "workflows.googleapis.com",
    "workflowexecutions.googleapis.com",
    "cloudbuild.googleapis.com",
    "storage.googleapis.com",
    "monitoring.googleapis.com",
  ])
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}
