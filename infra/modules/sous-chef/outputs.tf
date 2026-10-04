output "run_url" {
  description = "Cloud Run service URL — what the launcher's SOUS_CHEF_URL points at (internal ingress: reachable from the launcher's project/VPC or an internal LB only)."
  value       = google_cloud_run_v2_service.sous_chef.uri
}

output "run_service_account" {
  description = "Runtime service account email (sous-chef-run)."
  value       = google_service_account.run.email
}

output "sql_connection_name" {
  description = "Cloud SQL instance connection name (the /cloudsql/<name> socket)."
  value       = google_sql_database_instance.sous_chef.connection_name
}

output "artifact_repo" {
  description = "Artifact Registry Docker repository base URL to push the image to."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.sous_chef.repository_id}"
}

output "migrate_job" {
  description = "The migrate job the deploy executes before each roll-out."
  value       = google_cloud_run_v2_job.migrate.name
}

output "retention_job" {
  description = "The nightly retention job (Cloud Scheduler trigger of the same name)."
  value       = google_cloud_run_v2_job.retention.name
}

output "secrets_to_populate" {
  description = "Secret Manager secrets Terraform created WITHOUT a value; each needs a version before the first deploy (see infra/README.md)."
  value = concat(
    ["sous-chef-anthropic-api-key", "sous-chef-service-keys"],
    var.thermometer_api_url != "" ? ["sous-chef-thermometer-api-key"] : [],
  )
}
