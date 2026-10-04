output "repo" {
  description = "Repository base URL; push the image as <repo>/model:<tag>."
  value       = local.repo
}

output "service_account" {
  description = "The roleless service account the model runs as."
  value       = google_service_account.model.email
}

output "url" {
  description = "The GPU service's URL: the messages-compatible route's base URL and the ID token's audience in one. null until serve = true."
  value       = var.serve ? google_cloud_run_v2_service.model[0].uri : null
}
