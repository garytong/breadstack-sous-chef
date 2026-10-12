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

output "service_name" {
  description = "The GPU service's name (sous-chef-model); null until serve = true."
  value       = var.serve ? google_cloud_run_v2_service.model[0].name : null
}

output "region" {
  description = "The GPU service's region."
  value       = var.region
}

output "repository_id" {
  description = "The model repository's id."
  value       = google_artifact_registry_repository.model.repository_id
}

output "repo_location" {
  description = "The model repository's region."
  value       = google_artifact_registry_repository.model.location
}

output "image" {
  description = "The image the service was created with (<repo>/model:<image_tag>). The live image can differ: the pipeline and make model-rollout own it after creation."
  value       = local.image
}

output "invokers" {
  description = "The members of the authoritative run.invoker binding (empty until serve = true)."
  value       = var.serve ? google_cloud_run_v2_service_iam_binding.invoker[0].members : []
}
