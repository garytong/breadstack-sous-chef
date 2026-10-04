output "run_url" {
  value = module.sous_chef.run_url
}
output "run_service_account" {
  value = module.sous_chef.run_service_account
}
output "sql_connection_name" {
  value = module.sous_chef.sql_connection_name
}
output "artifact_repo" {
  value = module.sous_chef.artifact_repo
}
output "migrate_job" {
  value = module.sous_chef.migrate_job
}
output "retention_job" {
  value = module.sous_chef.retention_job
}
output "secrets_to_populate" {
  value = module.sous_chef.secrets_to_populate
}
output "gpu_model_url" {
  description = "Phase 5 model service URL (null until gpu_model_enabled and gpu_model_serve)."
  value       = var.gpu_model_enabled ? module.model_gpu[0].url : null
}
output "gpu_model_repo" {
  value = var.gpu_model_enabled ? module.model_gpu[0].repo : null
}
