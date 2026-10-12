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

output "pipeline" {
  description = "The Sous Chef pipeline (null while pipeline_enabled is off): bucket, GPU service URLs, jobs, workflows, accounts, secrets to populate."
  value = local.pipeline_on ? {
    bucket              = module.pipeline[0].bucket
    eval_url            = module.pipeline[0].eval_url
    cand_url            = module.pipeline[0].cand_url
    jobs                = module.pipeline[0].job_names
    workflows           = module.pipeline[0].workflow_names
    workflows_region    = module.pipeline[0].workflows_region
    service_accounts    = module.pipeline[0].service_accounts
    secrets_to_populate = module.pipeline[0].secrets_to_populate
    config_object       = module.pipeline[0].config_object
  } : null
}
