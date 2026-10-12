output "bucket" {
  description = "The pipeline bucket (gs://<name>)."
  value       = google_storage_bucket.pipeline.name
}

output "eval_url" {
  value = google_cloud_run_v2_service.gpu["eval"].uri
}

output "cand_url" {
  value = google_cloud_run_v2_service.gpu["cand"].uri
}

output "prod_model_url" {
  value = var.prod_model_url
}

output "int_service_url" {
  value = var.int_service_url
}

output "thermometer_url" {
  value = var.thermometer_url
}

output "job_names" {
  value = { for k, j in google_cloud_run_v2_job.job : k => j.name }
}

output "workflow_names" {
  value = { for k, w in google_workflows_workflow.wf : k => w.name }
}

output "workflows_region" {
  value = var.workflows_region
}

output "service_accounts" {
  value = local.sa_email
}

output "secrets_to_populate" {
  description = "Secrets created WITHOUT a value; add each (no-echo, docs/RETRAIN.md R0) before the full apply."
  value       = [for s in google_secret_manager_secret.pipeline : s.secret_id]
}

output "pipeline_invokers" {
  description = "The accounts the production model service must admit (modules/model-gpu extra_invokers)."
  value       = [local.sa_email["drift"], local.sa_email["workflow"]]
}

output "config_object" {
  value = "gs://${google_storage_bucket.pipeline.name}/${google_storage_bucket_object.config.name}"
}
