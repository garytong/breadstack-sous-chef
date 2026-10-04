module "sous_chef" {
  source = "../../modules/sous-chef"

  project_id = var.project_id
  region     = var.region
  env        = "int"
  image      = var.image

  run_cpu           = var.run_cpu
  run_memory        = var.run_memory
  run_min_instances = var.run_min_instances
  run_max_instances = var.run_max_instances
  run_concurrency   = var.run_concurrency
  run_timeout       = var.run_timeout
  ingress           = var.ingress
  deploy_nonce      = var.deploy_nonce

  model_routes          = var.model_routes
  default_route         = var.default_route
  model_route_allowlist = var.model_route_allowlist
  model_call_timeout    = var.model_call_timeout
  turn_ttl              = var.turn_ttl
  max_concurrent_turns  = var.max_concurrent_turns
  trace_retention_days  = var.trace_retention_days
  thermometer_api_url   = var.thermometer_api_url

  db_tier                 = var.db_tier
  sql_disk_size           = var.sql_disk_size
  sql_deletion_protection = var.sql_deletion_protection
  sql_private_ip          = var.sql_private_ip
  sql_private_network     = var.sql_private_network
  sql_private_subnetwork  = var.sql_private_subnetwork

  launcher_service_account = var.launcher_service_account
  public_invoker           = var.public_invoker
  ci_service_account       = var.ci_service_account
  retention_schedule       = var.retention_schedule
  scheduler_timezone       = var.scheduler_timezone
}

# Phase 5: the in-house GPU model service, behind gpu_model_enabled. Nothing
# is created while the flag is false. When it is on, add a messages-compatible
# route to model_routes pointing at module.model_gpu[0].url with auth: id-token.
module "model_gpu" {
  count  = var.gpu_model_enabled ? 1 : 0
  source = "../../modules/model-gpu"

  project_id              = var.project_id
  region                  = var.gpu_model_region
  serve                   = var.gpu_model_serve
  image_tag               = var.gpu_model_image_tag
  min_instances           = var.gpu_model_min_instances
  invoker_service_account = module.sous_chef.run_service_account
}
