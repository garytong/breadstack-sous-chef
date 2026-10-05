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

  model_routes          = local.model_routes
  default_route         = var.default_route
  model_route_allowlist = local.model_route_allowlist
  model_call_timeout    = local.model_call_timeout
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
# is created while the flag is false; the route to it is composed below once
# the service exists (gpu_model_serve), so no tfvar ever carries its URL.
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

# The in-house route exists only while the model service does: its URL is
# the route's url and the ID token's audience, and there is none before half
# 2 (thermometer run.tf: CHAT_SOURDOUGH_URL = sourdough_serve ? uri : null).
# With the route off the three model_* tfvars pass through untouched, so a
# plan with the flags off is the plan without Phase 5, and half 1 alone
# (repository + SA) does not touch the sous-chef service. default_route is
# never changed here: every caller that names no route keeps Claude.
locals {
  inhouse_live = var.gpu_model_enabled && var.gpu_model_serve

  # MODEL_ROUTES replaces the server's built-in table wholesale, so the
  # in-house route joins the table in force: the tfvar when set, else the
  # server's two defaults (service/internal/config/config.go DefaultModelRoutes).
  base_routes = var.model_routes != "" ? jsondecode(var.model_routes) : {
    default = { kind = "anthropic", auth = "api-key", model = "claude-sonnet-5-5", max_tokens = 8192, effort = "medium" }
    opus    = { kind = "anthropic", auth = "api-key", model = "claude-opus-5-5", max_tokens = 8192, effort = "medium" }
  }
  inhouse_route = {
    kind       = "messages-compatible"
    url        = local.inhouse_live ? module.model_gpu[0].url : null
    auth       = "id-token"
    model      = var.gpu_model_route.model
    max_tokens = var.gpu_model_route.max_tokens
    stream     = false # Ollama's /v1/messages is answered whole; the service applies plainRequest
    # Sampling the service puts in the body (the image's Modelfile sets the
    # rest): the base's chat default of temperature 1 cost 7 points on the
    # paired exam slice against 0.3 (2026-10-04, thinking already off).
    temperature = var.gpu_model_route.temperature
    # Hybrid: when the student's op is refused, it reports unsupported, or its
    # call fails, the same turn continues on this route (at most once). ""
    # turns it off. Needs a service build that knows the option (after 3b15f6b).
    fallback = var.gpu_model_route.fallback
    # What usage rows (the thermometer) and turn.done call this model: the
    # route's name, not the Ollama tag the server echoes back.
    report_as = var.gpu_model_route.name
  }

  model_routes          = local.inhouse_live ? jsonencode(merge(local.base_routes, { (var.gpu_model_route.name) = local.inhouse_route })) : var.model_routes
  model_route_allowlist = local.inhouse_live ? join(",", distinct(concat(split(",", var.model_route_allowlist), [var.gpu_model_route.name]))) : var.model_route_allowlist
  model_call_timeout    = local.inhouse_live ? var.gpu_model_route.call_timeout : var.model_call_timeout
}
