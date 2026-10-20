# Root variables: one-to-one with the sous-chef module's (see
# modules/sous-chef/variables.tf for the full descriptions) plus the Phase 5
# GPU model flags. Defaults here are the module's; the per-env tfvars file
# overrides what differs.

variable "project_id" { type = string }
variable "region" {
  type    = string
  default = "northamerica-northeast2"
}
variable "image" {
  type        = string
  description = "Image URI the service and jobs run; the deploy step passes it with -var=image=..."
}

variable "run_cpu" {
  type    = string
  default = "1"
}
variable "run_memory" {
  type    = string
  default = "512Mi"
}
variable "run_min_instances" {
  type    = number
  default = 0
}
variable "run_max_instances" {
  type    = number
  default = 4
}
variable "run_concurrency" {
  type    = number
  default = 20
}
variable "run_timeout" {
  type    = string
  default = "600s"
}
variable "ingress" {
  type    = string
  default = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"
}
variable "deploy_nonce" {
  type    = string
  default = ""
}

variable "model_routes" {
  type    = string
  default = ""
}
variable "default_route" {
  type    = string
  default = "default"
}
variable "gpu_model_default" {
  type        = bool
  default     = false
  description = "Make the in-house route (gpu_model_route.name) the service's DEFAULT_ROUTE while it is live, so every turn that names no route goes to the in-house model, which falls back to gpu_model_route.fallback (Claude) within the turn. Ignored while gpu_model_serve is off."
}
variable "model_route_allowlist" {
  type    = string
  default = "default,opus"
}
variable "model_call_timeout" {
  type    = string
  default = "60s"
}
variable "turn_ttl" {
  type    = string
  default = "10m"
}
variable "max_concurrent_turns" {
  type    = number
  default = 8
}
variable "trace_retention_days" {
  type    = number
  default = 90
}
variable "thermometer_api_url" {
  type    = string
  default = ""
}

variable "db_tier" {
  type    = string
  default = "db-f1-micro"
}
variable "sql_disk_size" {
  type    = number
  default = 10
}
variable "sql_deletion_protection" {
  type    = bool
  default = true
}
variable "sql_private_ip" {
  type    = bool
  default = false
}
variable "sql_private_network" {
  type    = string
  default = ""
}
variable "sql_private_subnetwork" {
  type    = string
  default = ""
}

variable "launcher_service_account" {
  type    = string
  default = ""
}
variable "public_invoker" {
  type    = bool
  default = false
}
variable "ci_service_account" {
  type    = string
  default = ""
}
variable "retention_schedule" {
  type    = string
  default = "0 3 * * *"
}
variable "scheduler_timezone" {
  type    = string
  default = "Etc/UTC"
}

# --- Phase 5 GPU model (modules/model-gpu) ---
variable "gpu_model_enabled" {
  type        = bool
  default     = false
  description = "Instantiate the model-gpu module (repository + roleless SA). Creating either needs a person's login, not a deployer key."
}
variable "gpu_model_serve" {
  type        = bool
  default     = false
  description = "Create the GPU Cloud Run service and its invoker binding (needs gpu_model_enabled and the image pushed)."
}
variable "gpu_model_region" {
  type    = string
  default = "us-east4"
}
variable "gpu_model_image_tag" {
  type    = string
  default = "lab1"
}
variable "gpu_model_min_instances" {
  type    = number
  default = 0
}
variable "gpu_model_route" {
  type = object({
    name         = string
    model        = string
    max_tokens   = number
    call_timeout = string
    temperature  = optional(number, 0.3)
    fallback     = optional(string, "default")
  })
  default = {
    name         = "blueberry-muffin" # the in-house model's name (2026-10-04): the route a turn's `model` names, the drawer badge, and what usage rows report
    model        = "sous-chef"        # the Ollama model name inside the image (infrastructure name, like the thermometer's "sourdough")
    max_tokens   = 1024               # 2026-10-10: real answers stay under ~450 tokens (eval p99 414); a runaway now hands off to Claude after ~25 s instead of ~110 s at 4096
    call_timeout = "60s"              # 2026-10-10: was 180s; a warm call is <40 s even at the 1024 cap, and a stuck or cold call should hand off to Claude within a minute (60s was the pre-GPU default)
    temperature  = 0.3
    fallback     = "default"
  }
  description = "The messages-compatible route main.tf adds to MODEL_ROUTES (and to MODEL_ROUTE_ALLOWLIST) while gpu_model_serve is on: its name (what a turn's `model` says), the Ollama model name inside the image, max_tokens, and the MODEL_CALL_TIMEOUT the service runs with meanwhile (the thermometer's L4 needed ~120 s per call; the model service's own timeout is 300 s). Its url is the module's output; its auth is id-token."
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]*$", var.gpu_model_route.name)) && var.gpu_model_route.max_tokens >= 1 && var.gpu_model_route.temperature >= 0 && var.gpu_model_route.temperature <= 2 && can(regex("^[0-9]+(ms|s|m|h)$", var.gpu_model_route.call_timeout))
    error_message = "gpu_model_route: name is lowercase [a-z0-9-], max_tokens >= 1, temperature 0..2, call_timeout a Go duration such as \"180s\"."
  }
}
variable "gpu_model_mistral_route" {
  type = object({
    enabled     = optional(bool, false)
    name        = optional(string, "chocolate-muffin")
    temperature = optional(number, 0.15)
    fallback    = optional(string, "default")
  })
  default     = {}
  description = "chocolate-muffin's route (docs/CHOCOLATE-MUFFIN.md §2): a second messages-compatible route on the SAME model service as gpu_model_route, with its Ollama model name, max_tokens and call timeout, and Ministral 3's settings: temperature 0.15, prompt-cache priming off, parser_retries off (needs a service build that knows the option). enabled adds it to MODEL_ROUTES and MODEL_ROUTE_ALLOWLIST while gpu_model_serve is on; it answers only turns that name it until model_route_aliases sends the in-house route to it. Off (the default) leaves both exactly as they were."
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]*$", var.gpu_model_mistral_route.name)) && var.gpu_model_mistral_route.name != var.gpu_model_route.name && var.gpu_model_mistral_route.temperature >= 0 && var.gpu_model_mistral_route.temperature <= 2
    error_message = "gpu_model_mistral_route: name is lowercase [a-z0-9-] and not gpu_model_route's, temperature 0..2."
  }
}
variable "model_route_aliases" {
  type        = map(string)
  default     = {}
  description = "MODEL_ROUTE_ALIASES, the switch between the two in-house routes (docs/CHOCOLATE-MUFFIN.md step 5): {\"blueberry-muffin\" = \"chocolate-muffin\"} answers every in-house turn — named, or on the default route — with chocolate-muffin's settings and report_as while DEFAULT_ROUTE stays blueberry-muffin. {} (the default) sets nothing. Passed only while gpu_model_serve is on; a check warns when an alias names a route that is not live."
}

# --- Sous Chef pipeline (modules/pipeline; docs/RETRAIN.md R0 for the order) ---
variable "pipeline_enabled" {
  type        = bool
  default     = false
  description = "Create the drift check and retraining pipeline: APIs, the pipeline bucket, seven service accounts, two secrets (no values), five Cloud Run jobs, two workflows, the eval and candidate GPU services (min 0), the monitoring backstop. Schedules nothing. Needs gpu_model_enabled, gpu_model_serve and pipeline_alert_emails."
}
variable "pipeline_schedule" {
  type        = bool
  default     = false
  description = "Cloud Scheduler triggers for souschef-drift: poll every 15 min, weekly Monday 05:00 UTC, the 09:00 UTC night window. Turn on after the SMOKE run and the first calibration."
}
variable "pipeline_retrain_auto" {
  type        = bool
  default     = false
  description = "A `retrain` drift verdict starts souschef-retrain (only with a calibrated baseline and no round in progress). false = notify only."
}
variable "pipeline_rollout" {
  type        = bool
  default     = false
  description = "An Approve on the thermometer rolls the candidate onto sous-chef-model. false = every Approve is a dry rollout (production untouched)."
}
variable "pipeline_l4_headroom" {
  type        = bool
  default     = false
  description = "true once the Cloud Run L4 quota in gpu_model_region is 5 (3 today): the rollout no longer waits ~20 min for the eval and candidate services to idle."
}
variable "pipeline_cleanup_candidates" {
  type        = bool
  default     = true
  description = "With the pipeline on: the model repository gets a cleanup policy for c-* candidate images older than 30 days (keeping the newest 10, every hand-built r* tag and every image ever rolled out, live-*). It deletes nothing while the repository's tags are immutable (modules/model-gpu immutable_tags = true): Artifact Registry never deletes a tagged artifact under immutable tags, so every candidate (~6 GB) stays until pruned by hand (docs/RETRAIN.md R7 \"Cleanup\")."
}
variable "pipeline_alert_emails" {
  type        = list(string)
  default     = []
  description = "Emails for the Cloud Monitoring backstop: a failed workflow execution, and a notification the thermometer did not take. Required (at least one) while pipeline_enabled: it is the only alert path that does not go through the thermometer."
  validation {
    condition     = !var.pipeline_enabled || length(var.pipeline_alert_emails) > 0
    error_message = "pipeline_alert_emails: at least one address while pipeline_enabled (the backstop when the thermometer or Slack is down)."
  }
  validation {
    condition     = alltrue([for e in var.pipeline_alert_emails : can(regex("^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$", e))])
    error_message = "pipeline_alert_emails: email addresses."
  }
}
variable "pipeline_base_family" {
  type        = string
  default     = "qwen"
  description = "The base model family the retrain pipeline trains: qwen (blueberry-muffin) or mistral (chocolate-muffin, Ministral 3 8B). Read by the dataset stage from infra/config.json; a round's trigger.base_family wins (docs/RETRAIN.md \"The base family\")."
  validation {
    condition     = contains(["qwen", "mistral"], var.pipeline_base_family)
    error_message = "pipeline_base_family: qwen or mistral."
  }
}
variable "pipeline_thermometer_url" {
  type        = string
  default     = "https://thermometer.breadstackcrm.com"
  description = "Where the pipeline posts notifications and reads decisions (the prod thermometer; Slack goes to its weekly digest channel)."
}
