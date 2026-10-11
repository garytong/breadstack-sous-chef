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
