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
