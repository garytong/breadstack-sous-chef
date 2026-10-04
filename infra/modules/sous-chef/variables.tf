variable "project_id" {
  type        = string
  description = "GCP project ID the service, database and secrets live in (one project per environment: int and prod)."
}

variable "region" {
  type        = string
  default     = "northamerica-northeast2"
  description = "Region for Cloud Run, Cloud SQL and Artifact Registry (Toronto, like the thermometer's)."
}

variable "env" {
  type        = string
  description = "Environment label: int or prod. Used in resource descriptions and the ENV env var only; resource NAMES are the same in both projects."
  validation {
    condition     = contains(["int", "prod"], var.env)
    error_message = "env must be \"int\" or \"prod\"."
  }
}

variable "image" {
  type        = string
  description = "Full Docker image URI the Run service and both jobs run (e.g. <region>-docker.pkg.dev/<project>/sous-chef/sous-chef:<tag>). The CI/deploy step passes it with -var; the tfvars example carries a placeholder."
}

# --- Cloud Run service -------------------------------------------------------

variable "run_cpu" {
  type        = string
  default     = "1"
  description = "vCPUs per Run instance. CPU only: the service is a thin loop between the browser and the model API; the GPU model (Phase 5) is a separate service."
}

variable "run_memory" {
  type        = string
  default     = "512Mi"
  description = "Memory per Run instance. The manifest registry holds a few 1.25 MB manifests in memory, so 512Mi is ample."
  validation {
    condition     = can(regex("^[1-9][0-9]*(Mi|Gi)$", var.run_memory))
    error_message = "run_memory must be a whole number of Mi or Gi (e.g. \"512Mi\", \"1Gi\")."
  }
}

variable "run_min_instances" {
  type        = number
  default     = 0
  description = "Minimum Run instances. 0 scales to zero (the task's decision); turns are short-lived and the launcher retries a cold start."
}

variable "run_max_instances" {
  type        = number
  default     = 4
  description = "Maximum Run instances (the task fixes 4; README section 9 said 10 — the task text is the later decision and wins). Each instance also caps itself at MAX_CONCURRENT_TURNS."
}

variable "run_concurrency" {
  type        = number
  default     = 20
  description = "max_instance_request_concurrency: SSE turns hold a connection open for seconds, so 20 per instance keeps a cold instance from being swamped."
}

variable "run_timeout" {
  type        = string
  default     = "600s"
  description = "Cloud Run request timeout. 600 s because a turn streams SSE for the whole model loop (up to 8 model calls); well under Cloud Run's 3600 s cap."
}

variable "ingress" {
  type        = string
  default     = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"
  description = "Cloud Run ingress. Default = internal + load balancer: only the launcher (same project/VPC or via an internal LB) reaches it; the service key is the second barrier. Set INGRESS_TRAFFIC_ALL in int if the launcher runs outside this project and no LB exists yet (IAM run.invoker still gates it)."
  validation {
    condition     = contains(["INGRESS_TRAFFIC_ALL", "INGRESS_TRAFFIC_INTERNAL_ONLY", "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"], var.ingress)
    error_message = "ingress must be one of INGRESS_TRAFFIC_ALL, INGRESS_TRAFFIC_INTERNAL_ONLY, INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER."
  }
}

variable "deploy_nonce" {
  type        = string
  default     = ""
  description = "Forces a fresh Run revision from an unchanged template (a timestamp). Never set it in a tfvars file; pass with -var when a roll-out needs re-minting (thermometer pattern)."
}

# --- Service env (README section 7 table) -------------------------------------

variable "model_routes" {
  type        = string
  default     = ""
  description = "MODEL_ROUTES JSON (route name -> {kind, url?, auth, model, max_tokens, effort?, stream?}). Empty = the server's built-in default table (default=claude-sonnet-5-5, opus=claude-opus-5-5). Set it to add a messages-compatible route pointing at the Phase 5 GPU service."
}

variable "default_route" {
  type        = string
  default     = "default"
  description = "DEFAULT_ROUTE: the route a turn uses when it names none."
}

variable "model_route_allowlist" {
  type        = string
  default     = "default,opus"
  description = "MODEL_ROUTE_ALLOWLIST: comma-separated routes a request may ask for."
}

variable "model_call_timeout" {
  type        = string
  default     = "60s"
  description = "MODEL_CALL_TIMEOUT per model call (Go duration)."
}

variable "turn_ttl" {
  type        = string
  default     = "10m"
  description = "TURN_TTL: how long a waiting/suspended turn (pick_asset) lives before the reaper expires it."
}

variable "max_concurrent_turns" {
  type        = number
  default     = 8
  description = "MAX_CONCURRENT_TURNS per instance; beyond it the service answers 503 busy."
}

variable "trace_retention_days" {
  type        = number
  default     = 90
  description = "TRACE_RETENTION_DAYS: the retention job deletes traces older than this (user decision: 90)."
}

variable "thermometer_api_url" {
  type        = string
  default     = ""
  description = "THERMOMETER_API_URL for usage shipping (POST /api/ingest/usage). Empty = shipping off, and the sous-chef-thermometer-api-key secret is NOT referenced by the Run template (two-halves gate: Cloud Run refuses to start a revision that references a version-less secret)."
}

# --- Cloud SQL ------------------------------------------------------------------

variable "db_tier" {
  type        = string
  default     = "db-f1-micro"
  description = "Cloud SQL machine type. db-f1-micro (shared core, smallest) for int; db-custom-1-3840 for prod per README section 9. A change restarts the instance."
}

variable "sql_disk_size" {
  type        = number
  default     = 10
  description = "Cloud SQL data disk in GB (PD-SSD, grows online, never shrinks)."
}

variable "sql_deletion_protection" {
  type        = bool
  default     = true
  description = "Cloud SQL deletion protection. Pass false in a targeted apply right before a destroy; never set it in a tfvars file."
}

variable "sql_private_ip" {
  type        = bool
  default     = false
  description = "Give the Cloud SQL instance a private IP on sql_private_network and no public IP. Needs sql_private_network; creates a Service Networking peering and gives the Run service and jobs direct VPC egress on sql_private_subnetwork. Default false: the Cloud SQL connector volume over the public IP (the thermometer's shape) needs no VPC at all."
}

variable "sql_private_network" {
  type        = string
  default     = ""
  description = "Self link of the VPC network for private-IP Cloud SQL (projects/<p>/global/networks/<name>). Required when sql_private_ip is true."
}

variable "sql_private_subnetwork" {
  type        = string
  default     = ""
  description = "Subnetwork (name or self link, same region) the Run service and jobs attach to for direct VPC egress when sql_private_ip is true. Required then."
}

# --- IAM ------------------------------------------------------------------------

variable "launcher_service_account" {
  type        = string
  default     = ""
  description = "Email of the pantry launcher's runtime service account (the reverse proxy that adds the service key). It gets roles/run.invoker on the service. Empty = no invoker binding yet (placeholder until the launcher's SA is known); nothing can then call the service through IAM except project owners/editors."
}

variable "scheduler_timezone" {
  type        = string
  default     = "Etc/UTC"
  description = "Time zone of the retention job's Cloud Scheduler trigger (README: 03:00 UTC daily)."
}

variable "retention_schedule" {
  type        = string
  default     = "0 3 * * *"
  description = "Cron for the nightly retention job (sous-chef -retention)."
}

variable "public_invoker" {
  type        = bool
  default     = false
  description = "Grant allUsers roles/run.invoker on the service (thermometer pattern), so callers need no Google ID token and the service key is the only gate. int sets true: the launcher runs on GKE in another project, sends only Authorization: Bearer <service key>, and has no Workload Identity GSA to bind yet. prod keeps false (IAM-gated: launcher_service_account + an ID token, or an internal LB path)."
}

# --- CI ----------------------------------------------------------------------------

variable "ci_service_account" {
  type        = string
  default     = ""
  description = "Email of the Cloud Build service account that runs cloudbuild.yaml (for a trigger in prod-devops-core-project like pantry's: <project-number>@cloudbuild.gserviceaccount.com). It gets roles/artifactregistry.writer on the sous-chef repository, roles/run.developer on the project (update + execute the jobs, update the service) and roles/iam.serviceAccountUser on sous-chef-run (deploy a revision that runs as it). Empty = no CI bindings; a person deploys with the Makefile."
}
