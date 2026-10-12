variable "project_id" {
  type        = string
  description = "GCP project ID (the same project as sous-chef)."
}

variable "region" {
  type        = string
  default     = "us-east4"
  description = "Region of the GPU service. Cloud Run has no GPU region in Canada: us-east4 is the nearest with the L4; us-central1 is the fallback when us-east4 has no capacity."
}

variable "repo_location" {
  type        = string
  default     = "us-east4"
  description = "Region of the private image repository. Stays put when region moves to the fallback: Cloud Run imports the image across regions at deploy time."
}

variable "repository_id" {
  type        = string
  default     = "sous-chef-model"
  description = "Artifact Registry repository id for the Ollama + weights images."
}

variable "serve" {
  type        = bool
  default     = false
  description = "Second half: create the GPU service and its invoker binding. Needs the image <repo>/model:<image_tag> pushed first (a service pointing at a missing image fails the apply). false = repository and service account only."
}

variable "image_tag" {
  type        = string
  default     = "lab1"
  description = "Tag in the repository the service runs. A new tag rolls a new revision."
  validation {
    condition     = can(regex("^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$", var.image_tag))
    error_message = "image_tag must be a Docker tag: letters, digits, _ . -, at most 128 characters, not starting with . or -."
  }
}

variable "min_instances" {
  type        = number
  default     = 0
  description = "0 scales to zero when idle (first request after that wakes the GPU, about a minute); 1 keeps the model loaded, billed every second (about US$1.05/hour for an L4 at us-east4 list price, 2026-09)."
  validation {
    condition     = contains([0, 1], var.min_instances)
    error_message = "min_instances must be 0 or 1; the service has one instance at most."
  }
}

variable "gpu" {
  type        = string
  default     = "nvidia-l4"
  description = "Accelerator for the node_selector, one per instance."
}

variable "cpu" {
  type        = string
  default     = "4"
  description = "vCPUs per instance: 4 is the documented minimum for a GPU service."
}

variable "memory" {
  type        = string
  default     = "16Gi"
  description = "Memory per instance: 16Gi is the documented minimum for a GPU service."
  validation {
    condition     = can(regex("^[1-9][0-9]*Gi$", var.memory))
    error_message = "memory must be a whole number of Gi, such as \"16Gi\"."
  }
}

variable "invoker_service_account" {
  type        = string
  description = "Email of the only identity allowed to call the model service: sous-chef's runtime SA (sous-chef-run), which sends a Google ID token with the service URL as audience."
}

variable "extra_invokers" {
  type        = list(string)
  default     = []
  description = "More service-account emails granted run.invoker on the model service beside invoker_service_account (the binding stays authoritative). The Sous Chef pipeline passes its drift runner and workflow accounts here."
  validation {
    condition     = alltrue([for e in var.extra_invokers : can(regex("^[a-z0-9-]+@[a-z0-9-]+\\.iam\\.gserviceaccount\\.com$", e))])
    error_message = "extra_invokers must be service-account emails (<name>@<project>.iam.gserviceaccount.com)."
  }
}

variable "cleanup_candidates" {
  type        = bool
  default     = false
  description = "Turn on the repository's cleanup policies for the pipeline's candidate images (main.tf). false = no policy: nothing is ever deleted."
}

variable "candidate_retention_days" {
  type        = number
  default     = 30
  description = "With cleanup_candidates: a c-* image older than this is deleted unless a keep rule holds it."
  validation {
    condition     = var.candidate_retention_days >= 7
    error_message = "candidate_retention_days must be at least 7 (a decision can wait 14 days; keep_recent covers the rest)."
  }
}

variable "keep_recent" {
  type        = number
  default     = 10
  description = "With cleanup_candidates: the newest N image versions are always kept."
  validation {
    condition     = var.keep_recent >= 3
    error_message = "keep_recent must be at least 3."
  }
}
