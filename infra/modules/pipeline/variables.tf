variable "project_id" {
  type        = string
  description = "The int project (breadstack-sous-chef-int). The pipeline is int only: Sous Chef runs only there."
}

variable "region" {
  type        = string
  default     = "northamerica-northeast2"
  description = "The service's region: the lab runtime image's repository (sous-chef) and the five Cloud Run jobs live here, next to int's Sous Chef service."
}

variable "workflows_region" {
  type        = string
  default     = "us-central1"
  description = "Cloud Workflows (and its scheduler triggers) region: next to the bucket and the first Vertex region."
}

variable "gpu_region" {
  type        = string
  default     = "us-east4"
  description = "Region of the eval and candidate GPU services: the production model service's region (L4 quota counts per region)."
}

variable "build_region" {
  type        = string
  default     = "us-east4"
  description = "Cloud Build region for the model image build: next to the model repository."
}

variable "bucket_name" {
  type        = string
  default     = ""
  description = "The pipeline bucket; empty = <project_id>-pipeline (breadstack-sous-chef-int-pipeline)."
}

variable "bucket_location" {
  type        = string
  default     = "US-CENTRAL1"
  description = "The pipeline bucket's location: beside Vertex's first region and the training bucket."
}

variable "run_retention_days" {
  type        = number
  default     = 180
  description = "runs/ and drift/ objects are deleted this many days after they were written; state/, baselines/, cases/, manifests/, engine/, base/ and infra/ never expire."
  validation {
    condition     = var.run_retention_days >= 30
    error_message = "run_retention_days must be at least 30."
  }
}

variable "bucket_force_destroy" {
  type        = bool
  default     = false
  description = "Allow a destroy to delete a non-empty bucket. Keep false: the bucket holds the case pool and the baselines."
}

variable "lab_repository_id" {
  type        = string
  default     = "sous-chef"
  description = "The service's Artifact Registry repository (modules/sous-chef registry.tf), where CI pushes sous-chef-lab:<tag> and sous-chef-trainer:<tag> beside sous-chef:<tag>."
}

variable "model_repository_id" {
  type        = string
  default     = "sous-chef-model"
  description = "The model repository (modules/model-gpu): candidate images are pushed there as model:c-<run id>."
}

variable "model_repo_location" {
  type    = string
  default = "us-east4"
}

variable "lab_image" {
  type        = string
  default     = "us-docker.pkg.dev/cloudrun/container/job:latest"
  description = "The image the jobs are CREATED with. Every workflow run re-points each job it runs at sous-chef-lab:<the int service's tag> before running it (jobs.tf ignores the image afterwards), so this matters only at creation; the default is Google's public sample job image, so the first apply does not wait on a lab image build."
}

variable "int_service_name" {
  type    = string
  default = "sous-chef"
}

variable "int_service_url" {
  type        = string
  description = "int's Sous Chef service URL (modules/sous-chef run_url): SOUS_CHEF_URL of the jobs (teacher collection, the deployed-manifest endpoint)."
}

variable "prod_model_service_name" {
  type        = string
  default     = "sous-chef-model"
  description = "The production GPU model service (modules/model-gpu) merchants use."
}

variable "prod_model_url" {
  type        = string
  description = "The production GPU model service's URL (modules/model-gpu url)."
}

variable "prod_model_image" {
  type        = string
  description = "An image that exists in the model repository (modules/model-gpu image): the eval and candidate services are created with it, and the pipeline re-points them before every use."
}

variable "model_service_account" {
  type        = string
  description = "The roleless account the model images run as (modules/model-gpu service_account, sous-chef-model-run); the eval and candidate services run as it too."
}

variable "anthropic_secret_id" {
  type        = string
  default     = "sous-chef-anthropic-api-key"
  description = "The existing Anthropic key secret (modules/sous-chef secrets.tf). The prepare job (case authoring) reads it; the pipeline adds one accessor member, the prep account."
}

variable "ollama_model" {
  type        = string
  default     = "sous-chef"
  description = "The Ollama model name inside the model image (the warm-up's /api/generate model)."
}

variable "inhouse_route_name" {
  type        = string
  default     = "blueberry-muffin"
  description = "The in-house route's name on int's service: the rollout reads /version to check it still points at the production model service."
}

variable "thermometer_url" {
  type        = string
  default     = "https://thermometer.breadstackcrm.com"
  description = "The thermometer (prod; employee analytics): notifications are POSTed to <url>/api/ingest/souschef/notifications and decisions read from <url>/api/ingest/souschef/decisions/<candidate id> with the pipeline thermometer key."
}

variable "schedule" {
  type        = bool
  default     = false
  description = "pipeline_schedule: create the Cloud Scheduler triggers (poll every 15 min, the weekly run, the night window). false = the workflows run only by hand."
}

variable "poll_cron" {
  type    = string
  default = "*/15 * * * *"
}

variable "weekly_cron" {
  type        = string
  default     = "0 5 * * 1"
  description = "Monday 05:00 UTC = Sunday 22:00 PT, before the thermometer's Monday digest."
}

variable "night_cron" {
  type        = string
  default     = "0 9 * * *"
  description = "09:00 UTC = 02:00 PT: a drift check deferred because the eval service had no L4 runs against the production service then."
}

variable "scheduler_timezone" {
  type    = string
  default = "Etc/UTC"
}

variable "retrain_auto" {
  type        = bool
  default     = false
  description = "pipeline_retrain_auto: a `retrain` drift verdict (with a calibrated baseline and no round in progress) starts souschef-retrain. false = notify only."
}

variable "rollout_enabled" {
  type        = bool
  default     = false
  description = "pipeline_rollout: an Approve rolls the candidate onto the production model service. false = every Approve is a dry rollout (permission check only, production untouched)."
}

variable "l4_headroom" {
  type        = bool
  default     = false
  description = "true once the Cloud Run L4 quota in gpu_region is 5 (it is 3 today): the rollout then patches production without first waiting 20 minutes for the eval and candidate services to idle (a revision swap briefly runs two production instances)."
}

variable "decision_ttl_days" {
  type        = number
  default     = 14
  description = "A candidate_ready decision expires (= Skip) this many days after it was sent. At most 21: the retrain execution polls the decision every 10 min (~3,600 Workflows steps a day) and an execution may take 100,000 steps."
  validation {
    condition     = var.decision_ttl_days >= 1 && var.decision_ttl_days <= 21 && floor(var.decision_ttl_days) == var.decision_ttl_days
    error_message = "decision_ttl_days must be a whole number of days, 1..21 (the decision wait must stay under Workflows' 100,000 steps per execution)."
  }
}

variable "eval_warm_minutes" {
  type        = number
  default     = 15
  description = "How long the drift workflow waits for the eval GPU service to get an L4 before deferring the check to the night window on production."
}

variable "l4_wait_minutes" {
  type        = number
  default     = 360
  description = "How long a retrain round's candidate and eval stages keep trying to get an L4 for the candidate and eval GPU services before the stage fails (resume_from candidate / eval). Timeliness first: hours, not the drift check's minutes."
  validation {
    condition     = var.l4_wait_minutes >= 15 && var.l4_wait_minutes <= 1440
    error_message = "l4_wait_minutes must be 15..1440."
  }
}

variable "lab_image_grace_minutes" {
  type        = number
  default     = 45
  description = "After the int service rolls to a new tag, how long a missing sous-chef-lab:<tag> is taken for CI still building it (the run is skipped quietly) before it is reported as stage_failed."
}

variable "train_ladder" {
  type = list(object({
    name             = string
    region           = string
    machine_type     = string
    accelerator_type = string
    strategy         = string
    attempts         = number
    max_wait_s       = optional(number, 0)
    max_pending_s    = optional(number, 3600)
  }))
  default = [
    { name = "h100-flex", region = "us-central1", machine_type = "a3-highgpu-1g", accelerator_type = "NVIDIA_H100_80GB", strategy = "FLEX_START", attempts = 1, max_wait_s = 14400 },
    { name = "h100-spot-usc1", region = "us-central1", machine_type = "a3-highgpu-1g", accelerator_type = "NVIDIA_H100_80GB", strategy = "SPOT", attempts = 4 },
    { name = "h100-spot-use4", region = "us-east4", machine_type = "a3-highgpu-1g", accelerator_type = "NVIDIA_H100_80GB", strategy = "SPOT", attempts = 4 },
    { name = "h100-spot-usw1", region = "us-west1", machine_type = "a3-highgpu-1g", accelerator_type = "NVIDIA_H100_80GB", strategy = "SPOT", attempts = 4 },
    { name = "a100-80-std", region = "us-central1", machine_type = "a2-ultragpu-1g", accelerator_type = "NVIDIA_A100_80GB", strategy = "STANDARD", attempts = 2 },
    { name = "a100-40-std", region = "us-central1", machine_type = "a2-highgpu-1g", accelerator_type = "NVIDIA_TESLA_A100", strategy = "STANDARD", attempts = 2 },
  ]
  description = "The Vertex training ladder (DESIGN §3.5), tried in order: a rung moves on after `attempts` preemptions, at once on no capacity (including a job that has not started after max_pending_s — Flex-start: after max_wait_s + 30 min), and after two failures of unknown cause (four in all fail the stage). A Spot rung's attempts after the first are resumes from the newest checkpoint. A rung Vertex refuses outright (400 on create, e.g. no FLEX_START for the machine) is skipped."
  validation {
    condition = length(var.train_ladder) > 0 && alltrue([
      for r in var.train_ladder : contains(["FLEX_START", "SPOT", "STANDARD"], r.strategy) && r.attempts >= 1 && can(regex("^[a-z0-9-]+$", r.name)) && r.max_pending_s >= 600
    ])
    error_message = "train_ladder: at least one rung; strategy FLEX_START | SPOT | STANDARD; attempts >= 1; name [a-z0-9-]; max_pending_s >= 600."
  }
}

variable "train_disk_gb" {
  type    = number
  default = 500
}

variable "train_timeout_hours" {
  type        = number
  default     = 12
  description = "Vertex scheduling.timeout per training attempt."
}

variable "alert_emails" {
  type        = list(string)
  default     = []
  description = "pipeline_alert_emails: the Cloud Monitoring backstop emails these on a failed workflow execution and on a notification the thermometer did not take (\"not delivered\" in the workflows' logs). The int root requires at least one while the pipeline is enabled; empty here = no policies (module tests)."
}

variable "jobs" {
  type = map(object({
    cpu       = string
    memory    = string
    timeout_s = number
  }))
  default = {
    drift   = { cpu = "4", memory = "16Gi", timeout_s = 36000 }
    eval    = { cpu = "4", memory = "16Gi", timeout_s = 64800 }
    prepare = { cpu = "4", memory = "16Gi", timeout_s = 21600 }
    collect = { cpu = "8", memory = "32Gi", timeout_s = 86400 }
    dataset = { cpu = "8", memory = "32Gi", timeout_s = 21600 }
  }
  description = "Shape of each job. Cloud Run jobs write to an in-memory file system, so memory covers the engine snapshot (~205 MB, extracted), Postgres and the run's logs."
}
