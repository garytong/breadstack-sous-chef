variable "project_id" {
  description = "The GCP project: sous-chef's int project (make -C lab passes it as GCP_PROJECT)."
  type        = string
  default     = "breadstack-sous-chef-int"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{4,28}[a-z0-9]$", var.project_id))
    error_message = "project_id must be a GCP project id."
  }
}

# --- The VM (off until make -C lab train-vm-up) ---

variable "training_vm_enabled" {
  description = "Whether the training VM exists. Only make -C lab train-vm-up / train-vm-down switch it; every other plan and apply passes whether it exists now, so a routine apply never creates or deletes it."
  type        = bool
  default     = false
}

variable "zone" {
  description = "The VM's zone, one where machine_type is offered: a2-highgpu-1g and a3-highgpu-1g in us-central1-a, -b, -c or -f (GCP named -f as the zone with A100 capacity during the 2026-10-04 stockouts) (vm.tf's local.shapes, whose precondition refuses any other pair). An A100 stockout in one zone: train-vm-down, another zone here, train-vm-up. Changing it replaces the VM and deletes its disk."
  type        = string
  default     = "us-central1-a"

  validation {
    condition     = contains(["us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f"], var.zone)
    error_message = "zone must be us-central1-a, -b or -c: the us-central1 zones with A2 Standard (A100 40 GB) and A3 High (H100). The data stays in us-central1."
  }
}

variable "machine_type" {
  description = "One GPU, two cards (vm.tf, local.shapes): a2-highgpu-1g, 1 × A100 40 GB, 12 vCPU, 85 GB of RAM (the default: the measured recipe — bf16 LoRA r16 at 32,768-token rows, 26.8 GiB peak on the 5090 — fits its 40 GB unchanged, and the int project holds 16 on-demand A100s of quota); a3-highgpu-1g, 1 × H100 80 GB, 26 vCPU, 234 GB, Spot only (the fast fallback: the thermometer trained on one on 2026-10-03)."
  type        = string
  default     = "a2-highgpu-1g"

  validation {
    condition     = contains(["a2-highgpu-1g", "a3-highgpu-1g"], var.machine_type)
    error_message = "machine_type must be a2-highgpu-1g or a3-highgpu-1g (1 GPU each): the 9B's SFT needs no more."
  }
}

variable "provisioning_model" {
  description = "STANDARD (the default: on demand, a2-highgpu-1g about US$3.67/h; no preemption, which matters because train.py has no --resume — a preempted run is lost) or SPOT (preemptible: a2-highgpu-1g about US$1.10/h, a3-highgpu-1g US$6.57/h; the A3 High's 1-GPU shape runs only as Spot or Flex-start). FLEX_START is not offered: provider 6.x cannot send its required requestValidForDuration."
  type        = string
  default     = "STANDARD"

  validation {
    condition     = contains(["SPOT", "STANDARD"], var.provisioning_model)
    error_message = "provisioning_model must be SPOT or STANDARD. FLEX_START (DWS) needs params.requestValidForDuration, which google_compute_instance cannot send in provider 6.x: create such a VM with gcloud if ever wanted (README.md)."
  }
}

variable "termination_action" {
  description = "What Google does to the VM on a Spot preemption and when max_run_hours runs out: STOP (the default: the VM and its disk stay, billing for the GPU stops, make -C lab train-vm-up starts it again with its environment intact) or DELETE (the VM and its disk go; the next train-vm-up rebuilds it from the bucket)."
  type        = string
  default     = "STOP"

  validation {
    condition     = contains(["STOP", "DELETE"], var.termination_action)
    error_message = "termination_action must be STOP or DELETE."
  }
}

variable "max_run_hours" {
  description = "The hard stop: Google stops (or deletes, per termination_action) the VM this many hours after each start, whatever runs on it, so a forgotten VM stops billing for its GPU. 1 to 168 (seven days). Restarting the VM starts a fresh period. 12 covers the bootstrap (~30 min), the ~900-row SFT (3–5 h estimated on the A100) and the export with room."
  type        = number
  default     = 12

  validation {
    condition     = var.max_run_hours >= 1 && var.max_run_hours <= 168 && floor(var.max_run_hours) == var.max_run_hours
    error_message = "max_run_hours must be a whole number from 1 to 168."
  }
}

variable "image" {
  description = "The boot image: Google's Deep Learning VM with the NVIDIA 580 driver and CUDA 12.9, Ubuntu 24.04, Python 3.12, no framework — the lab's uv environment brings its own PyTorch (the cu130 build, which needs driver R580+; this is the only image proven to run the pinned set, on the thermometer's H100 and RTX PRO 6000). A family, so a new VM takes the newest image; a running VM is never replaced for it."
  type        = string
  default     = "projects/deeplearning-platform-release/global/images/family/common-cu129-ubuntu-2404-nvidia-580"
}

variable "boot_disk_gb" {
  description = "The boot disk, GiB: SSD Persistent Disk on a2-highgpu-1g (which takes no Hyperdisk Balanced), Hyperdisk Balanced on a3-highgpu-1g. 200 holds the base's cache as Hugging Face ships it (19 GB), the uv environment with PyTorch (~5.3 GB), the dataset, the adapters, and the export's intermediates (the 18 GB bf16 merge folder, the 5.4 GB q4_k_m and ~9.5 GB q8_0 GGUFs, llama.cpp's build) twice over."
  type        = number
  default     = 200

  validation {
    condition     = var.boot_disk_gb >= 100 && var.boot_disk_gb <= 4000
    error_message = "boot_disk_gb must be from 100 to 4000."
  }
}

variable "boot_disk_iops" {
  description = "Provisioned IOPS of the Hyperdisk Balanced boot disk (a3-highgpu-1g only; an SSD Persistent Disk's performance follows its size): 3,000 are free; each one above costs about US$0.005/month."
  type        = number
  default     = 6000

  validation {
    condition     = var.boot_disk_iops >= 3000 && var.boot_disk_iops <= 120000
    error_message = "boot_disk_iops must be from 3000 (Hyperdisk Balanced's minimum) to 120000 (a3-highgpu-1g's limit)."
  }
}

variable "boot_disk_throughput" {
  description = "Provisioned throughput of the Hyperdisk Balanced boot disk, MiB/s (a3-highgpu-1g only): 140 are free; each one above costs about US$0.04/month. 600 loads the 19 GB base in well under a minute."
  type        = number
  default     = 600

  validation {
    condition     = var.boot_disk_throughput >= 140 && var.boot_disk_throughput <= 1800
    error_message = "boot_disk_throughput must be from 140 to 1800 MiB/s (a3-highgpu-1g's limit)."
  }
}

variable "secure_boot" {
  description = "Shielded VM Secure Boot. Off by default: the image's NVIDIA kernel modules are not known to be signed for it, and a refused module leaves a VM that bills for a GPU it cannot use. vTPM and integrity monitoring are always on."
  type        = bool
  default     = false
}

variable "uv_version" {
  description = "The uv the startup script installs (the workstation's, so the pinned install resolves the same)."
  type        = string
  default     = "0.12.19"

  validation {
    condition     = can(regex("^[0-9]+\\.[0-9]+\\.[0-9]+$", var.uv_version))
    error_message = "uv_version must be a version such as 0.12.19."
  }
}

variable "stage" {
  description = "A label on the VM naming the step, so the billing export splits spend by stage: smoke, sft, dpo or export."
  type        = string
  default     = "smoke"

  validation {
    condition     = contains(["smoke", "sft", "dpo", "export"], var.stage)
    error_message = "stage must be one of smoke, sft, dpo, export."
  }
}

variable "base_model" {
  description = "The base model the startup script puts in the shared Hugging Face cache (/work/hf) when it is not there yet: the bucket's copy (base/<slug>-hf.tar, slug = the id with / as -, in-region) if make -C lab train-remote-push made one, else Hugging Face through Cloud NAT. Public weights (unsloth/Qwen3.5-9B is neither gated nor private: no token). Empty: fetch nothing."
  type        = string
  default     = "unsloth/Qwen3.5-9B"

  validation {
    condition     = var.base_model == "" || can(regex("^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$", var.base_model))
    error_message = "base_model must be empty or a Hugging Face model id such as unsloth/Qwen3.5-9B."
  }
}

variable "lab_repo" {
  description = "Optional: a git URL the startup script clones (and pulls at every boot) into /work/sous-chef, the lab at /work/sous-chef/lab. Empty (the default, the repo has no remote yet): the lab's working tree comes from the bucket instead, kit/lab.tar.gz, which make -C lab train-remote-push uploads."
  type        = string
  default     = ""
}

variable "lab_ref" {
  description = "With lab_repo: the branch or tag to check out."
  type        = string
  default     = "main"
}

# --- The bucket ---

variable "operators" {
  description = "Who pushes and pulls (roles/storage.objectAdmin on the bucket, beside the VM's account): user: or group: principals. Project Owner alone does not imply object access."
  type        = list(string)
  default     = []

  validation {
    condition     = alltrue([for m in var.operators : can(regex("^(user|group):[^@\\s]+@[^@\\s]+$", m))])
    error_message = "Each operator must be user:<email> or group:<email>."
  }
}

variable "object_expiry_days" {
  description = "data/, kit/ and runs/ objects are deleted this many days after they were written (GCS objects are immutable, so an overwrite is a new write). 1 to 90."
  type        = number
  default     = 30

  validation {
    condition     = var.object_expiry_days >= 1 && var.object_expiry_days <= 90 && floor(var.object_expiry_days) == var.object_expiry_days
    error_message = "object_expiry_days must be a whole number from 1 to 90."
  }
}

variable "base_expiry_days" {
  description = "base/ objects (the public base weights, 19 GB) are deleted this many days after they were written. 1 to 365."
  type        = number
  default     = 180

  validation {
    condition     = var.base_expiry_days >= 1 && var.base_expiry_days <= 365 && floor(var.base_expiry_days) == var.base_expiry_days
    error_message = "base_expiry_days must be a whole number from 1 to 365."
  }
}

variable "kms_key_name" {
  description = "Optional CMEK for the bucket: a Cloud KMS key in us-central1 (projects/…/locations/us-central1/keyRings/…/cryptoKeys/…) that the project's Cloud Storage service agent may encrypt and decrypt with. Empty (the default): Google-managed encryption."
  type        = string
  default     = ""

  validation {
    condition     = var.kms_key_name == "" || can(regex("^projects/[^/]+/locations/us-central1/keyRings/[^/]+/cryptoKeys/[^/]+$", var.kms_key_name))
    error_message = "kms_key_name must be empty or a us-central1 Cloud KMS key name (projects/<p>/locations/us-central1/keyRings/<r>/cryptoKeys/<k>)."
  }
}

variable "bucket_force_destroy" {
  description = "Let terraform delete the bucket with objects still in it. Only a deliberate destroy sets it (README.md)."
  type        = bool
  default     = false
}

# --- The budget ---

variable "budget_enabled" {
  description = "The notify-only budget. Needs billing.budgets.create on the billing account (Billing Account Administrator or Costs Manager) on the applying login; false skips it."
  type        = bool
  default     = true
}

variable "billing_account" {
  description = "The project's billing account id (XXXXXX-XXXXXX-XXXXXX), for the budget."
  type        = string
  default     = ""

  validation {
    condition     = var.billing_account == "" || can(regex("^[0-9A-F]{6}-[0-9A-F]{6}-[0-9A-F]{6}$", var.billing_account))
    error_message = "billing_account must be empty or a billing account id such as 012345-6789AB-CDEF01."
  }
}

variable "budget_amount" {
  description = "The budget, in the billing account's own currency (no currency is sent, so a CAD account reads it as CAD): alerts at 50%, 90% and 100% of it. A notification only: nothing is stopped and there is no spending cap. 200 is a first figure (README.md, Cost: a 190-row run ~US$5–6, a 900-row run ~US$13–20 on the A100, each plus ~US$0.5 of export); set it to the user's."
  type        = number
  default     = 200

  validation {
    condition     = var.budget_amount >= 10 && var.budget_amount <= 5000 && floor(var.budget_amount) == var.budget_amount
    error_message = "budget_amount must be a whole number from 10 to 5000."
  }
}

variable "budget_start_date" {
  description = "The budget counts purpose=sous-chef-train spend from this day (YYYY-MM-DD) with no end: one running total, not a monthly one."
  type        = string
  default     = "2026-10-04"

  validation {
    condition     = can(regex("^20[0-9]{2}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])$", var.budget_start_date))
    error_message = "budget_start_date must be a date, YYYY-MM-DD."
  }
}

variable "budget_notification_channels" {
  description = "Optional Cloud Monitoring notification channels (projects/<p>/notificationChannels/<id>, at most 5) to alert beside the billing account's admins and users."
  type        = list(string)
  default     = []

  validation {
    condition     = length(var.budget_notification_channels) <= 5
    error_message = "A budget takes at most 5 notification channels."
  }
}
