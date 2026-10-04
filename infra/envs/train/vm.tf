# The training VM: one GPU in us-central1 — an A2 Standard (A100 40 GB) or
# an A3 High (H100 80 GB), as machine_type says (local.shapes below) — which
# exists only between make -C lab train-vm-up and train-vm-down. Off by
# default; every other plan and apply passes whether it exists now (the
# Makefile's TRAIN_VM_READ), so a routine apply never creates or deletes it.
#
#   on demand   STANDARD by default: train.py saves adapter-latest only, no
#               optimizer state and no --resume, so a Spot preemption loses
#               the run. The A100 40 GB has on-demand quota in int
#               (NVIDIA_A100_GPUS 16, read 2026-10-04) and needs no request.
#               SPOT for the H100 (its 1-GPU shape takes no on-demand VM):
#               the ~1 h runs where a preemption costs little.
#   STOP        on preemption, and when max_run_hours runs out, the VM stops:
#               the GPU stops billing at once; the disk (base weights, the uv
#               environment, adapters not yet synced) stays, at about
#               US$0.05-0.08 an hour, and train-vm-up starts it again where it
#               was. The disk is deleted by train-vm-down at the end of every
#               run either way (auto_delete).
#   hard stop   max_run_duration: Google stops the VM max_run_hours (12) after
#               each start, whatever runs on it — a forgotten VM stops billing
#               for its GPU even if nobody looks.
#   bootstrap   the startup script installs uv, unpacks the lab kit from the
#               bucket (or clones lab_repo), builds lab/.venv-train, puts the
#               base weights in /work/hf and runs train-gpu-check; the dataset
#               is copied by hand from the bucket (README.md).

locals {
  # The machine types this root allows: one GPU each, the GPU implied by the
  # type (no guest_accelerator). What differs between them is here; the
  # image, the network, the account, the hard stop and the startup script are
  # the same on both. Zones: where the type is offered in us-central1
  # (Google's GPU regions page, read 2026-10-04: A2 Standard and A3 High in
  # a, b, c). Prices: us-central1, whole VM, per hour. The A3's are from the
  # Cloud Billing catalog (thermometer, 2026-10-03); the A2's are published
  # list prices read on 2026-10-01 (not catalog-read: confirm in the console
  # before a long run). Spot prices move.
  shapes = {
    # 1 x A100 40 GB (SXM), 12 vCPU, 85 GB of RAM. It takes Persistent Disk
    # (pd-balanced, pd-ssd, pd-standard), no Hyperdisk Balanced: an SSD
    # Persistent Disk (200 GB: up to 6,000 read IOPS and 240 MiB/s, from its
    # size alone). No Local SSD unless asked for; none is.
    "a2-highgpu-1g" = {
      gpu          = "A100 40 GB"
      zones        = ["us-central1-a", "us-central1-b", "us-central1-c"]
      disk_type    = "pd-ssd"
      spot_only    = false
      quota        = { SPOT = "PREEMPTIBLE_NVIDIA_A100_GPUS (64 in int)", STANDARD = "NVIDIA_A100_GPUS (16 in int)" }
      usd_per_hour = { SPOT = 1.10, STANDARD = 3.67 }
    }
    # 1 x H100 SXM (80 GB), 26 vCPU, 234 GB of RAM. Spot or Flex-start only:
    # Google's rule for the 1-, 2- and 4-GPU A3 High (Flex-start is made with
    # gcloud, README.md). It takes pd-balanced, pd-ssd or Hyperdisk Balanced
    # (up to 120,000 IOPS and 1,800 MiB/s on this type). Google attaches
    # 2 x 375 GB of Local SSD to every A3 (scratch_disk below); nothing uses
    # them.
    "a3-highgpu-1g" = {
      gpu          = "H100 (80 GB)"
      zones        = ["us-central1-a", "us-central1-b", "us-central1-c"]
      disk_type    = "hyperdisk-balanced"
      spot_only    = true
      quota        = { SPOT = "PREEMPTIBLE_NVIDIA_H100_GPUS (64 in int)", STANDARD = "none: Spot or Flex-start only" }
      usd_per_hour = { SPOT = 6.571, STANDARD = 10.979 }
    }
  }
  shape     = local.shapes[var.machine_type]
  hyperdisk = local.shape.disk_type == "hyperdisk-balanced"

  vm_usd_per_hour = { for k, v in local.shapes : k => v.usd_per_hour }
  # Hyperdisk Balanced, us-central1: US$0.08/GiB-month, IOPS above 3,000 at
  # US$0.005 and MiB/s above 140 at US$0.04 a month each; SSD Persistent
  # Disk US$0.17/GiB-month, its performance included (730 h/month).
  disk_usd_per_hour = (local.hyperdisk
    ? var.boot_disk_gb * 0.08 + (var.boot_disk_iops - 3000) * 0.005 + (var.boot_disk_throughput - 140) * 0.04
    : var.boot_disk_gb * 0.17
  ) / 730
}

resource "google_compute_instance" "train" {
  count = var.training_vm_enabled ? 1 : 0

  name         = local.vm_name
  zone         = var.zone
  machine_type = var.machine_type # A2 and A3 types carry their GPU: no guest_accelerator
  description  = "Sous Chef training (infra/envs/train): the lab's int data, internal only"
  labels       = merge(local.labels, { stage = var.stage })

  deletion_protection       = false
  allow_stopping_for_update = true # a change of shape stops, resizes, starts

  boot_disk {
    auto_delete = true # the VM's disk goes with it (train-vm-down)
    initialize_params {
      image                  = var.image
      type                   = local.shape.disk_type # SSD PD on A2 (no Hyperdisk Balanced there), Hyperdisk Balanced on A3
      size                   = var.boot_disk_gb
      provisioned_iops       = local.hyperdisk ? var.boot_disk_iops : null # a Persistent Disk's follow its size
      provisioned_throughput = local.hyperdisk ? var.boot_disk_throughput : null
      labels                 = local.labels
    }
  }

  network_interface {
    subnetwork = google_compute_subnetwork.train.id
    nic_type   = "GVNIC"
    stack_type = "IPV4_ONLY"
    # no access_config: no external address; out through Cloud NAT
  }

  scheduling {
    provisioning_model          = var.provisioning_model
    preemptible                 = var.provisioning_model == "SPOT"
    automatic_restart           = false       # Spot requires it; a STANDARD VM that stops stays stopped
    on_host_maintenance         = "TERMINATE" # GPUs cannot live-migrate
    instance_termination_action = var.termination_action

    max_run_duration {
      seconds = var.max_run_hours * 3600
    }

    # A3 VMs carry Local SSDs (scratch_disk below), and with max_run_duration
    # set the API refuses to create one without saying what happens to them at
    # the stop ("undefined value for --discard-local-ssds-at-termination-
    # timestamp", measured 2026-10-04). Discard: nothing lives there, /work is
    # on the boot disk.
    dynamic "on_instance_stop_action" {
      for_each = startswith(var.machine_type, "a3-") ? [1] : []
      content {
        discard_local_ssd = true
      }
    }
  }

  # Never consume a reservation (there are none).
  reservation_affinity {
    type = "NO_RESERVATION"
  }

  service_account {
    email  = google_service_account.train.email
    scopes = ["cloud-platform"] # IAM decides: the bucket, logs and metrics only
  }

  shielded_instance_config {
    enable_secure_boot          = var.secure_boot
    enable_vtpm                 = true
    enable_integrity_monitoring = true
  }

  metadata = {
    enable-oslogin         = "TRUE" # SSH keys come from IAM (OS Login), never from metadata
    block-project-ssh-keys = "TRUE"
    serial-port-enable     = "FALSE"
    startup-script         = file("${path.module}/startup.sh")
    sous-chef-bucket       = google_storage_bucket.train.name
    sous-chef-uv-version   = var.uv_version
    sous-chef-base-model   = var.base_model
    sous-chef-lab-repo     = var.lab_repo
    sous-chef-lab-ref      = var.lab_ref
  }

  lifecycle {
    # A new image in the family must not replace a running VM (and its disk).
    # scratch_disk: Google attaches Local SSDs to every A3 VM by itself
    # (2 x 375 GB), and the provider reads them back into scratch_disk,
    # which forces a new VM on any difference: without this, the plan after
    # the create (the next train-vm-up after a preemption, or a routine
    # plan/apply) would replace the VM and delete its disk. Nothing uses
    # them, and Google wipes them at every stop. (A change that needs the VM
    # stopped, allow_stopping_for_update, fails on the A3: the provider's
    # stop cannot say to discard Local SSDs. train-vm-down then train-vm-up
    # instead.)
    ignore_changes = [boot_disk[0].initialize_params[0].image, scratch_disk]

    precondition {
      condition     = contains(local.shape.zones, var.zone)
      error_message = "${var.machine_type} is not offered in ${var.zone}: set zone to one of ${join(", ", local.shape.zones)} in terraform.tfvars."
    }
    precondition {
      condition     = !local.shape.spot_only || var.provisioning_model == "SPOT"
      error_message = "${var.machine_type} runs only as Spot or Flex-start (Google's rule for the 1-, 2- and 4-GPU A3 High): provisioning_model must be SPOT; a Flex-start VM is made with gcloud (README.md)."
    }
    precondition {
      condition     = !local.hyperdisk || var.boot_disk_iops <= 500 * var.boot_disk_gb
      error_message = "boot_disk_iops may be at most 500 per GiB of boot_disk_gb (Hyperdisk Balanced)."
    }
    precondition {
      condition     = !local.hyperdisk || (var.boot_disk_throughput <= var.boot_disk_iops / 4 && var.boot_disk_throughput >= max(140, var.boot_disk_iops / 256))
      error_message = "boot_disk_throughput must lie between max(140, IOPS/256) and IOPS/4 MiB/s (Hyperdisk Balanced)."
    }
  }

  # Everything the VM stands on, the firewall rules included. train-vm-up is
  # a targeted apply, which creates only what the VM depends on: after a
  # destroy of the network (NAT and all three rules gone, the bucket kept) it
  # must bring back the IAP SSH rule and the egress deny too, or the VM would
  # come up unreachable and with the implied allow-all egress.
  depends_on = [
    google_compute_router_nat.train, # the startup script's uv and pip downloads go out through NAT
    google_compute_firewall.iap_ssh,
    google_compute_firewall.egress_web,
    google_compute_firewall.egress_deny,
    google_storage_bucket_iam_binding.object_admin,
    google_project_iam_member.train_writer,
  ]
}

# train.py keeps no optimizer state and has no --resume (lab/souschef_lab/
# trainkit.py): a Spot preemption loses the run. Every plan says so when the
# VM is Spot, so the choice is a conscious one (the H100 has no other way).
check "spot_has_no_resume" {
  assert {
    condition     = !(var.training_vm_enabled && var.provisioning_model == "SPOT")
    error_message = "The VM is Spot (${var.machine_type}, ${local.shape.usd_per_hour.SPOT} USD/h): a preemption loses the run, since train.py has no --resume. Fine for a ~1 h run; for a multi-hour SFT use a2-highgpu-1g with provisioning_model = \"STANDARD\" (on-demand quota ${local.shapes["a2-highgpu-1g"].quota.STANDARD})."
  }
}
