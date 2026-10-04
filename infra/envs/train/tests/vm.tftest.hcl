# Offline checks of the training VM's shape per machine type (vm.tf's
# local.shapes) against mock providers: nothing is read from or applied to
# any project, and no credentials are needed.
#
#   terraform init -backend=false && terraform test -var-file=terraform.tfvars.example
#
# What it holds the root to: the default is an on-demand A100 40 GB in
# us-central1-a on a 200 GB SSD Persistent Disk with the 12 h hard stop; the
# A3 High takes a Hyperdisk Balanced boot disk of the same size and runs only
# as Spot; each type only in us-central1-a/b/c; the Local SSDs Google
# attaches to the A3 never force a new VM; and with the VM off nothing about
# its shape is checked or planned.

mock_provider "google" {}
mock_provider "google" {
  alias = "billing"
}

# The default: the A100 40 GB on demand, as the example file sets it.
run "a2_on_demand_from_the_file" {
  command = plan

  variables {
    training_vm_enabled = true
  }

  assert {
    condition     = google_compute_instance.train[0].machine_type == "a2-highgpu-1g" && google_compute_instance.train[0].zone == "us-central1-a"
    error_message = "terraform.tfvars.example shapes an a2-highgpu-1g in us-central1-a."
  }
  assert {
    condition = (
      google_compute_instance.train[0].boot_disk[0].initialize_params[0].type == "pd-ssd" &&
      google_compute_instance.train[0].boot_disk[0].initialize_params[0].size == 200
    )
    error_message = "The A2 boots from a 200 GB SSD Persistent Disk (it takes no Hyperdisk Balanced; provisioned IOPS/throughput are null, computed after apply)."
  }
  assert {
    condition = (
      google_compute_instance.train[0].scheduling[0].provisioning_model == "STANDARD" &&
      !google_compute_instance.train[0].scheduling[0].preemptible &&
      !google_compute_instance.train[0].scheduling[0].automatic_restart &&
      google_compute_instance.train[0].scheduling[0].on_host_maintenance == "TERMINATE" &&
      google_compute_instance.train[0].scheduling[0].instance_termination_action == "STOP" &&
      google_compute_instance.train[0].scheduling[0].max_run_duration[0].seconds == 12 * 3600
    )
    error_message = "On demand, not preemptible, STOP, TERMINATE on maintenance and the 12 h hard stop."
  }
  assert {
    condition     = length(google_compute_instance.train[0].scratch_disk) == 0 && length(google_compute_instance.train[0].guest_accelerator) == 0
    error_message = "No scratch disk and no guest_accelerator in the config: the machine type carries the GPU."
  }
  assert {
    condition     = google_compute_instance.train[0].labels["stage"] == "smoke" && google_compute_instance.train[0].labels["purpose"] == "sous-chef-train"
    error_message = "Labelled purpose=sous-chef-train and stage=smoke (the budget counts the first)."
  }
  assert {
    condition     = google_compute_instance.train[0].metadata["sous-chef-base-model"] == "unsloth/Qwen3.5-9B" && google_compute_instance.train[0].metadata["enable-oslogin"] == "TRUE"
    error_message = "The startup script's base model is unsloth/Qwen3.5-9B; OS Login is on."
  }
}

# The H100 fallback: Hyperdisk Balanced, Spot.
run "a3_spot_hyperdisk" {
  command = plan

  variables {
    training_vm_enabled = true
    machine_type        = "a3-highgpu-1g"
    zone                = "us-central1-b"
    provisioning_model  = "SPOT"
  }

  # Spot: the no-resume warning (vm.tf, check "spot_has_no_resume") is expected.
  expect_failures = [check.spot_has_no_resume]

  assert {
    condition = (
      google_compute_instance.train[0].boot_disk[0].initialize_params[0].type == "hyperdisk-balanced" &&
      google_compute_instance.train[0].boot_disk[0].initialize_params[0].size == 200 &&
      google_compute_instance.train[0].boot_disk[0].initialize_params[0].provisioned_iops == 6000 &&
      google_compute_instance.train[0].boot_disk[0].initialize_params[0].provisioned_throughput == 600
    )
    error_message = "The A3 High takes a 200 GB Hyperdisk Balanced boot disk at 6,000 IOPS and 600 MiB/s."
  }
  assert {
    condition = (
      google_compute_instance.train[0].scheduling[0].provisioning_model == "SPOT" &&
      google_compute_instance.train[0].scheduling[0].preemptible &&
      google_compute_instance.train[0].scheduling[0].instance_termination_action == "STOP"
    )
    error_message = "The H100 VM runs Spot with STOP."
  }
  assert {
    condition     = length(google_compute_instance.train[0].scratch_disk) == 0
    error_message = "The config asks for no Local SSD: Google attaches the A3's own, and lifecycle ignores them."
  }
}

run "prices_per_shape" {
  command = apply

  variables {
    training_vm_enabled = true
  }

  assert {
    condition     = output.vm_usd_per_hour.running == "3.72" && output.vm_usd_per_hour.stopped == "0.05"
    error_message = "The A100 on demand: US$3.67 plus the SSD Persistent Disk's ~US$0.05 an hour."
  }
}

run "a3_price" {
  command = apply

  variables {
    training_vm_enabled = true
    machine_type        = "a3-highgpu-1g"
    provisioning_model  = "SPOT"
  }

  # Spot: the no-resume warning (vm.tf, check "spot_has_no_resume") is expected.
  expect_failures = [check.spot_has_no_resume]

  assert {
    condition     = output.vm_usd_per_hour.running == "6.64" && output.vm_usd_per_hour.stopped == "0.07"
    error_message = "The A3 High on Spot: US$6.57 plus the 200 GB Hyperdisk's ~US$0.07 an hour."
  }
}

# The 1-GPU A3 High takes no on-demand VM.
run "a3_spot_only" {
  command = plan

  variables {
    training_vm_enabled = true
    machine_type        = "a3-highgpu-1g"
    provisioning_model  = "STANDARD"
  }

  expect_failures = [google_compute_instance.train]
}

# Only the three us-central1 zones with A2 Standard and A3 High.
run "no_zone_outside_the_three" {
  command = plan

  variables {
    zone = "us-central1-f"
  }

  expect_failures = [var.zone]
}

run "no_other_machine_type" {
  command = plan

  variables {
    machine_type = "g2-standard-8"
  }

  expect_failures = [var.machine_type]
}

# Spot on the A2 is allowed (the cheap smoke), with the no-resume warning.
run "a2_spot_allowed" {
  command = plan

  variables {
    training_vm_enabled = true
    provisioning_model  = "SPOT"
  }

  # Spot: the no-resume warning (vm.tf, check "spot_has_no_resume") is expected.
  expect_failures = [check.spot_has_no_resume]

  assert {
    condition     = google_compute_instance.train[0].scheduling[0].provisioning_model == "SPOT" && google_compute_instance.train[0].scheduling[0].preemptible
    error_message = "The A2 may run as Spot."
  }
}

# With the VM off (every plan but vm-up's), its shape is neither planned
# nor checked: a wrong pair in the file fails only the vm-up.
run "vm_off_plans_no_vm" {
  command = plan

  variables {
    training_vm_enabled = false
    machine_type        = "a3-highgpu-1g"
    provisioning_model  = "STANDARD"
  }

  assert {
    condition     = length(google_compute_instance.train) == 0 && output.vm_name == null
    error_message = "training_vm_enabled off plans no VM."
  }
}

# The bucket: private, short-lived data, the base kept longer.
run "bucket_posture" {
  command = plan

  assert {
    condition = (
      google_storage_bucket.train.name == "breadstack-sous-chef-int-training" &&
      google_storage_bucket.train.location == "US-CENTRAL1" &&
      google_storage_bucket.train.uniform_bucket_level_access &&
      google_storage_bucket.train.public_access_prevention == "enforced" &&
      !google_storage_bucket.train.force_destroy
    )
    error_message = "gs://breadstack-sous-chef-int-training in US-CENTRAL1, uniform access, public access prevented, not force-destroyable."
  }
  assert {
    condition     = length([for r in google_storage_bucket.train.lifecycle_rule : r if anytrue([for a in r.action : a.type == "Delete"])]) == 2
    error_message = "Two Delete lifecycle rules: data/kit/runs at object_expiry_days, base/ at base_expiry_days."
  }
}
