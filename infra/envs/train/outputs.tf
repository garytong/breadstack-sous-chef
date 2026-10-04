output "bucket" {
  description = "The private training bucket (make -C lab train-remote-push / train-remote-pull)"
  value       = "gs://${google_storage_bucket.train.name}"
}

output "service_account" {
  description = "The training VM's identity: objectAdmin on the bucket, log and metric writer, nothing else"
  value       = google_service_account.train.email
}

output "vm_name" {
  description = "The training VM (null while training_vm_enabled is off)"
  value       = var.training_vm_enabled ? google_compute_instance.train[0].name : null
}

output "vm_zone" {
  description = "The training VM's zone (null while training_vm_enabled is off)"
  value       = var.training_vm_enabled ? google_compute_instance.train[0].zone : null
}

output "ssh_command" {
  description = "SSH to the VM through IAP with your OS Login (make -C lab train-vm-ssh does the same)"
  value       = var.training_vm_enabled ? "gcloud compute ssh ${local.vm_name} --project ${var.project_id} --zone ${var.zone} --tunnel-through-iap" : null
}

output "vm_usd_per_hour" {
  description = "About what the VM costs per hour while it runs, disk included (us-central1 list prices, vm.tf's local.shapes; Spot prices move). Stopped: the disk alone."
  value = var.training_vm_enabled ? {
    running = format("%.2f", local.vm_usd_per_hour[var.machine_type][var.provisioning_model] + local.disk_usd_per_hour)
    stopped = format("%.2f", local.disk_usd_per_hour)
  } : null
}

output "budget" {
  description = "The notify-only budget's name (null without budget_enabled)"
  value       = var.budget_enabled ? google_billing_budget.train[0].name : null
}
