provider "google" {
  project = var.project_id
  region  = local.region
  zone    = var.zone
}

# The Billing Budget API refuses end-user credentials without a quota
# project ("…requires a quota project, which is not set by default"), and
# this root is applied on a person's Application Default Credentials. This
# alias bills its calls to the project itself, where
# billingbudgets.googleapis.com is enabled (apis.tf). Only the budget uses it.
provider "google" {
  alias                 = "billing"
  project               = var.project_id
  region                = local.region
  user_project_override = true
  billing_project       = var.project_id
}

data "google_project" "this" {
  project_id = var.project_id
}

locals {
  # us-central1: the only region where this project can have the A2 Standard
  # (A100 40 GB, zones a, b, c) and the A3 High (H100, zones a, b, c). The
  # service's home region, northamerica-northeast2, carries G2 (L4) only, and
  # the L4's 24 GB does not hold the measured recipe (26.8 GiB at 32k rows).
  # The int collection data leaves Canada for the run: the user approves that
  # before the first vm-up (README.md, "Before the first run").
  region = "us-central1"

  # On every resource that takes labels. The budget counts spend by
  # purpose=sous-chef-train (budget.tf); data=lab-int says what the bucket
  # and the disks hold: the lab's int collections and weights trained on
  # them — internal, not client data.
  labels = {
    purpose    = "sous-chef-train"
    data       = "lab-int"
    managed-by = "terraform-train"
  }

  bucket_name = "${var.project_id}-training"
  vm_name     = "sous-chef-train"
}
