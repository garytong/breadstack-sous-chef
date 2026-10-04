# Phase 5 stub: an in-house model served by Ollama from a private Cloud Run
# GPU service that sous-chef calls over a messages-compatible route with a
# Google ID token (README section 5, auth: id-token). Shape copied from
# breadstack-thermometer/infra/sourdough.tf (Muffin/Sourdough), reduced to
# what the stub needs: repository, roleless service account, GPU service,
# IAM-only access.
#
# IAM is the only barrier (ingress ALL; Cloud Run's front end rejects every
# request without an invoker's ID token, and rejected requests are not
# billed); behind it is Ollama's whole API, so the invoker binding is
# authoritative: only sous-chef's runtime SA may call it.
#
# Not applied by default: the env roots instantiate this module only when
# gpu_model_enabled = true. A service pointing at an image not yet pushed
# fails the apply — push first (`<repo>/model:<tag>`), then flip the flag.

terraform {
  required_version = ">= 1.7"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

locals {
  repo  = "${var.repo_location}-docker.pkg.dev/${var.project_id}/${var.repository_id}"
  image = "${local.repo}/model:${var.image_tag}"
}

# Private repository for Ollama + weights images. Immutable tags: a tag
# always means the same weights.
resource "google_artifact_registry_repository" "model" {
  project       = var.project_id
  repository_id = var.repository_id
  format        = "DOCKER"
  location      = var.repo_location
  description   = "sous-chef in-house model images (Ollama + trained weights; private)"

  docker_config {
    immutable_tags = true
  }
}

# The identity the model runs as, with no roles: it calls nothing and reads
# nothing (the weights are in the image).
resource "google_service_account" "model" {
  project      = var.project_id
  account_id   = "sous-chef-model-run"
  display_name = "sous-chef in-house model Cloud Run GPU service (no roles)"
}

resource "google_cloud_run_v2_service" "model" {
  count = var.serve ? 1 : 0

  project             = var.project_id
  name                = "sous-chef-model"
  location            = var.region
  deletion_protection = false
  ingress             = "INGRESS_TRAFFIC_ALL" # IAM is the barrier (header note)

  template {
    service_account                  = google_service_account.model.email
    execution_environment            = "EXECUTION_ENVIRONMENT_GEN2"
    max_instance_request_concurrency = 4
    timeout                          = "300s"
    gpu_zonal_redundancy_disabled    = true

    node_selector {
      accelerator = var.gpu
    }

    scaling {
      min_instance_count = var.min_instances
      max_instance_count = 1
    }

    containers {
      image = local.image

      resources {
        limits = {
          cpu              = var.cpu
          memory           = var.memory
          "nvidia.com/gpu" = "1"
        }
        cpu_idle = false # a GPU service must use instance-based billing
      }

      ports {
        container_port = 8080 # the image's OLLAMA_HOST
      }

      # "Ollama is running" on GET / as soon as the server listens, before
      # the model preload finishes; a request arriving during the load waits
      # inside Ollama.
      startup_probe {
        http_get {
          path = "/"
          port = 8080
        }
        period_seconds    = 2
        timeout_seconds   = 1
        failure_threshold = 60
      }
    }
  }

  lifecycle {
    # The API fills in a top-level scaling block the template never sets;
    # provider 6.x reports that as a change on every plan (thermometer note).
    ignore_changes = [scaling]
  }
}

# Authoritative run.invoker: sous-chef's runtime SA alone. Any other invoker
# added by hand is removed on the next apply.
resource "google_cloud_run_v2_service_iam_binding" "invoker" {
  count = var.serve ? 1 : 0

  project  = var.project_id
  name     = google_cloud_run_v2_service.model[0].name
  location = google_cloud_run_v2_service.model[0].location
  role     = "roles/run.invoker"
  members  = ["serviceAccount:${var.invoker_service_account}"]
}
