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

  # Candidate images (modules/pipeline: model:c-<run id>, ~5.4 GB each)
  # accumulate one per retrain round. With cleanup_candidates on, the policy
  # below names c-* versions older than candidate_retention_days, minus its
  # keep rules (the newest keep_recent versions, every hand-built r* tag and
  # every image ever rolled out: live-<run id>). It deletes NOTHING while
  # immutable_tags is true: "If a repository has immutable tags enabled,
  # tagged artifacts can't be deleted"
  # (https://docs.cloud.google.com/artifact-registry/docs/repositories/cleanup-policy).
  # Old candidates are pruned by hand (docs/RETRAIN.md R7 "Cleanup").
  cleanup_policy_dry_run = var.cleanup_candidates ? false : null

  dynamic "cleanup_policies" {
    for_each = var.cleanup_candidates ? [1] : []
    content {
      id     = "delete-old-candidates"
      action = "DELETE"
      condition {
        tag_state    = "TAGGED"
        tag_prefixes = ["c-"]
        older_than   = "${var.candidate_retention_days * 86400}s"
      }
    }
  }

  dynamic "cleanup_policies" {
    for_each = var.cleanup_candidates ? [1] : []
    content {
      id     = "keep-recent"
      action = "KEEP"
      most_recent_versions {
        keep_count = var.keep_recent
      }
    }
  }

  dynamic "cleanup_policies" {
    for_each = var.cleanup_candidates ? [1] : []
    content {
      id     = "keep-hand-built-and-live"
      action = "KEEP"
      condition {
        tag_state    = "TAGGED"
        tag_prefixes = ["r", "live-"]
      }
    }
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
    # - scaling: the API fills in a top-level scaling block the template
    #   never sets; provider 6.x reports that as a change on every plan
    #   (thermometer note).
    # - the image: from the first apply on, the pipeline (modules/pipeline,
    #   the souschef-retrain workflow's rollout on an Approve) and
    #   `make model-rollout TAG=<tag>` (rollbacks, manual rollouts) own it,
    #   so image_tag only sets the image the service is CREATED with; a
    #   later image_tag change plans nothing. Without this, the next apply
    #   would revert an approved candidate to the tfvar's tag.
    # - template annotations / labels, client, client_version: the rollout
    #   stamps souschef/candidate=<id>, gcloud stamps its client name and a
    #   nonce; Terraform sets none of its own.
    ignore_changes = [
      scaling,
      template[0].containers[0].image,
      template[0].annotations,
      template[0].labels,
      client,
      client_version,
    ]
  }
}

# Authoritative run.invoker: sous-chef's runtime SA plus extra_invokers (the
# Sous Chef pipeline's drift runner and workflow accounts, when the pipeline
# is on: the night-window drift check and the post-rollout warm-up call the
# production model). Any other invoker added by hand is removed on the next
# apply.
resource "google_cloud_run_v2_service_iam_binding" "invoker" {
  count = var.serve ? 1 : 0

  project  = var.project_id
  name     = google_cloud_run_v2_service.model[0].name
  location = google_cloud_run_v2_service.model[0].location
  role     = "roles/run.invoker"
  members  = concat(["serviceAccount:${var.invoker_service_account}"], [for e in var.extra_invokers : "serviceAccount:${e}"])
}
