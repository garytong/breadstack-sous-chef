# The eval and candidate GPU services (DESIGN §3.8): the production model
# service's shape (modules/model-gpu) with min 0, so drift exams and paired
# evals never touch the slot merchants use, and an idle service costs
# nothing.
#
#   sous-chef-model-eval  re-pointed by the workflows at the production
#                         service's current image DIGEST before every drift
#                         check and paired eval ("current")
#   sous-chef-model-cand  re-pointed at model:c-<run id> by the retrain
#                         workflow ("candidate")
#
# Created with the production image (var.prod_model_image, which exists);
# the pipeline owns the image afterwards (ignore_changes). Invokers: the
# drift runner and the workflow only (iam.tf, authoritative). Each holds at
# most one L4: with the production service that is 3, today's whole quota in
# us-east4 (measured), so the rollout waits for these two to idle unless
# l4_headroom says the quota is 5.
resource "google_cloud_run_v2_service" "gpu" {
  for_each = local.service_names

  project             = var.project_id
  name                = each.value
  location            = var.gpu_region
  deletion_protection = false
  ingress             = "INGRESS_TRAFFIC_ALL" # IAM is the barrier, as on the production service
  labels              = local.labels

  template {
    service_account                  = var.model_service_account
    execution_environment            = "EXECUTION_ENVIRONMENT_GEN2"
    max_instance_request_concurrency = 4
    timeout                          = "300s"
    gpu_zonal_redundancy_disabled    = true

    node_selector {
      accelerator = "nvidia-l4"
    }

    scaling {
      min_instance_count = 0
      max_instance_count = 1
    }

    containers {
      image = var.prod_model_image

      resources {
        limits = {
          cpu              = "4"
          memory           = "16Gi"
          "nvidia.com/gpu" = "1"
        }
        cpu_idle = false # a GPU service must use instance-based billing
      }

      ports {
        container_port = 8080
      }

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
