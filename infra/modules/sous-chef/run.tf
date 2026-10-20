# The Cloud Run v2 service. Shape copied from
# breadstack-thermometer/infra/run.tf (template layout, cloudsql volume,
# secret_key_ref env, two-halves gating with dynamic "env", startup probe);
# scaling and ingress per the sous-chef task: CPU only, min 0 / max 4,
# concurrency 20, 600 s timeout for SSE, ingress internal + load balancer.
#
# PORT is reserved on Cloud Run (injected from container_port = 8090).
# Schema changes are the migrate job's (job.tf); the service boots with
# MIGRATE_AT_BOOT=off and refuses to start while a migration is pending.

locals {
  # Every secret-backed env the service always reads.
  always_secret_env = {
    DATABASE_URL      = google_secret_manager_secret.database_url.secret_id
    SERVICE_KEYS      = google_secret_manager_secret.user["sous-chef-service-keys"].secret_id
    ANTHROPIC_API_KEY = google_secret_manager_secret.user["sous-chef-anthropic-api-key"].secret_id
  }

  # Plain env shared by the service and the jobs.
  common_env = {
    ENV                  = var.env
    MIGRATE_AT_BOOT      = "off"
    TRACE_RETENTION_DAYS = tostring(var.trace_retention_days)
  }

  # Plain env the service alone reads (model routing, turn limits, shipping).
  # MODEL_ROUTE_ALIASES is set only while an alias exists: with none the
  # template (and the running revision) is exactly what it was before the
  # option existed.
  service_env = merge(local.common_env, {
    DEPLOY_NONCE          = var.deploy_nonce
    MODEL_ROUTES          = var.model_routes
    DEFAULT_ROUTE         = var.default_route
    MODEL_ROUTE_ALLOWLIST = var.model_route_allowlist
    MODEL_CALL_TIMEOUT    = var.model_call_timeout
    TURN_TTL              = var.turn_ttl
    MAX_CONCURRENT_TURNS  = tostring(var.max_concurrent_turns)
    THERMOMETER_API_URL   = var.thermometer_api_url
    }, length(var.model_route_aliases) > 0 ? {
    MODEL_ROUTE_ALIASES = jsonencode(var.model_route_aliases)
  } : {})
}

resource "google_cloud_run_v2_service" "sous_chef" {
  project             = var.project_id
  name                = "sous-chef"
  location            = var.region
  deletion_protection = false # stateless; the data lives in Cloud SQL
  ingress             = var.ingress

  template {
    service_account                  = google_service_account.run.email
    execution_environment            = "EXECUTION_ENVIRONMENT_GEN2"
    max_instance_request_concurrency = var.run_concurrency
    timeout                          = var.run_timeout

    scaling {
      min_instance_count = var.run_min_instances
      max_instance_count = var.run_max_instances
    }

    volumes {
      name = "cloudsql"
      cloud_sql_instance {
        instances = [google_sql_database_instance.sous_chef.connection_name]
      }
    }

    # Private-IP Cloud SQL is reachable only from inside the VPC: direct VPC
    # egress for private ranges alone, so model API calls still leave through
    # Cloud Run's own egress.
    dynamic "vpc_access" {
      for_each = var.sql_private_ip ? [1] : []
      content {
        egress = "PRIVATE_RANGES_ONLY"
        network_interfaces {
          network    = var.sql_private_network
          subnetwork = var.sql_private_subnetwork
        }
      }
    }

    containers {
      image = var.image

      resources {
        limits = {
          cpu    = var.run_cpu
          memory = var.run_memory
        }
        cpu_idle          = true # request-based billing: no background work between turns beyond the in-request reaper
        startup_cpu_boost = true
      }

      ports {
        container_port = 8090
      }

      volume_mounts {
        name       = "cloudsql"
        mount_path = "/cloudsql"
      }

      dynamic "env" {
        for_each = local.service_env
        content {
          name  = env.key
          value = env.value
        }
      }

      # Secret-backed env vars — each referenced secret must have >= 1 version
      # or Cloud Run will fail to start the revision (secrets.tf header).
      dynamic "env" {
        for_each = local.always_secret_env
        content {
          name = env.key
          value_source {
            secret_key_ref {
              secret  = env.value
              version = "latest"
            }
          }
        }
      }

      # Usage shipping to the thermometer: the URL is a plain tfvar (empty =
      # off) and the key is a secret referenced ONLY when the URL is set.
      dynamic "env" {
        for_each = var.thermometer_api_url != "" ? [1] : []
        content {
          name = "THERMOMETER_API_KEY"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.user["sous-chef-thermometer-api-key"].secret_id
              version = "latest"
            }
          }
        }
      }

      startup_probe {
        http_get {
          path = "/healthz"
        }
        initial_delay_seconds = 5
        period_seconds        = 5
        # 5 + 12x5 = 65 s: the pgx connect retry plus the pending-migration
        # check before the listener starts; a cold Cloud SQL socket on first
        # apply eats most of that.
        failure_threshold = 12
      }
    }
  }

  # CI rolls images (cloudbuild.yaml: `gcloud run services update sous-chef
  # --image ...`), so Terraform owns the template EXCEPT the image: without
  # this, the next `terraform apply -var=image=<older>` would revert what CI
  # deployed, and a plan after every CI deploy would show drift. `client` and
  # `client_version` are stamped by gcloud on each update and ignored for the
  # same reason, as are the template labels/annotations gcloud adds
  # (client.knative.dev/nonce, run.googleapis.com/client-name) — Terraform
  # sets none of its own. var.image therefore matters only when the service is
  # first created (or recreated); afterwards the image is whatever CI last
  # rolled. To force a revision from Terraform anyway, pass -var=deploy_nonce=....
  # After the first CI deploy, run `terraform plan`: it must be empty; anything
  # else gcloud stamps belongs in this list.
  lifecycle {
    ignore_changes = [
      template[0].containers[0].image,
      template[0].labels,
      template[0].annotations,
      client,
      client_version,
      # The API reports a service-level scaling block (manual/min instance
      # counts = 0) that this config never sets; without this, every plan
      # wants to null it (measured on the first int apply, 2026-10-03).
      scaling,
    ]
  }

  depends_on = [
    google_project_service.services,
    google_secret_manager_secret_version.database_url,
    google_secret_manager_secret_iam_member.run_database_url,
    google_secret_manager_secret_iam_member.run_user_secrets,
    google_project_iam_member.run_cloudsql,
    google_cloud_run_v2_job.migrate, # the job exists (and the deploy has run it) before the service boots against the schema
  ]
}

# Unauthenticated invocation (thermometer run.tf pattern). Off by default:
# with it off, Cloud Run's IAM check requires a Google ID token from an
# identity holding roles/run.invoker, which the launcher does not send
# (launcher/server/souschef_proxy.go TODO(iam)). int turns it on so the
# launcher — on GKE in another project, with no Workload Identity GSA yet —
# can reach the service with the service key alone (SERVICE_KEYS is the gate,
# and the service refuses every /v1/* call without it). prod keeps it off.
resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  count = var.public_invoker ? 1 : 0

  project  = var.project_id
  name     = google_cloud_run_v2_service.sous_chef.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}
