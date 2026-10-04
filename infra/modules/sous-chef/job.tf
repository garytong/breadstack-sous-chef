# Cloud Run jobs. Shape copied from breadstack-thermometer/infra/job.tf.
#
# sous-chef-migrate    the service image run with -migrate, executed by the
#                      deploy step after the image is pushed and before the
#                      service rolls to it. max_retries = 0: a migration that
#                      failed half-way needs a person, not a second attempt.
# sous-chef-retention  the image run with -retention, nightly from Cloud
#                      Scheduler (README section 7: traces > 90 d, ended
#                      conversations > 90 d, shipped usage rows > 30 d).
#
# Both run as the runtime service account: the same Cloud SQL socket and the
# same DATABASE_URL secret, nothing more.

locals {
  jobs = {
    migrate = {
      name    = "sous-chef-migrate"
      args    = ["-migrate"]
      timeout = "3600s"
    }
    retention = {
      name    = "sous-chef-retention"
      args    = ["-retention"]
      timeout = "1800s"
    }
  }
}

resource "google_cloud_run_v2_job" "migrate" {
  project             = var.project_id
  name                = local.jobs.migrate.name
  location            = var.region
  deletion_protection = false

  template {
    template {
      service_account = google_service_account.run.email
      timeout         = local.jobs.migrate.timeout
      max_retries     = 0

      volumes {
        name = "cloudsql"
        cloud_sql_instance {
          instances = [google_sql_database_instance.sous_chef.connection_name]
        }
      }

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
        args  = local.jobs.migrate.args

        resources {
          limits = {
            cpu    = "1"
            memory = "512Mi"
          }
        }

        volume_mounts {
          name       = "cloudsql"
          mount_path = "/cloudsql"
        }

        dynamic "env" {
          for_each = local.common_env
          content {
            name  = env.key
            value = env.value
          }
        }

        env {
          name = "DATABASE_URL"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.database_url.secret_id
              version = "latest"
            }
          }
        }
      }
    }
  }

  # CI re-points the job with `gcloud run jobs update ... --image` (see
  # run.tf for the full note); Terraform ignores the image afterwards.
  lifecycle {
    ignore_changes = [
      template[0].template[0].containers[0].image,
      template[0].labels,
      template[0].annotations,
      client,
      client_version,
    ]
  }

  depends_on = [
    google_project_service.services,
    google_secret_manager_secret_version.database_url,
    google_secret_manager_secret_iam_member.run_database_url,
    google_project_iam_member.run_cloudsql,
    # The DSN names this database and user; a targeted apply of the job on a
    # fresh project must create them too.
    google_sql_database.sous_chef,
    google_sql_user.sous_chef,
  ]
}

resource "google_cloud_run_v2_job" "retention" {
  project             = var.project_id
  name                = local.jobs.retention.name
  location            = var.region
  deletion_protection = false

  template {
    template {
      service_account = google_service_account.run.email
      timeout         = local.jobs.retention.timeout
      max_retries     = 1

      volumes {
        name = "cloudsql"
        cloud_sql_instance {
          instances = [google_sql_database_instance.sous_chef.connection_name]
        }
      }

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
        args  = local.jobs.retention.args

        resources {
          limits = {
            cpu    = "1"
            memory = "512Mi"
          }
        }

        volume_mounts {
          name       = "cloudsql"
          mount_path = "/cloudsql"
        }

        dynamic "env" {
          for_each = local.common_env
          content {
            name  = env.key
            value = env.value
          }
        }

        env {
          name = "DATABASE_URL"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.database_url.secret_id
              version = "latest"
            }
          }
        }
      }
    }
  }

  # CI re-points the job with `gcloud run jobs update ... --image` (see
  # run.tf for the full note); Terraform ignores the image afterwards.
  lifecycle {
    ignore_changes = [
      template[0].template[0].containers[0].image,
      template[0].labels,
      template[0].annotations,
      client,
      client_version,
    ]
  }

  depends_on = [
    google_project_service.services,
    google_secret_manager_secret_version.database_url,
    google_secret_manager_secret_iam_member.run_database_url,
    google_project_iam_member.run_cloudsql,
    google_sql_database.sous_chef,
    google_sql_user.sous_chef,
  ]
}

# Nightly trigger: Cloud Scheduler calls the Run Admin API's jobs.run with an
# OAuth token of the scheduler SA (roles/run.invoker on the job, iam.tf).
resource "google_cloud_scheduler_job" "retention" {
  project     = var.project_id
  name        = "sous-chef-retention"
  region      = var.region
  description = "Runs the sous-chef-retention Cloud Run job (trace/conversation/usage pruning)"
  schedule    = var.retention_schedule
  time_zone   = var.scheduler_timezone

  attempt_deadline = "320s"

  retry_config {
    retry_count = 1
  }

  http_target {
    http_method = "POST"
    uri         = "https://${var.region}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${var.project_id}/jobs/${google_cloud_run_v2_job.retention.name}:run"

    oauth_token {
      service_account_email = google_service_account.scheduler.email
    }
  }

  depends_on = [
    google_project_service.services,
    google_cloud_run_v2_job_iam_member.scheduler_runs_retention,
  ]
}
