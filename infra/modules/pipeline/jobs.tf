# Five Cloud Run jobs on the lab runtime image (images/lab-runtime; DESIGN
# §6.1, I3). One entry, `python -m souschef_lab.pipeline <cmd>` (the image's
# ENTRYPOINT); the workflow passes each stage's args and per-run env as a
# run override, so the template's args are only the default (`selftest`).
#
#   souschef-drift    drift poll | static | check | weekly | calibrate   (drift SA)
#   souschef-eval     the paired eval, report and decision payload        (drift SA)
#   souschef-prepare  rebase every family, author new-field cases         (author SA, + Anthropic key)
#   souschef-collect  teacher collection through int's Claude routes      (prep SA)
#   souschef-dataset  r4dataset + the parity gate + context.tgz           (prep SA)
#
# max_retries = 0: retries are the workflow's (attempt n/N in the failure
# message; an exit 2 — configuration — is never retried). Every workflow run
# re-points the job it runs at sous-chef-lab:<int service tag> first, so the
# image is ignored after creation (var.lab_image is only the first one).
#
# Secrets arrive as secret_key_ref env only, never on argv or in the image.

locals {
  jobs = {
    drift   = { sa = "drift", anthropic = false }
    eval    = { sa = "drift", anthropic = false }
    prepare = { sa = "author", anthropic = true }
    collect = { sa = "prep", anthropic = false }
    dataset = { sa = "prep", anthropic = false }
  }

  job_env = {
    PIPELINE_BUCKET      = "gs://${local.bucket_name}"
    GOOGLE_CLOUD_PROJECT = var.project_id
    SOUS_CHEF_URL        = var.int_service_url
    PIPELINE_CONFIG      = "gs://${local.bucket_name}/${google_storage_bucket_object.config.name}"
    PROD_MODEL_URL       = var.prod_model_url
    EVAL_MODEL_URL       = google_cloud_run_v2_service.gpu["eval"].uri
    CAND_MODEL_URL       = google_cloud_run_v2_service.gpu["cand"].uri
    PYTHONUNBUFFERED     = "1"
  }
}

resource "google_cloud_run_v2_job" "job" {
  for_each = local.jobs

  project             = var.project_id
  name                = local.job_names[each.key]
  location            = var.region
  deletion_protection = false
  labels              = local.labels

  template {
    task_count  = 1
    parallelism = 1

    template {
      service_account       = local.sa_email[each.value.sa]
      timeout               = "${var.jobs[each.key].timeout_s}s"
      max_retries           = 0
      execution_environment = "EXECUTION_ENVIRONMENT_GEN2"

      containers {
        image = var.lab_image
        args  = ["selftest"]

        resources {
          limits = {
            cpu    = var.jobs[each.key].cpu
            memory = var.jobs[each.key].memory
          }
        }

        dynamic "env" {
          for_each = merge(local.job_env, { PIPELINE_JOB = local.job_names[each.key] })
          content {
            name  = env.key
            value = env.value
          }
        }

        env {
          name = "SOUS_CHEF_KEY"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.pipeline["service_key"].secret_id
              version = "latest"
            }
          }
        }

        dynamic "env" {
          for_each = each.value.anthropic ? [1] : []
          content {
            name = "ANTHROPIC_API_KEY"
            value_source {
              secret_key_ref {
                secret  = var.anthropic_secret_id
                version = "latest"
              }
            }
          }
        }
      }
    }
  }

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
    google_secret_manager_secret_iam_member.service_key,
    google_secret_manager_secret_iam_member.anthropic,
    google_storage_bucket_iam_member.jobs_write,
  ]
}
