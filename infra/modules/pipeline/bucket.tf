# The pipeline bucket (DESIGN §8.2 I2): one shared state for both workflows
# and every job. Layout:
#
#   infra/config.json, infra/cloudbuild-model.json   Terraform's (below)
#   config/thresholds.json                           optional override (by hand)
#   base/unsloth-qwen3.5-9b-hf.tar(.sha256)          copied once (RETRAIN.md R0)
#   engine/index.json, engine/<sha>/...              lab/tools/engine-vendor.sh
#   manifests/, state/, baselines/, cases/           the pipeline's memory
#   runs/<rt-run>/..., drift/<dr-run>/...            per run, expire
#   wf/outbox/, wf/runs/<rt-run>/...                 the workflow's own (outbox,
#                                                    decide.json, rollout.json);
#                                                    no job account writes wf/
#
#   private     uniform bucket-level access; public access prevention
#               enforced; object roles only through iam.tf's members
#   expiring    runs/, wf/runs/ and drift/ after run_retention_days (180); nothing
#               else ever expires (the case pool, the baselines and the
#               base weights must outlive any run)
#   soft delete the default 7 days stays on: a mistaken delete of state/ or
#               cases/ can be undone (gcloud storage restore)
resource "google_storage_bucket" "pipeline" {
  project       = var.project_id
  name          = local.bucket_name
  location      = var.bucket_location
  storage_class = "STANDARD"
  labels        = local.labels

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = var.bucket_force_destroy

  versioning {
    enabled = false
  }

  lifecycle_rule {
    condition {
      age            = var.run_retention_days
      matches_prefix = ["runs/", "wf/runs/", "drift/"]
    }
    action {
      type = "Delete"
    }
  }

  # Parts of an interrupted multipart upload (checkpoint rsyncs from a
  # preempted trainer) are billed until aborted.
  lifecycle_rule {
    condition {
      age = 1
    }
    action {
      type = "AbortIncompleteMultipartUpload"
    }
  }

  depends_on = [google_project_service.pipeline]
}

# What the workflows read at start: names, URLs, regions, switches, the
# training ladder. A config change is a Terraform apply of this object, never
# a workflow redeploy. No secret is in it.
resource "google_storage_bucket_object" "config" {
  bucket       = google_storage_bucket.pipeline.name
  name         = "infra/config.json"
  content_type = "application/json"
  content      = jsonencode(local.pipeline_config)
}

# The model image build (cloudbuild-model.yaml, the single source; the
# playbook submits the same file by hand). The retrain workflow reads this
# JSON form and fills in the run's source, image and substitutions.
#
# The context (runs/<id>/model/context.tar.gz) is job-written, so the spec pins
# the sha256 of the model-image files the build runs or ships; the build's
# first step refuses a context that differs (cloudbuild-model.yaml).
locals {
  model_image_dir    = "${path.module}/../../../lab/model-image"
  model_image_pinned = ["Dockerfile", ".dockerignore", "entrypoint.sh", "model-context.sh"]
  model_image_sums   = join(";", [for f in local.model_image_pinned : "${filesha256("${local.model_image_dir}/${f}")}  model-image/${f}"])
}

resource "google_storage_bucket_object" "cloudbuild_model" {
  bucket       = google_storage_bucket.pipeline.name
  name         = "infra/cloudbuild-model.json"
  content_type = "application/json"
  content      = jsonencode(yamldecode(replace(file("${path.module}/cloudbuild-model.yaml"), "__MODEL_IMAGE_SHA256SUMS__", local.model_image_sums)))
}
