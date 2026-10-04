# The one place the lab's training data lives in the cloud: the dataset
# pushed from the workstation, the lab kit's code, the base weights cached
# once, and the adapters and exports the VM writes for the workstation to
# pull back. It holds the int collections and weights trained on them:
# internal only.
#
#   private     uniform bucket-level access; public access prevention
#               enforced, so no allUsers / allAuthenticatedUsers binding
#               can ever take effect; the only object roles are the
#               authoritative binding in iam.tf (the VM's account and
#               var.operators)
#   encrypted   Google-managed keys by default; var.kms_key_name adds a
#               customer-managed key (CMEK)
#   short-lived data/, kit/ and runs/ objects are deleted object_expiry_days
#               (30) after they were written; base/ (public weights, 19 GB
#               that take a while to upload again) after base_expiry_days
#   deletable   no versioning and no soft delete: a deleted object is gone at
#               once (the default soft delete would keep it 7 days, still
#               billed and still restorable)
#
# Layout (make -C lab train-remote-push / train-remote-pull and the startup
# script follow it):
#   data/<set>/…       pushed from lab/data/<set>/ (dataset/, pairs/, cases)
#   kit/lab.tar.gz     the lab's working tree (code, fixtures, pyproject,
#                      constraints; no data, venvs or runs), one fixed name
#                      the startup script unpacks at every boot
#   base/<slug>-hf.tar the base model's Hugging Face cache folder
#                      (models--unsloth--Qwen3.5-9B/…), tarred once from the
#                      workstation; the VM extracts it into $HF_HOME/hub
#   runs/adapters/<name>, runs/exports/<name>
#                      written on the VM (gcloud storage rsync), pulled back
resource "google_storage_bucket" "train" {
  name          = local.bucket_name
  location      = "US-CENTRAL1"
  storage_class = "STANDARD"
  labels        = local.labels

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = var.bucket_force_destroy

  versioning {
    enabled = false
  }

  soft_delete_policy {
    retention_duration_seconds = 0
  }

  # "N days after creation or last update": for GCS the same rule, age.
  # Objects cannot be modified in place; rewriting one (a re-push of a
  # changed file) creates it anew, so age counts from the last write.
  # Deletion runs asynchronously, usually within a day of qualifying.
  lifecycle_rule {
    condition {
      age            = var.object_expiry_days
      matches_prefix = ["data/", "kit/", "runs/"]
    }
    action {
      type = "Delete"
    }
  }

  # The base weights are public and large: kept longer, so a run a few weeks
  # later does not upload 19 GB again (train-remote-push skips a tar that is
  # already there).
  lifecycle_rule {
    condition {
      age            = var.base_expiry_days
      matches_prefix = ["base/"]
    }
    action {
      type = "Delete"
    }
  }

  # Parts of an interrupted multipart upload are billed until aborted.
  lifecycle_rule {
    condition {
      age = 1
    }
    action {
      type = "AbortIncompleteMultipartUpload"
    }
  }

  dynamic "encryption" {
    for_each = var.kms_key_name == "" ? [] : [var.kms_key_name]
    content {
      default_kms_key_name = encryption.value
    }
  }

  depends_on = [google_project_service.train]
}
