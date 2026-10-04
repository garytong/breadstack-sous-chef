# The identity the training VM runs as. It reads and writes the training
# bucket and nothing else of the project's data: no project role but the two
# writer roles below, which can only add log entries and metric points (they
# are project-level by nature: Cloud Logging and Monitoring have no narrower
# place to grant them). Without its own account the VM would run as the
# default compute account, which holds Editor.
resource "google_service_account" "train" {
  account_id   = "sous-chef-train"
  display_name = "Sous Chef training VM (the training bucket only)"
  description  = "infra/envs/train: objectAdmin on ${local.bucket_name}, log and metric writer; nothing else"

  depends_on = [google_project_service.train]
}

# Authoritative for objectAdmin on this bucket: the VM's account and the
# operators, and nobody else — a grant added by hand (console, gcloud) is
# removed by the next apply. The legacy bucket roles Google gives every new
# bucket (project owners and editors: legacyBucketOwner; viewers:
# legacyBucketReader) are left as they are: with them project members see
# the bucket and can list object names, but not read an object. Reading
# needs this binding (project Owner alone no longer implies object access).
resource "google_storage_bucket_iam_binding" "object_admin" {
  bucket  = google_storage_bucket.train.name
  role    = "roles/storage.objectAdmin"
  members = concat(["serviceAccount:${google_service_account.train.email}"], var.operators)
}

resource "google_project_iam_member" "train_writer" {
  for_each = toset(["roles/logging.logWriter", "roles/monitoring.metricWriter"])
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.train.email}"
}
