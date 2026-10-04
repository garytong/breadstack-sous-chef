# Secret Manager. Shape copied from breadstack-thermometer/infra/secrets.tf.
#
# Terraform creates the SECRETS, not their values, except sous-chef-database-url
# whose value Terraform owns (random password + connection name). The two
# others the service needs MUST be given a version by hand before the first
# deploy or Cloud Run refuses to start the revision (see infra/README.md,
# "Secrets to populate"):
#   sous-chef-anthropic-api-key   ANTHROPIC_API_KEY
#   sous-chef-service-keys        SERVICE_KEYS (comma-separated; two = rotation)
# sous-chef-thermometer-api-key is referenced only when thermometer_api_url is
# set (two-halves gate), so an environment without usage shipping never blocks
# on a version-less secret. Secret Manager rejects empty payloads, so there is
# no placeholder version either.

locals {
  user_secret_ids = [
    "sous-chef-anthropic-api-key",
    "sous-chef-service-keys",
    "sous-chef-thermometer-api-key",
  ]
}

resource "google_secret_manager_secret" "user" {
  for_each  = toset(local.user_secret_ids)
  project   = var.project_id
  secret_id = each.value

  replication {
    auto {}
  }

  depends_on = [google_project_service.services]
}

resource "google_secret_manager_secret" "database_url" {
  project   = var.project_id
  secret_id = "sous-chef-database-url"

  replication {
    auto {}
  }

  depends_on = [google_project_service.services]
}

resource "google_secret_manager_secret_version" "database_url" {
  secret      = google_secret_manager_secret.database_url.id
  secret_data = local.database_url
}
