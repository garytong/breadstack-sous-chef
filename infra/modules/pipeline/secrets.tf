# Two new secrets, created WITHOUT a value (Secret Manager rejects empty
# payloads, and Terraform never holds these). The user adds each value with
# the no-echo procedure (docs/RETRAIN.md R0):
#
#   sous-chef-pipeline-service-key      a new int service key; the same value
#                                       is appended to sous-chef-service-keys
#                                       (two keys = rotation, auth.NewKeys), so
#                                       the pipeline's calls are attributable
#                                       and revocable on their own. Read by the
#                                       five jobs (SOUS_CHEF_KEY).
#   sous-chef-pipeline-thermometer-key  a tm_ key with Ingest scope minted on
#                                       the thermometer's API keys page, named
#                                       souschef-pipeline. Read by the
#                                       workflows only, never by a job.
#
# The jobs reference the service key, and Cloud Run checks a referenced
# secret when the job is created: add both values after a targeted apply of
# these two secrets and before the full pipeline_enabled apply (R0).
locals {
  pipeline_secrets = {
    service_key = "sous-chef-pipeline-service-key"
    thermometer = "sous-chef-pipeline-thermometer-key"
  }
}

resource "google_secret_manager_secret" "pipeline" {
  for_each  = local.pipeline_secrets
  project   = var.project_id
  secret_id = each.value
  labels    = local.labels

  replication {
    auto {}
  }
}
