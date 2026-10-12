# The two Cloud Workflows (DESIGN §3.1). Plain YAML, not templates: what the
# workflows need to know (names, URLs, regions, switches, the training
# ladder) is the bucket object infra/config.json (bucket.tf), read at the
# start of every execution, so the definitions carry no ${} of Terraform's
# and a configuration change is never a redeploy.
#
# Source: workflows/<name>.yaml + workflows/lib.yaml (the shared
# subworkflows; Workflows has no imports), deployed as compact JSON: the
# commented retrain YAML is ~117 KB against Workflows' 128 KB source limit,
# its JSON form ~79 KB. Lint offline with tests/lint_workflows.py.
#
# call_log_level LOG_ERRORS_ONLY: the thermometer key is read from Secret
# Manager into a local variable used only in an X-API-Key header, never
# logged, returned or put in a job's args.
locals {
  workflow_sources = {
    for k in keys(local.workflow_names) : k => jsonencode(yamldecode(join("\n", [
      file("${path.module}/workflows/${k}.yaml"),
      file("${path.module}/workflows/lib.yaml"),
    ])))
  }
}

resource "google_workflows_workflow" "wf" {
  for_each = local.workflow_names

  project         = var.project_id
  name            = each.value
  region          = var.workflows_region
  description     = each.key == "drift" ? "Sous Chef drift check: poll the deployed launcher manifest, static diff, behavioural exam, notify, start a retrain on a big drift (docs/DRIFT.md)" : "Sous Chef retrain round: prepare, collect, dataset, Vertex training, model image, candidate, paired eval, decision, rollout (docs/RETRAIN.md)"
  service_account = local.sa_email["workflow"]
  call_log_level  = "LOG_ERRORS_ONLY"
  labels          = local.labels
  source_contents = local.workflow_sources[each.key]

  user_env_vars = {
    PIPELINE_BUCKET = local.bucket_name
    PIPELINE_CONFIG = google_storage_bucket_object.config.name
  }

  deletion_protection = false

  depends_on = [
    google_project_service.pipeline,
    google_project_iam_member.workflow,
    google_storage_bucket_object.config,
  ]
}
