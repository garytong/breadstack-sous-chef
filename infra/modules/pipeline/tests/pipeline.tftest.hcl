# Offline checks of the pipeline module against a mock provider: nothing is
# read from or applied to any project, and no credentials are needed.
#
#   cd infra/modules/pipeline && terraform init -backend=false && terraform test
#
# What it holds the module to: every switch off schedules nothing, starts
# nothing automatically and rolls out nothing; the jobs never retry on
# their own and only the prepare job sees the Anthropic key; the GPU
# services idle at zero; the workflows log errors only and fit Workflows'
# source limit; IAM is what DESIGN §5 lists.

mock_provider "google" {}

variables {
  project_id            = "breadstack-sous-chef-int"
  int_service_url       = "https://sous-chef-abc-pd.a.run.app"
  prod_model_url        = "https://sous-chef-model-abc-uk.a.run.app"
  prod_model_image      = "us-east4-docker.pkg.dev/breadstack-sous-chef-int/sous-chef-model/model:r4"
  model_service_account = "sous-chef-model-run@breadstack-sous-chef-int.iam.gserviceaccount.com"
}

run "switches_off" {
  command = apply

  assert {
    condition     = length(google_cloud_scheduler_job.drift) == 0
    error_message = "pipeline_schedule off: no scheduler trigger."
  }
  assert {
    condition     = length(google_monitoring_alert_policy.failed_execution) == 0 && length(google_monitoring_notification_channel.email) == 0
    error_message = "No alert emails: no policy and no channel."
  }
  assert {
    condition = (
      jsondecode(google_storage_bucket_object.config.content).retrain_auto == false &&
      jsondecode(google_storage_bucket_object.config.content).rollout_enabled == false &&
      jsondecode(google_storage_bucket_object.config.content).l4_headroom == false
    )
    error_message = "config.json: retrain_auto, rollout_enabled and l4_headroom default off."
  }
  assert {
    condition     = google_storage_bucket.pipeline.name == "breadstack-sous-chef-int-pipeline" && google_storage_bucket.pipeline.location == "US-CENTRAL1" && google_storage_bucket.pipeline.public_access_prevention == "enforced" && google_storage_bucket.pipeline.uniform_bucket_level_access
    error_message = "The pipeline bucket: <project>-pipeline in US-CENTRAL1, private."
  }
  assert {
    condition     = length(google_secret_manager_secret.pipeline) == 2
    error_message = "Two secrets (service key, thermometer key), created without versions."
  }
}

run "jobs" {
  command = apply

  assert {
    condition     = length(google_cloud_run_v2_job.job) == 5 && alltrue([for j in google_cloud_run_v2_job.job : j.template[0].template[0].max_retries == 0])
    error_message = "Five jobs, max_retries 0 (the workflow retries, never Cloud Run)."
  }
  assert {
    condition = alltrue([
      for k, j in google_cloud_run_v2_job.job :
      contains([for e in j.template[0].template[0].containers[0].env : e.name], "ANTHROPIC_API_KEY") == (k == "prepare")
    ])
    error_message = "Only souschef-prepare (case authoring) gets ANTHROPIC_API_KEY."
  }
  assert {
    condition = alltrue([
      for j in google_cloud_run_v2_job.job :
      contains([for e in j.template[0].template[0].containers[0].env : e.name], "SOUS_CHEF_KEY") &&
      !contains([for e in j.template[0].template[0].containers[0].env : e.name], "THERMOMETER_API_KEY")
    ])
    error_message = "Every job reads the pipeline service key; none holds a thermometer key."
  }
  assert {
    condition = (
      google_cloud_run_v2_job.job["drift"].template[0].template[0].service_account == local.sa_email["drift"] &&
      google_cloud_run_v2_job.job["eval"].template[0].template[0].service_account == local.sa_email["drift"] &&
      google_cloud_run_v2_job.job["prepare"].template[0].template[0].service_account == local.sa_email["author"] &&
      google_cloud_run_v2_job.job["collect"].template[0].template[0].service_account == local.sa_email["prep"] &&
      google_cloud_run_v2_job.job["dataset"].template[0].template[0].service_account == local.sa_email["prep"]
    )
    error_message = "drift/eval run as souschef-drift, prepare as souschef-author, collect/dataset as souschef-prep."
  }
}

run "gpu_services" {
  command = apply

  assert {
    condition = alltrue([
      for s in google_cloud_run_v2_service.gpu :
      s.template[0].scaling[0].min_instance_count == 0 && s.template[0].scaling[0].max_instance_count == 1 &&
      s.template[0].node_selector[0].accelerator == "nvidia-l4" && s.location == "us-east4" &&
      s.template[0].containers[0].image == var.prod_model_image
    ])
    error_message = "Eval and candidate services: L4 in us-east4, min 0, max 1, created with the production image."
  }
  assert {
    condition     = alltrue([for b in google_cloud_run_v2_service_iam_binding.gpu_invoker : toset(b.members) == toset(["serviceAccount:${local.sa_email["drift"]}", "serviceAccount:${local.sa_email["workflow"]}"])])
    error_message = "Only the drift runner and the workflow may invoke the eval and candidate services."
  }
}

run "workflows" {
  command = apply

  assert {
    condition     = alltrue([for w in google_workflows_workflow.wf : w.call_log_level == "LOG_ERRORS_ONLY" && w.region == "us-central1"])
    error_message = "Workflows in us-central1, LOG_ERRORS_ONLY."
  }
  assert {
    condition     = alltrue([for w in google_workflows_workflow.wf : length(w.source_contents) < 131072 && can(jsondecode(w.source_contents).main.steps)])
    error_message = "Each definition is JSON with a main and fits Workflows' 128 KB source limit."
  }
  assert {
    condition     = can(jsondecode(google_workflows_workflow.wf["retrain"].source_contents).stage_train) && can(jsondecode(google_workflows_workflow.wf["drift"].source_contents).notify)
    error_message = "The shared subworkflows (lib.yaml) are appended to both definitions."
  }
  assert {
    condition     = can(jsondecode(google_storage_bucket_object.cloudbuild_model.content).steps[1].args) && jsondecode(google_storage_bucket_object.cloudbuild_model.content).options.logging == "CLOUD_LOGGING_ONLY"
    error_message = "infra/cloudbuild-model.json is cloudbuild-model.yaml as JSON."
  }
}

run "iam" {
  command = apply

  assert {
    condition     = toset([for m in google_project_iam_member.workflow : m.role]) == toset(["roles/aiplatform.user", "roles/cloudbuild.builds.editor", "roles/workflows.invoker", "roles/logging.logWriter"])
    error_message = "The workflow's project roles: Vertex, Cloud Build, Workflows invoker, logs; nothing broader."
  }
  assert {
    condition     = google_secret_manager_secret_iam_member.thermometer_key.member == "serviceAccount:${local.sa_email["workflow"]}"
    error_message = "Only the workflow reads the thermometer key."
  }
  assert {
    condition     = google_secret_manager_secret_iam_member.anthropic.member == "serviceAccount:${local.sa_email["author"]}" && google_secret_manager_secret_iam_member.anthropic.secret_id == "sous-chef-anthropic-api-key"
    error_message = "Only the author account (the prepare job) is added to the Anthropic key; collect and dataset (souschef-prep) are not."
  }
  assert {
    condition     = google_storage_bucket_iam_member.admin.member == "serviceAccount:${local.sa_email["workflow"]}" && length(google_storage_bucket_iam_member.admin.condition) == 0
    error_message = "Bucket: only the workflow account holds an unconditional objectAdmin."
  }
  assert {
    condition = alltrue([
      for k, m in google_storage_bucket_iam_member.jobs_write :
      m.role == "roles/storage.objectAdmin" && length(m.condition) == 1 &&
      strcontains(m.condition[0].expression, "!resource.name.startsWith(\"projects/_/buckets/${local.bucket_name}/objects/infra/\")") &&
      strcontains(m.condition[0].expression, "!resource.name.startsWith(\"projects/_/buckets/${local.bucket_name}/objects/config/\")") &&
      strcontains(m.condition[0].expression, "!resource.name.startsWith(\"projects/_/buckets/${local.bucket_name}/objects/wf/\")")
    ]) && toset(keys(google_storage_bucket_iam_member.jobs_write)) == toset(["drift", "prep", "author"])
    error_message = "The job accounts write every object but infra/, config/ and wf/ (what the workflows act on, and the workflow's outbox and decision records)."
  }
  assert {
    condition = (
      strcontains(google_storage_bucket_object.cloudbuild_model.content, "${filesha256("${path.module}/../../../lab/model-image/model-context.sh")}  model-image/model-context.sh") &&
      strcontains(google_storage_bucket_object.cloudbuild_model.content, "${filesha256("${path.module}/../../../lab/model-image/Dockerfile")}  model-image/Dockerfile") &&
      !strcontains(google_storage_bucket_object.cloudbuild_model.content, "__MODEL_IMAGE_SHA256SUMS__") &&
      anytrue([for r in google_storage_bucket.pipeline.lifecycle_rule : anytrue([for c in r.condition : contains(coalesce(c.matches_prefix, []), "wf/runs/")])])
    )
    error_message = "The build spec pins the model-image files' sha256 (the job-written context is checked), and wf/runs/ expires with runs/."
  }
  assert {
    condition = (
      google_storage_bucket_iam_member.trainer_write.role == "roles/storage.objectAdmin" &&
      strcontains(google_storage_bucket_iam_member.trainer_write.condition[0].expression, "/objects/runs/{run}/train/") &&
      google_storage_bucket_iam_member.trainer_read.role == "roles/storage.objectViewer" &&
      strcontains(google_storage_bucket_iam_member.trainer_read.condition[0].expression, "/objects/base/") &&
      google_storage_bucket_iam_member.builder_viewer.role == "roles/storage.objectViewer" &&
      strcontains(google_storage_bucket_iam_member.builder_viewer.condition[0].expression, "/objects/runs/") &&
      !strcontains(google_storage_bucket_iam_member.builder_viewer.condition[0].expression, "/objects/infra/")
    )
    error_message = "The trainer writes runs/<id>/train/ only and reads runs/ and base/; the builder reads runs/ only."
  }
  assert {
    condition     = toset(keys(google_cloud_run_v2_service_iam_member.workflow_develops_gpu)) == toset(["eval", "cand", "prod"]) && google_cloud_run_v2_service_iam_member.workflow_develops_gpu["prod"].name == "sous-chef-model"
    error_message = "The workflow may patch the three GPU services (the image), production included."
  }
}

run "switches_on" {
  command = apply

  variables {
    schedule        = true
    retrain_auto    = true
    rollout_enabled = true
    alert_emails    = ["gary.tong@breadstack.com"]
  }

  assert {
    condition     = toset(keys(google_cloud_scheduler_job.drift)) == toset(["poll", "weekly", "night"])
    error_message = "pipeline_schedule: poll, weekly and night triggers."
  }
  assert {
    condition = (
      google_cloud_scheduler_job.drift["poll"].schedule == "*/15 * * * *" &&
      google_cloud_scheduler_job.drift["weekly"].schedule == "0 5 * * 1" &&
      google_cloud_scheduler_job.drift["night"].schedule == "0 9 * * *" &&
      jsondecode(jsondecode(base64decode(google_cloud_scheduler_job.drift["weekly"].http_target[0].body)).argument).mode == "weekly"
    )
    error_message = "Triggers: */15 poll, Monday 05:00 UTC weekly, 09:00 UTC night; each passes its mode."
  }
  assert {
    condition     = jsondecode(google_storage_bucket_object.config.content).retrain_auto && jsondecode(google_storage_bucket_object.config.content).rollout_enabled
    error_message = "The switches reach config.json."
  }
  assert {
    condition     = length(google_monitoring_alert_policy.failed_execution) == 1 && length(google_monitoring_notification_channel.email) == 1
    error_message = "alert_emails: one policy, one channel per address."
  }
  assert {
    condition     = length(google_monitoring_alert_policy.not_delivered) == 1 && strcontains(google_logging_metric.not_delivered[0].filter, "not delivered")
    error_message = "alert_emails: the log-based alert on notifications the thermometer did not take."
  }
  assert {
    condition     = jsondecode(google_storage_bucket_object.config.content).l4_wait_minutes == 360 && alltrue([for r in jsondecode(google_storage_bucket_object.config.content).trainer.ladder : r.max_pending_s >= 600])
    error_message = "config.json carries the L4 wait and each rung's pending deadline."
  }
}

run "decision_ttl_capped" {
  command = plan

  variables {
    decision_ttl_days = 30
  }

  expect_failures = [var.decision_ttl_days]
}

run "ladder_validation" {
  command = plan

  variables {
    train_ladder = [{ name = "bad", region = "us-central1", machine_type = "a3-highgpu-1g", accelerator_type = "NVIDIA_H100_80GB", strategy = "ON_DEMAND", attempts = 1 }]
  }

  expect_failures = [var.train_ladder]
}
