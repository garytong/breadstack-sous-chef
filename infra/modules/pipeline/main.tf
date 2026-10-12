# The Sous Chef drift check and automated retraining pipeline (int only;
# docs/DRIFT.md, docs/RETRAIN.md). Instantiated by infra/envs/int only while
# pipeline_enabled is true; every switch inside defaults off.
#
#   apis.tf        the APIs the pipeline needs (never turned off by a destroy)
#   bucket.tf      the pipeline bucket (state, baselines, cases, runs, drift)
#                  and the two objects Terraform owns in it: infra/config.json
#                  (what the workflows read at start) and
#                  infra/cloudbuild-model.json (the model image build)
#   iam.tf         six service accounts, resource-scoped grants
#   secrets.tf     the pipeline service key and thermometer key (no values)
#   jobs.tf        five Cloud Run jobs on the lab runtime image
#   services.tf    sous-chef-model-eval and sous-chef-model-cand (L4, min 0)
#   workflows.tf   souschef-drift and souschef-retrain (Cloud Workflows)
#   scheduler.tf   poll / weekly / night triggers (pipeline_schedule)
#   monitoring.tf  the failed-execution backstop (alert_emails)
#
# Nothing here holds a secret value, and no image takes one at build time.

terraform {
  required_version = ">= 1.7"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

locals {
  labels = {
    purpose = "sous-chef-pipeline"
    env     = "int"
  }

  bucket_name = var.bucket_name != "" ? var.bucket_name : "${var.project_id}-pipeline"

  # The lab runtime image's repository (the service's own, ne2) and the model
  # repository (us-east4). The workflow resolves the lab and trainer images
  # per run from the int service's image tag (DESIGN §6.3), so no tag lives
  # here; lab_image is only the image the jobs are created with.
  lab_repo   = "${var.region}-docker.pkg.dev/${var.project_id}/${var.lab_repository_id}"
  model_repo = "${var.model_repo_location}-docker.pkg.dev/${var.project_id}/${var.model_repository_id}"

  job_names = {
    drift   = "souschef-drift"
    eval    = "souschef-eval"
    prepare = "souschef-prepare"
    collect = "souschef-collect"
    dataset = "souschef-dataset"
  }

  service_names = {
    eval = "sous-chef-model-eval"
    cand = "sous-chef-model-cand"
  }

  workflow_names = {
    drift   = "souschef-drift"
    retrain = "souschef-retrain"
  }

  # What both workflows read from gs://<bucket>/infra/config.json at start
  # (bucket.tf). Plain configuration only: the thermometer key is a secret
  # the workflow reads itself, by name.
  pipeline_config = {
    project            = var.project_id
    bucket             = local.bucket_name
    workflows_region   = var.workflows_region
    workflows          = local.workflow_names
    jobs_region        = var.region
    jobs               = local.job_names
    int_service        = { name = var.int_service_name, region = var.region, url = var.int_service_url }
    prod_model         = { name = var.prod_model_service_name, region = var.gpu_region, url = var.prod_model_url }
    eval_model         = { name = local.service_names.eval, region = var.gpu_region, url = google_cloud_run_v2_service.gpu["eval"].uri }
    cand_model         = { name = local.service_names.cand, region = var.gpu_region, url = google_cloud_run_v2_service.gpu["cand"].uri }
    lab_repo           = { location = var.region, repository = var.lab_repository_id, image = "${local.lab_repo}/sous-chef-lab", trainer = "${local.lab_repo}/sous-chef-trainer" }
    model_repo         = { location = var.model_repo_location, repository = var.model_repository_id, image = "${local.model_repo}/model" }
    ollama_model       = var.ollama_model
    route_name         = var.inhouse_route_name
    thermometer_url    = trimsuffix(var.thermometer_url, "/")
    thermometer_secret = google_secret_manager_secret.pipeline["thermometer"].secret_id
    retrain_auto       = var.retrain_auto
    rollout_enabled    = var.rollout_enabled
    l4_headroom        = var.l4_headroom
    decision_ttl_days  = var.decision_ttl_days
    smoke_ttl_days     = 1
    build              = { region = var.build_region, service_account = local.sa_email["builder"], spec = "infra/cloudbuild-model.json" }
    trainer            = { service_account = local.sa_email["trainer"], ladder = var.train_ladder, disk_gb = var.train_disk_gb, timeout_s = var.train_timeout_hours * 3600 }
    eval_warm_minutes  = var.eval_warm_minutes
    l4_wait_minutes    = var.l4_wait_minutes
    lab_image_grace_m  = var.lab_image_grace_minutes
    console_project    = var.project_id
  }
}
