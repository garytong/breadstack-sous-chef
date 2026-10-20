# Offline checks of how infra/envs/int wires the Sous Chef pipeline, against
# mock providers (nothing is read from or applied to any project):
#
#   cd infra/envs/int && terraform init -backend=false && terraform test
#
# Every variable that matters is set here, so a local terraform.tfvars
# (gitignored, auto-loaded by `terraform test`) changes nothing.
#
# What it holds the root to: with pipeline_enabled off nothing of the
# pipeline is planned and the production model service's invokers are
# sous-chef-run alone; on, the drift runner and workflow join them; the
# pipeline refuses to exist without the production model service; and the
# model service's image belongs to the pipeline after creation.

mock_provider "google" {}
mock_provider "random" {}

# Mocked emails are random strings; IAM resources validate the form.
override_resource {
  target = module.model_gpu[0].google_service_account.model
  values = {
    email = "sous-chef-model-run@breadstack-sous-chef-int.iam.gserviceaccount.com"
    name  = "projects/breadstack-sous-chef-int/serviceAccounts/sous-chef-model-run@breadstack-sous-chef-int.iam.gserviceaccount.com"
  }
}

override_resource {
  target = module.sous_chef.google_service_account.run
  values = {
    email = "sous-chef-run@breadstack-sous-chef-int.iam.gserviceaccount.com"
    name  = "projects/breadstack-sous-chef-int/serviceAccounts/sous-chef-run@breadstack-sous-chef-int.iam.gserviceaccount.com"
  }
}

variables {
  project_id               = "breadstack-sous-chef-int"
  image                    = "northamerica-northeast2-docker.pkg.dev/breadstack-sous-chef-int/sous-chef/sous-chef:dev"
  public_invoker           = true
  ingress                  = "INGRESS_TRAFFIC_ALL"
  launcher_service_account = ""
  ci_service_account       = ""
  thermometer_api_url      = ""
  gpu_model_enabled        = true
  gpu_model_serve          = true
  gpu_model_image_tag      = "r4"
  gpu_model_min_instances  = 1
  gpu_model_default        = true
  pipeline_enabled         = false
  pipeline_schedule        = false
  pipeline_retrain_auto    = false
  pipeline_rollout         = false
  pipeline_alert_emails    = [] # the live tfvars sets one; pipeline_needs_alert_emails needs none
}

run "pipeline_off" {
  command = apply

  assert {
    condition     = length(module.pipeline) == 0 && output.pipeline == null
    error_message = "pipeline_enabled off: nothing of the pipeline."
  }
  assert {
    condition     = length(module.model_gpu[0].invokers) == 1
    error_message = "Without the pipeline, sous-chef-run alone invokes the production model."
  }
}

run "pipeline_on" {
  command = apply

  variables {
    pipeline_enabled      = true
    pipeline_alert_emails = ["gary.tong@breadstack.com"]
  }

  assert {
    condition     = length(module.pipeline) == 1 && output.pipeline.bucket == "breadstack-sous-chef-int-pipeline"
    error_message = "pipeline_enabled: the module and its bucket."
  }
  assert {
    condition = toset(module.model_gpu[0].invokers) == toset([
      "serviceAccount:${module.sous_chef.run_service_account}",
      "serviceAccount:souschef-drift@breadstack-sous-chef-int.iam.gserviceaccount.com",
      "serviceAccount:souschef-workflow@breadstack-sous-chef-int.iam.gserviceaccount.com",
    ])
    error_message = "With the pipeline, the drift runner and the workflow also invoke the production model."
  }
  assert {
    condition     = toset(output.pipeline.secrets_to_populate) == toset(["sous-chef-pipeline-service-key", "sous-chef-pipeline-thermometer-key"])
    error_message = "Two secrets to populate by hand."
  }
}

run "pipeline_needs_model_service" {
  command = plan

  variables {
    pipeline_enabled      = true
    gpu_model_serve       = false
    pipeline_alert_emails = ["gary.tong@breadstack.com"]
  }

  expect_failures = [check.pipeline_needs_the_model_service]

  assert {
    condition     = length(module.pipeline) == 0
    error_message = "No pipeline without the production model service."
  }
}

run "model_image_owned_by_the_pipeline" {
  command = plan

  assert {
    condition = alltrue([
      for s in ["template[0].containers[0].image", "template[0].annotations", "client_version"] :
      strcontains(file("${path.module}/../../modules/model-gpu/main.tf"), s)
    ])
    error_message = "modules/model-gpu ignores the image (and the rollout's annotation): the pipeline and make model-rollout own it after creation."
  }
}

run "pipeline_needs_alert_emails" {
  command = plan

  variables {
    pipeline_enabled = true
  }

  expect_failures = [var.pipeline_alert_emails]
}

# --- the in-house routes: blueberry-muffin, chocolate-muffin and the switch ---
# (docs/CHOCOLATE-MUFFIN.md §2 and step 5). With the new variables at their
# defaults, MODEL_ROUTES, the allowlist and the aliases are what they were.

run "inhouse_routes_as_today" {
  command = apply

  assert {
    condition     = keys(jsondecode(local.model_routes)) == ["blueberry-muffin", "default", "opus"] && local.model_route_allowlist == "default,opus,blueberry-muffin" && length(local.model_route_aliases) == 0 && local.default_route == "blueberry-muffin"
    error_message = "Defaults: blueberry-muffin alone beside Claude, allowlisted, the default route, no alias."
  }
  assert {
    condition = jsondecode(local.model_routes)["blueberry-muffin"] == {
      kind        = "messages-compatible"
      url         = module.model_gpu[0].url
      auth        = "id-token"
      model       = "sous-chef"
      max_tokens  = 1024
      stream      = false
      temperature = 0.3
      fallback    = "default"
      prime       = "on"
      fast_finish = "on"
      report_as   = "blueberry-muffin"
    }
    error_message = "blueberry-muffin's route is exactly today's (no parser_retries, no fields_context key)."
  }
}

run "the_chocolate_muffin_route" {
  command = apply

  variables {
    gpu_model_mistral_route = { enabled = true }
  }

  assert {
    condition     = keys(jsondecode(local.model_routes)) == ["blueberry-muffin", "chocolate-muffin", "default", "opus"] && local.model_route_allowlist == "default,opus,blueberry-muffin,chocolate-muffin"
    error_message = "chocolate-muffin joins MODEL_ROUTES and the allowlist."
  }
  assert {
    condition = jsondecode(local.model_routes)["chocolate-muffin"] == {
      kind           = "messages-compatible"
      url            = module.model_gpu[0].url
      auth           = "id-token"
      model          = "sous-chef"
      max_tokens     = 1024
      stream         = false
      temperature    = 0.15
      fallback       = "default"
      prime          = "off"
      fast_finish    = "on"
      parser_retries = "off"
      fields_context = "on"
      report_as      = "chocolate-muffin"
    }
    error_message = "chocolate-muffin's route (CHOCOLATE-MUFFIN.md §2; service/internal/config/chocolate_test.go loads the same JSON): the same model service, 0.15, priming off, parser_retries off."
  }
  assert {
    condition     = jsondecode(local.model_routes)["blueberry-muffin"].prime == "on" && jsondecode(local.model_routes)["blueberry-muffin"].temperature == 0.3 && local.default_route == "blueberry-muffin" && length(local.model_route_aliases) == 0
    error_message = "Adding the route changes neither blueberry-muffin nor the default route, and switches nothing."
  }
}

run "the_switch" {
  command = apply

  variables {
    gpu_model_mistral_route = { enabled = true }
    model_route_aliases     = { "blueberry-muffin" = "chocolate-muffin" }
  }

  assert {
    condition     = local.model_route_aliases == tomap({ "blueberry-muffin" = "chocolate-muffin" }) && local.default_route == "blueberry-muffin"
    error_message = "The switch is the alias; DEFAULT_ROUTE stays blueberry-muffin."
  }
}

# The break-glass (docs/CHOCOLATE-MUFFIN.md §5): every in-house turn to Claude's
# default route, which int's allowlist already holds, with or without the
# chocolate-muffin route; no spare L4 needed.
run "the_break_glass" {
  command = apply

  variables {
    gpu_model_mistral_route = { enabled = true }
    model_route_aliases     = { "blueberry-muffin" = "default" }
  }

  assert {
    condition     = local.model_route_aliases == tomap({ "blueberry-muffin" = "default" }) && local.default_route == "blueberry-muffin" && contains(split(",", local.model_route_allowlist), "default")
    error_message = "The break-glass alias passes the check: default is a route of MODEL_ROUTES and allowlisted."
  }
}

run "an_alias_needs_its_route" {
  command = apply

  variables {
    model_route_aliases = { "blueberry-muffin" = "chocolate-muffin" }
  }

  expect_failures = [check.model_route_aliases_name_live_routes]
}

run "no_alias_without_the_model_service" {
  command = plan

  variables {
    gpu_model_serve     = false
    model_route_aliases = { "blueberry-muffin" = "chocolate-muffin" }
  }

  assert {
    condition     = length(local.model_route_aliases) == 0
    error_message = "Without the model service there are no in-house routes, so no alias is passed."
  }
}
