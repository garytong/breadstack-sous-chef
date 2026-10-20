# Offline checks of the service's routing env (mock providers; nothing is read
# or applied):
#
#   cd infra/modules/sous-chef && terraform init -backend=false && terraform test
#
# MODEL_ROUTE_ALIASES (docs/CHOCOLATE-MUFFIN.md step 5) is in the service's
# env only while an alias exists, so with model_route_aliases at its default
# the revision template is what it was before the option.

mock_provider "google" {}
mock_provider "random" {}

variables {
  project_id            = "breadstack-sous-chef-int"
  env                   = "int"
  image                 = "northamerica-northeast2-docker.pkg.dev/breadstack-sous-chef-int/sous-chef/sous-chef:dev"
  default_route         = "blueberry-muffin"
  model_route_allowlist = "default,opus,blueberry-muffin,chocolate-muffin"
}

run "no_alias_no_env" {
  command = plan

  assert {
    condition     = !contains([for e in google_cloud_run_v2_service.sous_chef.template[0].containers[0].env : e.name], "MODEL_ROUTE_ALIASES")
    error_message = "model_route_aliases {}: no MODEL_ROUTE_ALIASES in the service's env."
  }
  assert {
    condition     = one([for e in google_cloud_run_v2_service.sous_chef.template[0].containers[0].env : e.value if e.name == "DEFAULT_ROUTE"]) == "blueberry-muffin"
    error_message = "DEFAULT_ROUTE is passed as given."
  }
}

run "an_alias_is_passed_as_json" {
  command = plan

  variables {
    model_route_aliases = { "blueberry-muffin" = "chocolate-muffin" }
  }

  assert {
    condition     = one([for e in google_cloud_run_v2_service.sous_chef.template[0].containers[0].env : e.value if e.name == "MODEL_ROUTE_ALIASES"]) == "{\"blueberry-muffin\":\"chocolate-muffin\"}"
    error_message = "MODEL_ROUTE_ALIASES is the aliases' JSON."
  }
  assert {
    condition     = one([for e in google_cloud_run_v2_service.sous_chef.template[0].containers[0].env : e.value if e.name == "DEFAULT_ROUTE"]) == "blueberry-muffin"
    error_message = "The switch leaves DEFAULT_ROUTE alone."
  }
}

run "an_alias_to_itself_is_refused" {
  command = plan

  variables {
    model_route_aliases = { "blueberry-muffin" = "blueberry-muffin" }
  }

  expect_failures = [var.model_route_aliases]
}
