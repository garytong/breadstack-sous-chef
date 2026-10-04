# A notify-only budget on the training spend: e-mails at 50%, 90% and 100%
# of budget_amount. It stops nothing — the hard stop is the VM's
# max_run_duration (vm.tf) — and Cloud Billing reports with a delay of hours,
# so it is a record, not a brake.
#
#   scope    this project AND the label purpose=sous-chef-train: the VM (its
#            vCPUs, memory and GPU), its disk and the bucket. Not the
#            sous-chef service, its Cloud SQL or the GPU model service
#            (modules/model-gpu), which carry no such label. Cloud NAT and
#            network egress carry no labels, so they are not counted (cents
#            per run).
#   period   one running total from budget_start_date, no end: the whole
#            project, not a month at a time.
#   currency none is sent, so the amount is in the billing account's own
#            currency (a CAD account reads 200 as CAD 200).
#   who      the billing account's admins and users (Google's default
#            recipients), plus any budget_notification_channels.
#
# Needs billing.budgets.create on the billing account for the applying login
# (Billing Account Administrator or Billing Account Costs Manager); without
# it, set budget_enabled = false and make a budget in the console instead.
resource "google_billing_budget" "train" {
  count    = var.budget_enabled ? 1 : 0
  provider = google.billing

  billing_account = var.billing_account
  display_name    = "Sous Chef training (purpose=sous-chef-train)"

  budget_filter {
    projects               = ["projects/${data.google_project.this.number}"]
    labels                 = { purpose = local.labels.purpose }
    credit_types_treatment = "INCLUDE_ALL_CREDITS"

    custom_period {
      start_date {
        year  = tonumber(split("-", var.budget_start_date)[0])
        month = tonumber(split("-", var.budget_start_date)[1])
        day   = tonumber(split("-", var.budget_start_date)[2])
      }
    }
  }

  amount {
    specified_amount {
      units = tostring(var.budget_amount)
    }
  }

  threshold_rules {
    threshold_percent = 0.5
  }
  threshold_rules {
    threshold_percent = 0.9
  }
  threshold_rules {
    threshold_percent = 1.0
  }

  all_updates_rule {
    monitoring_notification_channels = var.budget_notification_channels
    disable_default_iam_recipients   = false
  }

  lifecycle {
    precondition {
      condition     = var.billing_account != ""
      error_message = "budget_enabled needs billing_account (the project's billing account id, terraform.tfvars), or set budget_enabled = false."
    }
  }

  depends_on = [google_project_service.train]
}
