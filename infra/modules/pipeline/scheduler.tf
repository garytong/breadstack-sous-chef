# The drift workflow's triggers (DESIGN §2.1), only with pipeline_schedule.
# UTC like the module's scheduler_timezone. Each creates an execution of
# souschef-drift through the Workflow Executions API with an OAuth token of
# souschef-scheduler (roles/workflows.invoker).
#
#   souschef-drift-poll    */15 * * * *  mode poll   (a manifest change seen in <= 15 min;
#                                                     a due check runs in the same execution)
#   souschef-drift-weekly  0 5 * * 1     mode weekly (baseline refresh + arm D + probe + F0H)
#   souschef-drift-night   0 9 * * *     mode night  (a check deferred for lack of an L4 runs
#                                                     on production; nothing to do = exits)
locals {
  triggers = var.schedule ? {
    poll   = { cron = var.poll_cron, mode = "poll" }
    weekly = { cron = var.weekly_cron, mode = "weekly" }
    night  = { cron = var.night_cron, mode = "night" }
  } : {}
}

resource "google_cloud_scheduler_job" "drift" {
  for_each = local.triggers

  project     = var.project_id
  name        = "souschef-drift-${each.key}"
  region      = var.workflows_region
  description = "Starts the souschef-drift workflow with mode ${each.value.mode}"
  schedule    = each.value.cron
  time_zone   = var.scheduler_timezone

  attempt_deadline = "180s"

  retry_config {
    retry_count = 1
  }

  http_target {
    http_method = "POST"
    uri         = "https://workflowexecutions.googleapis.com/v1/${google_workflows_workflow.wf["drift"].id}/executions"
    headers     = { "Content-Type" = "application/json" }
    body        = base64encode(jsonencode({ argument = jsonencode({ mode = each.value.mode }) }))

    oauth_token {
      service_account_email = local.sa_email["scheduler"]
      scope                 = "https://www.googleapis.com/auth/cloud-platform"
    }
  }

  depends_on = [google_project_iam_member.scheduler_invoker]
}
