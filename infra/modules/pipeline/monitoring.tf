# The backstop (DESIGN §1.6): every workflow step already posts stage_failed
# to the thermometer; this catches what cannot (the thermometer down, the
# key revoked, a workflow that fails before its first try block). Email to
# pipeline_alert_emails; none = no policy.
#
#   failed_execution  a souschef-* execution ended FAILED
#   not_delivered     a workflow logged "notification … not delivered": the
#                     thermometer refused or could not take a message (it
#                     waits in the bucket's outbox, state/outbox/, which the
#                     next drift poll re-sends; a refusal is dropped)
resource "google_monitoring_notification_channel" "email" {
  for_each     = toset(var.alert_emails)
  project      = var.project_id
  display_name = "Sous Chef pipeline: ${each.value}"
  type         = "email"
  labels = {
    email_address = each.value
  }

  depends_on = [google_project_service.pipeline]
}

resource "google_monitoring_alert_policy" "failed_execution" {
  count = length(var.alert_emails) > 0 ? 1 : 0

  project      = var.project_id
  display_name = "Sous Chef pipeline: workflow execution failed"
  combiner     = "OR"

  conditions {
    display_name = "souschef-drift or souschef-retrain execution FAILED"
    condition_threshold {
      filter          = "resource.type = \"workflows.googleapis.com/Workflow\" AND metric.type = \"workflows.googleapis.com/finished_execution_count\" AND metric.label.status = \"FAILED\" AND resource.label.workflow_id = monitoring.regex.full_match(\"souschef-.*\")"
      comparison      = "COMPARISON_GT"
      threshold_value = 0
      duration        = "0s"

      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_SUM"
      }

      trigger {
        count = 1
      }
    }
  }

  documentation {
    content   = "A Sous Chef pipeline workflow execution failed. Open the execution (Workflows > souschef-drift / souschef-retrain > Executions), then docs/RETRAIN.md R8 (debugging) or docs/DRIFT.md."
    mime_type = "text/markdown"
  }

  notification_channels = [for c in google_monitoring_notification_channel.email : c.id]

  alert_strategy {
    auto_close = "86400s"
  }
}

# Workflows' sys.log lines land in Cloud Logging as the workflow's own entries;
# notify / notify_send log "notification <id> (<kind>) not delivered" at ERROR.
resource "google_logging_metric" "not_delivered" {
  count = length(var.alert_emails) > 0 ? 1 : 0

  project     = var.project_id
  name        = "souschef_notification_not_delivered"
  description = "Sous Chef pipeline notifications the thermometer did not take (workflow log lines \"not delivered\")"
  filter      = "resource.type=\"workflows.googleapis.com/Workflow\" AND severity>=ERROR AND textPayload:\"not delivered\""

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
  }

  depends_on = [google_project_service.pipeline]
}

resource "google_monitoring_alert_policy" "not_delivered" {
  count = length(var.alert_emails) > 0 ? 1 : 0

  project      = var.project_id
  display_name = "Sous Chef pipeline: notification not delivered to the thermometer"
  combiner     = "OR"

  conditions {
    display_name = "a souschef-* workflow could not post to the thermometer"
    condition_threshold {
      filter          = "resource.type = \"workflows.googleapis.com/Workflow\" AND metric.type = \"logging.googleapis.com/user/${google_logging_metric.not_delivered[0].name}\""
      comparison      = "COMPARISON_GT"
      threshold_value = 0
      duration        = "0s"

      aggregations {
        alignment_period   = "600s"
        per_series_aligner = "ALIGN_SUM"
      }

      trigger {
        count = 1
      }
    }
  }

  documentation {
    content   = "A Sous Chef pipeline message did not reach the thermometer (down, its key revoked or out of SOUSCHEF_INGEST_KEY_IDS, or a refusal). Undelivered messages wait in gs://<pipeline bucket>/state/outbox/ and the next drift poll re-sends them. Check https://thermometer.breadstackcrm.com and the workflow's logs; docs/PIPELINE.md \"When Slack goes quiet\"."
    mime_type = "text/markdown"
  }

  notification_channels = [for c in google_monitoring_notification_channel.email : c.id]

  alert_strategy {
    auto_close = "86400s"
  }
}
