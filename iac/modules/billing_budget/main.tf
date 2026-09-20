# A budget is a *notification*, not a cap. Nothing here stops spend — GCP has no such
# control — so this exists to make a runaway visible within hours instead of at the end of
# a billing cycle. The actual blast-radius limits live on the resources themselves:
# max_instance_count on the Cloud Run service, MAX_BYTES_BILLED on every dashboard query,
# max_retries on the jobs.

resource "google_monitoring_notification_channel" "email" {
  for_each = toset(var.notification_emails)

  project      = var.project_id
  display_name = "Budget alerts — ${each.value}"
  type         = "email"

  labels = {
    email_address = each.value
  }
}

resource "google_billing_budget" "this" {
  billing_account = var.billing_account_id
  display_name    = var.display_name

  budget_filter {
    # Scoped to this project by number, which is the only form the Budgets API accepts.
    # Without it the budget covers every project on the billing account, and an alert
    # would say nothing about whether *this* platform is the thing costing money.
    projects        = ["projects/${var.project_number}"]
    calendar_period = "MONTH"
  }

  amount {
    specified_amount {
      # Omitted rather than guessed: a budget whose currency differs from the billing
      # account's is rejected by the API, and the account's own currency is always right.
      currency_code = var.currency_code != "" ? var.currency_code : null
      units         = tostring(var.amount)
    }
  }

  dynamic "threshold_rules" {
    for_each = var.threshold_percents
    content {
      threshold_percent = threshold_rules.value
      spend_basis       = "CURRENT_SPEND"
    }
  }

  # The one that matters for a platform designed to sit inside the free tier. Actual spend
  # crossing 50% tells you the month is already half spent; a *forecast* crossing 100%
  # fires on the trend, which is what catches a service that started billing yesterday and
  # will keep billing until someone looks.
  threshold_rules {
    threshold_percent = 1.0
    spend_basis       = "FORECASTED_SPEND"
  }

  # Left absent when no address is configured, which is not the same as "no alerts": with
  # no all_updates_rule the Budgets API falls back to emailing the billing account's admins
  # and users. Declaring the block with an empty channel list would instead silence it.
  dynamic "all_updates_rule" {
    for_each = length(var.notification_emails) > 0 ? [1] : []
    content {
      monitoring_notification_channels = [
        for channel in google_monitoring_notification_channel.email : channel.id
      ]
      # Billing admins keep getting the mail as well. The named recipient is an addition to
      # that audience, not a replacement for it.
      disable_default_iam_recipients = false
    }
  }
}
