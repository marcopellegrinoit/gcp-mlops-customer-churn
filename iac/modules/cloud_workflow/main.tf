resource "google_service_account" "workflow" {
  project      = var.project_id
  account_id   = "${var.workflow_name}-sa"
  display_name = "${var.workflow_name} Cloud Workflow service account"
}

resource "google_project_iam_member" "workflow_sa" {
  for_each = toset(var.service_account_project_roles)

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.workflow.email}"
}

resource "google_storage_bucket_iam_member" "this" {
  for_each = var.gcs_bucket_roles

  bucket = each.key
  role   = each.value
  member = "serviceAccount:${google_service_account.workflow.email}"
}

resource "google_bigquery_dataset_iam_member" "this" {
  for_each = var.bq_dataset_roles

  project    = var.project_id
  dataset_id = each.key
  role       = each.value
  member     = "serviceAccount:${google_service_account.workflow.email}"
}

resource "google_service_account_iam_member" "act_as" {
  for_each = toset(var.act_as_service_account_emails)

  service_account_id = "projects/${var.project_id}/serviceAccounts/${each.value}"
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.workflow.email}"
}

resource "google_workflows_workflow" "this" {
  project         = var.project_id
  region          = var.region
  name            = var.workflow_name
  description     = var.description
  service_account = google_service_account.workflow.email
  source_contents = var.source_contents

  # workflow-trigger (Cloud Build) deploys source_contents on every push to
  # workflows/**, so it — not Terraform — owns ongoing content updates. This
  # var still seeds the initial revision on a fresh project. Same pattern as
  # cloud_run_job's `template` ignore_changes for CI/CD-pushed images.
  lifecycle {
    ignore_changes = [source_contents]
  }
}

# Dedicated SA for Cloud Scheduler to invoke the workflow
resource "google_service_account" "scheduler" {
  project      = var.project_id
  account_id   = "${var.workflow_name}-sched-sa"
  display_name = "${var.workflow_name} Cloud Scheduler service account"
}

resource "google_project_iam_member" "scheduler_invoker" {
  project = var.project_id
  role    = "roles/workflows.invoker"
  member  = "serviceAccount:${google_service_account.scheduler.email}"
}

resource "google_cloud_scheduler_job" "this" {
  count     = var.schedule != "" ? 1 : 0
  project   = var.project_id
  region    = var.region
  name      = "${var.workflow_name}-scheduler"
  schedule  = var.schedule
  time_zone = "UTC"

  http_target {
    http_method = "POST"
    uri         = "https://workflowexecutions.googleapis.com/v1/${google_workflows_workflow.this.id}/executions"

    body = base64encode(jsonencode({
      argument = jsonencode({
        project_id       = var.project_id
        region           = var.region
        alert_email      = var.alert_email
        alert_from_email = var.alert_from_email
      })
    }))

    headers = {
      "Content-Type" = "application/json"
    }

    oauth_token {
      service_account_email = google_service_account.scheduler.email
    }
  }
}
