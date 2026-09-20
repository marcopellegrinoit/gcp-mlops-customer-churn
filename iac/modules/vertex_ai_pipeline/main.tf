resource "google_service_account" "this" {
  project      = var.project_id
  account_id   = var.account_id
  display_name = var.display_name
}

resource "google_project_iam_member" "this" {
  for_each = toset(var.service_account_project_roles)

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.this.email}"
}

resource "google_bigquery_dataset_iam_member" "this" {
  for_each = var.bq_dataset_roles

  project    = var.project_id
  dataset_id = each.key
  role       = each.value
  member     = "serviceAccount:${google_service_account.this.email}"
}

resource "google_storage_bucket_iam_member" "this" {
  for_each = var.gcs_bucket_roles

  bucket = each.key
  role   = each.value
  member = "serviceAccount:${google_service_account.this.email}"
}

resource "google_service_account_iam_member" "act_as" {
  for_each = toset(var.act_as_service_account_emails)

  service_account_id = "projects/${var.project_id}/serviceAccounts/${each.value}"
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.this.email}"
}
