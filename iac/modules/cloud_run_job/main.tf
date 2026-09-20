resource "google_service_account" "this" {
  project      = var.project_id
  account_id   = "${var.job_name}-sa"
  display_name = "${var.job_name} Cloud Run Job service account"
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

resource "google_service_account_iam_member" "deployer_act_as" {
  count = var.deployer_sa_email != "" ? 1 : 0

  service_account_id = google_service_account.this.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${var.deployer_sa_email}"
}

resource "google_cloud_run_v2_job" "this" {
  name                = var.job_name
  location            = var.region
  project             = var.project_id
  deletion_protection = false

  template {
    template {
      service_account = google_service_account.this.email
      max_retries     = var.max_retries
      timeout         = var.timeout

      containers {
        image = var.image

        dynamic "env" {
          for_each = var.env_vars
          content {
            name  = env.key
            value = env.value
          }
        }

        resources {
          limits = {
            cpu    = var.cpu
            memory = var.memory
          }
        }
      }
    }
  }

  lifecycle {
    ignore_changes = [template[0].template[0].containers[0].image, client, client_version]
  }
}
