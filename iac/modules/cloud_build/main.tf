resource "google_service_account" "cloudbuild" {
  project      = var.project_id
  account_id   = "cloudbuild-sa"
  display_name = "Cloud Build service account"
}

resource "google_project_iam_member" "sa" {
  for_each = toset(var.sa_roles)

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.cloudbuild.email}"
}

resource "google_service_account_iam_member" "self_token_creator" {
  service_account_id = google_service_account.cloudbuild.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "serviceAccount:${google_service_account.cloudbuild.email}"
}

resource "google_service_account_iam_member" "self_account_user" {
  service_account_id = google_service_account.cloudbuild.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.cloudbuild.email}"
}

resource "google_cloudbuildv2_repository" "repo" {
  project           = var.project_id
  location          = var.region
  name              = var.github_repo
  parent_connection = "projects/${var.project_id}/locations/${var.region}/connections/${var.connection_name}"
  remote_uri        = "https://github.com/${var.github_owner}/${var.github_repo}.git"
}

resource "google_cloudbuild_trigger" "this" {
  for_each = var.triggers

  project         = var.project_id
  name            = each.key
  description     = each.value.description
  location        = var.region
  service_account = google_service_account.cloudbuild.id

  repository_event_config {
    repository = google_cloudbuildv2_repository.repo.id
    push {
      branch = each.value.branch
    }
  }

  included_files = each.value.included_files
  filename       = each.value.filename
  substitutions  = each.value.substitutions
}
