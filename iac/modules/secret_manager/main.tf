# The value is added by hand (gcloud/console) after creation, never through
# Terraform state, so this module only creates the secret container — no
# google_secret_manager_secret_version resource.
resource "google_secret_manager_secret" "this" {
  project   = var.project_id
  secret_id = var.secret_id

  replication {
    auto {}
  }
}
