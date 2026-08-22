output "trigger_ids" {
  value = { for k, v in google_cloudbuild_trigger.this : k => v.trigger_id }
}

output "repository_id" {
  value = google_cloudbuildv2_repository.repo.id
}

output "sa_email" {
  value = google_service_account.cloudbuild.email
}
