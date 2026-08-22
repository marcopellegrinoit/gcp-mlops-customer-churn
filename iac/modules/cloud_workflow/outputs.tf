output "workflow_id" {
  value = google_workflows_workflow.this.id
}

output "workflow_sa_email" {
  value = google_service_account.workflow.email
}
