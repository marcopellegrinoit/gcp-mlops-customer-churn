output "bq_dataset_ids" {
  description = "Created BigQuery dataset IDs."
  value       = { for k, v in module.bq_dataset : k => v.dataset_id }
}

output "bq_table_ids" {
  description = "Created BigQuery table IDs."
  value       = { for k, v in module.bq_table : k => v.table_id }
}

output "artifact_registry_urls" {
  description = "Artifact Registry repository URLs."
  value       = { for k, v in module.artifact_registry : k => v.repository_url }
}

output "cloud_run_job_names" {
  description = "Provisioned Cloud Run Job names."
  value       = { for k, v in module.cloud_run_job : k => v.job_name }
}

output "cloud_run_job_service_accounts" {
  description = "Service account emails for Cloud Run Jobs."
  value       = { for k, v in module.cloud_run_job : k => v.service_account_email }
}

output "cloud_build_trigger_ids" {
  description = "Cloud Build trigger IDs."
  value       = module.cloud_build.trigger_ids
}

output "gcs_bucket_names" {
  description = "Provisioned GCS bucket names."
  value       = { for k, v in module.gcs_bucket : k => v.bucket_name }
}

output "vertex_ai_pipeline_service_account" {
  description = "Service account email used by the Vertex AI training pipeline."
  value       = module.vertex_ai_pipeline.service_account_email
}
