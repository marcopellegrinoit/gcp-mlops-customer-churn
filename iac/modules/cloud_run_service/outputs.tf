output "service_name" {
  value = google_cloud_run_v2_service.this.name
}

output "service_url" {
  description = "The run.app URL. With IAP enabled this is the address users visit; IAP intercepts it."
  value       = google_cloud_run_v2_service.this.uri
}

output "service_account_email" {
  value = google_service_account.this.email
}
