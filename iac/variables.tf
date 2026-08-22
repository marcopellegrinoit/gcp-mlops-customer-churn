variable "project_id" {
  description = "GCP project to deploy resources into."
  type        = string
}

variable "region" {
  description = "GCP provider region."
  type        = string
  default     = "us-central1"
}

variable "alert_email" {
  description = "Email address for operational alert notifications."
  type        = string
  default     = ""
}

variable "alert_from_email" {
  description = "Verified SendGrid sender address operational alert emails are sent from."
  type        = string
  default     = "noreply@mlops-alerts.com"
}

variable "alerts_enabled" {
  description = "Whether the orchestrator workflow sends SendGrid failure/drift alert emails. Keep false until the sendgrid-api-key secret has a version."
  type        = bool
  default     = false
}
