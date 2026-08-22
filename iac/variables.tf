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
