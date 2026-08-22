variable "project_id" {
  type = string
}

variable "account_id" {
  type = string
}

variable "display_name" {
  type = string
}

variable "service_account_project_roles" {
  type    = list(string)
  default = []
}

variable "bq_dataset_roles" {
  type    = map(string)
  default = {}
}

variable "gcs_bucket_roles" {
  description = "Map of GCS bucket name to IAM role to grant the pipeline service account."
  type        = map(string)
  default     = {}
}

variable "act_as_service_account_emails" {
  description = "Emails of other service accounts this one is allowed to actAs (roles/iam.serviceAccountUser), e.g. to launch jobs under a narrower-scoped SA."
  type        = list(string)
  default     = []
}
