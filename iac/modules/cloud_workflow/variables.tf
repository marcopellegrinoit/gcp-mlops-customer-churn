variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "workflow_name" {
  type = string
}

variable "description" {
  type    = string
  default = ""
}

variable "source_contents" {
  type        = string
  description = "Full YAML source of the Cloud Workflow."
}

variable "schedule" {
  type        = string
  default     = ""
  description = "Cron schedule for the Cloud Scheduler job. Leave empty to skip scheduler creation."
}

variable "alert_email" {
  type    = string
  default = ""
}

variable "alert_from_email" {
  type    = string
  default = ""
}

variable "service_account_project_roles" {
  type    = list(string)
  default = []
}

variable "gcs_bucket_roles" {
  description = "Map of bucket_name => role to grant the workflow's service account."
  type        = map(string)
  default     = {}
}

variable "act_as_service_account_emails" {
  description = "SA emails this workflow's service account may impersonate (roles/iam.serviceAccountUser), e.g. to submit a PipelineJob under another module's runtime SA."
  type        = list(string)
  default     = []
}
