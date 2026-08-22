variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "job_name" {
  type = string
}

variable "image" {
  type     = string
  nullable = false
  default  = "gcr.io/cloudrun/placeholder"
}

variable "cpu" {
  type    = string
  default = "1"
}

variable "memory" {
  type    = string
  default = "512Mi"
}

variable "timeout" {
  type    = string
  default = "600s"
}

variable "max_retries" {
  type    = number
  default = 1
}

variable "env_vars" {
  type    = map(string)
  default = {}
}

variable "bq_dataset_roles" {
  type    = map(string)
  default = {}
}

variable "gcs_bucket_roles" {
  description = "Map of bucket_name => role to grant this job's service account."
  type        = map(string)
  default     = {}
}

variable "service_account_project_roles" {
  type    = list(string)
  default = []
}

variable "deployer_sa_email" {
  type    = string
  default = ""
}
