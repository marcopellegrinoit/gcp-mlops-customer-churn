variable "project_id" {
  type = string
}

variable "location" {
  type = string
}

variable "repository_id" {
  type = string
}

variable "description" {
  type    = string
  default = ""
}

variable "format" {
  type    = string
  default = "DOCKER"
}

variable "reader_service_account_emails" {
  description = "Emails of service accounts to grant roles/artifactregistry.reader on this repository, e.g. so a job's execution SA can pull its own container image."
  type        = list(string)
  default     = []
}
