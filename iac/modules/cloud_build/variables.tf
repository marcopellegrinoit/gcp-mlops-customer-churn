variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "github_owner" {
  type = string
}

variable "github_repo" {
  type = string
}

variable "connection_name" {
  type        = string
  description = "Name of the Cloud Build v2 connection created in GCP Console."
}

variable "sa_roles" {
  type    = list(string)
  default = []
}

variable "triggers" {
  type = map(object({
    description    = string
    branch         = string
    included_files = list(string)
    filename       = string
  }))
  default = {}
}
