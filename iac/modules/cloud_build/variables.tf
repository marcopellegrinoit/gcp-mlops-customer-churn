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
  description = <<-EOT
    Build triggers, keyed by trigger name. `substitutions` are passed to the build as
    _-prefixed variables; the pipeline-compile build forwards them into the compile step's
    environment, which is how the model policy a pipeline is compiled with is set from here
    rather than from a constant in the image.
  EOT
  type = map(object({
    description    = string
    branch         = string
    included_files = list(string)
    filename       = string
    substitutions  = optional(map(string), {})
  }))
  default = {}
}
