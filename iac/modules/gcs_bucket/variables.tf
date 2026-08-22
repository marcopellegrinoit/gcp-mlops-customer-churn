variable "project_id" {
  type = string
}

variable "location" {
  type = string
}

variable "bucket_name" {
  type = string
}

variable "force_destroy" {
  type    = bool
  default = false
}

variable "lifecycle_age_days" {
  description = "If set, objects older than this many days are automatically deleted."
  type        = number
  default     = null
}

variable "iam_bindings" {
  description = "Bucket-level IAM bindings to grant, e.g. [{ role = \"roles/storage.objectAdmin\", member = \"serviceAccount:...\" }]"
  type = list(object({
    role   = string
    member = string
  }))
  default = []
}
