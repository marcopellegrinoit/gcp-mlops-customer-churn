variable "project_id" {
  type = string
}

variable "dataset_id" {
  type = string
}

variable "table_id" {
  type = string
}

variable "description" {
  type = string
}

variable "friendly_name" {
  type    = string
  default = null
}

variable "schema" {
  type = list(object({
    name        = string
    type        = string
    mode        = string
    description = optional(string)
  }))
}

variable "time_partitioning" {
  type = object({
    type          = string
    field         = optional(string)
    expiration_ms = optional(number)
  })
  default = null
}

variable "clustering_fields" {
  type    = list(string)
  default = null
}
