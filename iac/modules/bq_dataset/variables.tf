variable "project_id" {
  type = string
}

variable "location" {
  type = string
}

variable "dataset_id" {
  type = string
}

variable "description" {
  type = string
}

variable "default_table_expiration_ms" {
  type    = number
  default = null
}
