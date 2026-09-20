resource "google_bigquery_dataset" "this" {
  project                     = var.project_id
  dataset_id                  = var.dataset_id
  location                    = var.location
  description                 = var.description
  default_table_expiration_ms = var.default_table_expiration_ms
}
