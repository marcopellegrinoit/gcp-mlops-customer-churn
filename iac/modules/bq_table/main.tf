resource "google_bigquery_table" "this" {
  project             = var.project_id
  dataset_id          = var.dataset_id
  table_id            = var.table_id
  description         = var.description
  deletion_protection = false

  friendly_name = var.friendly_name
  schema        = jsonencode(var.schema)

  dynamic "time_partitioning" {
    for_each = var.time_partitioning != null ? [var.time_partitioning] : []
    content {
      type          = time_partitioning.value.type
      field         = try(time_partitioning.value.field, null)
      expiration_ms = try(time_partitioning.value.expiration_ms, null)
    }
  }

  clustering = var.clustering_fields
}
