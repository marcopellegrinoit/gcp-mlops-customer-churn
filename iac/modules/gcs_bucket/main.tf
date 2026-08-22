resource "google_storage_bucket" "this" {
  project                     = var.project_id
  name                        = "${var.project_id}-${var.bucket_name}"
  location                    = var.location
  force_destroy               = var.force_destroy
  uniform_bucket_level_access = true

  dynamic "lifecycle_rule" {
    for_each = var.lifecycle_age_days != null ? [1] : []
    content {
      condition {
        age = var.lifecycle_age_days
      }
      action {
        type = "Delete"
      }
    }
  }
}

resource "google_storage_bucket_iam_member" "this" {
  for_each = { for b in var.iam_bindings : "${b.role}-${b.member}" => b }

  bucket = google_storage_bucket.this.name
  role   = each.value.role
  member = each.value.member
}
