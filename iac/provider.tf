terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source = "hashicorp/google"
      # Floor raised from 4.0 for the dashboard's Cloud Run service: `iap_enabled` on
      # google_cloud_run_v2_service and google_iap_web_cloud_run_service_iam_member are what
      # make IAP-without-a-load-balancer expressible, and both postdate the 4.x line. On an
      # older provider the IAP field is silently unknown and the service deploys unprotected.
      version = ">= 7.0"
    }
    # Declared for exactly one resource: google_project_service_identity, which provisions
    # IAP's service agent and has no GA equivalent. GCP creates that agent lazily, so
    # without it the invoker binding in modules/cloud_run_service can reference a principal
    # that does not exist yet and fail the first apply. Nothing else in this configuration
    # uses the beta provider — check before reaching for it.
    google-beta = {
      source  = "hashicorp/google-beta"
      version = ">= 7.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# Used by exactly one module: billing_budget. Same discipline as the beta provider above —
# scoped to what needs it rather than applied globally.
#
# The Budgets API is the only API in this configuration that requires a *quota project* on
# the caller's credentials. User Application Default Credentials (`gcloud auth
# application-default login`) carry none, so without this the request is quota-attributed to
# Google's own shared Cloud SDK project (764086051850) and fails with a SERVICE_DISABLED 403
# naming a project id nobody here recognises. user_project_override makes the provider send
# X-Goog-User-Project, pointing the quota at this project instead.
#
# Deliberately not set on the default provider: every other resource applies cleanly as-is,
# and changing the request path for all of them to fix one API would trade a known-good
# configuration for an untested one. The cost of the header is that the caller needs
# serviceusage.services.use on the billing project, which a project owner has.
provider "google" {
  alias   = "billing"
  project = var.project_id
  region  = var.region

  user_project_override = true
  billing_project       = var.project_id
}
