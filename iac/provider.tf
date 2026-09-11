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
