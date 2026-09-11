resource "google_service_account" "this" {
  project      = var.project_id
  account_id   = "${var.service_name}-sa"
  display_name = "${var.service_name} Cloud Run Service service account"
}

resource "google_project_iam_member" "this" {
  for_each = toset(var.service_account_project_roles)

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.this.email}"
}

resource "google_bigquery_dataset_iam_member" "this" {
  for_each = var.bq_dataset_roles

  project    = var.project_id
  dataset_id = each.key
  role       = each.value
  member     = "serviceAccount:${google_service_account.this.email}"
}

resource "google_service_account_iam_member" "deployer_act_as" {
  count = var.deployer_sa_email != "" ? 1 : 0

  service_account_id = google_service_account.this.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${var.deployer_sa_email}"
}

resource "google_cloud_run_v2_service" "this" {
  name                = var.service_name
  location            = var.region
  project             = var.project_id
  description         = var.description
  deletion_protection = false

  # INGRESS_TRAFFIC_ALL is required, not a relaxation: with IAP enabled directly on Cloud
  # Run, IAP fronts the run.app URL itself and forwards over the public ingress path. The
  # service is not thereby public — invocation still requires roles/run.invoker, which only
  # IAP's service agent holds (see google_cloud_run_v2_service_iam_member below). Setting
  # this to INTERNAL would instead require the load-balancer IAP topology this module exists
  # to avoid, which costs money and is the configuration Streamlit's WebSocket upgrade is
  # known to break under.
  ingress = "INGRESS_TRAFFIC_ALL"

  # The whole point of the direct integration: authentication at the edge, no load balancer,
  # no forwarding rule, no static IP — none of which have a free tier.
  iap_enabled = var.iap_enabled

  template {
    service_account = google_service_account.this.email

    scaling {
      # Zero, deliberately. Cloud Run's always-free allowance is 180k vCPU-seconds a month;
      # one instance pinned warm for a 730-hour month is roughly 2.6M, so min_instance_count
      # = 1 would take this service out of the free tier by an order of magnitude. The cost
      # is a cold start on the first request of the day, which startup CPU boost absorbs.
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    containers {
      image = var.image

      ports {
        container_port = var.container_port
      }

      dynamic "env" {
        for_each = var.env_vars
        content {
          name  = env.key
          value = env.value
        }
      }

      resources {
        limits = {
          cpu    = var.cpu
          memory = var.memory
        }
        # Streamlit holds the scored snapshot in a per-process cache and serves every
        # interaction from it over a long-lived WebSocket, so the instance must keep running
        # between requests. Throttling CPU outside a request would stall the session.
        cpu_idle          = false
        startup_cpu_boost = true
      }

      startup_probe {
        # Streamlit's own health endpoint. Without an explicit probe Cloud Run uses a TCP
        # check, which succeeds as soon as the port is bound — before the app can serve —
        # so the first request of a cold start lands on a server that is not ready yet.
        http_get {
          path = "/_stcore/health"
          port = var.container_port
        }
        initial_delay_seconds = 5
        timeout_seconds       = 3
        period_seconds        = 5
        failure_threshold     = 12
      }
    }

    # Streamlit's session is a WebSocket held open for as long as the tab is; the default
    # 5-minute request timeout would drop an idle reader's connection mid-session.
    timeout = var.request_timeout
  }

  lifecycle {
    # Cloud Build deploys new revisions by digest; Terraform owns the shape of the service,
    # not which image is current. Same split as cloud_run_job.
    ignore_changes = [template[0].containers[0].image, client, client_version]
  }
}

# IAP calls Cloud Run as its own service agent, so that agent — and nothing else — is what
# holds invoker on this service. This is the authorisation boundary: a request that has not
# come through IAP has no identity that can invoke the service at all.
resource "google_cloud_run_v2_service_iam_member" "iap_invoker" {
  count = var.iap_enabled ? 1 : 0

  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.this.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:service-${var.project_number}@gcp-sa-iap.iam.gserviceaccount.com"
}

# Who IAP will let through. Members are ordinary IAM principals, so in a real deployment
# this is a Google Group ("group:retention-team@example.com") and team membership becomes
# the access control — no user list living in Terraform.
resource "google_iap_web_cloud_run_service_iam_member" "viewers" {
  for_each = var.iap_enabled ? toset(var.iap_members) : toset([])

  project                = var.project_id
  location               = var.region
  cloud_run_service_name = google_cloud_run_v2_service.this.name
  role                   = "roles/iap.httpsResourceAccessor"
  member                 = each.value
}

# The OAuth client IAP runs its sign-in flow with.
#
# This is separate from everything above because it cannot be fully automated in this
# project. IAP will use a *Google-managed* OAuth client only when the project belongs to a
# Cloud organization — the managed client scopes access to "users within the organization",
# and a standalone project has no such set. Without an organization IAP has no client at
# all, and every request to the service returns 502 "Empty Google Account OAuth client
# ID(s)/secret(s)".
#
# The old escape hatch — creating a brand and client through google_iap_brand /
# google_iap_client — is gone: the IAP OAuth Admin APIs were permanently shut down in March
# 2026, and they refused no-organization projects before that anyway. So the client is
# created once by hand in the console (see docs/setup.md) and only *attached* here.
#
# Left unset, this resource is not created and IAP falls back to the managed client, which
# is the correct behaviour for an organization-owned project.
resource "google_iap_settings" "oauth" {
  count = var.iap_enabled && var.oauth_client_id != "" ? 1 : 0

  # Addressed by project *number*, which is what the IAP API returns. Using the project id
  # here would leave a permanent diff on every plan.
  name = "projects/${var.project_number}/iap_web/cloud_run-${var.region}/services/${google_cloud_run_v2_service.this.name}"

  access_settings {
    oauth_settings {
      client_id     = var.oauth_client_id
      client_secret = var.oauth_client_secret
    }
  }

  depends_on = [google_cloud_run_v2_service.this]
}
