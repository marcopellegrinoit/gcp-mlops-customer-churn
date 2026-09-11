variable "project_id" {
  description = "GCP project to deploy resources into."
  type        = string
}

variable "region" {
  description = "GCP provider region."
  type        = string
  default     = "us-central1"
}

variable "alert_email" {
  description = "Email address for operational alert notifications."
  type        = string
  default     = ""
}

variable "alert_from_email" {
  description = "Verified SendGrid sender address operational alert emails are sent from."
  type        = string
  default     = "noreply@mlops-alerts.com"
}

variable "alerts_enabled" {
  description = "Whether the orchestrator workflow sends SendGrid failure/drift alert emails. Keep false until the sendgrid-api-key secret has a version."
  type        = bool
  default     = false
}

variable "dashboard_viewers" {
  description = <<-EOT
    IAM principals allowed through Identity-Aware Proxy to the churn dashboard, e.g.
    ["group:retention-team@example.com", "user:analyst@example.com"]. A group is the
    intended form: team membership then controls access without a Terraform apply per joiner.
    Empty by default — the service deploys reachable by nobody, which is the safe state to
    fail into for a page showing customer-level data.
  EOT
  type        = list(string)
  default     = []
}

variable "dashboard_oauth_client_id" {
  description = <<-EOT
    OAuth 2.0 client ID IAP uses to sign users in to the dashboard.

    Required because this project has no Cloud organization: IAP's Google-managed OAuth
    client only admits "users within the organization in which the resource is contained",
    so with no organization IAP has no client and every request returns 502 "Empty Google
    Account OAuth client ID(s)/secret(s)".

    The client cannot be created by Terraform — the IAP OAuth Admin APIs shut down in March
    2026 and rejected no-organization projects before that. Create it once in the console
    (docs/setup.md), then set this. Leave empty on an organization-owned project to use the
    managed client instead.
  EOT
  type        = string
  default     = ""
}

variable "dashboard_oauth_client_secret" {
  description = "Secret paired with dashboard_oauth_client_id. Lives in terraform.tfvars, which is gitignored."
  type        = string
  default     = ""
  sensitive   = true
}
