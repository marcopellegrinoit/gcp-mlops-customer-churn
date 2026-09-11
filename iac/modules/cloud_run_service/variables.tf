variable "project_id" {
  type = string
}

variable "project_number" {
  description = "Numeric project id, used to address the IAP service agent."
  type        = string
}

variable "region" {
  type = string
}

variable "service_name" {
  type = string
}

variable "description" {
  type    = string
  default = ""
}

variable "image" {
  type     = string
  nullable = false
  default  = "gcr.io/cloudrun/placeholder"
}

variable "cpu" {
  type    = string
  default = "1"
}

variable "memory" {
  type    = string
  default = "1Gi"
}

variable "container_port" {
  type    = number
  default = 8080
}

variable "min_instances" {
  description = "Keep at 0 unless cold starts are genuinely unacceptable — see the scaling block."
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Ceiling on concurrent instances. Also the ceiling on a runaway bill."
  type        = number
  default     = 3
}

variable "max_concurrency" {
  description = "Requests one instance serves at once. For a Streamlit container a request is a session-long WebSocket, so this is really a ceiling on simultaneous readers."
  type        = number
  default     = 20
}

variable "request_timeout" {
  type    = string
  default = "3600s"
}

variable "env_vars" {
  type    = map(string)
  default = {}
}

variable "bq_dataset_roles" {
  type    = map(string)
  default = {}
}

variable "service_account_project_roles" {
  type    = list(string)
  default = []
}

variable "iap_enabled" {
  description = "Front the service with Identity-Aware Proxy. Disabling it leaves the service invocable by nobody until an invoker binding is added by hand."
  type        = bool
  default     = true
}

variable "iap_members" {
  description = "IAM principals allowed through IAP, e.g. [\"group:retention@example.com\"]."
  type        = list(string)
  default     = []
}

variable "deployer_sa_email" {
  type    = string
  default = ""
}

variable "oauth_client_id" {
  description = "OAuth 2.0 client ID for IAP's sign-in flow. Required for projects with no Cloud organization, where the Google-managed client is unavailable. Empty leaves IAP on the managed client."
  type        = string
  default     = ""
}

variable "oauth_client_secret" {
  description = "Secret paired with oauth_client_id."
  type        = string
  default     = ""
  sensitive   = true
}
