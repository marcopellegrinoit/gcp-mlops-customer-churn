variable "project_id" {
  type = string
}

variable "project_number" {
  description = "Numeric project id. The Budgets API scopes a filter by number, not by id."
  type        = string
}

variable "billing_account_id" {
  description = "Billing account the budget is created under, e.g. \"01471E-834F87-328FE5\"."
  type        = string
}

variable "display_name" {
  type    = string
  default = "MLOps platform monthly budget"
}

variable "amount" {
  description = "Monthly budget in the billing account's currency."
  type        = number
}

variable "currency_code" {
  description = "ISO 4217 code. Empty inherits the billing account's own currency, which is the only value guaranteed to be accepted."
  type        = string
  default     = ""
}

variable "threshold_percents" {
  description = "Fractions of the budget at which an actual-spend alert fires. A forecasted-spend rule at 100% is always added on top."
  type        = list(number)
  default     = [0.5, 0.9, 1.0]
}

variable "notification_emails" {
  description = "Addresses to alert. Empty falls back to the billing account's admins and users, which the Budgets API emails by default."
  type        = list(string)
  default     = []
}
