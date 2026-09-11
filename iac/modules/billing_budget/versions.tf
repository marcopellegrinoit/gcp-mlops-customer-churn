# The only module here that declares its own required_providers, because it is the only one
# that receives a non-default provider configuration: main.tf passes the `google.billing`
# alias in, and without this declaration Terraform only infers the mapping and warns.
terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
    }
  }
}
