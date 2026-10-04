# Provider constraints for the sous-chef module. The root module (envs/<env>)
# configures the providers; this block only declares what the module needs.
# Pinned to the same major versions the thermometer runs on (google 6.50.0,
# random 3.9.0 in its lock file) so the Cloud Run v2 / Cloud SQL shapes
# copied from it behave identically.
terraform {
  required_version = ">= 1.7"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}
