terraform {
  backend "gcs" {
    # Terraform backends cannot interpolate variables: pass the bucket at init
    # (the bucket is created once by hand, see infra/README.md "Bootstrap"):
    #   terraform init -backend-config="bucket=<project>-sous-chef-tfstate"
    prefix = "sous-chef/int"
  }
}
