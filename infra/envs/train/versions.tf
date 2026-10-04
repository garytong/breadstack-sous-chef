# Sous Chef's training infrastructure: a root of its own, beside envs/int and
# envs/prod, so nothing here can plan a change to the live service and no
# service apply can touch a training VM. It shares the int project
# (breadstack-sous-chef-int) and the int state bucket, under its own prefix.
#
# Ported from breadstack-thermometer/infra/training (the Focaccia VM) with
# the shapes the 9B needs: an A100 40 GB on demand by default, an H100 Spot
# as the fallback (vm.tf, local.shapes). Runbook: README.md in this folder;
# the lab's wrappers are make -C lab train-vm-up / train-vm-ssh /
# train-vm-down / train-remote-push / train-remote-pull.
#
# Everything here is applied on a person's login: the deployer's roles cannot
# create a network, a service account or a budget. Never from an agent.

terraform {
  required_version = ">= 1.7"
  required_providers {
    google = {
      # The same pin as envs/int (locked there at 6.50.0). Provider 6.x cannot
      # request a Flex-start VM (see variables.tf, provisioning_model).
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }

  backend "gcs" {
    # The int state bucket, <project>-sous-chef-tfstate, passed at init as
    # envs/int does (make -C lab train-vm-init):
    #   terraform init -backend-config="bucket=breadstack-sous-chef-int-sous-chef-tfstate"
    # Its own prefix, so the two states never meet.
    prefix = "sous-chef/train"
  }
}
