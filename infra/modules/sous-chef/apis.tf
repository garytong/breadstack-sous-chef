# Project services the module needs. Shape copied from
# breadstack-thermometer/infra/apis.tf (disable_on_destroy = false so a
# destroy of this module never switches an API off under another workload).
locals {
  apis = concat([
    "run.googleapis.com",
    "sqladmin.googleapis.com",
    "secretmanager.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudscheduler.googleapis.com",
    "compute.googleapis.com",
    "iam.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    ],
    # Private-IP Cloud SQL needs a VPC peering with the Service Networking API.
    var.sql_private_ip ? ["servicenetworking.googleapis.com", "vpcaccess.googleapis.com"] : [],
  )
}

resource "google_project_service" "services" {
  for_each           = toset(local.apis)
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}
