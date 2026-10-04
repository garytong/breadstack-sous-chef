# Docker repository for the service image. Shape copied from
# breadstack-thermometer/infra/artifact.tf. Project-private: the deployer
# pushes with project-level artifactregistry.writer and Cloud Run's service
# agent pulls with its own role; no repository-level IAM.
resource "google_artifact_registry_repository" "sous_chef" {
  project       = var.project_id
  repository_id = "sous-chef"
  format        = "DOCKER"
  location      = var.region
  description   = "Docker images for sous-chef (${var.env})"

  depends_on = [google_project_service.services]
}
