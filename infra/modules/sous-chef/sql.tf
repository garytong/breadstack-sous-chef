# Cloud SQL Postgres. Shape copied from breadstack-thermometer/infra/sql.tf
# (MySQL there; POSTGRES_16 here per README section 9), with an optional
# private-IP mode behind sql_private_ip. In-transit encryption is required
# (ssl_mode = ENCRYPTED_ONLY) in both modes.

# --- Optional private IP: a reserved range peered to Service Networking ---------

resource "google_compute_global_address" "sql_private_range" {
  count = var.sql_private_ip ? 1 : 0

  project       = var.project_id
  name          = "sous-chef-sql-private-range"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = var.sql_private_network

  depends_on = [google_project_service.services]
}

resource "google_service_networking_connection" "sql_private" {
  count = var.sql_private_ip ? 1 : 0

  network                 = var.sql_private_network
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.sql_private_range[0].name]

  depends_on = [google_project_service.services]
}

# --- Instance -----------------------------------------------------------------------

resource "google_sql_database_instance" "sous_chef" {
  project          = var.project_id
  name             = "sous-chef"
  database_version = "POSTGRES_16"
  region           = var.region

  settings {
    # Shared-core tiers (db-f1-micro, db-g1-small) exist only in the Enterprise
    # edition; the provider otherwise defaults POSTGRES_16 to ENTERPRISE_PLUS,
    # which rejects them ("Invalid Tier (db-f1-micro) for (ENTERPRISE_PLUS)").
    edition   = "ENTERPRISE"
    tier      = var.db_tier
    disk_size = var.sql_disk_size
    disk_type = "PD_SSD"

    backup_configuration {
      enabled                        = true
      point_in_time_recovery_enabled = true
    }

    ip_configuration {
      # Default mode (sql_private_ip = false, int): a public IPv4 with no
      # authorized networks. The Run service reaches it through the Cloud SQL
      # connector volume (/cloudsql/<conn>), which authenticates with the
      # runtime SA's roles/cloudsql.client and encrypts in transit; ssl_mode
      # ENCRYPTED_ONLY additionally rejects any unencrypted direct client on
      # the public address. The sslmode=disable in the DSN below is for the
      # unix socket to the connector, not for the instance.
      # Private mode (sql_private_ip = true, prod): no public IP at all; the
      # instance lives on sql_private_network and the Run service/jobs reach it
      # over direct VPC egress. ENCRYPTED_ONLY is kept there too.
      ipv4_enabled    = !var.sql_private_ip
      private_network = var.sql_private_ip ? var.sql_private_network : null
      ssl_mode        = "ENCRYPTED_ONLY"
    }

    insights_config {
      query_insights_enabled = true
    }
  }

  deletion_protection = var.sql_deletion_protection

  lifecycle {
    precondition {
      condition     = !var.sql_private_ip || (var.sql_private_network != "" && var.sql_private_subnetwork != "")
      error_message = "sql_private_ip = true needs sql_private_network (VPC self link) and sql_private_subnetwork (for the Run service's direct VPC egress)."
    }
  }

  depends_on = [
    google_project_service.services,
    google_service_networking_connection.sql_private,
  ]
}

resource "google_sql_database" "sous_chef" {
  project  = var.project_id
  name     = "sous_chef"
  instance = google_sql_database_instance.sous_chef.name
}

resource "random_password" "db" {
  length  = 32
  special = false
}

resource "google_sql_user" "sous_chef" {
  project  = var.project_id
  name     = "sous_chef"
  instance = google_sql_database_instance.sous_chef.name
  password = random_password.db.result
}

locals {
  # pgx DSN over the Cloud SQL connector's unix socket. pgx accepts the
  # socket DIRECTORY as host (it appends /.s.PGSQL.5432 itself).
  database_url = format(
    "postgres://sous_chef:%s@/sous_chef?host=/cloudsql/%s&sslmode=disable",
    random_password.db.result,
    google_sql_database_instance.sous_chef.connection_name,
  )
}
