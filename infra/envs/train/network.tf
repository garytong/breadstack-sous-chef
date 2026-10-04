# A small network of the training VM's own, not the project's default one.
# The default network (auto mode) lets anything in 10.128.0.0/9 reach every
# port of its VMs (default-allow-internal) and opens SSH and RDP to the whole
# internet (default-allow-ssh, -rdp). Here the only way in is SSH from IAP,
# and nothing else shares the network.
#
#   no external address  the VM is unreachable from the internet
#   IAP SSH only         tcp:22 from IAP's TCP-forwarding range, 35.235.240.0/20;
#                        who may connect is IAM's decision (OS Login + IAP)
#   egress: web only     tcp 80/443 out (apt, PyPI, PyTorch's index, Hugging
#                        Face, GitHub for llama.cpp, Google APIs), everything
#                        else refused; the metadata server (DNS, NTP, tokens)
#                        is always reachable
#   Cloud NAT            the way out for 80/443
#   Private Google Access  the bucket is reached on Google's network, not
#                        through NAT: no NAT processing charge on the 19 GB
#                        base copy, and no internet path for the data

resource "google_compute_network" "train" {
  name                    = "sous-chef-train"
  description             = "Sous Chef training VM only (infra/envs/train): IAP SSH in, web out through Cloud NAT"
  auto_create_subnetworks = false
  routing_mode            = "REGIONAL"

  depends_on = [google_project_service.train]
}

resource "google_compute_subnetwork" "train" {
  name                     = "sous-chef-train-${local.region}"
  region                   = local.region
  network                  = google_compute_network.train.id
  ip_cidr_range            = "10.20.0.0/24"
  private_ip_google_access = true
  stack_type               = "IPV4_ONLY"
}

resource "google_compute_firewall" "iap_ssh" {
  name        = "sous-chef-train-allow-iap-ssh"
  network     = google_compute_network.train.id
  description = "SSH from IAP's TCP forwarding only (make -C lab train-vm-ssh)"
  direction   = "INGRESS"
  priority    = 1000

  source_ranges           = ["35.235.240.0/20"]
  target_service_accounts = [google_service_account.train.email]

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
}

resource "google_compute_firewall" "egress_web" {
  name        = "sous-chef-train-allow-web-out"
  network     = google_compute_network.train.id
  description = "Out on 80/443 only: apt, PyPI, PyTorch, Hugging Face, GitHub, Google APIs"
  direction   = "EGRESS"
  priority    = 1000

  destination_ranges      = ["0.0.0.0/0"]
  target_service_accounts = [google_service_account.train.email]

  allow {
    protocol = "tcp"
    ports    = ["80", "443"]
  }
}

resource "google_compute_firewall" "egress_deny" {
  name        = "sous-chef-train-deny-other-out"
  network     = google_compute_network.train.id
  description = "Everything else out is refused (the metadata server stays reachable)"
  direction   = "EGRESS"
  priority    = 65000

  destination_ranges = ["0.0.0.0/0"]

  deny {
    protocol = "all"
  }
}

resource "google_compute_router" "train" {
  name    = "sous-chef-train"
  region  = local.region
  network = google_compute_network.train.id
}

# Billed only while a VM uses it (about US$0.0014/hour per VM) plus
# US$0.045/GB it carries: the uv install (5.1 GB from a cold cache, ~US$0.25)
# and, when the bucket has no base/ copy, the 19 GB base from Hugging Face
# (~US$0.85). make -C lab train-remote-push puts the base in the bucket once,
# and the startup script reads it over Private Google Access instead.
resource "google_compute_router_nat" "train" {
  name                               = "sous-chef-train"
  region                             = local.region
  router                             = google_compute_router.train.name
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "LIST_OF_SUBNETWORKS"

  subnetwork {
    name                    = google_compute_subnetwork.train.id
    source_ip_ranges_to_nat = ["ALL_IP_RANGES"]
  }

  # uv downloads up to 50 files at once, many from the same CDN address;
  # the default 64 ports per VM would throttle it. Ports cost nothing.
  min_ports_per_vm = 1024

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}
