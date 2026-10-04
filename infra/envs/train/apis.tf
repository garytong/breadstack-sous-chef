# The APIs this root uses. compute and storage are on in int already (read
# 2026-10-04; modules/sous-chef manages storage too) and are listed here so a
# fresh project works; iap and billingbudgets are not on yet:
#   iap.googleapis.com            make -C lab train-vm-ssh: SSH through IAP's
#                                 TCP forwarding (the VM has no external address)
#   billingbudgets.googleapis.com the budget (budget.tf), called with this
#                                 project as its quota project
# None is ever turned off by a destroy.
resource "google_project_service" "train" {
  for_each = toset([
    "compute.googleapis.com",
    "storage.googleapis.com",
    "iap.googleapis.com",
    "billingbudgets.googleapis.com",
  ])
  service            = each.value
  disable_on_destroy = false
}
