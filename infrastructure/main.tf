resource "google_storage_bucket" "mlops_artifacts_bucket" {
  name                        = "diabetes_disease_bucket_neat_chain_464913"
  location                    = "US"
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  public_access_prevention    = "inherited"
  soft_delete_policy {
    retention_duration_seconds = 604800
  }
}

resource "google_storage_bucket" "dvc_artifacts_bucket" {
  name                        = "diabetes_disease_dvc"
  location                    = "US"
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  public_access_prevention    = "inherited"
  soft_delete_policy {
    retention_duration_seconds = 604800
  }
}