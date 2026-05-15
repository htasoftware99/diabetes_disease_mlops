resource "google_service_account" "diabetes_disease" {
  account_id   = "diabetes-disease-mlops"
  display_name = "diabetes Disease MLOps Pipeline Service Account"
}

resource "google_project_iam_member" "ht_storage_admin" {
  project = "neat-chain-464913-k3"
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.diabetes_disease.email}"
}

resource "google_project_iam_member" "ht_storage_viewer" {
  project = "neat-chain-464913-k3"
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.diabetes_disease.email}"
}

resource "google_project_iam_member" "ht_owner" {
  project = "neat-chain-464913-k3"
  role    = "roles/owner"
  member  = "serviceAccount:${google_service_account.diabetes_disease.email}"
}

resource "google_service_account_key" "diabetes_disease_key" {
  service_account_id = google_service_account.diabetes_disease.name
}

resource "local_file" "ht_key_file" {
  content  = base64decode(google_service_account_key.diabetes_disease_key.private_key)
  filename = "${path.module}/diabetes-disease-gcp-key.json"
}

resource "google_storage_bucket_iam_member" "bucket_object_admin" {
  bucket = google_storage_bucket.mlops_artifacts_bucket.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.diabetes_disease.email}"
}

resource "google_storage_bucket_iam_member" "bucket_object_viewer_bucket_level" {
  bucket = google_storage_bucket.mlops_artifacts_bucket.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.diabetes_disease.email}"
}

resource "google_storage_bucket_iam_member" "dvc_bucket_object_admin" {
  bucket = google_storage_bucket.dvc_artifacts_bucket.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.diabetes_disease.email}"
}

resource "google_storage_bucket_iam_member" "dvc_bucket_object_viewer" {
  bucket = google_storage_bucket.dvc_artifacts_bucket.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.diabetes_disease.email}"
}