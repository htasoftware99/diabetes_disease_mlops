terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "7.29.0"
    }
  }
}

provider "google" {
  project = "neat-chain-464913-k3"
  region  = "us-central1"
  zone    = "us-central1-a"
}