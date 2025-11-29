terraform {
  required_version = ">= 1.3.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 4.0"
    }
  }
}

provider "google" {
  project = "your-gcp-project-id"
  region  = "asia-northeast1"
  zone    = "asia-northeast1-b"
}

resource "google_container_cluster" "llm_scale_lab" {
  name     = "llm-scale-lab-gke"
  location = "asia-northeast1-b"

  remove_default_node_pool = true
  initial_node_count       = 1

  networking_mode = "VPC_NATIVE"

  ip_allocation_policy {}
}

resource "google_container_node_pool" "primary_nodes" {
  name       = "primary-nodes"
  location   = google_container_cluster.llm_scale_lab.location
  cluster    = google_container_cluster.llm_scale_lab.name
  node_count = 1

  node_config {
    machine_type = "e2-standard-2"
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]
  }
}