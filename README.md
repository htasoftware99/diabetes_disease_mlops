# Diabetes Disease MLOps Project

This project implements a robust, end-to-end MLOps pipeline for predicting diabetes using the Pima Indians Diabetes Dataset. It encompasses data ingestion, feature engineering, model training, experiment tracking, automated orchestration, deployment, and real-time monitoring.

## 🚀 Project Overview

The goal of this project is to build a scalable and production-ready machine learning system that not only predicts diabetes but also monitors data drift and model performance in real-time.

## 🏗️ System Architecture

<p align="center">
  <img src="structure.png" width="700"/>
</p>

### Key Features:
- **Automated ETL**: Orchestrated using Apache Airflow (Astro CLI).
- **Orchestration**: Scalable ML pipelines defined using **Kubeflow Pipelines (KFP)** for containerized execution.
- **Data Versioning**: Managed via DVC (Data Version Control) with GCS backend.
- **Feature Store**: Redis-based feature store for low latency data retrieval.
- **Experiment Tracking**: Comprehensive logging of metrics and models using Comet ML.
- **Drift Detection**: Real-time data drift monitoring using Alibi Detect (Kolmogorov-Smirnov test).
- **Observability**: Prometheus and Graphana integration for monitoring prediction counts, latency, and drift events.
- **Infrastructure as Code**: Cloud resources (GCP) provisioned via Terraform.
- **Containerization**: Fully Dockerized application for consistent deployment.

---

## 🛠 Tech Stack

| Category | Tools |
| :--- | :--- |
| **Language** | Python 3.x |
| **Machine Learning** | Scikit-learn, XGBoost, Pandas, NumPy |
| **API Framework** | FastAPI, Uvicorn |
| **Orchestration** | Apache Airflow, Kubeflow |
| **Database/Storage** | PostgreSQL, Redis, Google Cloud Storage (GCS) |
| **MLOps & DevOps** | DVC, Docker, Terraform, Comet ML |
| **Monitoring** | Prometheus, Graphana, Alibi Detect |

---

## 📁 Project Structure

```text
├── .astro/                 # Airflow configuration (Astro CLI)
├── .dvc/                   # Data Version Control configuration
├── artifacts/              # Model and data artifacts (DVC-tracked)
├── config/                 # Configuration files (DB, Paths)
├── dags/                   # Apache Airflow DAGs for ETL
├── infrastructure/         # Terraform files for GCP resources
├── kubeflow_pipeline/      # Kubeflow pipeline definitions
├── notebooks/              # Exploratory Data Analysis (EDA)
├── src/                    # Core source code
│   ├── data_ingestion.py   # Extraction from Postgres/GCS
│   ├── data_processing.py  # Cleaning, scaling, and Redis storage
│   ├── feature_store.py    # Redis Feature Store implementation
│   ├── model_training.py   # Training with Comet ML tracking
│   └── logger.py           # Custom logging module
├── static/ & templates/    # Frontend for FastAPI web interface
├── main.py                 # FastAPI application entry point
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
└── setup.py                # Project packaging
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd diabetes_disease_mlops
```

### 2. Environment Variables
Create a `.env` file in the root directory and add your credentials:
```env
COMET_API_KEY=your_comet_api_key
COMET_PROJECT_NAME=diabetes-mlops
COMET_WORKSPACE=your_workspace
# Add other DB/Cloud credentials as needed
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Infrastructure Setup (Terraform)
```bash
cd infrastructure
terraform init
terraform plan
terraform apply
```

### 5. Data Versioning (DVC)
```bash
dvc pull
```

---

## 🔄 Pipeline Workflow

1.  **Ingestion**: Airflow DAG `extract_diabetes_data` pulls raw data from GCS and loads it into a PostgreSQL database.
2.  **Processing**: `src/data_processing.py` cleans the data, scales features, and stores them in the **Redis Feature Store**.
3.  **Training**: `src/model_training.py` retrieves features from Redis, trains a Logistic Regression model, and logs everything to **Comet ML**.
4.  **Deployment**: The FastAPI app (`main.py`) serves the model.
5.  **Monitoring**:
    -   **Drift**: `Alibi Detect` compares incoming request distributions with training data.
    -   **Metrics**: Prometheus scrapes `/metrics` for real-time dashboarding.

---

## 🖥 Running the Application

### Locally with FastAPI
```bash
python main.py
```
The application will be available at `http://localhost:5000`.

### With Docker
```bash
docker build -t diabetes-mlops .
docker run -p 5000:5000 diabetes-mlops
```

---

## 📊 Monitoring & Drift Detection

The application monitors for **Data Drift** using the Kolmogorov-Smirnov (KS) test. If the statistical distribution of incoming features significantly deviates from the training data, a warning is triggered.

Metrics exposed via `/metrics`:
- `prediction_count`: Total requests served.
- `drift_count`: Number of drift events detected.
- `prediction_latency_seconds`: Histogram of processing time.
