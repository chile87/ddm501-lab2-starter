"""
Port and Server Configuration for DDM501 Lab 2 MLOps Services.
Centralizes all service ports and default URLs used across MLflow, Airflow, and PostgreSQL.
"""

import os

# =============================================================================
# Default Port Settings
# =============================================================================

# MLflow Tracking Server & Model Registry UI
MLFLOW_PORT = int(os.getenv("MLFLOW_PORT", 5000))
MLFLOW_HOST = os.getenv("MLFLOW_HOST", "0.0.0.0")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"http://localhost:{MLFLOW_PORT}")

# Airflow Webserver UI
AIRFLOW_PORT = int(os.getenv("AIRFLOW_PORT", 8080))
AIRFLOW_HOST = os.getenv("AIRFLOW_HOST", "0.0.0.0")
AIRFLOW_WEBSERVER_URL = os.getenv("AIRFLOW_WEBSERVER_URL", f"http://localhost:{AIRFLOW_PORT}")

# PostgreSQL Database (Airflow Backend Metadata Store)
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")

# =============================================================================
# Summary Mapping for System Verification / Debugging
# =============================================================================
SERVICE_PORTS = {
    "mlflow": {
        "port": MLFLOW_PORT,
        "host": MLFLOW_HOST,
        "url": MLFLOW_TRACKING_URI,
        "description": "MLflow Tracking Server & Model Registry UI",
    },
    "airflow": {
        "port": AIRFLOW_PORT,
        "host": AIRFLOW_HOST,
        "url": AIRFLOW_WEBSERVER_URL,
        "description": "Apache Airflow Webserver UI",
    },
    "postgres": {
        "port": POSTGRES_PORT,
        "host": POSTGRES_HOST,
        "url": f"postgresql://airflow:airflow@{POSTGRES_HOST}:{POSTGRES_PORT}/airflow",
        "description": "PostgreSQL Database for Airflow Metadata",
    },
}
