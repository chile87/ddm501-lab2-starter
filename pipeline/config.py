"""
Configuration settings for ML Pipeline.
"""

import os
from pathlib import Path

# =============================================================================
# Paths
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

# =============================================================================
# MLflow Configuration
# =============================================================================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "movie-rating-prediction")

# =============================================================================
# Data Configuration
# =============================================================================
DATASET_NAME = "ml-100k"  # MovieLens 100K
TEST_SIZE = 0.2
RANDOM_STATE = 42

# =============================================================================
# Model Configuration
# =============================================================================
# Available model types: 'svd', 'nmf', 'knn'
DEFAULT_MODEL_TYPE = "svd"

# Default hyperparameters for each model type
MODEL_CONFIGS = {
    "svd": {
        "n_factors": 100,
        "n_epochs": 20,
        "lr_all": 0.005,
        "reg_all": 0.02,
    },
    "nmf": {
        "n_factors": 50,
        "n_epochs": 50,
    },
    "knn": {
        "k": 40,
        "sim_options": {
            "name": "cosine",
            "user_based": True,
        },
    },
}

# =============================================================================
# Experiment Configurations for Hyperparameter Tuning
# =============================================================================
EXPERIMENT_CONFIGS = [
    # SVD experiments
    {"model_type": "svd", "n_factors": 50, "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02},
    {"model_type": "svd", "n_factors": 100, "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02},
    {"model_type": "svd", "n_factors": 100, "n_epochs": 50, "lr_all": 0.005, "reg_all": 0.02},
    {"model_type": "svd", "n_factors": 150, "n_epochs": 30, "lr_all": 0.01, "reg_all": 0.02},
    
    # NMF experiments
    {"model_type": "nmf", "n_factors": 50, "n_epochs": 50},
    {"model_type": "nmf", "n_factors": 100, "n_epochs": 50},
    
    # KNN experiments
    {"model_type": "knn", "k": 20, "sim_options": {"name": "cosine", "user_based": True}},
    {"model_type": "knn", "k": 40, "sim_options": {"name": "cosine", "user_based": True}},
    {"model_type": "knn", "k": 40, "sim_options": {"name": "pearson", "user_based": True}},
]

# =============================================================================
# Airflow Configuration
# =============================================================================
AIRFLOW_DAG_ID = "movie_rating_training"
AIRFLOW_SCHEDULE = "@weekly"  # Run weekly
