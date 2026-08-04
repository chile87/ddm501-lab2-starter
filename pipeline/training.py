"""
Model Training Stage for ML Pipeline.

This module handles:
- Model initialization
- Model training
- MLflow experiment tracking

TODO: Complete the functions marked with TODO.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import mlflow
from surprise import SVD, NMF, KNNBasic

from pipeline.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_CONFIGS,
    MODELS_DIR,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Model Classes Registry
# =============================================================================
MODEL_CLASSES = {
    "svd": SVD,
    "nmf": NMF,
    "knn": KNNBasic,
}


def setup_mlflow(
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME
) -> None:
    """
    Setup MLflow tracking.
    
    Args:
        tracking_uri: MLflow tracking server URI
        experiment_name: Name of the experiment
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow configured: URI={tracking_uri}, Experiment={experiment_name}")


# =============================================================================
# TODO 1: Implement train_model function
# =============================================================================
def train_model(
    trainset: Any,
    model_type: str = "svd",
    run_name: Optional[str] = None,
    **model_params
) -> Tuple[Any, str]:
    """
    Train a recommendation model and log to MLflow.
    
    TODO: Implement this function with the following requirements:
    
    1. Start an MLflow run (with optional run_name)
    2. Log all model parameters to MLflow:
       - model_type
       - All hyperparameters from model_params
    3. Initialize the model using MODEL_CLASSES[model_type]
    4. Train the model using model.fit(trainset)
    5. Save the model to a pickle file
    6. Log the model file as an MLflow artifact
    7. Return the trained model and the run_id
    
    Args:
        trainset: Surprise trainset object
        model_type: Type of model ('svd', 'nmf', 'knn')
        run_name: Optional name for the MLflow run
        **model_params: Model hyperparameters
        
    Returns:
        Tuple of (trained_model, run_id)
        
    Example:
        model, run_id = train_model(
            trainset, 
            model_type='svd',
            n_factors=100,
            n_epochs=20
        )
    """
    # TODO: Implement this function
    #
    # Hints:
    # 1. Use mlflow.start_run(run_name=run_name) to start a run
    # 2. Use mlflow.log_param(key, value) to log parameters
    # 3. Use mlflow.log_artifact(file_path) to log the model file
    # 4. Use mlflow.active_run().info.run_id to get the run_id
    #
    # Example structure:
    # with mlflow.start_run(run_name=run_name):
    #     # Log parameters
    #     mlflow.log_param("model_type", model_type)
    #     for key, value in model_params.items():
    #         mlflow.log_param(key, value)
    #     
    #     # Initialize model
    #     model_class = MODEL_CLASSES[model_type]
    #     model = model_class(**model_params)
    #     
    #     # Train model
    #     logger.info(f"Training {model_type} model...")
    #     model.fit(trainset)
    #     
    #     # Save and log model artifact
    #     model_path = MODELS_DIR / f"model_{model_type}.pkl"
    #     with open(model_path, "wb") as f:
    #         pickle.dump(model, f)
    #     mlflow.log_artifact(str(model_path))
    #     
    #     run_id = mlflow.active_run().info.run_id
    #     logger.info(f"Training complete. Run ID: {run_id}")
    #     
    #     return model, run_id
    
    pass  # Remove this and implement the function


# =============================================================================
# TODO 2: Implement train_with_config function
# =============================================================================
def train_with_config(trainset: Any, config: Dict[str, Any]) -> Tuple[Any, str]:
    """
    Train model using a configuration dictionary.
    
    TODO: Implement this function that:
    1. Extracts model_type from config
    2. Passes remaining config items as model_params
    3. Calls train_model() with the extracted parameters
    
    Args:
        trainset: Surprise trainset object
        config: Configuration dictionary with model_type and hyperparameters
        
    Returns:
        Tuple of (trained_model, run_id)
        
    Example:
        config = {"model_type": "svd", "n_factors": 100, "n_epochs": 20}
        model, run_id = train_with_config(trainset, config)
    """
    # TODO: Implement this function
    #
    # Hints:
    # 1. Make a copy of config to avoid modifying the original
    # 2. Pop 'model_type' from the config
    # 3. Pass remaining items to train_model
    #
    # config_copy = config.copy()
    # model_type = config_copy.pop("model_type")
    # return train_model(trainset, model_type=model_type, **config_copy)
    
    pass  # Remove this and implement the function


# =============================================================================
# TODO 3: Implement get_model_class function (BONUS)
# =============================================================================
def get_model_class(model_type: str):
    """
    Get the model class for a given model type.
    
    TODO: Implement this function that:
    1. Validates model_type is in MODEL_CLASSES
    2. Returns the corresponding class
    3. Raises ValueError for invalid model types
    
    Args:
        model_type: Type of model ('svd', 'nmf', 'knn')
        
    Returns:
        Model class from Surprise library
        
    Raises:
        ValueError: If model_type is not supported
    """
    # TODO: Implement this function
    #
    # if model_type not in MODEL_CLASSES:
    #     raise ValueError(f"Unknown model type: {model_type}. "
    #                      f"Supported types: {list(MODEL_CLASSES.keys())}")
    # return MODEL_CLASSES[model_type]
    
    pass  # Remove this and implement the function


# =============================================================================
# Helper functions (PROVIDED)
# =============================================================================
def get_default_params(model_type: str) -> Dict[str, Any]:
    """
    Get default parameters for a model type.
    
    Args:
        model_type: Type of model
        
    Returns:
        Dictionary of default parameters
    """
    return MODEL_CONFIGS.get(model_type, {})


def list_available_models() -> list:
    """
    List all available model types.
    
    Returns:
        List of model type names
    """
    return list(MODEL_CLASSES.keys())


# =============================================================================
# Main execution for testing
# =============================================================================
if __name__ == "__main__":
    from pipeline.data_ingestion import load_and_split
    
    print("Testing Training Module")
    print("=" * 50)
    
    # Setup MLflow
    setup_mlflow()
    
    # Load data
    trainset, testset, _ = load_and_split()
    
    # Test training (uncomment after implementing)
    # model, run_id = train_model(
    #     trainset,
    #     model_type="svd",
    #     run_name="test_run",
    #     n_factors=50,
    #     n_epochs=10
    # )
    # print(f"Model trained. Run ID: {run_id}")
    
    print("\nAvailable models:", list_available_models())
    print("Default SVD params:", get_default_params("svd"))
