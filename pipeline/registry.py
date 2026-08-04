"""
Model Registry Stage for ML Pipeline.

This module handles:
- Finding the best model from experiments
- Registering models to MLflow Model Registry
- Managing model versions and stages

TODO: Complete the functions marked with TODO.
"""

import logging
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient

from pipeline.config import MLFLOW_EXPERIMENT_NAME

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# TODO 1: Implement find_best_run function
# =============================================================================
def find_best_run(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    metric: str = "rmse",
    ascending: bool = True
) -> Dict[str, Any]:
    """
    Find the best run from an experiment based on a metric.
    
    TODO: Implement this function with the following requirements:
    
    1. Get the MLflow client
    2. Get the experiment by name
    3. Search runs and order by the specified metric
    4. Return information about the best run
    
    Args:
        experiment_name: Name of the MLflow experiment
        metric: Metric to optimize (default: 'rmse')
        ascending: If True, lower is better (default: True for RMSE)
        
    Returns:
        Dictionary with best run information:
        {
            'run_id': str,
            'metrics': dict,
            'params': dict,
            'artifact_uri': str
        }
        
    Example:
        best = find_best_run(metric='rmse', ascending=True)
        print(f"Best RMSE: {best['metrics']['rmse']}")
    """
    # TODO: Implement this function
    #
    # Hints:
    # 1. Create client: client = MlflowClient()
    # 2. Get experiment: experiment = client.get_experiment_by_name(experiment_name)
    # 3. Search runs with ordering:
    #    order = "ASC" if ascending else "DESC"
    #    runs = client.search_runs(
    #        experiment_ids=[experiment.experiment_id],
    #        order_by=[f"metrics.{metric} {order}"],
    #        max_results=1
    #    )
    # 4. Extract info from best_run = runs[0]
    #
    # Example structure:
    # client = MlflowClient()
    # experiment = client.get_experiment_by_name(experiment_name)
    # 
    # if experiment is None:
    #     raise ValueError(f"Experiment '{experiment_name}' not found")
    # 
    # order = "ASC" if ascending else "DESC"
    # runs = client.search_runs(
    #     experiment_ids=[experiment.experiment_id],
    #     order_by=[f"metrics.{metric} {order}"],
    #     max_results=1
    # )
    # 
    # if not runs:
    #     raise ValueError(f"No runs found in experiment '{experiment_name}'")
    # 
    # best_run = runs[0]
    # return {
    #     "run_id": best_run.info.run_id,
    #     "metrics": best_run.data.metrics,
    #     "params": best_run.data.params,
    #     "artifact_uri": best_run.info.artifact_uri,
    # }
    
    pass  # Remove this and implement the function


# =============================================================================
# TODO 2: Implement register_model function
# =============================================================================
def register_model(
    run_id: str,
    model_name: str,
    artifact_path: str = "model"
) -> str:
    """
    Register a model from an MLflow run to the Model Registry.
    
    TODO: Implement this function with the following requirements:
    
    1. Create the model URI from the run_id
    2. Register the model using mlflow.register_model()
    3. Return the model version
    
    Args:
        run_id: MLflow run ID containing the model
        model_name: Name for the registered model
        artifact_path: Path to the model artifact within the run
        
    Returns:
        Version number of the registered model (as string)
        
    Example:
        version = register_model(run_id, "movie-rating-model")
        print(f"Registered model version: {version}")
    """
    # TODO: Implement this function
    #
    # Hints:
    # 1. Model URI format: f"runs:/{run_id}/{artifact_path}"
    # 2. Use mlflow.register_model(model_uri, model_name)
    # 3. The result has a 'version' attribute
    #
    # Example structure:
    # model_uri = f"runs:/{run_id}/{artifact_path}"
    # logger.info(f"Registering model from {model_uri} as '{model_name}'")
    # 
    # result = mlflow.register_model(model_uri, model_name)
    # 
    # logger.info(f"Model registered: {model_name} version {result.version}")
    # return result.version
    
    pass  # Remove this and implement the function


# =============================================================================
# TODO 3: Implement transition_model_stage function
# =============================================================================
def transition_model_stage(
    model_name: str,
    version: str,
    stage: str = "Production"
) -> None:
    """
    Transition a model version to a new stage.
    
    TODO: Implement this function with the following requirements:
    
    1. Create MLflow client
    2. Transition the model version to the specified stage
    3. Log the transition
    
    Args:
        model_name: Name of the registered model
        version: Version number to transition
        stage: Target stage ('Staging', 'Production', 'Archived')
        
    Valid stages:
        - 'None': No stage
        - 'Staging': For testing
        - 'Production': For production use
        - 'Archived': Retired models
        
    Example:
        transition_model_stage("movie-rating-model", "1", "Production")
    """
    # TODO: Implement this function
    #
    # Hints:
    # 1. Create client: client = MlflowClient()
    # 2. Use client.transition_model_version_stage(name, version, stage)
    #
    # Example structure:
    # client = MlflowClient()
    # 
    # logger.info(f"Transitioning {model_name} v{version} to {stage}")
    # 
    # client.transition_model_version_stage(
    #     name=model_name,
    #     version=version,
    #     stage=stage
    # )
    # 
    # logger.info(f"Model {model_name} v{version} is now in {stage}")
    
    pass  # Remove this and implement the function


# =============================================================================
# TODO 4: Implement register_best_model function (combines all above)
# =============================================================================
def register_best_model(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    model_name: str = "movie-rating-model",
    metric: str = "rmse",
    stage: str = "Production"
) -> Dict[str, Any]:
    """
    Find the best model and register it to the Model Registry.
    
    TODO: Implement this function that:
    
    1. Finds the best run using find_best_run()
    2. Registers the model using register_model()
    3. Transitions to the specified stage using transition_model_stage()
    4. Returns information about the registered model
    
    Args:
        experiment_name: Name of the MLflow experiment
        model_name: Name for the registered model
        metric: Metric to optimize (default: 'rmse')
        stage: Stage to transition to (default: 'Production')
        
    Returns:
        Dictionary with registration info:
        {
            'run_id': str,
            'model_name': str,
            'version': str,
            'stage': str,
            'metrics': dict
        }
        
    Example:
        result = register_best_model()
        print(f"Registered {result['model_name']} v{result['version']}")
    """
    # TODO: Implement this function
    #
    # Example structure:
    # # Find best run
    # best_run = find_best_run(experiment_name, metric, ascending=True)
    # logger.info(f"Best run: {best_run['run_id']} with {metric}={best_run['metrics'].get(metric)}")
    # 
    # # Register model
    # version = register_model(best_run['run_id'], model_name)
    # 
    # # Transition to stage
    # transition_model_stage(model_name, version, stage)
    # 
    # return {
    #     "run_id": best_run['run_id'],
    #     "model_name": model_name,
    #     "version": version,
    #     "stage": stage,
    #     "metrics": best_run['metrics']
    # }
    
    pass  # Remove this and implement the function


# =============================================================================
# Helper Functions (PROVIDED)
# =============================================================================
def list_registered_models() -> List[Dict[str, Any]]:
    """
    List all registered models.
    
    Returns:
        List of model information dictionaries
    """
    client = MlflowClient()
    models = client.search_registered_models()
    
    return [
        {
            "name": model.name,
            "latest_versions": [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                }
                for v in model.latest_versions
            ]
        }
        for model in models
    ]


def get_production_model(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get the current production version of a model.
    
    Args:
        model_name: Name of the registered model
        
    Returns:
        Dictionary with model info or None if not found
    """
    client = MlflowClient()
    
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            v = versions[0]
            return {
                "name": model_name,
                "version": v.version,
                "stage": v.current_stage,
                "run_id": v.run_id,
            }
    except Exception as e:
        logger.error(f"Error getting production model: {e}")
    
    return None


def compare_runs(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    metric: str = "rmse",
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Get top N runs from an experiment.
    
    Args:
        experiment_name: Name of the experiment
        metric: Metric to sort by
        top_n: Number of runs to return
        
    Returns:
        List of run information sorted by metric
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        return []
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} ASC"],
        max_results=top_n
    )
    
    return [
        {
            "run_id": run.info.run_id,
            "metrics": run.data.metrics,
            "params": run.data.params,
        }
        for run in runs
    ]


# =============================================================================
# Main execution for testing
# =============================================================================
if __name__ == "__main__":
    print("Testing Registry Module")
    print("=" * 50)
    
    # Test helper functions
    print("\nRegistered models:", list_registered_models())
    
    # After implementing TODO functions, test:
    # result = register_best_model()
    # print(f"Registered: {result}")
    
    print("\nRegistry module loaded successfully.")
    print("Implement TODO functions and test after running experiments.")
