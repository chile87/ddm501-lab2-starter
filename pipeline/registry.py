"""
Model Registry Stage for ML Pipeline.

This module handles:
- Finding the best model from experiments
- Registering models to MLflow Model Registry
- Managing model versions and stages

Registration targets the pyfunc model logged by ``pipeline.training`` under the
``model`` artifact path, so every registered version is a loadable MLflow model
rather than a pointer at a loose pickle file.
"""

import logging
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from pipeline.config import MLFLOW_EXPERIMENT_NAME
from pipeline.training import MODEL_ARTIFACT_PATH

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default registered-model name used across the pipeline and the Airflow DAG.
DEFAULT_REGISTERED_MODEL_NAME = "movie-rating-model"


def find_best_run(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    metric: str = "rmse",
    ascending: bool = True
) -> Dict[str, Any]:
    """
    Find the best run from an experiment based on a metric.

    Only finished runs that actually recorded the metric are considered, so a
    crashed or still-running run cannot be promoted by accident.

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
            'artifact_uri': str,
            'run_name': str
        }

    Raises:
        ValueError: If the experiment does not exist or has no qualifying runs

    Example:
        best = find_best_run(metric='rmse', ascending=True)
        print(f"Best RMSE: {best['metrics']['rmse']}")
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(
            f"Experiment '{experiment_name}' not found. Run the pipeline or "
            f"experiment suite before attempting registration."
        )

    order = "ASC" if ascending else "DESC"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        # Exclude unfinished runs and runs that never logged the metric.
        filter_string=f"attributes.status = 'FINISHED' and metrics.{metric} > -1e30",
        order_by=[f"metrics.{metric} {order}"],
        max_results=1,
    )

    if not runs:
        raise ValueError(
            f"No finished runs with metric '{metric}' found in experiment "
            f"'{experiment_name}'. Did the evaluation stage run?"
        )

    best_run = runs[0]
    logger.info(
        f"Best run in '{experiment_name}': {best_run.info.run_id} "
        f"({metric}={best_run.data.metrics.get(metric)})"
    )

    return {
        "run_id": best_run.info.run_id,
        "metrics": dict(best_run.data.metrics),
        "params": dict(best_run.data.params),
        "artifact_uri": best_run.info.artifact_uri,
        "run_name": best_run.data.tags.get("mlflow.runName", ""),
    }


def register_model(
    run_id: str,
    model_name: str,
    artifact_path: str = MODEL_ARTIFACT_PATH
) -> str:
    """
    Register a model from an MLflow run to the Model Registry.

    Args:
        run_id: MLflow run ID containing the model
        model_name: Name for the registered model
        artifact_path: Path to the model artifact within the run
            (defaults to the path ``pipeline.training`` logs to)

    Returns:
        Version number of the registered model (as string)

    Raises:
        ValueError: If ``run_id`` is empty or the run has no model at
            ``artifact_path``
        MlflowException: If the registry rejects the registration

    Example:
        version = register_model(run_id, "movie-rating-model")
        print(f"Registered model version: {version}")
    """
    if not run_id:
        raise ValueError("run_id must not be empty")

    # Fail loudly here rather than registering a version that cannot be loaded.
    _assert_model_artifact_exists(run_id, artifact_path)

    model_uri = f"runs:/{run_id}/{artifact_path}"
    logger.info(f"Registering model from {model_uri} as '{model_name}'")

    result = mlflow.register_model(model_uri, model_name)

    logger.info(f"Model registered: {model_name} version {result.version}")
    return str(result.version)


def _assert_model_artifact_exists(run_id: str, artifact_path: str) -> None:
    """
    Verify a run contains a model at the given artifact path.

    Args:
        run_id: MLflow run ID to inspect
        artifact_path: Artifact path expected to hold the MLflow model

    Raises:
        ValueError: If the run has no artifact at ``artifact_path``
    """
    client = MlflowClient()

    try:
        artifacts = {a.path for a in client.list_artifacts(run_id)}
    except MlflowException as e:
        raise ValueError(f"Cannot inspect artifacts of run '{run_id}': {e}") from e

    if artifact_path not in artifacts:
        raise ValueError(
            f"Run '{run_id}' has no model at artifact path '{artifact_path}'. "
            f"Found artifacts: {sorted(artifacts)}. The training stage must log "
            f"a pyfunc model before registration."
        )


def transition_model_stage(
    model_name: str,
    version: str,
    stage: str = "Production"
) -> None:
    """
    Transition a model version to a new stage.

    Also sets a registry *alias* mirroring the stage (e.g. ``production``).
    Stages are deprecated from MLflow 2.9 onwards in favour of aliases; setting
    both keeps this lab's stage-based workflow working while remaining
    forward-compatible.

    Args:
        model_name: Name of the registered model
        version: Version number to transition
        stage: Target stage ('None', 'Staging', 'Production', 'Archived')

    Raises:
        ValueError: If ``stage`` is not a valid MLflow stage

    Example:
        transition_model_stage("movie-rating-model", "1", "Production")
    """
    valid_stages = {"None", "Staging", "Production", "Archived"}
    if stage not in valid_stages:
        raise ValueError(
            f"Invalid stage '{stage}'. Valid stages: {sorted(valid_stages)}"
        )

    client = MlflowClient()
    logger.info(f"Transitioning {model_name} v{version} to {stage}")

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        # Demote any previous Production/Staging version so exactly one version
        # is live per stage.
        archive_existing_versions=stage in {"Production", "Staging"},
    )

    # Alias is best-effort: older registries do not support it.
    if stage != "None":
        try:
            client.set_registered_model_alias(
                name=model_name,
                alias=stage.lower(),
                version=version,
            )
        except (MlflowException, AttributeError) as e:
            logger.warning(f"Could not set alias '{stage.lower()}': {e}")

    logger.info(f"Model {model_name} v{version} is now in {stage}")


def register_best_model(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
    metric: str = "rmse",
    stage: str = "Production",
    ascending: bool = True,
) -> Dict[str, Any]:
    """
    Find the best model and register it to the Model Registry.

    Args:
        experiment_name: Name of the MLflow experiment
        model_name: Name for the registered model
        metric: Metric to optimize (default: 'rmse')
        stage: Stage to transition to (default: 'Production')
        ascending: If True, lower metric is better

    Returns:
        Dictionary with registration info:
        {
            'run_id': str,
            'model_name': str,
            'version': str,
            'stage': str,
            'metrics': dict,
            'params': dict
        }

    Raises:
        ValueError: If no suitable run exists in the experiment

    Example:
        result = register_best_model()
        print(f"Registered {result['model_name']} v{result['version']}")
    """
    logger.info(
        f"Registering best model from '{experiment_name}' by {metric} "
        f"({'lower' if ascending else 'higher'} is better)"
    )

    best_run = find_best_run(experiment_name, metric, ascending=ascending)

    version = register_model(best_run["run_id"], model_name)
    transition_model_stage(model_name, version, stage)

    result = {
        "run_id": best_run["run_id"],
        "model_name": model_name,
        "version": version,
        "stage": stage,
        "metrics": best_run["metrics"],
        "params": best_run["params"],
    }

    logger.info(
        f"Registered {model_name} v{version} in {stage} "
        f"({metric}={best_run['metrics'].get(metric)})"
    )
    return result


def load_registered_model(
    model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
    stage: str = "Production",
) -> Any:
    """
    Load a registered model from the Model Registry for inference.

    Args:
        model_name: Name of the registered model
        stage: Stage to load from

    Returns:
        An ``mlflow.pyfunc`` model wrapping the Surprise estimator

    Raises:
        MlflowException: If no model exists for the given name/stage

    Example:
        model = load_registered_model()
        model.predict(pd.DataFrame({"user_id": ["196"], "item_id": ["242"]}))
    """
    model_uri = f"models:/{model_name}/{stage}"
    logger.info(f"Loading registered model from {model_uri}")
    return mlflow.pyfunc.load_model(model_uri)


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
        logger.warning(f"Experiment '{experiment_name}' not found")
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"attributes.status = 'FINISHED' and metrics.{metric} > -1e30",
        order_by=[f"metrics.{metric} ASC"],
        max_results=top_n
    )

    return [
        {
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName", ""),
            "metrics": dict(run.data.metrics),
            "params": dict(run.data.params),
        }
        for run in runs
    ]


# =============================================================================
# Main execution for testing
# =============================================================================
if __name__ == "__main__":
    print("Testing Registry Module")
    print("=" * 50)

    print("\nRegistered models:", list_registered_models())

    print("\nRegistry module loaded successfully.")
    print("Run experiments first, then: register_best_model()")
