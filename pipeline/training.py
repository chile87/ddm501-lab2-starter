"""
Model Training Stage for ML Pipeline.

This module handles:
- Model initialization
- Model training
- MLflow experiment tracking (parameters + artifacts)

Every trained model is logged twice on purpose:
1. as a raw pickle artifact (fast to reload inside the pipeline), and
2. as an MLflow pyfunc model under the ``model`` artifact path, which is what
   the Model Registry stage promotes.
"""

import logging
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
from surprise import SVD, NMF, KNNBasic

from pipeline.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_CONFIGS,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    DATASET_NAME,
)
from pipeline.model_wrapper import (
    SURPRISE_ARTIFACT_KEY,
    SurpriseRecommender,
    build_input_example,
    build_signature,
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

# Artifact path (inside an MLflow run) under which the pyfunc model is stored.
# The Model Registry resolves models as "runs:/<run_id>/<MODEL_ARTIFACT_PATH>".
MODEL_ARTIFACT_PATH = "model"

# Matrix-factorisation models accept a random_state; the neighbourhood model
# (KNNBasic) is deterministic and rejects the kwarg.
_SEEDED_MODELS = {"svd", "nmf"}


def setup_mlflow(
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    allow_local_fallback: bool = True,
) -> str:
    """
    Setup MLflow tracking.

    If the configured tracking server is unreachable and ``allow_local_fallback``
    is True, the pipeline degrades to a local ``file:./mlruns`` store instead of
    crashing. This keeps the pipeline runnable (and reproducible) on a machine
    where the tracking server has not been started.

    Args:
        tracking_uri: MLflow tracking server URI
        experiment_name: Name of the experiment
        allow_local_fallback: Fall back to a local file store on connection error

    Returns:
        The tracking URI that ended up being used

    Raises:
        mlflow.exceptions.MlflowException: If the server is unreachable and
            ``allow_local_fallback`` is False
    """
    mlflow.set_tracking_uri(tracking_uri)

    try:
        # Cheapest call that forces a round-trip to the backend store.
        mlflow.search_experiments(max_results=1)
    except Exception as e:
        if not allow_local_fallback:
            logger.error(f"MLflow tracking server unreachable at {tracking_uri}: {e}")
            raise
        logger.warning(
            f"MLflow tracking server unreachable at {tracking_uri} ({e}). "
            f"Falling back to local file store 'file:./mlruns'."
        )
        tracking_uri = "file:./mlruns"
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow configured: URI={tracking_uri}, Experiment={experiment_name}")
    return tracking_uri


def flatten_params(params: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Flatten nested parameter dictionaries into MLflow-friendly scalar params.

    MLflow parameters are flat key/value strings, so a nested config such as
    ``{"sim_options": {"name": "cosine"}}`` would be logged as an opaque
    ``"{'name': 'cosine'}"`` string and become unsearchable in the UI. Flattening
    turns it into ``sim_options.name = cosine``, which *is* filterable.

    Args:
        params: Possibly nested parameter dictionary
        prefix: Key prefix used during recursion

    Returns:
        Flat dictionary of parameters

    Example:
        >>> flatten_params({"k": 40, "sim_options": {"name": "cosine"}})
        {'k': 40, 'sim_options.name': 'cosine'}
    """
    flat: Dict[str, Any] = {}

    for key, value in params.items():
        full_key = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_params(value, prefix=f"{full_key}."))
        else:
            flat[full_key] = value

    return flat


def get_model_class(model_type: str):
    """
    Get the model class for a given model type.

    Args:
        model_type: Type of model ('svd', 'nmf', 'knn')

    Returns:
        Model class from Surprise library

    Raises:
        ValueError: If model_type is not supported

    Example:
        >>> get_model_class('svd').__name__
        'SVD'
    """
    if not isinstance(model_type, str):
        raise ValueError(
            f"model_type must be a string, got {type(model_type).__name__}"
        )

    key = model_type.lower().strip()
    if key not in MODEL_CLASSES:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Supported types: {sorted(MODEL_CLASSES.keys())}"
        )

    return MODEL_CLASSES[key]


def build_model(model_type: str, **model_params) -> Any:
    """
    Instantiate a Surprise model, injecting a seed for reproducibility.

    Args:
        model_type: Type of model ('svd', 'nmf', 'knn')
        **model_params: Model hyperparameters

    Returns:
        Unfitted Surprise estimator

    Raises:
        ValueError: If model_type is unsupported or a hyperparameter is invalid
    """
    model_class = get_model_class(model_type)
    params = dict(model_params)

    # Pin the seed so repeated runs of the same config produce the same model.
    if model_type.lower() in _SEEDED_MODELS:
        params.setdefault("random_state", RANDOM_STATE)

    try:
        return model_class(**params)
    except TypeError as e:
        raise ValueError(
            f"Invalid hyperparameters for model '{model_type}': {e}. "
            f"Received: {sorted(params.keys())}"
        ) from e


def train_model(
    trainset: Any,
    model_type: str = "svd",
    run_name: Optional[str] = None,
    **model_params
) -> Tuple[Any, str]:
    """
    Train a recommendation model and log it to MLflow.

    Logged to the MLflow run:
        - params: model_type, every (flattened) hyperparameter, dataset context
        - metrics: training_time_seconds
        - artifacts: raw pickle + a pyfunc model under the 'model' path
        - tags: stage=training, model_type, training_status

    Args:
        trainset: Surprise trainset object
        model_type: Type of model ('svd', 'nmf', 'knn')
        run_name: Optional name for the MLflow run
        **model_params: Model hyperparameters

    Returns:
        Tuple of (trained_model, run_id)

    Raises:
        ValueError: If model_type is unsupported or trainset is None
        Exception: Any training failure is re-raised after being tagged in MLflow

    Example:
        model, run_id = train_model(
            trainset,
            model_type='svd',
            n_factors=100,
            n_epochs=20
        )
    """
    if trainset is None:
        raise ValueError("trainset must not be None - run data ingestion first")

    # Validate before opening an MLflow run so a bad config does not litter the
    # experiment with empty failed runs.
    get_model_class(model_type)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        try:
            # -----------------------------------------------------------------
            # 1. Log parameters (hyperparameters + data context)
            # -----------------------------------------------------------------
            mlflow.log_param("model_type", model_type)
            for key, value in flatten_params(model_params).items():
                mlflow.log_param(key, value)

            # Data context makes a run reproducible from the UI alone.
            mlflow.log_params(
                {
                    "dataset": DATASET_NAME,
                    "test_size": TEST_SIZE,
                    "random_state": RANDOM_STATE,
                    "n_train_users": trainset.n_users,
                    "n_train_items": trainset.n_items,
                    "n_train_ratings": trainset.n_ratings,
                }
            )

            mlflow.set_tags(
                {
                    "stage": "training",
                    "model_type": model_type,
                    "model_library": "scikit-surprise",
                }
            )

            # -----------------------------------------------------------------
            # 2. Train
            # -----------------------------------------------------------------
            model = build_model(model_type, **model_params)

            logger.info(f"Training {model_type} model with params={model_params}...")
            start = time.perf_counter()
            model.fit(trainset)
            training_time = time.perf_counter() - start

            mlflow.log_metric("training_time_seconds", training_time)
            logger.info(f"Training finished in {training_time:.2f}s")

            # -----------------------------------------------------------------
            # 3. Log artifacts: raw pickle (pipeline-internal reuse)
            # -----------------------------------------------------------------
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            model_path = MODELS_DIR / f"model_{model_type}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(str(model_path), artifact_path="pickle")

            # -----------------------------------------------------------------
            # 4. Log artifacts: pyfunc model (registry-ready, loadable, served)
            # -----------------------------------------------------------------
            _log_pyfunc_model(model, model_type)

            mlflow.set_tag("training_status", "success")
            logger.info(f"Training complete. Run ID: {run_id}")

            return model, run_id

        except Exception as e:
            # Tag the run so failures are visible in the MLflow UI rather than
            # showing up as a silently empty run, then propagate.
            mlflow.set_tag("training_status", "failed")
            mlflow.set_tag("error", str(e)[:500])
            logger.error(f"Training failed for run {run_id}: {e}")
            raise


def _log_pyfunc_model(model: Any, model_type: str) -> None:
    """
    Log a trained Surprise model as an MLflow pyfunc model.

    The estimator is pickled into a temporary directory and passed to MLflow as
    a wrapper artifact, so the resulting MLflow model is self-contained and can
    be loaded with ``mlflow.pyfunc.load_model("runs:/<run_id>/model")``.

    Args:
        model: Trained Surprise estimator
        model_type: Type of model, used in the artifact filename
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        pickle_path = Path(tmp_dir) / f"{model_type}_estimator.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(model, f)

        mlflow.pyfunc.log_model(
            artifact_path=MODEL_ARTIFACT_PATH,
            python_model=SurpriseRecommender(),
            artifacts={SURPRISE_ARTIFACT_KEY: str(pickle_path)},
            signature=build_signature(),
            input_example=build_input_example(),
        )

    logger.info(f"Logged pyfunc model at artifact path '{MODEL_ARTIFACT_PATH}'")


def train_with_config(trainset: Any, config: Dict[str, Any]) -> Tuple[Any, str]:
    """
    Train model using a configuration dictionary.

    Args:
        trainset: Surprise trainset object
        config: Configuration dictionary with model_type and hyperparameters

    Returns:
        Tuple of (trained_model, run_id)

    Raises:
        ValueError: If ``config`` does not contain a 'model_type' key

    Example:
        config = {"model_type": "svd", "n_factors": 100, "n_epochs": 20}
        model, run_id = train_with_config(trainset, config)
    """
    if not config or "model_type" not in config:
        raise ValueError(
            f"config must contain a 'model_type' key. Received keys: "
            f"{sorted(config.keys()) if config else []}"
        )

    # Copy so the caller's config (often a shared module-level constant) is not
    # mutated by the pop below.
    config_copy = dict(config)
    model_type = config_copy.pop("model_type")
    run_name = config_copy.pop("run_name", None)

    return train_model(
        trainset,
        model_type=model_type,
        run_name=run_name,
        **config_copy,
    )


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

    model, run_id = train_model(
        trainset,
        model_type="svd",
        run_name="test_run",
        n_factors=50,
        n_epochs=10,
    )
    print(f"Model trained. Run ID: {run_id}")

    print("\nAvailable models:", list_available_models())
    print("Default SVD params:", get_default_params("svd"))
