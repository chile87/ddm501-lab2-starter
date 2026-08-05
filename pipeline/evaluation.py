"""
Model Evaluation Stage for ML Pipeline.

This module handles:
- Making predictions on test data
- Calculating evaluation metrics
- Logging metrics to MLflow
- Creating evaluation visualizations

Metrics are logged back onto the *same* MLflow run that produced the model, so
each run in the UI carries both its hyperparameters and its scores and can be
sorted/compared directly.
"""

import logging
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import matplotlib

# Use a non-interactive backend: evaluation runs headless (Airflow workers, CI)
# where no display is attached and the default backend would fail.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
from surprise import accuracy  # noqa: E402

from pipeline.config import ARTIFACTS_DIR  # noqa: E402

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(
    model: Any,
    testset: List,
    run_id: Optional[str] = None,
    log_to_mlflow: bool = True
) -> Dict[str, float]:
    """
    Evaluate model and log metrics to MLflow.

    Logged to the MLflow run (when ``log_to_mlflow``):
        - metrics: rmse, mae, mse, mape, coverage, n_predictions, ...
        - artifacts: prediction_distribution.png, error_by_rating.png,
          evaluation_report.txt
        - tags: stage=evaluation

    Args:
        model: Trained Surprise model
        testset: Test set as list of (user, item, rating) tuples
        run_id: MLflow run ID to log metrics to. Required when ``log_to_mlflow``
            is True so metrics land on the run that trained the model.
        log_to_mlflow: Whether to log metrics to MLflow

    Returns:
        Dictionary with evaluation metrics, e.g.
        {'rmse': 0.93, 'mae': 0.73, 'mse': 0.87, 'coverage': 100.0, ...}

    Raises:
        ValueError: If model/testset are empty, or run_id is missing while
            MLflow logging is requested

    Example:
        metrics = evaluate_model(model, testset, run_id)
        print(f"RMSE: {metrics['rmse']:.4f}")
    """
    if model is None:
        raise ValueError("model must not be None - run the training stage first")
    if not testset:
        raise ValueError("testset must not be empty - cannot evaluate a model")
    if log_to_mlflow and not run_id:
        raise ValueError(
            "run_id is required when log_to_mlflow=True so that metrics are "
            "attached to the training run. Pass log_to_mlflow=False for an "
            "offline evaluation."
        )

    logger.info(f"Evaluating model on {len(testset)} test ratings...")

    predictions = model.test(testset)

    # Core accuracy metrics (verbose=False: we log rather than print).
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    metrics: Dict[str, float] = {"rmse": float(rmse), "mae": float(mae)}
    metrics.update(calculate_additional_metrics(predictions))

    if log_to_mlflow:
        try:
            _log_evaluation_to_mlflow(metrics, predictions, run_id)
        except Exception as e:
            # A tracking-server hiccup must not throw away a valid evaluation:
            # return the metrics and let the caller decide.
            logger.error(f"Failed to log evaluation to MLflow run {run_id}: {e}")
            raise

    logger.info(
        f"Evaluation complete. RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}"
    )
    return metrics


def _log_evaluation_to_mlflow(
    metrics: Dict[str, float],
    predictions: List,
    run_id: str,
) -> None:
    """
    Attach metrics, plots and a text report to an existing MLflow run.

    Args:
        metrics: Metric dictionary to log
        predictions: List of Surprise Prediction objects, used for the plots
        run_id: MLflow run ID to resume
    """
    # nested=False + explicit run_id resumes the training run instead of
    # creating a second, disconnected run.
    with mlflow.start_run(run_id=run_id):
        for name, value in metrics.items():
            if value is None:
                logger.warning(f"Skipping metric '{name}': value is None")
                continue
            mlflow.log_metric(name, float(value))

        mlflow.set_tag("stage", "evaluation")

        # Figures: logged straight from memory, no temp files needed.
        fig = create_prediction_distribution_plot(predictions)
        mlflow.log_figure(fig, "plots/prediction_distribution.png")
        plt.close(fig)

        fig = create_error_by_rating_plot(predictions)
        mlflow.log_figure(fig, "plots/error_by_rating.png")
        plt.close(fig)

        # Human-readable report alongside the machine-readable metrics.
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = ARTIFACTS_DIR / f"evaluation_report_{run_id[:8]}.txt"
        save_evaluation_report(metrics, str(report_path))
        mlflow.log_artifact(str(report_path), artifact_path="reports")

    logger.info(f"Logged {len(metrics)} metrics and 3 artifacts to run {run_id}")


def calculate_additional_metrics(predictions: List) -> Dict[str, float]:
    """
    Calculate additional evaluation metrics beyond RMSE and MAE.

    Computes:
        - mse: mean squared error
        - mape: mean absolute percentage error (zero actuals excluded)
        - coverage: percentage of test pairs predicted without falling back to
          the global mean (``was_impossible``); a low value means the model
          could not personalise for many cold-start users/items
        - n_predictions / n_impossible: prediction counts
        - mean_predicted_rating / mean_actual_rating: bias check
        - max_error: worst single-prediction error

    Args:
        predictions: List of Surprise Prediction objects

    Returns:
        Dictionary with additional metrics

    Raises:
        ValueError: If ``predictions`` is empty
    """
    if not predictions:
        raise ValueError("predictions must not be empty")

    actuals = np.array([pred.r_ui for pred in predictions], dtype=float)
    estimated = np.array([pred.est for pred in predictions], dtype=float)

    errors = estimated - actuals
    mse = float(np.mean(errors ** 2))

    # MAPE is undefined where the actual rating is 0; exclude those rows rather
    # than emitting inf. MovieLens ratings start at 1, so this is a guard for
    # other datasets.
    non_zero = actuals != 0
    if np.any(non_zero):
        mape = float(
            np.mean(np.abs(errors[non_zero] / actuals[non_zero])) * 100
        )
    else:
        mape = None
        logger.warning("MAPE undefined: all actual ratings are zero")

    # Surprise flags a prediction as "impossible" when it had to fall back to
    # the global mean (unknown user or item).
    n_impossible = sum(
        1 for pred in predictions if pred.details.get("was_impossible", False)
    )
    coverage = 100.0 * (len(predictions) - n_impossible) / len(predictions)

    return {
        "mse": mse,
        "mape": mape,
        "coverage": coverage,
        "n_predictions": float(len(predictions)),
        "n_impossible": float(n_impossible),
        "mean_predicted_rating": float(np.mean(estimated)),
        "mean_actual_rating": float(np.mean(actuals)),
        "max_error": float(np.max(np.abs(errors))),
    }


# =============================================================================
# Visualization Functions (PROVIDED)
# =============================================================================
def create_prediction_distribution_plot(predictions: List) -> plt.Figure:
    """
    Create a plot showing prediction vs actual rating distribution.

    Args:
        predictions: List of Surprise Prediction objects

    Returns:
        Matplotlib figure
    """
    actuals = [pred.r_ui for pred in predictions]
    estimated = [pred.est for pred in predictions]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Scatter plot of actual vs predicted
    axes[0].scatter(actuals, estimated, alpha=0.1, s=1)
    axes[0].plot([1, 5], [1, 5], 'r--', label='Perfect prediction')
    axes[0].set_xlabel('Actual Rating')
    axes[0].set_ylabel('Predicted Rating')
    axes[0].set_title('Actual vs Predicted Ratings')
    axes[0].legend()

    # Plot 2: Distribution of actual ratings
    axes[1].hist(actuals, bins=20, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Rating')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Actual Ratings')

    # Plot 3: Distribution of prediction errors
    errors = np.array(estimated) - np.array(actuals)
    axes[2].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[2].axvline(x=0, color='r', linestyle='--')
    axes[2].set_xlabel('Prediction Error')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Distribution of Prediction Errors')

    plt.tight_layout()
    return fig


def create_error_by_rating_plot(predictions: List) -> plt.Figure:
    """
    Create a plot showing error distribution by actual rating.

    Args:
        predictions: List of Surprise Prediction objects

    Returns:
        Matplotlib figure
    """
    # Group predictions by actual rating
    rating_groups = {}
    for pred in predictions:
        rating = round(pred.r_ui)
        if rating not in rating_groups:
            rating_groups[rating] = []
        rating_groups[rating].append(pred.est - pred.r_ui)

    fig, ax = plt.subplots(figsize=(10, 6))

    ratings = sorted(rating_groups.keys())
    positions = range(len(ratings))

    bp = ax.boxplot(
        [rating_groups[r] for r in ratings],
        positions=positions,
        widths=0.6
    )

    ax.set_xticklabels([str(r) for r in ratings])
    ax.set_xlabel('Actual Rating')
    ax.set_ylabel('Prediction Error')
    ax.set_title('Prediction Error by Actual Rating')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    return fig


def save_evaluation_report(metrics: Dict, filepath: str) -> None:
    """
    Save evaluation metrics to a text file.

    Args:
        metrics: Dictionary of metrics
        filepath: Path to save the report
    """
    with open(filepath, 'w') as f:
        f.write("Model Evaluation Report\n")
        f.write("=" * 40 + "\n\n")

        for name, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{name}: {value:.4f}\n")
            else:
                f.write(f"{name}: {value}\n")

    logger.info(f"Evaluation report saved to {filepath}")


# =============================================================================
# Main execution for testing
# =============================================================================
if __name__ == "__main__":
    print("Testing Evaluation Module")
    print("=" * 50)

    from pipeline.data_ingestion import load_and_split
    from pipeline.training import train_model, setup_mlflow

    setup_mlflow()
    trainset, testset, _ = load_and_split()
    model, run_id = train_model(trainset, model_type="svd", n_factors=50, n_epochs=10)
    metrics = evaluate_model(model, testset, run_id)

    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value}")
