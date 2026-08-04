"""
Model Evaluation Stage for ML Pipeline.

This module handles:
- Making predictions on test data
- Calculating evaluation metrics
- Logging metrics to MLflow
- Creating evaluation visualizations

TODO: Complete the functions marked with TODO.
"""

import logging
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import matplotlib.pyplot as plt
from surprise import accuracy

from pipeline.config import ARTIFACTS_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# TODO 1: Implement evaluate_model function
# =============================================================================
def evaluate_model(
    model: Any,
    testset: List,
    run_id: str,
    log_to_mlflow: bool = True
) -> Dict[str, float]:
    """
    Evaluate model and log metrics to MLflow.
    
    TODO: Implement this function with the following requirements:
    
    1. Make predictions using model.test(testset)
    2. Calculate RMSE and MAE using surprise.accuracy
    3. If log_to_mlflow is True:
       - Resume the MLflow run using the run_id
       - Log RMSE and MAE as metrics
       - Create and log evaluation plots
    4. Return dictionary with metrics
    
    Args:
        model: Trained Surprise model
        testset: Test set as list of (user, item, rating) tuples
        run_id: MLflow run ID to log metrics to
        log_to_mlflow: Whether to log metrics to MLflow
        
    Returns:
        Dictionary with evaluation metrics {'rmse': float, 'mae': float}
        
    Example:
        metrics = evaluate_model(model, testset, run_id)
        print(f"RMSE: {metrics['rmse']:.4f}")
    """
    # TODO: Implement this function
    #
    # Hints:
    # 1. Use model.test(testset) to get predictions
    # 2. Use accuracy.rmse(predictions, verbose=False) for RMSE
    # 3. Use accuracy.mae(predictions, verbose=False) for MAE
    # 4. Use mlflow.start_run(run_id=run_id) to resume a run
    # 5. Use mlflow.log_metric(name, value) to log metrics
    #
    # Example structure:
    # logger.info("Evaluating model...")
    # predictions = model.test(testset)
    # 
    # rmse = accuracy.rmse(predictions, verbose=False)
    # mae = accuracy.mae(predictions, verbose=False)
    # 
    # metrics = {"rmse": rmse, "mae": mae}
    # 
    # if log_to_mlflow:
    #     with mlflow.start_run(run_id=run_id):
    #         mlflow.log_metric("rmse", rmse)
    #         mlflow.log_metric("mae", mae)
    #         
    #         # Create and log plots
    #         fig = create_prediction_distribution_plot(predictions)
    #         mlflow.log_figure(fig, "prediction_distribution.png")
    #         plt.close(fig)
    # 
    # logger.info(f"Evaluation complete. RMSE={rmse:.4f}, MAE={mae:.4f}")
    # return metrics
    
    pass  # Remove this and implement the function


# =============================================================================
# TODO 2: Implement calculate_additional_metrics function
# =============================================================================
def calculate_additional_metrics(predictions: List) -> Dict[str, float]:
    """
    Calculate additional evaluation metrics beyond RMSE and MAE.
    
    TODO: Implement this function that calculates:
    1. Mean Squared Error (MSE)
    2. Mean Absolute Percentage Error (MAPE) - if applicable
    3. Coverage (percentage of user-item pairs that can be predicted)
    4. Any other relevant metrics
    
    Args:
        predictions: List of Surprise Prediction objects
        
    Returns:
        Dictionary with additional metrics
        
    Hints:
        - Each prediction has: prediction.r_ui (actual) and prediction.est (predicted)
        - Be careful with division by zero for MAPE
    """
    # TODO: Implement this function
    #
    # Example structure:
    # actuals = [pred.r_ui for pred in predictions]
    # estimated = [pred.est for pred in predictions]
    # 
    # actuals = np.array(actuals)
    # estimated = np.array(estimated)
    # 
    # mse = np.mean((actuals - estimated) ** 2)
    # 
    # # MAPE (handle division by zero)
    # non_zero_mask = actuals != 0
    # if np.any(non_zero_mask):
    #     mape = np.mean(np.abs((actuals[non_zero_mask] - estimated[non_zero_mask]) / actuals[non_zero_mask])) * 100
    # else:
    #     mape = None
    # 
    # return {
    #     "mse": mse,
    #     "mape": mape,
    #     "n_predictions": len(predictions),
    # }
    
    pass  # Remove this and implement the function


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
    
    # This will work after training.py is implemented
    # from pipeline.data_ingestion import load_and_split
    # from pipeline.training import train_model, setup_mlflow
    # 
    # setup_mlflow()
    # trainset, testset, _ = load_and_split()
    # model, run_id = train_model(trainset, model_type="svd", n_factors=50)
    # metrics = evaluate_model(model, testset, run_id)
    # print(f"Metrics: {metrics}")
    
    print("Evaluation module loaded successfully.")
    print("Implement the TODO functions and test with the training module.")
