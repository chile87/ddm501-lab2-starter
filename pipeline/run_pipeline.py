"""
Pipeline Runner - Orchestrates all pipeline stages.

This script runs the complete ML pipeline:
1. Data Ingestion
2. Preprocessing
3. Training (with MLflow tracking)
4. Evaluation
5. Model Registration (optional)

Usage:
    python -m pipeline.run_pipeline
    python -m pipeline.run_pipeline --model-type svd --n-factors 100
"""

import argparse
import logging
from typing import Dict, Any, Optional

from pipeline.config import DEFAULT_MODEL_TYPE, MODEL_CONFIGS
from pipeline.data_ingestion import load_and_split
from pipeline.preprocessing import preprocess_data
from pipeline.training import train_model, setup_mlflow
from pipeline.evaluation import evaluate_model
from pipeline.registry import register_best_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline(
    model_type: str = DEFAULT_MODEL_TYPE,
    register: bool = False,
    **model_params
) -> Dict[str, Any]:
    """
    Run the complete ML pipeline.
    
    Args:
        model_type: Type of model to train ('svd', 'nmf', 'knn')
        register: Whether to register the model after training
        **model_params: Model hyperparameters
        
    Returns:
        Dictionary with pipeline results
    """
    logger.info("=" * 60)
    logger.info("Starting ML Pipeline")
    logger.info("=" * 60)
    
    results = {
        "status": "started",
        "stages": {},
    }
    
    try:
        # Stage 1: Setup MLflow
        logger.info("\n[Stage 1/5] Setting up MLflow...")
        setup_mlflow()
        results["stages"]["mlflow_setup"] = "success"
        
        # Stage 2: Data Ingestion
        logger.info("\n[Stage 2/5] Loading data...")
        trainset, testset, data_stats = load_and_split()
        results["stages"]["data_ingestion"] = {
            "status": "success",
            "stats": data_stats
        }
        
        # Stage 3: Preprocessing
        logger.info("\n[Stage 3/5] Preprocessing data...")
        preprocess_report = preprocess_data(trainset, testset)
        results["stages"]["preprocessing"] = {
            "status": "success" if preprocess_report["preprocessing_successful"] else "warning",
            "report": preprocess_report
        }
        
        # Stage 4: Training
        logger.info(f"\n[Stage 4/5] Training {model_type} model...")
        
        # Use default params if none provided
        if not model_params:
            model_params = MODEL_CONFIGS.get(model_type, {})
        
        model, run_id = train_model(
            trainset,
            model_type=model_type,
            run_name=f"pipeline_run_{model_type}",
            **model_params
        )
        results["stages"]["training"] = {
            "status": "success",
            "run_id": run_id,
            "model_type": model_type,
            "params": model_params
        }
        
        # Stage 5: Evaluation
        logger.info("\n[Stage 5/5] Evaluating model...")
        metrics = evaluate_model(model, testset, run_id)
        results["stages"]["evaluation"] = {
            "status": "success",
            "metrics": metrics
        }
        
        # Optional: Register model
        if register:
            logger.info("\n[Optional] Registering model...")
            reg_result = register_best_model()
            results["stages"]["registration"] = {
                "status": "success",
                "info": reg_result
            }
        
        results["status"] = "completed"
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline Completed Successfully!")
        logger.info("=" * 60)
        logger.info(f"Model Type: {model_type}")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"RMSE: {metrics.get('rmse', 'N/A'):.4f}")
        logger.info(f"MAE: {metrics.get('mae', 'N/A'):.4f}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        results["status"] = "failed"
        results["error"] = str(e)
        raise
    
    return results


def main():
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run the ML Pipeline for Movie Rating Prediction"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        default=DEFAULT_MODEL_TYPE,
        choices=["svd", "nmf", "knn"],
        help="Type of model to train"
    )
    
    parser.add_argument(
        "--n-factors",
        type=int,
        default=None,
        help="Number of factors (for SVD/NMF)"
    )
    
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=None,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register the model after training"
    )
    
    args = parser.parse_args()
    
    # Build model params from CLI args
    model_params = {}
    if args.n_factors:
        model_params["n_factors"] = args.n_factors
    if args.n_epochs:
        model_params["n_epochs"] = args.n_epochs
    
    # Run pipeline
    results = run_pipeline(
        model_type=args.model_type,
        register=args.register,
        **model_params
    )
    
    return results


if __name__ == "__main__":
    main()
