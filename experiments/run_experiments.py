"""
Experiment Runner - Run multiple experiments for hyperparameter tuning.

This script runs multiple experiments with different configurations
and logs all results to MLflow for comparison.

TODO: Complete the functions marked with TODO.

Usage:
    python -m experiments.run_experiments
"""

import logging
from typing import Dict, Any, List
import json
from datetime import datetime

import mlflow

from pipeline.config import EXPERIMENT_CONFIGS, MLFLOW_EXPERIMENT_NAME
from pipeline.data_ingestion import load_and_split
from pipeline.training import train_model, setup_mlflow
from pipeline.evaluation import evaluate_model
from pipeline.registry import compare_runs

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# TODO 1: Implement run_single_experiment function
# =============================================================================
def run_single_experiment(
    trainset: Any,
    testset: Any,
    config: Dict[str, Any],
    experiment_name: str = "hyperparameter-tuning"
) -> Dict[str, Any]:
    """
    Run a single experiment with the given configuration.
    
    TODO: Implement this function that:
    1. Sets the MLflow experiment
    2. Extracts model_type from config
    3. Trains the model using train_model()
    4. Evaluates the model using evaluate_model()
    5. Returns experiment results
    
    Args:
        trainset: Training data
        testset: Test data
        config: Configuration dictionary with model_type and hyperparameters
        experiment_name: Name of the MLflow experiment
        
    Returns:
        Dictionary with experiment results:
        {
            'config': dict,
            'run_id': str,
            'metrics': dict
        }
    """
    # TODO: Implement this function
    #
    # Example structure:
    # mlflow.set_experiment(experiment_name)
    # 
    # config_copy = config.copy()
    # model_type = config_copy.pop("model_type")
    # 
    # # Create descriptive run name
    # run_name = f"{model_type}_" + "_".join(f"{k}={v}" for k, v in config_copy.items() if not isinstance(v, dict))
    # 
    # # Train model
    # model, run_id = train_model(
    #     trainset,
    #     model_type=model_type,
    #     run_name=run_name,
    #     **config_copy
    # )
    # 
    # # Evaluate model
    # metrics = evaluate_model(model, testset, run_id)
    # 
    # return {
    #     "config": config,
    #     "run_id": run_id,
    #     "metrics": metrics
    # }
    
    pass  # Remove this and implement the function


# =============================================================================
# TODO 2: Implement run_all_experiments function
# =============================================================================
def run_all_experiments(
    configs: List[Dict[str, Any]] = EXPERIMENT_CONFIGS,
    experiment_name: str = "hyperparameter-tuning"
) -> List[Dict[str, Any]]:
    """
    Run all experiments defined in configs.
    
    TODO: Implement this function that:
    1. Loads data once (for efficiency)
    2. Iterates through all configs
    3. Runs each experiment using run_single_experiment()
    4. Collects and returns all results
    
    Args:
        configs: List of configuration dictionaries
        experiment_name: Name of the MLflow experiment
        
    Returns:
        List of experiment results
    """
    # TODO: Implement this function
    #
    # Example structure:
    # logger.info(f"Running {len(configs)} experiments...")
    # 
    # # Load data once
    # trainset, testset, _ = load_and_split()
    # 
    # results = []
    # for i, config in enumerate(configs):
    #     logger.info(f"\nExperiment {i+1}/{len(configs)}: {config}")
    #     try:
    #         result = run_single_experiment(trainset, testset, config, experiment_name)
    #         results.append(result)
    #         logger.info(f"  RMSE: {result['metrics']['rmse']:.4f}")
    #     except Exception as e:
    #         logger.error(f"  Failed: {e}")
    #         results.append({"config": config, "error": str(e)})
    # 
    # return results
    
    pass  # Remove this and implement the function


# =============================================================================
# TODO 3: Implement generate_experiment_report function
# =============================================================================
def generate_experiment_report(
    results: List[Dict[str, Any]],
    output_path: str = "experiment_report.md"
) -> str:
    """
    Generate a markdown report from experiment results.
    
    TODO: Implement this function that creates a markdown report with:
    1. Summary statistics
    2. Table of all experiments with metrics
    3. Best performing model details
    4. Recommendations
    
    Args:
        results: List of experiment results
        output_path: Path to save the report
        
    Returns:
        Report content as string
    """
    # TODO: Implement this function
    #
    # Example structure:
    # report = []
    # report.append("# Experiment Report")
    # report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    # 
    # # Summary
    # successful = [r for r in results if 'metrics' in r]
    # report.append(f"## Summary\n")
    # report.append(f"- Total experiments: {len(results)}")
    # report.append(f"- Successful: {len(successful)}")
    # report.append(f"- Failed: {len(results) - len(successful)}\n")
    # 
    # # Results table
    # report.append("## Results\n")
    # report.append("| Model | Parameters | RMSE | MAE |")
    # report.append("|-------|------------|------|-----|")
    # 
    # for r in successful:
    #     model_type = r['config'].get('model_type', 'unknown')
    #     params = {k: v for k, v in r['config'].items() if k != 'model_type'}
    #     rmse = r['metrics'].get('rmse', 'N/A')
    #     mae = r['metrics'].get('mae', 'N/A')
    #     report.append(f"| {model_type} | {params} | {rmse:.4f} | {mae:.4f} |")
    # 
    # # Best model
    # if successful:
    #     best = min(successful, key=lambda x: x['metrics'].get('rmse', float('inf')))
    #     report.append(f"\n## Best Model\n")
    #     report.append(f"- Configuration: {best['config']}")
    #     report.append(f"- RMSE: {best['metrics']['rmse']:.4f}")
    #     report.append(f"- Run ID: {best['run_id']}")
    # 
    # content = "\n".join(report)
    # 
    # with open(output_path, 'w') as f:
    #     f.write(content)
    # 
    # return content
    
    pass  # Remove this and implement the function


# =============================================================================
# Main Execution
# =============================================================================
def main():
    """Run all experiments and generate report."""
    
    logger.info("=" * 60)
    logger.info("Starting Experiment Runner")
    logger.info("=" * 60)
    
    # Setup MLflow
    setup_mlflow()
    
    # Run experiments
    results = run_all_experiments(
        configs=EXPERIMENT_CONFIGS,
        experiment_name="hyperparameter-tuning"
    )
    
    # Generate report
    report = generate_experiment_report(results, "experiment_report.md")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Experiment Summary")
    logger.info("=" * 60)
    
    successful = [r for r in results if 'metrics' in r]
    if successful:
        best = min(successful, key=lambda x: x['metrics'].get('rmse', float('inf')))
        logger.info(f"Total experiments: {len(results)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Best RMSE: {best['metrics']['rmse']:.4f}")
        logger.info(f"Best config: {best['config']}")
    
    # Compare top runs
    logger.info("\nTop 5 runs:")
    top_runs = compare_runs(metric="rmse", top_n=5)
    for i, run in enumerate(top_runs, 1):
        logger.info(f"  {i}. RMSE={run['metrics'].get('rmse', 'N/A'):.4f} - {run['params']}")
    
    logger.info(f"\nReport saved to: experiment_report.md")
    logger.info("View experiments in MLflow UI: http://localhost:5000")
    
    return results


if __name__ == "__main__":
    main()
