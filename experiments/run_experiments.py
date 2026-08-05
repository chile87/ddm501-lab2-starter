"""
Experiment Runner - Run multiple experiments for hyperparameter tuning.

This script runs multiple experiments with different configurations
and logs all results to MLflow for comparison.

Design notes:
- Data is loaded and split **once** so every configuration is scored on the
  identical train/test split; comparing RMSE across runs is only meaningful
  under that condition.
- A failing configuration is recorded and skipped, never fatal: one bad
  hyperparameter combination must not discard the whole sweep.

Usage:
    python -m experiments.run_experiments
    python -m experiments.run_experiments --experiment-name my-sweep
"""

import argparse
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pipeline.config import (  # noqa: E402
    EXPERIMENT_CONFIGS,
    MLFLOW_EXPERIMENT_NAME,
    ARTIFACTS_DIR,
)
from pipeline.data_ingestion import load_and_split  # noqa: E402
from pipeline.training import train_model, setup_mlflow, flatten_params  # noqa: E402
from pipeline.evaluation import evaluate_model  # noqa: E402
from pipeline.registry import compare_runs  # noqa: E402

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MLflow experiment that the hyperparameter sweep writes into.
DEFAULT_SWEEP_EXPERIMENT = "hyperparameter-tuning"


def build_run_name(model_type: str, params: Dict[str, Any]) -> str:
    """
    Build a human-readable MLflow run name from a configuration.

    Nested params are flattened first so that a KNN config's similarity metric
    shows up in the run name instead of being dropped.

    Args:
        model_type: Type of model
        params: Hyperparameters (without model_type)

    Returns:
        Run name such as ``svd_n_factors=100_n_epochs=20``

    Example:
        >>> build_run_name("knn", {"k": 40, "sim_options": {"name": "cosine"}})
        'knn_k=40_sim_options.name=cosine'
    """
    flat = flatten_params(params)
    suffix = "_".join(f"{k}={v}" for k, v in flat.items())
    return f"{model_type}_{suffix}" if suffix else model_type


def run_single_experiment(
    trainset: Any,
    testset: Any,
    config: Dict[str, Any],
    experiment_name: str = DEFAULT_SWEEP_EXPERIMENT
) -> Dict[str, Any]:
    """
    Run a single experiment with the given configuration.

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
            'run_name': str,
            'metrics': dict
        }

    Raises:
        ValueError: If ``config`` has no 'model_type' key
    """
    if not config or "model_type" not in config:
        raise ValueError(
            f"Experiment config must contain 'model_type'. Got: {config}"
        )

    mlflow.set_experiment(experiment_name)

    config_copy = dict(config)
    model_type = config_copy.pop("model_type")
    run_name = build_run_name(model_type, config_copy)

    logger.info(f"Starting run '{run_name}'")

    model, run_id = train_model(
        trainset,
        model_type=model_type,
        run_name=run_name,
        **config_copy
    )

    metrics = evaluate_model(model, testset, run_id)

    # Fold in metrics logged by the training stage (e.g. training_time_seconds)
    # so the report can show them without a second source of truth.
    metrics = {**_fetch_run_metrics(run_id), **metrics}

    return {
        "config": config,
        "run_id": run_id,
        "run_name": run_name,
        "metrics": metrics,
    }


def _fetch_run_metrics(run_id: str) -> Dict[str, float]:
    """
    Read all metrics recorded on an MLflow run.

    Args:
        run_id: MLflow run ID

    Returns:
        Metric dictionary, or an empty dict if the run cannot be read
    """
    try:
        return dict(MlflowClient().get_run(run_id).data.metrics)
    except Exception as e:
        logger.warning(f"Could not read metrics back from run {run_id}: {e}")
        return {}


def run_all_experiments(
    configs: List[Dict[str, Any]] = EXPERIMENT_CONFIGS,
    experiment_name: str = DEFAULT_SWEEP_EXPERIMENT
) -> List[Dict[str, Any]]:
    """
    Run all experiments defined in configs.

    Args:
        configs: List of configuration dictionaries
        experiment_name: Name of the MLflow experiment

    Returns:
        List of experiment results. Failed experiments are included with an
        ``'error'`` key instead of ``'metrics'`` so the report can account for
        them.

    Raises:
        ValueError: If ``configs`` is empty
    """
    if not configs:
        raise ValueError("configs must not be empty - nothing to run")

    logger.info(f"Running {len(configs)} experiments into '{experiment_name}'...")

    # Load once: identical split across all configs makes the comparison fair.
    trainset, testset, data_stats = load_and_split()
    logger.info(
        f"Shared split: {trainset.n_ratings} train / {len(testset)} test ratings"
    )

    results: List[Dict[str, Any]] = []

    for i, config in enumerate(configs, start=1):
        logger.info(f"\n--- Experiment {i}/{len(configs)}: {config} ---")
        try:
            result = run_single_experiment(trainset, testset, config, experiment_name)
            results.append(result)
            logger.info(
                f"  OK  RMSE={result['metrics']['rmse']:.4f} "
                f"MAE={result['metrics']['mae']:.4f}"
            )
        except Exception as e:
            # Keep sweeping: record the failure and move to the next config.
            logger.error(f"  FAILED: {e}")
            results.append({"config": config, "error": str(e)})

    n_ok = sum(1 for r in results if "metrics" in r)
    logger.info(f"\nSweep finished: {n_ok}/{len(results)} experiments succeeded")

    return results


def create_comparison_chart(
    results: List[Dict[str, Any]],
    output_path: Optional[str] = None,
) -> Optional[str]:
    """
    Create a bar chart comparing RMSE and MAE across all successful experiments.

    Args:
        results: List of experiment results
        output_path: Where to save the PNG (defaults to artifacts/)

    Returns:
        Path to the saved chart, or None if there was nothing to plot
    """
    successful = [r for r in results if "metrics" in r]
    if not successful:
        logger.warning("No successful experiments - skipping comparison chart")
        return None

    successful = sorted(successful, key=lambda r: r["metrics"]["rmse"])

    labels = [r["run_name"] for r in successful]
    rmses = [r["metrics"]["rmse"] for r in successful]
    maes = [r["metrics"]["mae"] for r in successful]

    if output_path is None:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(ARTIFACTS_DIR / "experiment_comparison.png")

    y = range(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, 0.5 * len(labels) + 2)))

    axes[0].barh(list(y), rmses, color="steelblue")
    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("RMSE (lower is better)")
    axes[0].set_title("RMSE by configuration")
    axes[0].set_xlim(min(rmses) * 0.95, max(rmses) * 1.02)

    axes[1].barh(list(y), maes, color="indianred")
    axes[1].set_yticks(list(y))
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("MAE (lower is better)")
    axes[1].set_title("MAE by configuration")
    axes[1].set_xlim(min(maes) * 0.95, max(maes) * 1.02)

    plt.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Comparison chart saved to {output_path}")
    return output_path


def _summarise_by_model_family(
    successful: List[Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate RMSE per model family (svd / nmf / knn).

    Args:
        successful: Successful experiment results

    Returns:
        Mapping of model_type -> {'best', 'worst', 'mean', 'n'}
    """
    families: Dict[str, List[float]] = {}
    for r in successful:
        families.setdefault(r["config"]["model_type"], []).append(r["metrics"]["rmse"])

    return {
        family: {
            "best": min(values),
            "worst": max(values),
            "mean": sum(values) / len(values),
            "n": float(len(values)),
        }
        for family, values in families.items()
    }


def generate_experiment_report(
    results: List[Dict[str, Any]],
    output_path: str = "experiment_report.md"
) -> str:
    """
    Generate a markdown report from experiment results.

    The report contains a summary, a full results table sorted by RMSE, a
    per-model-family breakdown, the winning configuration, and recommendations.
    It is regenerated from scratch on every sweep, so it always reflects the
    latest run.

    Args:
        results: List of experiment results
        output_path: Path to save the report

    Returns:
        Report content as string
    """
    successful = [r for r in results if "metrics" in r]
    failed = [r for r in results if "metrics" not in r]
    successful_sorted = sorted(successful, key=lambda r: r["metrics"]["rmse"])

    chart_path = create_comparison_chart(results)

    lines: List[str] = []
    lines.append("# Experiment Report - Movie Rating Prediction")
    lines.append("")
    lines.append(
        f"_Auto-generated by `experiments/run_experiments.py` on "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}._"
    )
    lines.append("")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    lines.append("## 1. Summary")
    lines.append("")
    lines.append(f"- Total experiments: **{len(results)}**")
    lines.append(f"- Successful: **{len(successful)}**")
    lines.append(f"- Failed: **{len(failed)}**")
    if successful_sorted:
        best = successful_sorted[0]
        worst = successful_sorted[-1]
        spread = worst["metrics"]["rmse"] - best["metrics"]["rmse"]
        lines.append(
            f"- Best RMSE: **{best['metrics']['rmse']:.4f}** (`{best['run_name']}`)"
        )
        lines.append(
            f"- Worst RMSE: **{worst['metrics']['rmse']:.4f}** (`{worst['run_name']}`)"
        )
        lines.append(f"- Spread between best and worst: **{spread:.4f} RMSE**")
    lines.append("")
    lines.append(
        "All configurations were trained and scored on the **same train/test "
        "split** (`test_size=0.2`, `random_state=42`), so the metrics below are "
        "directly comparable."
    )
    lines.append("")

    # -------------------------------------------------------------------------
    # Results table
    # -------------------------------------------------------------------------
    lines.append("## 2. Results")
    lines.append("")
    lines.append("Sorted by RMSE (lower is better).")
    lines.append("")
    lines.append(
        "| # | Model | Hyperparameters | RMSE | MAE | MSE | Coverage % | Train time (s) |"
    )
    lines.append(
        "|---|-------|-----------------|------|-----|-----|-----------|----------------|"
    )

    for rank, r in enumerate(successful_sorted, start=1):
        model_type = r["config"].get("model_type", "unknown")
        params = {k: v for k, v in r["config"].items() if k != "model_type"}
        params_str = ", ".join(f"{k}={v}" for k, v in flatten_params(params).items())
        m = r["metrics"]
        train_time = m.get("training_time_seconds")
        train_time_str = f"{train_time:.1f}" if train_time is not None else "n/a"
        lines.append(
            f"| {rank} | {model_type} | {params_str} | "
            f"{m['rmse']:.4f} | {m['mae']:.4f} | {m.get('mse', float('nan')):.4f} | "
            f"{m.get('coverage', float('nan')):.1f} | {train_time_str} |"
        )

    if failed:
        lines.append("")
        lines.append("### Failed experiments")
        lines.append("")
        lines.append("| Model | Config | Error |")
        lines.append("|-------|--------|-------|")
        for r in failed:
            model_type = r["config"].get("model_type", "unknown")
            lines.append(f"| {model_type} | `{r['config']}` | {r['error']} |")
    lines.append("")

    # -------------------------------------------------------------------------
    # Per-family analysis
    # -------------------------------------------------------------------------
    if successful:
        lines.append("## 3. Analysis by model family")
        lines.append("")
        lines.append("| Model family | Runs | Best RMSE | Mean RMSE | Worst RMSE |")
        lines.append("|--------------|------|-----------|-----------|------------|")

        families = _summarise_by_model_family(successful)
        for family, stats in sorted(families.items(), key=lambda kv: kv[1]["best"]):
            lines.append(
                f"| {family} | {int(stats['n'])} | {stats['best']:.4f} | "
                f"{stats['mean']:.4f} | {stats['worst']:.4f} |"
            )
        lines.append("")

        ranked = sorted(families.items(), key=lambda kv: kv[1]["best"])
        if len(ranked) > 1:
            winner, runner_up = ranked[0], ranked[1]
            delta = runner_up[1]["best"] - winner[1]["best"]
            lines.append(
                f"**{winner[0].upper()}** produced the lowest RMSE overall, "
                f"beating the next-best family (**{runner_up[0].upper()}**) by "
                f"{delta:.4f} RMSE."
            )
            lines.append("")

    # -------------------------------------------------------------------------
    # Best model
    # -------------------------------------------------------------------------
    if successful_sorted:
        best = successful_sorted[0]
        lines.append("## 4. Best model")
        lines.append("")
        lines.append(f"- Run name: `{best['run_name']}`")
        lines.append(f"- Run ID: `{best['run_id']}`")
        lines.append(f"- Configuration: `{best['config']}`")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for name, value in sorted(best["metrics"].items()):
            if value is None:
                continue
            lines.append(f"| {name} | {value:.4f} |")
        lines.append("")

    # -------------------------------------------------------------------------
    # Visualisation
    # -------------------------------------------------------------------------
    if chart_path:
        lines.append("## 5. Visualisation")
        lines.append("")
        lines.append(
            f"![Experiment comparison]({chart_path.replace(str(ARTIFACTS_DIR.parent) + '/', '')})"
        )
        lines.append("")
        lines.append(
            "Per-run diagnostic plots (actual vs predicted, error distribution, "
            "error by rating) are attached to each MLflow run under `plots/`."
        )
        lines.append("")

    # -------------------------------------------------------------------------
    # Recommendation
    # -------------------------------------------------------------------------
    if successful_sorted:
        best = successful_sorted[0]
        lines.append("## 6. Recommendation")
        lines.append("")
        lines.append(
            f"Promote `{best['run_name']}` (RMSE **{best['metrics']['rmse']:.4f}**, "
            f"MAE **{best['metrics']['mae']:.4f}**) to the **Production** stage of "
            f"the MLflow Model Registry:"
        )
        lines.append("")
        lines.append("```bash")
        lines.append(
            "python -c \"from pipeline.registry import register_best_model; "
            "print(register_best_model(experiment_name='hyperparameter-tuning'))\""
        )
        lines.append("```")
        lines.append("")

    content = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(content)

    logger.info(f"Report written to {output_path}")
    return content


def log_sweep_summary(
    results: List[Dict[str, Any]],
    report_path: str,
    chart_path: Optional[str],
    experiment_name: str = DEFAULT_SWEEP_EXPERIMENT,
) -> Optional[str]:
    """
    Log a summary MLflow run holding the sweep-level report and chart.

    Individual runs carry their own params/metrics; this extra run gives the
    sweep a single place in the UI where the comparison artifacts live.

    Args:
        results: List of experiment results
        report_path: Path to the generated markdown report
        chart_path: Path to the comparison chart (may be None)
        experiment_name: MLflow experiment to log into

    Returns:
        The summary run id, or None if logging failed
    """
    successful = [r for r in results if "metrics" in r]
    if not successful:
        return None

    try:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="experiment_summary") as run:
            mlflow.set_tag("stage", "sweep_summary")
            mlflow.log_param("n_experiments", len(results))
            mlflow.log_param("n_successful", len(successful))
            mlflow.log_metric(
                "best_rmse", min(r["metrics"]["rmse"] for r in successful)
            )
            mlflow.log_metric(
                "best_mae", min(r["metrics"]["mae"] for r in successful)
            )
            mlflow.log_artifact(report_path, artifact_path="reports")
            if chart_path:
                mlflow.log_artifact(chart_path, artifact_path="reports")
            return run.info.run_id
    except Exception as e:
        logger.error(f"Could not log sweep summary run: {e}")
        return None


# =============================================================================
# Main Execution
# =============================================================================
def main(argv: Optional[List[str]] = None):
    """Run all experiments and generate report."""
    parser = argparse.ArgumentParser(
        description="Run hyperparameter tuning experiments for movie rating prediction"
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_SWEEP_EXPERIMENT,
        help="MLflow experiment name to log runs into",
    )
    parser.add_argument(
        "--report-path",
        default="experiment_report.md",
        help="Where to write the generated markdown report",
    )
    args = parser.parse_args(argv)

    logger.info("=" * 60)
    logger.info("Starting Experiment Runner")
    logger.info("=" * 60)

    # Setup MLflow (falls back to a local store if the server is down)
    setup_mlflow(experiment_name=args.experiment_name)

    # Run experiments
    results = run_all_experiments(
        configs=EXPERIMENT_CONFIGS,
        experiment_name=args.experiment_name,
    )

    # Generate report (also produces the comparison chart)
    generate_experiment_report(results, args.report_path)
    chart_path = str(ARTIFACTS_DIR / "experiment_comparison.png")
    log_sweep_summary(results, args.report_path, chart_path, args.experiment_name)

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
    else:
        logger.warning("No experiment succeeded - check the logs above")

    # Compare top runs
    logger.info("\nTop 5 runs:")
    top_runs = compare_runs(
        experiment_name=args.experiment_name, metric="rmse", top_n=5
    )
    for i, run in enumerate(top_runs, 1):
        logger.info(
            f"  {i}. RMSE={run['metrics'].get('rmse', float('nan')):.4f} "
            f"- {run['run_name']}"
        )

    logger.info(f"\nReport saved to: {args.report_path}")
    logger.info("View experiments in MLflow UI: http://localhost:5000")

    return results


if __name__ == "__main__":
    main()
