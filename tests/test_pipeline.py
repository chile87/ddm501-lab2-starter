"""
Unit tests for ML Pipeline.

Run tests with:
    pytest tests/ -v
    pytest tests/ -v --cov=pipeline --cov-report=term-missing
    pytest tests/ -v -m "not slow"   # skip tests that need a real MLflow server
"""

import pytest
from unittest.mock import MagicMock, patch, ANY


# =============================================================================
# Data Ingestion
# =============================================================================
class TestDataIngestion:
    """Tests for data ingestion module."""

    def test_load_data_returns_dataset(self):
        """load_data should return a non-None Surprise Dataset."""
        from pipeline.data_ingestion import load_data

        data = load_data("ml-100k")
        assert data is not None

    def test_split_data_returns_train_test(self):
        """split_data should return a non-empty trainset and testset."""
        from pipeline.data_ingestion import load_data, split_data

        data = load_data("ml-100k")
        trainset, testset = split_data(data, test_size=0.2)

        assert trainset is not None
        assert testset is not None
        assert len(testset) > 0

    def test_get_data_stats(self):
        """get_data_stats should return the expected keys with sane values."""
        from pipeline.data_ingestion import load_data, get_data_stats

        data = load_data("ml-100k")
        stats = get_data_stats(data)

        assert "n_users" in stats
        assert "n_items" in stats
        assert "n_ratings" in stats
        assert stats["n_users"] > 0
        assert stats["n_items"] > 0

    def test_load_and_split_returns_three_items(self):
        """load_and_split should return (trainset, testset, stats)."""
        from pipeline.data_ingestion import load_and_split

        trainset, testset, stats = load_and_split()
        assert trainset is not None
        assert testset is not None
        assert isinstance(stats, dict)


# =============================================================================
# Preprocessing
# =============================================================================
class TestPreprocessing:
    """Tests for preprocessing module."""

    def test_validate_trainset(self):
        """validate_trainset should mark a real trainset as valid."""
        from pipeline.data_ingestion import load_and_split
        from pipeline.preprocessing import validate_trainset

        trainset, _, _ = load_and_split()
        report = validate_trainset(trainset)

        assert "is_valid" in report
        assert report["is_valid"] is True

    def test_get_rating_distribution(self):
        """get_rating_distribution should return mean in the valid rating range."""
        from pipeline.data_ingestion import load_and_split
        from pipeline.preprocessing import get_rating_distribution

        trainset, _, _ = load_and_split()
        dist = get_rating_distribution(trainset)

        assert "mean" in dist
        assert "std" in dist
        assert 1.0 <= dist["mean"] <= 5.0


# =============================================================================
# Training
# =============================================================================
class TestTraining:
    """Tests for training module."""

    def test_list_available_models(self):
        """list_available_models should include svd, nmf and knn."""
        from pipeline.training import list_available_models

        models = list_available_models()
        assert "svd" in models
        assert "nmf" in models
        assert "knn" in models

    def test_get_default_params(self):
        """get_default_params should return a dict with the key hyperparameters."""
        from pipeline.training import get_default_params

        params = get_default_params("svd")
        assert "n_factors" in params
        assert "n_epochs" in params

    def test_get_model_class_svd(self):
        """get_model_class should return the SVD class for 'svd'."""
        from pipeline.training import get_model_class
        from surprise import SVD

        assert get_model_class("svd") is SVD

    def test_get_model_class_invalid_raises(self):
        """get_model_class should raise ValueError for unknown model types."""
        from pipeline.training import get_model_class

        with pytest.raises(ValueError, match="Unknown model type"):
            get_model_class("unknown_model")

    def test_flatten_params_nested(self):
        """flatten_params should flatten nested dicts with dot-notation keys."""
        from pipeline.training import flatten_params

        result = flatten_params({"k": 40, "sim_options": {"name": "cosine"}})
        assert result["k"] == 40
        assert result["sim_options.name"] == "cosine"

    def test_flatten_params_flat(self):
        """flatten_params should leave already-flat dicts unchanged."""
        from pipeline.training import flatten_params

        result = flatten_params({"n_factors": 100, "n_epochs": 20})
        assert result == {"n_factors": 100, "n_epochs": 20}

    def test_build_model_svd(self):
        """build_model should return an untrained SVD estimator."""
        from pipeline.training import build_model
        from surprise import SVD

        model = build_model("svd", n_factors=10, n_epochs=5)
        assert isinstance(model, SVD)

    def test_build_model_invalid_params_raises(self):
        """build_model should raise ValueError for invalid hyperparameter names."""
        from pipeline.training import build_model

        with pytest.raises(ValueError, match="Invalid hyperparameters"):
            build_model("svd", nonexistent_param=999)

    @pytest.mark.slow
    def test_train_model(self):
        """train_model should return a fitted model and a non-empty run_id."""
        from pipeline.data_ingestion import load_and_split
        from pipeline.training import train_model, setup_mlflow

        setup_mlflow(allow_local_fallback=True)
        trainset, _, _ = load_and_split()
        model, run_id = train_model(
            trainset, model_type="svd", n_factors=10, n_epochs=5
        )

        assert model is not None
        assert run_id is not None
        assert len(run_id) > 0

    def test_train_model_none_trainset_raises(self):
        """train_model should raise ValueError when trainset is None."""
        from pipeline.training import train_model

        with pytest.raises(ValueError, match="trainset must not be None"):
            train_model(None, model_type="svd")

    def test_train_with_config_missing_model_type_raises(self):
        """train_with_config should raise ValueError when config has no model_type."""
        from pipeline.training import train_with_config

        with pytest.raises(ValueError, match="model_type"):
            train_with_config(MagicMock(), {"n_factors": 50})


# =============================================================================
# Evaluation
# =============================================================================
class TestEvaluation:
    """Tests for evaluation module."""

    def test_create_prediction_distribution_plot(self):
        """create_prediction_distribution_plot should return a Figure."""
        from pipeline.evaluation import create_prediction_distribution_plot
        from surprise import SVD, Dataset
        from surprise.model_selection import train_test_split

        data = Dataset.load_builtin("ml-100k")
        trainset, testset = train_test_split(data, test_size=0.1)
        model = SVD(n_factors=10, n_epochs=5)
        model.fit(trainset)
        predictions = model.test(testset[:100])

        fig = create_prediction_distribution_plot(predictions)
        assert fig is not None

    def test_create_error_by_rating_plot(self):
        """create_error_by_rating_plot should return a Figure."""
        from pipeline.evaluation import create_error_by_rating_plot
        from surprise import SVD, Dataset
        from surprise.model_selection import train_test_split

        data = Dataset.load_builtin("ml-100k")
        trainset, testset = train_test_split(data, test_size=0.1)
        model = SVD(n_factors=10, n_epochs=5)
        model.fit(trainset)
        predictions = model.test(testset[:100])

        fig = create_error_by_rating_plot(predictions)
        assert fig is not None

    def test_calculate_additional_metrics(self):
        """calculate_additional_metrics should return the expected metric keys."""
        from pipeline.evaluation import calculate_additional_metrics
        from surprise import SVD, Dataset
        from surprise.model_selection import train_test_split

        data = Dataset.load_builtin("ml-100k")
        trainset, testset = train_test_split(data, test_size=0.1)
        model = SVD(n_factors=10, n_epochs=5)
        model.fit(trainset)
        predictions = model.test(testset[:200])

        metrics = calculate_additional_metrics(predictions)
        for key in ("mse", "coverage", "n_predictions", "max_error"):
            assert key in metrics, f"Missing metric: {key}"

    def test_calculate_additional_metrics_empty_raises(self):
        """calculate_additional_metrics should raise ValueError on empty input."""
        from pipeline.evaluation import calculate_additional_metrics

        with pytest.raises(ValueError, match="must not be empty"):
            calculate_additional_metrics([])

    def test_evaluate_model_no_run_id_raises(self):
        """evaluate_model should raise ValueError when run_id is missing."""
        from pipeline.evaluation import evaluate_model

        with pytest.raises(ValueError, match="run_id is required"):
            evaluate_model(MagicMock(), [("u", "i", 4.0)], run_id=None, log_to_mlflow=True)

    def test_evaluate_model_none_model_raises(self):
        """evaluate_model should raise ValueError when model is None."""
        from pipeline.evaluation import evaluate_model

        with pytest.raises(ValueError, match="model must not be None"):
            evaluate_model(None, [("u", "i", 4.0)])

    def test_evaluate_model_empty_testset_raises(self):
        """evaluate_model should raise ValueError when testset is empty."""
        from pipeline.evaluation import evaluate_model

        with pytest.raises(ValueError, match="testset must not be empty"):
            evaluate_model(MagicMock(), [])

    def test_save_evaluation_report(self, tmp_path):
        """save_evaluation_report should create a readable text file."""
        from pipeline.evaluation import save_evaluation_report

        metrics = {"rmse": 0.93, "mae": 0.74, "coverage": 100.0}
        report_path = tmp_path / "report.txt"
        save_evaluation_report(metrics, str(report_path))

        assert report_path.exists()
        content = report_path.read_text()
        assert "rmse" in content
        assert "0.9300" in content

    @pytest.mark.slow
    def test_evaluate_model_offline(self):
        """evaluate_model should return correct metric keys when log_to_mlflow=False."""
        from pipeline.data_ingestion import load_and_split
        from pipeline.evaluation import evaluate_model
        from surprise import SVD

        trainset, testset, _ = load_and_split()
        model = SVD(n_factors=10, n_epochs=5)
        model.fit(trainset)

        metrics = evaluate_model(model, testset, log_to_mlflow=False)
        assert "rmse" in metrics
        assert "mae" in metrics
        assert 0 < metrics["rmse"] < 5


# =============================================================================
# Registry
# =============================================================================
class TestRegistry:
    """Tests for registry module."""

    def test_list_registered_models(self):
        """list_registered_models should return a list (may be empty)."""
        from pipeline.registry import list_registered_models

        models = list_registered_models()
        assert isinstance(models, list)

    def test_register_model_empty_run_id_raises(self):
        """register_model should raise ValueError when run_id is empty."""
        from pipeline.registry import register_model

        with pytest.raises(ValueError, match="run_id must not be empty"):
            register_model("", "test-model")

    def test_find_best_run_missing_experiment_raises(self):
        """find_best_run should raise ValueError for a non-existent experiment."""
        from pipeline.registry import find_best_run

        with pytest.raises(ValueError, match="not found"):
            find_best_run(experiment_name="__nonexistent_experiment__9999")

    def test_transition_model_stage_invalid_raises(self):
        """transition_model_stage should raise ValueError for invalid stage names."""
        from pipeline.registry import transition_model_stage

        with pytest.raises(ValueError, match="Invalid stage"):
            transition_model_stage("some-model", "1", stage="InvalidStage")


# =============================================================================
# Experiments
# =============================================================================
class TestExperiments:
    """Tests for the experiment runner module."""

    def test_build_run_name_flat_params(self):
        """build_run_name should produce a readable string from flat params."""
        from experiments.run_experiments import build_run_name

        name = build_run_name("svd", {"n_factors": 100, "n_epochs": 20})
        assert name.startswith("svd_")
        assert "n_factors=100" in name
        assert "n_epochs=20" in name

    def test_build_run_name_nested_params(self):
        """build_run_name should flatten nested params in the run name."""
        from experiments.run_experiments import build_run_name

        name = build_run_name("knn", {"k": 40, "sim_options": {"name": "cosine"}})
        assert "sim_options.name=cosine" in name

    def test_build_run_name_no_params(self):
        """build_run_name with empty params should just return the model type."""
        from experiments.run_experiments import build_run_name

        assert build_run_name("svd", {}) == "svd"

    def test_run_single_experiment_missing_model_type_raises(self):
        """run_single_experiment should raise ValueError when config has no model_type."""
        from experiments.run_experiments import run_single_experiment

        with pytest.raises(ValueError, match="model_type"):
            run_single_experiment(MagicMock(), MagicMock(), {"n_factors": 50})

    def test_run_all_experiments_empty_raises(self):
        """run_all_experiments should raise ValueError when configs list is empty."""
        from experiments.run_experiments import run_all_experiments

        with pytest.raises(ValueError, match="must not be empty"):
            run_all_experiments(configs=[])

    def test_create_comparison_chart_no_results(self):
        """create_comparison_chart should return None when there are no successes."""
        from experiments.run_experiments import create_comparison_chart

        result = create_comparison_chart([{"config": {"model_type": "svd"}, "error": "oops"}])
        assert result is None


# =============================================================================
# Config
# =============================================================================
class TestConfig:
    """Tests for configuration module."""

    def test_config_values(self):
        """Core config constants should have the expected types and values."""
        from pipeline.config import (
            DATASET_NAME,
            TEST_SIZE,
            DEFAULT_MODEL_TYPE,
            MODEL_CONFIGS,
            AIRFLOW_DAG_ID,
            AIRFLOW_SCHEDULE,
        )

        assert DATASET_NAME == "ml-100k"
        assert 0 < TEST_SIZE < 1
        assert DEFAULT_MODEL_TYPE in {"svd", "nmf", "knn"}
        assert "svd" in MODEL_CONFIGS
        assert "nmf" in MODEL_CONFIGS
        assert "knn" in MODEL_CONFIGS
        assert isinstance(AIRFLOW_DAG_ID, str)
        assert isinstance(AIRFLOW_SCHEDULE, str)

    def test_experiment_configs_have_model_type(self):
        """Every entry in EXPERIMENT_CONFIGS must have a 'model_type' key."""
        from pipeline.config import EXPERIMENT_CONFIGS

        assert len(EXPERIMENT_CONFIGS) >= 5, "Need at least 5 experiments"
        for cfg in EXPERIMENT_CONFIGS:
            assert "model_type" in cfg, f"Missing 'model_type' in config: {cfg}"


# =============================================================================
# Integration tests (marked slow — skip with: pytest -m "not slow")
# =============================================================================
class TestPipelineIntegration:
    """End-to-end integration tests for the complete pipeline."""

    @pytest.mark.slow
    def test_full_pipeline_svd(self):
        """Full pipeline run with SVD should complete without errors."""
        from pipeline.run_pipeline import run_pipeline

        results = run_pipeline(model_type="svd", register=False,
                               n_factors=10, n_epochs=5)
        assert results["status"] == "completed"
        assert "training" in results["stages"]
        assert "evaluation" in results["stages"]
        rmse = results["stages"]["evaluation"]["metrics"]["rmse"]
        assert 0 < rmse < 5

    @pytest.mark.slow
    def test_full_pipeline_knn(self):
        """Full pipeline run with KNN should complete without errors."""
        from pipeline.run_pipeline import run_pipeline

        results = run_pipeline(
            model_type="knn",
            register=False,
            k=10,
            sim_options={"name": "cosine", "user_based": True},
        )
        assert results["status"] == "completed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
