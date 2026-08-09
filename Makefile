.PHONY: help mlflow train sweep test clean

help:
	@echo "=========================================================="
	@echo "  DDM501 Lab 2 - ML Pipeline Commands (Make)"
	@echo "=========================================================="
	@echo "  make mlflow  - Start MLflow Tracking Server"
	@echo "  make train   - Run single pipeline pass"
	@echo "  make sweep   - Run hyperparameter sweep"
	@echo "  make test    - Run pytest test suite"
	@echo "  make clean   - Remove cache files"

MLFLOW_HOST ?= 0.0.0.0
MLFLOW_PORT ?= 5000

mlflow:
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host $(MLFLOW_HOST) --port $(MLFLOW_PORT)

train:
	python -m pipeline.run_pipeline

sweep:
	python -m experiments.run_experiments

test:
	pytest tests/test_pipeline.py -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
