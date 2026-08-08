# Lab 2: ML Pipeline & Experiment Tracking

> **DDM501 · Lab 2 · 15% of grade**
> _Building Reproducible ML Pipelines with MLflow and Airflow_

## Overview

This project transforms the movie rating prediction system (Lab 1) into a **production-ready ML pipeline** with:

- 📊 **MLflow** experiment tracking, model versioning, and Model Registry
- 🔁 **Apache Airflow** workflow orchestration with weekly retraining
- 🐳 **Docker Compose** for one-command environment setup
- 🧪 Modular, tested, reproducible pipeline code

---

## Project Structure

```
ddm501-lab2-starter/
├── pipeline/
│   ├── config.py           # All configuration in one place
│   ├── data_ingestion.py   # Load & split MovieLens 100K
│   ├── preprocessing.py    # Data validation and statistics
│   ├── training.py         # Model training with full MLflow logging
│   ├── evaluation.py       # Metrics, plots, evaluation report
│   ├── registry.py         # MLflow Model Registry management
│   ├── model_wrapper.py    # pyfunc wrapper for Surprise models
│   └── run_pipeline.py     # CLI entry-point (end-to-end run)
├── dags/
│   └── ml_training_dag.py  # Airflow DAG (7-task pipeline + branching)
├── experiments/
│   └── run_experiments.py  # Hyperparameter sweep + comparison report
├── tests/
│   └── test_pipeline.py    # Unit & integration tests
├── scripts/
│   └── setup_mlflow.py     # Helper to initialise MLflow server
├── artifacts/              # Generated charts & reports
├── models/                 # Saved model pickles
├── docker-compose.yml      # MLflow + Airflow + PostgreSQL
├── Dockerfile              # ML pipeline image
├── Dockerfile.airflow      # Airflow image
├── requirements.txt
└── experiment_report.md    # Auto-generated experiment comparison report
```

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.10+ |
| Docker & Docker Compose | 24+ |
| pip | latest |

---

## Quick Start (Local — no Docker)

### 1. Create virtual environment and install dependencies

```bash
cd ddm501-lab2-starter

python -m venv venv
source venv/bin/activate       # macOS / Linux
# venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

### 2. Start MLflow Tracking Server

```bash
# macOS / Linux
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 --port 5000

# Windows (PowerShell)
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

Open MLflow UI at **<http://localhost:5000>**

### 3. Run a single pipeline pass

```bash
# Default: SVD with standard hyperparameters
python -m pipeline.run_pipeline

# Custom model and hyperparameters
python -m pipeline.run_pipeline --model-type nmf --n-factors 50 --n-epochs 30

# Train + register best model to registry
python -m pipeline.run_pipeline --model-type svd --register
```

### 4. Run hyperparameter sweep (≥ 9 experiments)

```bash
python -m experiments.run_experiments
```

This will:
- Train SVD, NMF, and KNN models with various hyperparameters
- Log every run to MLflow (params, metrics, artifacts, plots)
- Auto-generate `experiment_report.md` with comparison table and chart

### 5. Register best model to Production

```bash
python -c "
from pipeline.training import setup_mlflow
from pipeline.registry import register_best_model
setup_mlflow()
result = register_best_model(experiment_name='hyperparameter-tuning')
print(f\"Registered: {result['model_name']} v{result['version']} → {result['stage']}\")
"
```

---

### 💡 Helper Scripts

For convenience, helper scripts are provided for both Windows PowerShell (`run.ps1`) and Linux/macOS (`Makefile`):

| Command | PowerShell (Windows) | Makefile (Linux/Mac) | Description |
|---------|-----------------------|----------------------|-------------|
| Start MLflow Server | `.\run.ps1 mlflow` | `make mlflow` | Starts tracking server on port 5000 |
| Run Pipeline Pass | `.\run.ps1 train` | `make train` | Runs SVD training pipeline |
| Run 9 Experiments | `.\run.ps1 sweep` | `make sweep` | Sweeps hyperparameters & auto-generates report |
| Run Unit Tests | `.\run.ps1 test` | `make test` | Runs pytest test suite |
| Clean Caches | `.\run.ps1 clean` | `make clean` | Removes cache folders |


---

## Quick Start (Docker — recommended)

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env if you want to change passwords/keys (defaults work for local dev)
```

### 2. Start all services

```bash
docker-compose up -d
```

| Service | URL | Credentials |
|---------|-----|-------------|
| MLflow UI | <http://localhost:5000> | — |
| Airflow UI | <http://localhost:8080> | admin / admin |

### 3. Run ML pipeline inside Docker

```bash
# Full pipeline
docker-compose --profile tools run --rm ml-pipeline

# Hyperparameter sweep
docker-compose --profile tools run --rm ml-pipeline \
  python -m experiments.run_experiments
```

### 4. Stop services

```bash
docker-compose down        # keep data volumes
docker-compose down -v     # also delete volumes (clean slate)
```

---

## ML Pipeline Architecture

```
Data Ingestion → Preprocessing → Training → Evaluation → Model Registry
     │                │              │           │              │
  MovieLens        Validate       MLflow      RMSE/MAE      Production
   100K           Quality        Run Log      + Plots        Stage
```

### Pipeline stages

| Stage | File | Responsibility |
|-------|------|---------------|
| Data Ingestion | `pipeline/data_ingestion.py` | Load MovieLens 100K, train/test split |
| Preprocessing | `pipeline/preprocessing.py` | Validate data quality, compute stats |
| Training | `pipeline/training.py` | Train SVD/NMF/KNN, log to MLflow |
| Evaluation | `pipeline/evaluation.py` | RMSE/MAE/MSE/MAPE, plots, report |
| Registry | `pipeline/registry.py` | Find best run, register to Production |

---

## MLflow Experiment Tracking

Every training run logs:

**Parameters** (searchable in UI)
```python
mlflow.log_param("model_type", "svd")
mlflow.log_param("n_factors", 100)
mlflow.log_param("n_epochs", 20)
mlflow.log_param("dataset", "ml-100k")
```

**Metrics** (sortable / comparable)
```python
mlflow.log_metric("rmse", 0.9348)
mlflow.log_metric("mae", 0.7377)
mlflow.log_metric("coverage", 100.0)
mlflow.log_metric("training_time_seconds", 0.23)
```

**Artifacts**
- Raw model pickle (`pickle/model_svd.pkl`)
- Loadable pyfunc model (`model/`) — registered in Model Registry
- Prediction distribution plot (`plots/prediction_distribution.png`)
- Error-by-rating boxplot (`plots/error_by_rating.png`)
- Text evaluation report (`reports/evaluation_report_*.txt`)

---

## Airflow DAG

DAG ID: `movie_rating_training` · Schedule: **@weekly** (every Sunday 00:00)

```
load_data → preprocess_data → train_model → evaluate_model
                                                    ↓
                                          decide_registration
                                         ↙               ↘
                                  register_model    skip_registration
                                         ↘               ↙
                                              cleanup
```

**Quality gate**: the model is only promoted to the registry when `RMSE < 1.0` (configurable via `AIRFLOW_RMSE_THRESHOLD` env var).

### Test the DAG locally

```bash
export AIRFLOW_HOME=$(pwd)/airflow_home
airflow db migrate
airflow dags test movie_rating_training 2024-01-07
```

---

## Experiment Results

9 experiments were run across 3 model families (SVD, NMF, KNN):

| Rank | Model | Key Params | RMSE | MAE |
|------|-------|-----------|------|-----|
| 🥇 1 | SVD | factors=50, epochs=20, lr=0.005 | **0.9348** | 0.7377 |
| 2 | SVD | factors=100, epochs=20, lr=0.005 | 0.9352 | 0.7375 |
| 3 | SVD | factors=150, epochs=30, lr=0.01 | 0.9596 | 0.7535 |
| 4 | SVD | factors=100, epochs=50, lr=0.005 | 0.9667 | 0.7583 |
| 5 | KNN | k=40, pearson, user-based | 1.0150 | 0.8037 |
| 6 | KNN | k=40, cosine, user-based | 1.0194 | 0.8038 |
| 7 | KNN | k=20, cosine, user-based | 1.0284 | 0.8099 |
| 8 | NMF | factors=50, epochs=50 | 1.0294 | 0.7851 |
| 9 | NMF | factors=100, epochs=50 | 1.1017 | 0.8394 |

**Conclusion**: SVD consistently outperforms KNN and NMF on MovieLens 100K.
See [`experiment_report.md`](experiment_report.md) for the full analysis and charts.

---

## Running Tests

```bash
# All tests (unit + integration markers)
pytest tests/ -v

# Skip slow tests (no MLflow server needed)
pytest tests/ -v -m "not slow"

# With coverage report
pytest tests/ -v --cov=pipeline --cov-report=term-missing
```

---

## Configuration Reference

All tunable settings are in [`pipeline/config.py`](pipeline/config.py):

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URL |
| `MLFLOW_EXPERIMENT_NAME` | `movie-rating-prediction` | Main experiment name |
| `DATASET_NAME` | `ml-100k` | Surprise built-in dataset |
| `TEST_SIZE` | `0.2` | Train/test split ratio |
| `RANDOM_STATE` | `42` | Global random seed |
| `AIRFLOW_RMSE_THRESHOLD` | `1.0` | Quality gate for model registration |
| `REGISTERED_MODEL_NAME` | `movie-rating-model` | Name in Model Registry |


---

## Submission Checklist

- [x] Complete ML pipeline with modular stages
- [x] MLflow tracking configured and running
- [x] ≥ 5 experiments with different configurations logged
- [x] Airflow DAG with full task graph and @weekly schedule
- [x] Experiment comparison report (`experiment_report.md`)
- [x] Best model registered to Production stage in Model Registry
- [x] README with setup and usage instructions
- [ ] Screenshots of MLflow UI showing experiment runs _(add before submitting)_
- [ ] GitHub repository link submitted

---

## Troubleshooting

**MLflow server not reachable** — the pipeline falls back to a local `file:./mlruns` store automatically, so you can still run experiments without the server.

**Airflow import error in DAG** — make sure the project root is on `PYTHONPATH`:
```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```

**MovieLens dataset download fails** — set a custom download folder:
```bash
export SURPRISE_DATA_FOLDER=~/.surprise_data
```