# Lab 2: ML Pipeline & Experiment Tracking

## Overview

Build reproducible ML pipelines with MLflow experiment tracking and Airflow workflow orchestration for the movie rating prediction system.


## Project Structure

```
ddm501-lab2-starter/
├── pipeline/
│   ├── __init__.py
│   ├── config.py           # Configuration parameters
│   ├── data_ingestion.py   # Load and split data
│   ├── preprocessing.py    # Data preprocessing
│   ├── training.py         # Model training with MLflow (TODO)
│   ├── evaluation.py       # Model evaluation (TODO)
│   └── registry.py         # Model registration (TODO)
├── dags/
│   └── ml_training_dag.py  # Airflow DAG (TODO)
├── experiments/
│   └── run_experiments.py  # Hyperparameter tuning (TODO)
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py    # Pipeline tests
├── scripts/
│   └── setup_mlflow.py     # MLflow setup script
├── docker-compose.yml      # MLflow + Airflow services (TODO)
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Lab 1 completed (familiarity with the movie rating model)

## Quick Start

### 1. Clone and Setup

```bash
unzip ddm501-lab2-starter.zip
cd ddm501-lab2-starter

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Start MLflow Server

```bash
# Option 1: Simple UI
mlflow ui --host 0.0.0.0 --port 5000

# Option 2: With backend store
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 --port 5000
```

Access MLflow UI at: http://localhost:5000

### 3. Run Pipeline

```bash
# Run complete pipeline
python -m pipeline.run_pipeline

# Or run individual stages
python -c "from pipeline.data_ingestion import load_data; load_data()"
```

### 4. Start Airflow (Optional)

```bash
# Initialize Airflow
airflow db init
airflow users create --username admin --password admin \
  --firstname Admin --lastname User --role Admin --email admin@example.com

# Start services (in separate terminals)
airflow webserver --port 8080
airflow scheduler
```

Access Airflow UI at: http://localhost:8080

### 5. Run with Docker

```bash
docker-compose up -d
```

## TODO Tasks

Complete the following files:

- [ ] `pipeline/training.py` - Implement `train_model()` with MLflow logging
- [ ] `pipeline/evaluation.py` - Implement `evaluate_model()` with metrics
- [ ] `pipeline/registry.py` - Implement `register_best_model()`
- [ ] `dags/ml_training_dag.py` - Create Airflow DAG
- [ ] `experiments/run_experiments.py` - Run hyperparameter experiments
- [ ] `docker-compose.yml` - Configure MLflow and Airflow services

## MLflow Tracking

### Logging Parameters
```python
mlflow.log_param("model_type", "svd")
mlflow.log_param("n_factors", 100)
```

### Logging Metrics
```python
mlflow.log_metric("rmse", 0.95)
mlflow.log_metric("mae", 0.75)
```

### Logging Artifacts
```python
mlflow.log_artifact("model.pkl")
mlflow.log_figure(fig, "plot.png")
```

## Experiment Report

Your experiment report should include:

1. **Experiment Setup**: Description of models and hyperparameters tested
2. **Results Table**: Metrics for all experiments
3. **Analysis**: Which model/parameters performed best and why
4. **Visualizations**: Charts comparing experiments
5. **Recommendations**: Selected model for production

## Running Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=pipeline
```

## Grading Rubric

| Criteria | Weight |
|----------|--------|
| Pipeline Quality | 35% |
| Experiment Tracking (MLflow) | 25% |
| Airflow Automation | 20% |
| Documentation | 20% |

## Submission

1. Complete all TODO tasks
2. Run at least 5 experiments with different configurations
3. Generate experiment comparison report
4. Include MLflow UI screenshots
5. Push to GitHub and submit link