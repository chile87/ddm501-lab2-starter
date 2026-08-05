# =============================================================================
# ML Pipeline image - runs the pipeline / experiment sweep against MLflow.
# =============================================================================
FROM python:3.11-slim

# Build toolchain for scikit-surprise's Cython extension.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Airflow is only needed by the Airflow containers, so it is stripped here to
# keep this image small and avoid dependency conflicts with mlflow.
COPY requirements.txt .
RUN pip install --no-cache-dir "setuptools<81" "numpy==1.26.2" \
    && grep -viE "^apache-airflow" requirements.txt > requirements-pipeline.txt \
    && pip install --no-cache-dir -r requirements-pipeline.txt

COPY pipeline/ ./pipeline/
COPY experiments/ ./experiments/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY pytest.ini .

# Talk to the MLflow service on the compose network by default.
ENV MLFLOW_TRACKING_URI=http://mlflow:5000 \
    PYTHONUNBUFFERED=1

CMD ["python", "-m", "pipeline.run_pipeline"]
