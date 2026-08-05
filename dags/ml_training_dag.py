"""
Airflow DAG for ML Training Pipeline.

This DAG orchestrates the movie rating prediction training pipeline:

    load_data -> preprocess_data -> train_model -> evaluate_model
              -> decide_registration -> [register_model | skip_registration]
              -> cleanup

Design notes:
- **Task isolation**: each task is a thin Airflow adapter around a
  ``pipeline.*`` function. All ML logic lives in the pipeline package so it can
  be tested and run without Airflow.
- **Run-scoped scratch space**: intermediate artifacts go to a directory keyed
  by the Airflow ``run_id``, not a fixed ``/tmp`` path, so concurrent or
  backfilled DAG runs cannot overwrite each other's data.
- **Quality gate**: the model is only promoted to the registry when its RMSE
  beats ``AIRFLOW_RMSE_THRESHOLD``.

Usage:
    # Local: point Airflow at this folder, then
    airflow dags test movie_rating_training 2024-01-07

    # Docker: dags/ is mounted into the scheduler/webserver containers
    docker-compose up -d      # Airflow UI at http://localhost:8080
"""

from datetime import datetime, timedelta
import logging
import os
import pickle
import shutil
import sys

# Make the ``pipeline`` package importable when Airflow loads this file from the
# dags/ folder (dags/../ is the project root, which contains pipeline/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator

from pipeline.config import (
    AIRFLOW_DAG_ID,
    AIRFLOW_MODEL_CONFIG,
    AIRFLOW_RMSE_THRESHOLD,
    AIRFLOW_SCHEDULE,
    MLFLOW_EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
)

logger = logging.getLogger(__name__)

# Task IDs referenced by XCom pulls; named constants avoid silent typos.
TASK_LOAD_DATA = "load_data"
TASK_PREPROCESS = "preprocess_data"
TASK_TRAIN = "train_model"
TASK_EVALUATE = "evaluate_model"
TASK_DECIDE = "decide_registration"
TASK_REGISTER = "register_model"
TASK_SKIP = "skip_registration"
TASK_CLEANUP = "cleanup"

# =============================================================================
# Default Arguments
# =============================================================================
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=1),
}

# =============================================================================
# DAG Definition
# =============================================================================
dag = DAG(
    AIRFLOW_DAG_ID,
    default_args=default_args,
    description='ML Training Pipeline for Movie Rating Prediction',
    # Weekly retraining, every Sunday at 00:00. Equivalent cron: '0 0 * * 0'.
    schedule=AIRFLOW_SCHEDULE,
    start_date=datetime(2024, 1, 1),
    # Don't replay every week since start_date on first deploy.
    catchup=False,
    # Training is stateful (writes to the registry); never overlap runs.
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=2),
    tags=['ml', 'training', 'movie-rating'],
    doc_md=__doc__,
)


# =============================================================================
# Helpers
# =============================================================================
def _run_dir(context) -> str:
    """
    Return the scratch directory for the current DAG run.

    Keying the path by run_id keeps concurrent/backfilled runs from sharing
    (and corrupting) the same pickle files.

    Args:
        context: Airflow task context

    Returns:
        Absolute path to this run's scratch directory
    """
    # run_id can contain characters that are awkward in paths (':', '+').
    safe_run_id = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in context["run_id"]
    )
    return f"/tmp/airflow_ml_pipeline/{safe_run_id}"


def _load_pickle(tmp_dir: str, name: str):
    """
    Load a pickle produced by an upstream task.

    Args:
        tmp_dir: This run's scratch directory
        name: File name inside ``tmp_dir``

    Returns:
        The unpickled object

    Raises:
        AirflowException: If the file is missing, i.e. the upstream task did not
            produce it
    """
    path = os.path.join(tmp_dir, name)
    if not os.path.exists(path):
        raise AirflowException(
            f"Expected artifact '{path}' from an upstream task was not found. "
            f"Check that the previous task completed successfully."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# =============================================================================
# Task Functions
# =============================================================================

def load_data_task(**context):
    """
    Task 1: Load and prepare data.

    This function:
    1. Loads the dataset
    2. Splits into train/test
    3. Saves to this run's scratch directory
    4. Pushes metadata via XCom
    """
    from pipeline.data_ingestion import load_and_split

    print("Loading data...")
    trainset, testset, stats = load_and_split()

    tmp_dir = _run_dir(context)
    os.makedirs(tmp_dir, exist_ok=True)

    with open(f'{tmp_dir}/trainset.pkl', 'wb') as f:
        pickle.dump(trainset, f)
    with open(f'{tmp_dir}/testset.pkl', 'wb') as f:
        pickle.dump(testset, f)

    # Push stats via XCom
    context['ti'].xcom_push(key='data_stats', value=stats)
    context['ti'].xcom_push(key='data_path', value=tmp_dir)

    print(f"Data loaded: {stats['n_ratings']} ratings -> {tmp_dir}")
    return "Data loaded successfully"


def preprocess_data_task(**context):
    """
    Task 2: Preprocess and validate data.

    Loads the split produced upstream, runs validation/statistics, and fails the
    DAG run if the data does not pass validation - training on invalid data
    would waste an hour and pollute the experiment history.
    """
    from pipeline.preprocessing import preprocess_data

    tmp_dir = context['ti'].xcom_pull(task_ids=TASK_LOAD_DATA, key='data_path')
    if not tmp_dir:
        raise AirflowException(
            f"No 'data_path' in XCom from '{TASK_LOAD_DATA}' - did it run?"
        )

    trainset = _load_pickle(tmp_dir, 'trainset.pkl')
    testset = _load_pickle(tmp_dir, 'testset.pkl')

    report = preprocess_data(trainset, testset)

    context['ti'].xcom_push(key='preprocess_report', value=report)

    # Data-quality gate: stop the pipeline rather than train on bad data.
    if not report["preprocessing_successful"]:
        issues = (
            report["trainset_validation"]["issues"]
            + report["testset_validation"]["issues"]
        )
        raise AirflowException(f"Data validation failed: {issues}")

    print(
        f"Preprocessing complete. "
        f"mean_rating={report['rating_distribution']['mean']:.3f}"
    )
    return "Preprocessing complete"


def train_model_task(**context):
    """
    Task 3: Train the model with MLflow tracking.

    Uses ``AIRFLOW_MODEL_CONFIG`` from pipeline.config, and pushes the resulting
    MLflow ``run_id`` so the evaluation task logs its metrics onto the same run.
    """
    from pipeline.training import setup_mlflow, train_with_config

    tmp_dir = context['ti'].xcom_pull(task_ids=TASK_LOAD_DATA, key='data_path')
    trainset = _load_pickle(tmp_dir, 'trainset.pkl')

    # Scheduled retrains log into the main experiment, keeping them separate
    # from the ad-hoc hyperparameter sweep.
    setup_mlflow(experiment_name=MLFLOW_EXPERIMENT_NAME)

    config = dict(AIRFLOW_MODEL_CONFIG)
    config["run_name"] = f"airflow_run_{context['ds']}"

    model, run_id = train_with_config(trainset, config)

    with open(f'{tmp_dir}/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    context['ti'].xcom_push(key='run_id', value=run_id)
    context['ti'].xcom_push(key='model_config', value=AIRFLOW_MODEL_CONFIG)

    print(f"Model trained with {AIRFLOW_MODEL_CONFIG}. Run ID: {run_id}")
    return f"Model trained. Run ID: {run_id}"


def evaluate_model_task(**context):
    """
    Task 4: Evaluate the trained model.

    Metrics and evaluation plots are attached to the MLflow run created by the
    training task, then the metrics are pushed to XCom for the branch decision.
    """
    from pipeline.evaluation import evaluate_model

    tmp_dir = context['ti'].xcom_pull(task_ids=TASK_LOAD_DATA, key='data_path')
    run_id = context['ti'].xcom_pull(task_ids=TASK_TRAIN, key='run_id')

    if not run_id:
        raise AirflowException(
            f"No 'run_id' in XCom from '{TASK_TRAIN}' - training must run first"
        )

    model = _load_pickle(tmp_dir, 'model.pkl')
    testset = _load_pickle(tmp_dir, 'testset.pkl')

    metrics = evaluate_model(model, testset, run_id)

    context['ti'].xcom_push(key='metrics', value=metrics)

    print(f"Evaluation complete. RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f}")
    return f"Evaluation complete. RMSE: {metrics['rmse']:.4f}"


def decide_registration(**context):
    """
    Branch task: Decide whether to register model based on performance.

    Returns ``register_model`` when RMSE beats AIRFLOW_RMSE_THRESHOLD,
    otherwise ``skip_registration``.
    """
    metrics = context['ti'].xcom_pull(task_ids=TASK_EVALUATE, key='metrics')

    if not metrics:
        print("No metrics found in XCom - skipping registration")
        return TASK_SKIP

    rmse = metrics.get('rmse', float('inf'))

    if rmse < AIRFLOW_RMSE_THRESHOLD:
        print(f"RMSE {rmse:.4f} < {AIRFLOW_RMSE_THRESHOLD} -> registering model")
        return TASK_REGISTER

    print(
        f"RMSE {rmse:.4f} >= threshold {AIRFLOW_RMSE_THRESHOLD} "
        f"-> skipping registration"
    )
    return TASK_SKIP


def register_model_task(**context):
    """
    Task 5: Register the best model and promote it to Production.
    """
    from pipeline.registry import register_best_model
    from pipeline.training import setup_mlflow

    setup_mlflow(experiment_name=MLFLOW_EXPERIMENT_NAME)

    result = register_best_model(
        experiment_name=MLFLOW_EXPERIMENT_NAME,
        model_name=REGISTERED_MODEL_NAME,
        metric="rmse",
        stage="Production",
    )

    print(f"Model registered: {result['model_name']} v{result['version']}")
    context['ti'].xcom_push(key='registration', value=result)

    # Returned value is stored as the task's XCom return_value.
    return {
        "model_name": result["model_name"],
        "version": result["version"],
        "stage": result["stage"],
        "run_id": result["run_id"],
    }


def cleanup_task(**context):
    """
    Final task: Cleanup this run's temporary files.

    Runs with ``trigger_rule='none_failed'`` so it executes on both branches.
    """
    tmp_dir = context['ti'].xcom_pull(task_ids=TASK_LOAD_DATA, key='data_path')

    if tmp_dir and os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"Cleaned up: {tmp_dir}")
    else:
        print(f"Nothing to clean up (path: {tmp_dir})")

    return "Cleanup complete"


# =============================================================================
# Task Definitions
# =============================================================================

# Task 1: Load Data
t_load_data = PythonOperator(
    task_id=TASK_LOAD_DATA,
    python_callable=load_data_task,
    doc_md="Load MovieLens 100K and split into train/test.",
    dag=dag,
)

# Task 2: Preprocess Data
t_preprocess = PythonOperator(
    task_id=TASK_PREPROCESS,
    python_callable=preprocess_data_task,
    doc_md="Validate the split and compute data statistics. Fails on bad data.",
    dag=dag,
)

# Task 3: Train Model
t_train = PythonOperator(
    task_id=TASK_TRAIN,
    python_callable=train_model_task,
    doc_md="Train the configured model and log params/artifacts to MLflow.",
    dag=dag,
)

# Task 4: Evaluate Model
t_evaluate = PythonOperator(
    task_id=TASK_EVALUATE,
    python_callable=evaluate_model_task,
    doc_md="Score the model on the test set and log metrics/plots to MLflow.",
    dag=dag,
)

# Task 5: Branch - Decide Registration
t_decide = BranchPythonOperator(
    task_id=TASK_DECIDE,
    python_callable=decide_registration,
    doc_md=f"Quality gate: register only when RMSE < {AIRFLOW_RMSE_THRESHOLD}.",
    dag=dag,
)

# Task 6a: Register Model
t_register = PythonOperator(
    task_id=TASK_REGISTER,
    python_callable=register_model_task,
    doc_md="Register the best run and promote it to the Production stage.",
    dag=dag,
)

# Task 6b: Skip Registration
t_skip = EmptyOperator(
    task_id=TASK_SKIP,
    dag=dag,
)

# Task 7: Cleanup
t_cleanup = PythonOperator(
    task_id=TASK_CLEANUP,
    python_callable=cleanup_task,
    trigger_rule='none_failed',  # Run even if the branch skipped a task
    doc_md="Remove this run's scratch directory.",
    dag=dag,
)


# =============================================================================
# Task Dependencies
# =============================================================================
# load_data -> preprocess -> train -> evaluate -> decide -> [register|skip] -> cleanup
t_load_data >> t_preprocess >> t_train >> t_evaluate >> t_decide
t_decide >> [t_register, t_skip]
[t_register, t_skip] >> t_cleanup
