"""
Airflow DAG for ML Training Pipeline.

This DAG orchestrates the movie rating prediction training pipeline:
1. Load Data
2. Preprocess Data
3. Train Model
4. Evaluate Model
5. Register Model (conditional)

TODO: Complete the DAG definition and task functions.

Usage:
    Copy this file to your Airflow dags/ folder
    Access Airflow UI at http://localhost:8080
"""

from datetime import datetime, timedelta
import pickle
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator

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
}

# =============================================================================
# DAG Definition
# =============================================================================
# TODO: Complete the DAG definition
#
# Requirements:
# - DAG ID: 'movie_rating_training'
# - Description: 'ML Training Pipeline for Movie Rating Prediction'
# - Schedule: Weekly (@weekly) or use cron expression '0 0 * * 0' for every Sunday
# - Start date: January 1, 2024
# - Catchup: False (don't run for past dates)
# - Tags: ['ml', 'training', 'movie-rating']

dag = DAG(
    'movie_rating_training',
    default_args=default_args,
    description='ML Training Pipeline for Movie Rating Prediction',
    schedule_interval='@weekly',  # Or '0 0 * * 0' for every Sunday
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'training', 'movie-rating'],
)


# =============================================================================
# Task Functions
# =============================================================================

def load_data_task(**context):
    """
    Task 1: Load and prepare data.
    
    This function:
    1. Loads the dataset
    2. Splits into train/test
    3. Saves to temporary location
    4. Pushes metadata via XCom
    """
    from pipeline.data_ingestion import load_and_split
    
    print("Loading data...")
    trainset, testset, stats = load_and_split()
    
    # Save data to temporary files
    tmp_dir = '/tmp/airflow_ml_pipeline'
    os.makedirs(tmp_dir, exist_ok=True)
    
    with open(f'{tmp_dir}/trainset.pkl', 'wb') as f:
        pickle.dump(trainset, f)
    with open(f'{tmp_dir}/testset.pkl', 'wb') as f:
        pickle.dump(testset, f)
    
    # Push stats via XCom
    context['ti'].xcom_push(key='data_stats', value=stats)
    context['ti'].xcom_push(key='data_path', value=tmp_dir)
    
    print(f"Data loaded: {stats['n_ratings']} ratings")
    return "Data loaded successfully"


# =============================================================================
# TODO 1: Implement preprocess_data_task
# =============================================================================
def preprocess_data_task(**context):
    """
    Task 2: Preprocess and validate data.
    
    TODO: Implement this function that:
    1. Retrieves data path from XCom
    2. Loads the saved trainset and testset
    3. Runs preprocessing using preprocess_data()
    4. Pushes preprocessing report via XCom
    
    Hints:
    - Use context['ti'].xcom_pull(key='data_path') to get the path
    - Load pickle files with pickle.load()
    - Push results with context['ti'].xcom_push()
    """
    # TODO: Implement this function
    #
    # from pipeline.preprocessing import preprocess_data
    # 
    # # Get data path from previous task
    # tmp_dir = context['ti'].xcom_pull(key='data_path')
    # 
    # # Load data
    # with open(f'{tmp_dir}/trainset.pkl', 'rb') as f:
    #     trainset = pickle.load(f)
    # with open(f'{tmp_dir}/testset.pkl', 'rb') as f:
    #     testset = pickle.load(f)
    # 
    # # Preprocess
    # report = preprocess_data(trainset, testset)
    # 
    # # Push report
    # context['ti'].xcom_push(key='preprocess_report', value=report)
    # 
    # return "Preprocessing complete"
    
    pass  # Remove this and implement


# =============================================================================
# TODO 2: Implement train_model_task
# =============================================================================
def train_model_task(**context):
    """
    Task 3: Train the model with MLflow tracking.
    
    TODO: Implement this function that:
    1. Retrieves trainset from temporary storage
    2. Sets up MLflow
    3. Trains the model using train_model()
    4. Pushes run_id via XCom for evaluation task
    
    Configuration:
    - model_type: 'svd'
    - n_factors: 100
    - n_epochs: 20
    """
    # TODO: Implement this function
    #
    # from pipeline.training import train_model, setup_mlflow
    # 
    # # Get data path
    # tmp_dir = context['ti'].xcom_pull(key='data_path')
    # 
    # # Load trainset
    # with open(f'{tmp_dir}/trainset.pkl', 'rb') as f:
    #     trainset = pickle.load(f)
    # 
    # # Setup MLflow
    # setup_mlflow()
    # 
    # # Train model
    # model, run_id = train_model(
    #     trainset,
    #     model_type='svd',
    #     run_name=f"airflow_run_{context['ds']}",
    #     n_factors=100,
    #     n_epochs=20
    # )
    # 
    # # Save model for evaluation
    # with open(f'{tmp_dir}/model.pkl', 'wb') as f:
    #     pickle.dump(model, f)
    # 
    # # Push run_id
    # context['ti'].xcom_push(key='run_id', value=run_id)
    # 
    # return f"Model trained. Run ID: {run_id}"
    
    pass  # Remove this and implement


# =============================================================================
# TODO 3: Implement evaluate_model_task
# =============================================================================
def evaluate_model_task(**context):
    """
    Task 4: Evaluate the trained model.
    
    TODO: Implement this function that:
    1. Retrieves model and testset from storage
    2. Retrieves run_id from XCom
    3. Evaluates using evaluate_model()
    4. Pushes metrics via XCom
    """
    # TODO: Implement this function
    #
    # from pipeline.evaluation import evaluate_model
    # 
    # # Get data path and run_id
    # tmp_dir = context['ti'].xcom_pull(key='data_path')
    # run_id = context['ti'].xcom_pull(key='run_id')
    # 
    # # Load model and testset
    # with open(f'{tmp_dir}/model.pkl', 'rb') as f:
    #     model = pickle.load(f)
    # with open(f'{tmp_dir}/testset.pkl', 'rb') as f:
    #     testset = pickle.load(f)
    # 
    # # Evaluate
    # metrics = evaluate_model(model, testset, run_id)
    # 
    # # Push metrics
    # context['ti'].xcom_push(key='metrics', value=metrics)
    # 
    # return f"Evaluation complete. RMSE: {metrics['rmse']:.4f}"
    
    pass  # Remove this and implement


def decide_registration(**context):
    """
    Branch task: Decide whether to register model based on performance.
    
    Returns 'register_model' if RMSE < 1.0, otherwise 'skip_registration'
    """
    metrics = context['ti'].xcom_pull(key='metrics')
    
    if metrics and metrics.get('rmse', float('inf')) < 1.0:
        return 'register_model'
    return 'skip_registration'


def register_model_task(**context):
    """
    Task 5: Register the best model.
    """
    from pipeline.registry import register_best_model
    
    result = register_best_model()
    print(f"Model registered: {result['model_name']} v{result['version']}")
    return result


def cleanup_task(**context):
    """
    Final task: Cleanup temporary files.
    """
    import shutil
    
    tmp_dir = context['ti'].xcom_pull(key='data_path')
    if tmp_dir and os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
        print(f"Cleaned up: {tmp_dir}")
    
    return "Cleanup complete"


# =============================================================================
# Task Definitions
# =============================================================================

# Task 1: Load Data
t_load_data = PythonOperator(
    task_id='load_data',
    python_callable=load_data_task,
    dag=dag,
)

# Task 2: Preprocess Data
t_preprocess = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data_task,
    dag=dag,
)

# Task 3: Train Model
t_train = PythonOperator(
    task_id='train_model',
    python_callable=train_model_task,
    dag=dag,
)

# Task 4: Evaluate Model
t_evaluate = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model_task,
    dag=dag,
)

# Task 5: Branch - Decide Registration
t_decide = BranchPythonOperator(
    task_id='decide_registration',
    python_callable=decide_registration,
    dag=dag,
)

# Task 6a: Register Model
t_register = PythonOperator(
    task_id='register_model',
    python_callable=register_model_task,
    dag=dag,
)

# Task 6b: Skip Registration
t_skip = DummyOperator(
    task_id='skip_registration',
    dag=dag,
)

# Task 7: Cleanup
t_cleanup = PythonOperator(
    task_id='cleanup',
    python_callable=cleanup_task,
    trigger_rule='none_failed',  # Run even if branch skipped
    dag=dag,
)


# =============================================================================
# TODO 4: Define Task Dependencies
# =============================================================================
# Define the task execution order using >> operator
#
# The flow should be:
# load_data -> preprocess -> train -> evaluate -> decide -> [register OR skip] -> cleanup
#
# Hint:
# t_load_data >> t_preprocess >> t_train >> t_evaluate >> t_decide
# t_decide >> [t_register, t_skip]
# [t_register, t_skip] >> t_cleanup

# TODO: Define the dependencies
t_load_data >> t_preprocess >> t_train >> t_evaluate >> t_decide
t_decide >> [t_register, t_skip]
[t_register, t_skip] >> t_cleanup
