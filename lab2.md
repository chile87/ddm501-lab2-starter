**LAB 2**

**ML PIPELINE & EXPERIMENT TRACKING**

_Building Reproducible ML Pipelines with MLflow and Airflow_

| Course        | DDM501                          |
| ------------- | ------------------------------- |
| Weight        | 15%                             |
| Format        | Team Lab (3-4 members per team) |
| Prerequisites | Lab 1 completed                 |

# 1\. OVERVIEW

## 1.1. Introduction

In this lab, you will transform the movie rating prediction system from Lab 1 into a production-ready ML pipeline with experiment tracking and workflow orchestration. This is a critical step in MLOps - moving from ad-hoc model training to systematic.

You will learn to track experiments systematically using MLflow, version your models, and automate the entire training workflow using Apache Airflow.

## 1.2. Scenario: Scaling the ML System

Your movie rating prediction API from Lab 1 is now in production. The Data Science team wants to:

- Experiment with different model architectures (SVD, NMF, KNN)
- Track all experiments to compare results systematically
- Automate weekly model retraining with new data
- Version models

Your task is to build the MLOps infrastructure to support these requirements.

# 2\. BACKGROUND KNOWLEDGE

## 2.1. ML Pipeline Architecture

A typical ML pipeline consists of sequential stages, each with specific responsibilities:

| **Stage**           | **Description**                                     |
| ------------------- | --------------------------------------------------- |
| Data Ingestion      | Load raw data from sources (databases, files, APIs) |
| Data Validation     | Check data quality, schema, and statistics          |
| Data Preprocessing  | Clean, transform, and normalize data                |
| Feature Engineering | Create and select features for model training       |
| Model Training      | Train model with hyperparameter tuning              |
| Model Evaluation    | Evaluate model performance on test data             |
| Model Registry      | Version and register model for deployment           |

## 2.2. MLflow Overview

MLflow is an open-source platform for managing the ML lifecycle. It has four main components:

- MLflow Tracking: Log parameters, metrics, and artifacts during experiments
- MLflow Projects: Package ML code in a reusable, reproducible format
- MLflow Models: Deploy models to various serving platforms
- MLflow Registry: Centralized model store with versioning and stage management

_Key MLflow Concepts:_

- Experiment: Collection of runs for a specific task
- Run: Single execution of training code
- Parameters: Input configuration (hyperparameters)
- Metrics: Output measurements (RMSE, MAE, accuracy)
- Artifacts: Output files (models, plots, data)

## 2.3. Apache Airflow Overview

Apache Airflow is a platform to programmatically author, schedule, and monitor workflows. _Key concepts:_

- DAG (Directed Acyclic Graph): Collection of tasks with dependencies
- Task: Single unit of work (Python function, Bash command)
- Operator: Template for tasks (PythonOperator, BashOperator)
- Schedule: Cron expression defining when DAG runs

# 3\. HANDS-ON GUIDE

## Task 1: Setup MLflow Environment

1.1. Install MLflow

1.2. Start MLflow Tracking Server

Access MLflow UI at: <http://localhost:5000>

1.3. Configure MLflow in Python

## Task 2: Build Modular ML Pipeline

2.1. Pipeline Structure

Create modular pipeline with separate stages

2.2. Data Ingestion Stage

2.3. Training Stage with MLflow

2.4. Evaluation Stage

## Task 3: Experiment Tracking with MLflow

3.1. Run Multiple Experiments

3.2. Compare Experiments in MLflow UI

After running experiments, use MLflow UI to:

- Compare runs side-by-side
- Sort by metrics (RMSE, MAE)
- View parameter vs metric charts
- Download artifacts (models, plots)

  3.3. Model Registry

## Task 4: Airflow Pipeline Orchestration

4.1. Install and Setup Airflow

4.2. Create Training DAG

4.3. DAG Task Functions

# 4\. STARTER CODE TEMPLATE

Unzip starter file: _unzip ddm501-lab2-starter.zip_

Files to complete:

| **File**                       | **TODO**                                        |
| ------------------------------ | ----------------------------------------------- |
| pipeline/training.py           | Implement train_model() with MLflow logging     |
| pipeline/evaluation.py         | Implement evaluate_model() with metrics logging |
| pipeline/registry.py           | Implement register_best_model()                 |
| dags/ml_training_dag.py        | Create complete Airflow DAG with all tasks      |
| experiments/run_experiments.py | Run hyperparameter tuning experiments           |
| docker-compose.yml             | Add MLflow and Airflow services                 |

# 5\. DELIVERABLES & GRADING

## 5.1. Deliverables

Submit GitHub repository link containing:

- Complete ML Pipeline: Modular, reusable pipeline code
- MLflow Tracking Setup: Configured experiment tracking with logged runs
- Airflow DAG: Working DAG definition for training pipeline
- Experiment Report: Comparison of at least 5 experiments with analysis
- Documentation: README with setup and usage instructions

## 5.2. Grading Rubric

| **Criteria**        | **Weight** | **Description**                                                                                                                            |
| ------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Pipeline Quality    | 35%        | Modular structure (10%)<br><br>Reproducible execution (10%)<br><br>Error handling (8%)<br><br>Code quality (7%)                            |
| Experiment Tracking | 25%        | MLflow setup correct (8%)<br><br>Parameters logged (5%)<br><br>Metrics logged (5%)<br><br>Artifacts logged (4%)<br><br>Model registry (3%) |
| Airflow Automation  | 20%        | DAG structure correct (8%)<br><br>Tasks execute properly (7%)<br><br>Schedule configured (5%)                                              |
| Documentation       | 20%        | Experiment report (10%)<br><br>README complete (5%)<br><br>Code documentation (5%)                                                         |

## 5.3. Submission

Deadline: 1 week after the lab session

Format: GitHub repository link

Required: Screenshots of MLflow UI showing experiments
