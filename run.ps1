<#
.SYNOPSIS
    ML Pipeline & MLOps Helper Script for Windows PowerShell (DDM501 Lab 2).

.DESCRIPTION
    Provides shortcut commands for starting MLflow server, running training pipelines,
    executing hyperparameter sweeps, and running unit tests.

.EXAMPLE
    .\run.ps1 mlflow
    .\run.ps1 train
    .\run.ps1 sweep
    .\run.ps1 test
#>

param (
    [Parameter(Position=0)]
    [string]$Command = "help",

    [Parameter(Position=1)]
    [string]$Option = ""
)

# Resolve python and mlflow executables inside venv automatically if available
$VenvPython = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
$VenvMlflow = Join-Path $PSScriptRoot "venv\Scripts\mlflow.exe"

if (Test-Path $VenvPython) {
    $PythonCmd = $VenvPython
} else {
    $PythonCmd = "python"
}

if (Test-Path $VenvMlflow) {
    $MlflowCmd = $VenvMlflow
} else {
    $MlflowCmd = "mlflow"
}

switch ($Command.ToLower()) {
    "mlflow" {
        Write-Host "Starting MLflow Tracking Server on http://localhost:5000 ..." -ForegroundColor Green
        & $MlflowCmd server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
    }
    "train" {
        Write-Host "Running ML Pipeline training..." -ForegroundColor Cyan
        if ($Option) {
            & $PythonCmd -m pipeline.run_pipeline $Option
        } else {
            & $PythonCmd -m pipeline.run_pipeline
        }
    }
    "sweep" {
        Write-Host "Running Hyperparameter Sweep (9 Experiments)..." -ForegroundColor Yellow
        & $PythonCmd -m experiments.run_experiments
    }
    "test" {
        Write-Host "Running Unit Tests..." -ForegroundColor Magenta
        & $PythonCmd -m pytest tests/test_pipeline.py -v
    }
    "clean" {
        Write-Host "Cleaning temporary artifacts and caches..." -ForegroundColor Gray
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue .pytest_cache, __pycache__, pipeline\__pycache__, experiments\__pycache__, tests\__pycache__
        Write-Host "Done." -ForegroundColor Green
    }
    default {
        Write-Host "==========================================================" -ForegroundColor Cyan
        Write-Host "  DDM501 Lab 2 - ML Pipeline Helper Commands (PowerShell) " -ForegroundColor Yellow
        Write-Host "==========================================================" -ForegroundColor Cyan
        Write-Host "  Usage: .\run.ps1 [command]" -ForegroundColor White
        Write-Host ""
        Write-Host "  Commands:" -ForegroundColor White
        Write-Host "    mlflow   : Start MLflow Tracking Server (http://localhost:5000)" -ForegroundColor Green
        Write-Host "    train    : Run single training pipeline pass (SVD default)" -ForegroundColor Green
        Write-Host "    sweep    : Run hyperparameter sweep over 9 experiments" -ForegroundColor Green
        Write-Host "    test     : Run pytest unit test suite" -ForegroundColor Green
        Write-Host "    clean    : Remove temporary cache directories" -ForegroundColor Green
        Write-Host "==========================================================" -ForegroundColor Cyan
    }
}
