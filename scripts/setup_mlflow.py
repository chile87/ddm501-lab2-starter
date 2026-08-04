"""
MLflow Setup Script.

This script helps set up MLflow for the project:
1. Verifies MLflow installation
2. Creates experiment
3. Tests tracking functionality

Usage:
    python scripts/setup_mlflow.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_mlflow_installation():
    """Check if MLflow is installed."""
    try:
        import mlflow
        print(f"✓ MLflow installed: version {mlflow.__version__}")
        return True
    except ImportError:
        print("✗ MLflow not installed. Run: pip install mlflow")
        return False


def setup_tracking_uri(uri: str = "http://localhost:5000"):
    """Set up MLflow tracking URI."""
    import mlflow
    
    mlflow.set_tracking_uri(uri)
    print(f"✓ Tracking URI set to: {uri}")
    
    # Test connection
    try:
        mlflow.search_experiments()
        print("✓ Connection to MLflow server successful")
        return True
    except Exception as e:
        print(f"✗ Cannot connect to MLflow server: {e}")
        print("  Make sure MLflow server is running:")
        print("  mlflow ui --host 0.0.0.0 --port 5000")
        return False


def create_experiment(name: str = "movie-rating-prediction"):
    """Create or get experiment."""
    import mlflow
    
    experiment = mlflow.get_experiment_by_name(name)
    
    if experiment is None:
        experiment_id = mlflow.create_experiment(name)
        print(f"✓ Created experiment: {name} (ID: {experiment_id})")
    else:
        print(f"✓ Experiment exists: {name} (ID: {experiment.experiment_id})")
    
    mlflow.set_experiment(name)
    return name


def test_logging():
    """Test MLflow logging functionality."""
    import mlflow
    
    print("\nTesting MLflow logging...")
    
    with mlflow.start_run(run_name="test_run"):
        # Log parameters
        mlflow.log_param("test_param", "value")
        print("  ✓ Parameter logging works")
        
        # Log metrics
        mlflow.log_metric("test_metric", 0.95)
        print("  ✓ Metric logging works")
        
        # Log artifact (create temp file)
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test artifact")
            temp_path = f.name
        
        mlflow.log_artifact(temp_path)
        os.unlink(temp_path)
        print("  ✓ Artifact logging works")
        
        run_id = mlflow.active_run().info.run_id
    
    print(f"\n✓ Test run completed: {run_id}")
    return run_id


def main():
    """Main setup function."""
    print("=" * 60)
    print("MLflow Setup for DDM501 Lab 2")
    print("=" * 60)
    
    # Step 1: Check installation
    print("\n[Step 1] Checking MLflow installation...")
    if not check_mlflow_installation():
        sys.exit(1)
    
    # Step 2: Setup tracking URI
    print("\n[Step 2] Setting up tracking URI...")
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    
    # Try local file-based tracking if server not available
    if not setup_tracking_uri(uri):
        print("\n  Falling back to local file-based tracking...")
        import mlflow
        mlflow.set_tracking_uri("file:./mlruns")
        print("  ✓ Using local tracking: ./mlruns")
    
    # Step 3: Create experiment
    print("\n[Step 3] Creating experiment...")
    create_experiment()
    
    # Step 4: Test logging
    print("\n[Step 4] Testing logging functionality...")
    test_logging()
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start MLflow UI: mlflow ui --port 5000")
    print("2. Open browser: http://localhost:5000")
    print("3. Run experiments: python -m experiments.run_experiments")


if __name__ == "__main__":
    main()
