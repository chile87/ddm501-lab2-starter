"""
Unit tests for ML Pipeline.

Run tests with:
    pytest tests/ -v
    pytest tests/ -v --cov=pipeline
"""

import pytest
from unittest.mock import MagicMock, patch


class TestDataIngestion:
    """Tests for data ingestion module."""
    
    def test_load_data_returns_dataset(self):
        """Test that load_data returns a dataset object."""
        from pipeline.data_ingestion import load_data
        
        data = load_data('ml-100k')
        assert data is not None
    
    def test_split_data_returns_train_test(self):
        """Test that split_data returns trainset and testset."""
        from pipeline.data_ingestion import load_data, split_data
        
        data = load_data('ml-100k')
        trainset, testset = split_data(data, test_size=0.2)
        
        assert trainset is not None
        assert testset is not None
        assert len(testset) > 0
    
    def test_get_data_stats(self):
        """Test that get_data_stats returns correct statistics."""
        from pipeline.data_ingestion import load_data, get_data_stats
        
        data = load_data('ml-100k')
        stats = get_data_stats(data)
        
        assert 'n_users' in stats
        assert 'n_items' in stats
        assert 'n_ratings' in stats
        assert stats['n_users'] > 0
        assert stats['n_items'] > 0


class TestPreprocessing:
    """Tests for preprocessing module."""
    
    def test_validate_trainset(self):
        """Test trainset validation."""
        from pipeline.data_ingestion import load_and_split
        from pipeline.preprocessing import validate_trainset
        
        trainset, _, _ = load_and_split()
        report = validate_trainset(trainset)
        
        assert 'is_valid' in report
        assert report['is_valid'] == True
    
    def test_get_rating_distribution(self):
        """Test rating distribution calculation."""
        from pipeline.data_ingestion import load_and_split
        from pipeline.preprocessing import get_rating_distribution
        
        trainset, _, _ = load_and_split()
        dist = get_rating_distribution(trainset)
        
        assert 'mean' in dist
        assert 'std' in dist
        assert 1.0 <= dist['mean'] <= 5.0


class TestTraining:
    """Tests for training module."""
    
    def test_list_available_models(self):
        """Test that available models are listed."""
        from pipeline.training import list_available_models
        
        models = list_available_models()
        assert 'svd' in models
        assert 'nmf' in models
        assert 'knn' in models
    
    def test_get_default_params(self):
        """Test getting default parameters."""
        from pipeline.training import get_default_params
        
        params = get_default_params('svd')
        assert 'n_factors' in params
        assert 'n_epochs' in params
    
    # TODO: Add test for train_model after implementation
    # def test_train_model(self):
    #     """Test model training."""
    #     from pipeline.data_ingestion import load_and_split
    #     from pipeline.training import train_model, setup_mlflow
    #     
    #     setup_mlflow()
    #     trainset, _, _ = load_and_split()
    #     model, run_id = train_model(trainset, model_type='svd', n_factors=10, n_epochs=5)
    #     
    #     assert model is not None
    #     assert run_id is not None


class TestEvaluation:
    """Tests for evaluation module."""
    
    def test_create_prediction_distribution_plot(self):
        """Test plot creation."""
        from pipeline.evaluation import create_prediction_distribution_plot
        from surprise import SVD, Dataset
        from surprise.model_selection import train_test_split
        
        # Quick training for test
        data = Dataset.load_builtin('ml-100k')
        trainset, testset = train_test_split(data, test_size=0.1)
        model = SVD(n_factors=10, n_epochs=5)
        model.fit(trainset)
        predictions = model.test(testset[:100])
        
        fig = create_prediction_distribution_plot(predictions)
        assert fig is not None


class TestRegistry:
    """Tests for registry module."""
    
    def test_list_registered_models(self):
        """Test listing registered models."""
        from pipeline.registry import list_registered_models
        
        # This should not raise an error
        models = list_registered_models()
        assert isinstance(models, list)


class TestConfig:
    """Tests for configuration."""
    
    def test_config_values(self):
        """Test that config has required values."""
        from pipeline.config import (
            DATASET_NAME,
            TEST_SIZE,
            DEFAULT_MODEL_TYPE,
            MODEL_CONFIGS,
        )
        
        assert DATASET_NAME == 'ml-100k'
        assert 0 < TEST_SIZE < 1
        assert DEFAULT_MODEL_TYPE in ['svd', 'nmf', 'knn']
        assert 'svd' in MODEL_CONFIGS


# =============================================================================
# Integration Tests
# =============================================================================
class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.mark.slow
    def test_full_pipeline(self):
        """Test running the complete pipeline."""
        # This test is marked as slow and can be skipped in quick test runs
        # pytest tests/ -v -m "not slow"
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
