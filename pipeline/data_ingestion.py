"""
Data Ingestion Stage for ML Pipeline.

This module handles:
- Loading data from sources
- Splitting data into train/test sets
- Basic data validation

This file is PROVIDED - no TODOs required.
"""

import logging
from typing import Tuple, Any

from surprise import Dataset
from surprise.model_selection import train_test_split

from pipeline.config import DATASET_NAME, TEST_SIZE, RANDOM_STATE

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(dataset_name: str = DATASET_NAME) -> Any:
    """
    Load dataset from surprise built-in datasets.
    
    Args:
        dataset_name: Name of the dataset to load (default: 'ml-100k')
        
    Returns:
        Surprise Dataset object
        
    Supported datasets:
        - 'ml-100k': MovieLens 100K (100,000 ratings)
        - 'ml-1m': MovieLens 1M (1,000,000 ratings)
        - 'jester': Jester jokes dataset
    """
    logger.info(f"Loading dataset: {dataset_name}")
    
    try:
        data = Dataset.load_builtin(dataset_name)
        logger.info(f"Dataset '{dataset_name}' loaded successfully")
        return data
    except Exception as e:
        logger.error(f"Failed to load dataset '{dataset_name}': {e}")
        raise


def split_data(
    data: Any, 
    test_size: float = TEST_SIZE, 
    random_state: int = RANDOM_STATE
) -> Tuple[Any, Any]:
    """
    Split data into training and test sets.
    
    Args:
        data: Surprise Dataset object
        test_size: Proportion of data for testing (default: 0.2)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (trainset, testset)
    """
    logger.info(f"Splitting data with test_size={test_size}, random_state={random_state}")
    
    trainset, testset = train_test_split(
        data, 
        test_size=test_size, 
        random_state=random_state
    )
    
    logger.info(f"Training set size: {trainset.n_ratings} ratings")
    logger.info(f"Test set size: {len(testset)} ratings")
    
    return trainset, testset


def get_data_stats(data: Any) -> dict:
    """
    Get basic statistics about the dataset.
    
    Args:
        data: Surprise Dataset object
        
    Returns:
        Dictionary with dataset statistics
    """
    # Build full trainset to get statistics
    trainset = data.build_full_trainset()
    
    stats = {
        "n_users": trainset.n_users,
        "n_items": trainset.n_items,
        "n_ratings": trainset.n_ratings,
        "rating_scale": (trainset.rating_scale[0], trainset.rating_scale[1]),
        "global_mean": trainset.global_mean,
    }
    
    logger.info(f"Dataset statistics: {stats}")
    return stats


def load_and_split(
    dataset_name: str = DATASET_NAME,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[Any, Any, dict]:
    """
    Convenience function to load and split data in one call.
    
    Args:
        dataset_name: Name of the dataset to load
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (trainset, testset, stats)
    """
    data = load_data(dataset_name)
    stats = get_data_stats(data)
    trainset, testset = split_data(data, test_size, random_state)
    
    return trainset, testset, stats


# =============================================================================
# Main execution for testing
# =============================================================================
if __name__ == "__main__":
    # Test the data ingestion module
    print("Testing Data Ingestion Module")
    print("=" * 50)
    
    trainset, testset, stats = load_and_split()
    
    print(f"\nDataset Statistics:")
    print(f"  - Users: {stats['n_users']}")
    print(f"  - Items: {stats['n_items']}")
    print(f"  - Ratings: {stats['n_ratings']}")
    print(f"  - Rating scale: {stats['rating_scale']}")
    print(f"  - Global mean: {stats['global_mean']:.2f}")
    
    print(f"\nData split:")
    print(f"  - Training: {trainset.n_ratings} ratings")
    print(f"  - Test: {len(testset)} ratings")
