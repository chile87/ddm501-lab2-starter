"""
Data Preprocessing Stage for ML Pipeline.

This module handles:
- Data cleaning
- Feature engineering (if needed)
- Data transformation

For the Surprise library recommendation system, 
most preprocessing is handled internally.
This file is provided as a template for more complex scenarios.

This file is PROVIDED - no TODOs required.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_trainset(trainset: Any) -> Dict[str, Any]:
    """
    Validate training set and return validation report.
    
    Args:
        trainset: Surprise trainset object
        
    Returns:
        Dictionary with validation results
    """
    validation_report = {
        "is_valid": True,
        "n_users": trainset.n_users,
        "n_items": trainset.n_items,
        "n_ratings": trainset.n_ratings,
        "issues": [],
    }
    
    # Check for minimum data requirements
    if trainset.n_users < 10:
        validation_report["is_valid"] = False
        validation_report["issues"].append("Too few users (< 10)")
    
    if trainset.n_items < 10:
        validation_report["is_valid"] = False
        validation_report["issues"].append("Too few items (< 10)")
    
    if trainset.n_ratings < 100:
        validation_report["is_valid"] = False
        validation_report["issues"].append("Too few ratings (< 100)")
    
    # Check rating scale
    min_rating, max_rating = trainset.rating_scale
    if min_rating < 0:
        validation_report["issues"].append(f"Negative ratings found (min: {min_rating})")
    
    logger.info(f"Validation report: {validation_report}")
    return validation_report


def validate_testset(testset: List) -> Dict[str, Any]:
    """
    Validate test set.
    
    Args:
        testset: List of (user_id, item_id, rating) tuples
        
    Returns:
        Dictionary with validation results
    """
    validation_report = {
        "is_valid": True,
        "n_ratings": len(testset),
        "issues": [],
    }
    
    if len(testset) < 10:
        validation_report["is_valid"] = False
        validation_report["issues"].append("Too few test ratings (< 10)")
    
    # Check for missing values
    for uid, iid, rating in testset[:100]:  # Sample check
        if rating is None:
            validation_report["is_valid"] = False
            validation_report["issues"].append("Missing ratings found")
            break
    
    logger.info(f"Test set validation: {validation_report}")
    return validation_report


def get_rating_distribution(trainset: Any) -> Dict[str, Any]:
    """
    Calculate rating distribution statistics.
    
    Args:
        trainset: Surprise trainset object
        
    Returns:
        Dictionary with distribution statistics
    """
    ratings = []
    for uid in trainset.all_users():
        for iid, rating in trainset.ur[uid]:
            ratings.append(rating)
    
    ratings = np.array(ratings)
    
    distribution = {
        "mean": float(np.mean(ratings)),
        "std": float(np.std(ratings)),
        "min": float(np.min(ratings)),
        "max": float(np.max(ratings)),
        "median": float(np.median(ratings)),
        "q25": float(np.percentile(ratings, 25)),
        "q75": float(np.percentile(ratings, 75)),
    }
    
    logger.info(f"Rating distribution: mean={distribution['mean']:.2f}, std={distribution['std']:.2f}")
    return distribution


def get_user_activity_stats(trainset: Any) -> Dict[str, Any]:
    """
    Calculate user activity statistics.
    
    Args:
        trainset: Surprise trainset object
        
    Returns:
        Dictionary with user activity stats
    """
    user_ratings_counts = [len(trainset.ur[uid]) for uid in trainset.all_users()]
    user_ratings_counts = np.array(user_ratings_counts)
    
    stats = {
        "mean_ratings_per_user": float(np.mean(user_ratings_counts)),
        "median_ratings_per_user": float(np.median(user_ratings_counts)),
        "min_ratings_per_user": int(np.min(user_ratings_counts)),
        "max_ratings_per_user": int(np.max(user_ratings_counts)),
        "users_with_few_ratings": int(np.sum(user_ratings_counts < 5)),
    }
    
    logger.info(f"User activity stats: mean={stats['mean_ratings_per_user']:.1f} ratings/user")
    return stats


def get_item_popularity_stats(trainset: Any) -> Dict[str, Any]:
    """
    Calculate item popularity statistics.
    
    Args:
        trainset: Surprise trainset object
        
    Returns:
        Dictionary with item popularity stats
    """
    item_ratings_counts = [len(trainset.ir[iid]) for iid in trainset.all_items()]
    item_ratings_counts = np.array(item_ratings_counts)
    
    stats = {
        "mean_ratings_per_item": float(np.mean(item_ratings_counts)),
        "median_ratings_per_item": float(np.median(item_ratings_counts)),
        "min_ratings_per_item": int(np.min(item_ratings_counts)),
        "max_ratings_per_item": int(np.max(item_ratings_counts)),
        "items_with_few_ratings": int(np.sum(item_ratings_counts < 5)),
    }
    
    logger.info(f"Item popularity stats: mean={stats['mean_ratings_per_item']:.1f} ratings/item")
    return stats


def preprocess_data(trainset: Any, testset: List) -> Dict[str, Any]:
    """
    Run all preprocessing steps and return comprehensive report.
    
    Args:
        trainset: Surprise trainset object
        testset: Test set as list of tuples
        
    Returns:
        Dictionary with all preprocessing results
    """
    logger.info("Starting data preprocessing...")
    
    report = {
        "trainset_validation": validate_trainset(trainset),
        "testset_validation": validate_testset(testset),
        "rating_distribution": get_rating_distribution(trainset),
        "user_activity": get_user_activity_stats(trainset),
        "item_popularity": get_item_popularity_stats(trainset),
    }
    
    # Overall status
    report["preprocessing_successful"] = (
        report["trainset_validation"]["is_valid"] and 
        report["testset_validation"]["is_valid"]
    )
    
    logger.info(f"Preprocessing completed. Success: {report['preprocessing_successful']}")
    return report


# =============================================================================
# Main execution for testing
# =============================================================================
if __name__ == "__main__":
    from pipeline.data_ingestion import load_and_split
    
    print("Testing Preprocessing Module")
    print("=" * 50)
    
    trainset, testset, _ = load_and_split()
    report = preprocess_data(trainset, testset)
    
    print(f"\nPreprocessing Report:")
    print(f"  - Successful: {report['preprocessing_successful']}")
    print(f"  - Mean rating: {report['rating_distribution']['mean']:.2f}")
    print(f"  - Ratings per user: {report['user_activity']['mean_ratings_per_user']:.1f}")
    print(f"  - Ratings per item: {report['item_popularity']['mean_ratings_per_item']:.1f}")
