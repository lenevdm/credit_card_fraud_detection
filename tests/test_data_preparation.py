"""Tests for data preparation pipeline"""

import pytest
import numpy as np
from src.data.data_preparation import DataPreparation

def test_data_preparation_initialization():
    """Test DataPreparation class initializes correctly"""
    prep = DataPreparation()
    assert hasattr(prep, 'primary_features')
    assert len(prep.primary_features) == 9
    assert prep.random_state == prep.config.RANDOM_SEED

def test_load_and_split_proportions(data_preparation, sample_data, tmp_path):
    """Test data splitting maintains correct proportions"""
    # Save sample data to temp file
    data_path = tmp_path / "test_data.csv"
    sample_data.to_csv(data_path, index=False)
    
    # Load and split data
    data = data_preparation.load_and_split_data(data_path)
    
    # Check all expected keys are present
    expected_keys = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test',
                    'class_distribution', 'feature_names', 'scaler']
    assert all(key in data for key in expected_keys)
    
    # Check split proportions
    n_total = len(sample_data)
    
    # First split takes 20% for test
    expected_test_size = int(n_total * data_preparation.config.TEST_SIZE)
    
    # Second split takes 20% of remaining 80% for validation
    remaining_samples = n_total - expected_test_size
    expected_val_size = int(remaining_samples * data_preparation.config.VAL_SIZE)
    
    # Check sizes
    assert len(data['X_test']) == expected_test_size
    assert len(data['X_val']) == expected_val_size
    assert len(data['X_train']) == n_total - expected_test_size - expected_val_size

def test_class_distribution_maintenance(data_preparation, sample_data, tmp_path):
    """Test that class distributions are maintained in splits"""
    # Save sample data to temp file
    data_path = tmp_path / "test_data.csv"
    sample_data.to_csv(data_path, index=False)
    
    # Get original fraud ratio
    original_fraud_ratio = (sample_data['Class'] == 1).sum() / len(sample_data)
    
    # Load and split data
    data = data_preparation.load_and_split_data(data_path)
    
    # Check fraud ratios in splits
    train_fraud_ratio = (data['y_train'] == 1).sum() / len(data['y_train'])
    val_fraud_ratio = (data['y_val'] == 1).sum() / len(data['y_val'])
    test_fraud_ratio = (data['y_test'] == 1).sum() / len(data['y_test'])
    
    # Allow for small variations due to random splitting
    tolerance = 0.02
    assert abs(train_fraud_ratio - original_fraud_ratio) < tolerance
    assert abs(val_fraud_ratio - original_fraud_ratio) < tolerance
    assert abs(test_fraud_ratio - original_fraud_ratio) < tolerance

def test_feature_scaling(data_preparation, sample_data, tmp_path):
    """Test that feature scaling is applied correctly"""
    # Save sample data to temp file
    data_path = tmp_path / "test_data.csv"
    sample_data.to_csv(data_path, index=False)
    
    # Load and split data
    data = data_preparation.load_and_split_data(data_path)
    
    # Check that scaled data has zero mean and unit variance
    tolerance = 0.1
    for X in [data['X_train'], data['X_val'], data['X_test']]:
        assert np.abs(X.mean()) < tolerance
        assert abs(X.std() - 1) < tolerance

def test_no_data_leakage(data_preparation, sample_data, tmp_path):
    """Test that there's no data leakage between splits"""
    # Save sample data to temp file
    data_path = tmp_path / "test_data.csv"
    sample_data.to_csv(data_path, index=False)
    
    # Load and split data
    data = data_preparation.load_and_split_data(data_path)
    
    # Check that there's no overlap between splits
    train_indices = set(map(tuple, data['X_train']))
    val_indices = set(map(tuple, data['X_val']))
    test_indices = set(map(tuple, data['X_test']))
    
    assert len(train_indices.intersection(val_indices)) == 0
    assert len(train_indices.intersection(test_indices)) == 0
    assert len(val_indices.intersection(test_indices)) == 0