"""Tests for class balancing techniques"""

import pytest
import numpy as np
import pandas as pd
from src.experiments.smote_experiment import SMOTEExperiment
from src.experiments.random_undersampling_experiment import RandomUndersamplingExperiment
from src.experiments.smoteenn_experiment import SMOTEENNExperiment
from src.experiments.class_weight_experiment import ClassWeightExperiment
from config.experiment_config import ExperimentConfig

@pytest.fixture
def sample_imbalanced_data():
    """Create sample imbalanced dataset for testing"""
    np.random.seed(42)
    n_samples = 1000
    n_features = 9

    # Create synthetic features
    X = np.random.randn(n_samples, n_features)

    # Create imbalanced classes (1% fraud)
    y = np.zeros((n_samples, 1))
    fraud_indices = np.random.choice(n_samples, size=int(n_samples * 0.01), replace=False)
    y[fraud_indices] = 1
    
    # Create data dictionary matching your format
    return {
        'X_train': X,
        'y_train': y,
        'X_val': X[:100],  # Small validation set
        'y_val': y[:100],
        'X_test': X[:100],  # Small test set
        'y_test': y[:100]
    }

def test_smote_generation(sample_imbalanced_data):
    """Test SMOTE generates correct number of synthetic samples"""
    experiment = SMOTEExperiment()
    
    # Process data with SMOTE
    processed_data = experiment.preprocess_data(sample_imbalanced_data)
    
    # Get class distributions
    original_class_counts = np.bincount(sample_imbalanced_data['y_train'].ravel())
    resampled_class_counts = np.bincount(processed_data['y_train'].ravel())
    
    # Check majority class unchanged
    assert resampled_class_counts[0] == original_class_counts[0]
    
    # Check minority class increased to match majority
    assert resampled_class_counts[1] == original_class_counts[0]
    
    # Check data integrity
    assert not np.any(np.isnan(processed_data['X_train']))
    assert not np.any(np.isnan(processed_data['y_train']))
    
    # Check validation and test sets unchanged
    assert np.array_equal(processed_data['X_val'], sample_imbalanced_data['X_val'])
    assert np.array_equal(processed_data['y_val'], sample_imbalanced_data['y_val'])
    assert np.array_equal(processed_data['X_test'], sample_imbalanced_data['X_test'])
    assert np.array_equal(processed_data['y_test'], sample_imbalanced_data['y_test'])

def test_random_undersampling(sample_imbalanced_data):
    """Test random undersampling reduces majority class correctly"""
    experiment = RandomUndersamplingExperiment()
    
    # Process data with random undersampling
    processed_data = experiment.preprocess_data(sample_imbalanced_data)
    
    # Get class distributions
    original_class_counts = np.bincount(sample_imbalanced_data['y_train'].ravel())
    resampled_class_counts = np.bincount(processed_data['y_train'].ravel())
    
    # Check minority class unchanged
    assert resampled_class_counts[1] == original_class_counts[1]
    
    # Check majority class reduced to specified ratio
    expected_majority = int(original_class_counts[1] / ExperimentConfig.RandomUndersampling.SAMPLING_STRATEGY)
    assert resampled_class_counts[0] == expected_majority
    
    # Check data integrity
    assert not np.any(np.isnan(processed_data['X_train']))
    assert not np.any(np.isnan(processed_data['y_train']))
    
    # Check validation and test sets unchanged
    assert np.array_equal(processed_data['X_val'], sample_imbalanced_data['X_val'])
    assert np.array_equal(processed_data['y_val'], sample_imbalanced_data['y_val'])
    assert np.array_equal(processed_data['X_test'], sample_imbalanced_data['X_test'])
    assert np.array_equal(processed_data['y_test'], sample_imbalanced_data['y_test'])

