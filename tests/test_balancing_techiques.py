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
    
    # Create imbalanced classes (1% fraud) as integers
    y = np.zeros(n_samples, dtype=np.int32)  # Explicitly set dtype
    fraud_indices = np.random.choice(n_samples, size=int(n_samples * 0.01), replace=False)
    y[fraud_indices] = 1
    
    # Reshape y to match your format (n_samples, 1)
    y = y.reshape(-1, 1)
    
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

def test_smoteenn_integrity(sample_imbalanced_data):
    """Test SMOTE-ENN maintains data integrity"""
    experiment = SMOTEENNExperiment()
    
    # Process data with SMOTE-ENN
    processed_data = experiment.preprocess_data(sample_imbalanced_data)
    
    # Get class distributions
    resampled_class_counts = np.bincount(processed_data['y_train'].ravel())
    
    # Check both classes have samples
    assert resampled_class_counts[0] > 0
    assert resampled_class_counts[1] > 0
    
    # Check class ratio is within expected range
    class_ratio = resampled_class_counts[0] / resampled_class_counts[1]
    expected_ratio = 1/ExperimentConfig.SMOTEENN.SAMPLING_STRATEGY
    # Allow for larger tolerance since ENN cleaning affects the ratio
    assert abs(class_ratio - expected_ratio) < 0.5  # Increased tolerance to 50%
    
    # Check data integrity
    assert not np.any(np.isnan(processed_data['X_train']))
    assert not np.any(np.isnan(processed_data['y_train']))
    
    # Check validation and test sets unchanged
    assert np.array_equal(processed_data['X_val'], sample_imbalanced_data['X_val'])
    assert np.array_equal(processed_data['y_val'], sample_imbalanced_data['y_val'])
    assert np.array_equal(processed_data['X_test'], sample_imbalanced_data['X_test'])
    assert np.array_equal(processed_data['y_test'], sample_imbalanced_data['y_test'])

def test_class_weight_calculation(sample_imbalanced_data):
    """Test class weights are calculated correctly"""
    experiment = ClassWeightExperiment()
    
    # Process data with class weights
    processed_data = experiment.preprocess_data(sample_imbalanced_data)
    
    # Check class weights are present
    assert 'class_weights' in processed_data
    
    class_weights = processed_data['class_weights']
    
    # Check class weight structure
    assert 0 in class_weights
    assert 1 in class_weights
    
    # Check class weights are proportional to class imbalance
    class_counts = np.bincount(sample_imbalanced_data['y_train'].ravel())
    expected_ratio = class_counts[0] / class_counts[1]
    actual_ratio = class_weights[1] / class_weights[0]
    
    # Allow for some numerical precision differences
    assert abs(actual_ratio - expected_ratio) < 1.0
    
    # Check data is unchanged
    assert np.array_equal(processed_data['X_train'], sample_imbalanced_data['X_train'])
    assert np.array_equal(processed_data['y_train'], sample_imbalanced_data['y_train'])
    assert np.array_equal(processed_data['X_val'], sample_imbalanced_data['X_val'])
    assert np.array_equal(processed_data['y_val'], sample_imbalanced_data['y_val'])
    assert np.array_equal(processed_data['X_test'], sample_imbalanced_data['X_test'])
    assert np.array_equal(processed_data['y_test'], sample_imbalanced_data['y_test'])

def test_metadata_logging(sample_imbalanced_data):
    """Test all techniques log appropriate metadata"""
    experiments = [
        SMOTEExperiment(),
        RandomUndersamplingExperiment(),
        SMOTEENNExperiment(),
        ClassWeightExperiment()
    ]
    
    for experiment in experiments:
        # Process data
        processed_data = experiment.preprocess_data(sample_imbalanced_data)
        
        # Check metadata exists
        assert hasattr(experiment, 'current_data')
        
        # Check technique-specific metadata
        if isinstance(experiment, ClassWeightExperiment):
            metadata = processed_data.get('weight_metadata')
            assert 'calculation_time' in metadata
        else:
            metadata = processed_data.get('resampling_metadata')
            assert 'resampling_time' in metadata or 'processing_time' in metadata
            
        assert metadata is not None
        assert 'peak_memory_usage' in metadata