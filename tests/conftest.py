"""Shared fixtures for testing fraud detection components"""

import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def sample_data():
    """Create a small synthetic dataset for testing"""
    np.random.seed(42)
    n_samples = 1000
    n_features = 9 # To match feature selection

    # Create synthetic features
    X = np.random.randn(n_samples, n_features)

    # Create imbalanced classes (1% fraud)
    y = np.zeros(n_samples)
    fraud_indices = np.random.choice(n_samples, size=init(n_samples * 0.01), replace=False)
    y[fraud_indices] = 1

    # Create DataFrame with feature names
    feature_names = ['V14', 'V17', 'V12', 'V10', 'V3', 'V7', 'V4', 'V16', 'V11']
    df = pd.DataFrame(X, columns=feature_names)
    df['Class'] = y
    
    return df

@pytest.fixture
def data_preparation():
    """Create DataPreparation instance for testing"""
    from src.data.data_preparation import DataPreparation
    return DataPreparation()