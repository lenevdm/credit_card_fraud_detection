"""Tests for evaluation metrics and statistical functions"""

import pytest
import numpy as np
from scipy import stats
import pandas as pd
from typing import Dict, List

from src.utils.statistical_analysis import (
    cohens_d, interpret_cohens_d, paired_t_test, adjust_pvalues, compare_techniques
)
from src.experiments.base_experiment import BaseExperiment


@pytest.fixture
def mock_metrics_list():
    """Create mock metrics for two techniques"""
    np.random.seed(42)
    
    # Create metrics with controlled differences for testing
    technique1_metrics = []
    technique2_metrics = []
    
    for i in range(30):  # 30 runs
        # Base values with controlled differences
        base_precision = 0.8 + np.random.normal(0, 0.01)
        base_recall = 0.7 + np.random.normal(0, 0.01)
        base_f1 = 0.75 + np.random.normal(0, 0.01)
        base_auc = 0.85 + np.random.normal(0, 0.01)
        
        # Technique 1: baseline performance
        technique1_metrics.append({
            'precision': base_precision,
            'recall': base_recall,
            'f1_score': base_f1,
            'roc_auc': base_auc,
            'g_mean': np.sqrt(base_recall * 0.99),
            'training_time': 1.0 + np.random.normal(0, 0.1),
            'peak_memory_usage': 10.0 + np.random.normal(0, 1.0)
        })
        
        # Technique 2: slightly better recall, worse precision
        technique2_metrics.append({
            'precision': base_precision - 0.05,
            'recall': base_recall + 0.15,
            'f1_score': base_f1 + 0.05,  # Slightly better F1
            'roc_auc': base_auc + 0.01,  # Slightly better AUC
            'g_mean': np.sqrt((base_recall + 0.15) * 0.97),
            'training_time': 1.5 + np.random.normal(0, 0.1),  # Longer training time
            'peak_memory_usage': 15.0 + np.random.normal(0, 1.0)  # More memory
        })
    
    return technique1_metrics, technique2_metrics

def test_cohens_d_calculation():
    """Test Cohen's d effect size calculation"""
    # Create two samples with known effect size
    group1 = np.random.normal(10, 2, 100)  # mean=10, sd=2
    group2 = np.random.normal(12, 2, 100)  # mean=12, sd=2
    
    # Expected d = (12-10)/2 = 1.0 (large effect)
    d = cohens_d(group1, group2)
    
    # Allow for some sampling variation
    assert 0.9 < d < 1.1
    
    # Test interpretation
    interpretation = interpret_cohens_d(d)
    assert interpretation == "large"
    
    # Test with smaller effect
    group3 = np.random.normal(10, 2, 100)
    group4 = np.random.normal(10.5, 2, 100)  # mean difference of 0.5, d should be ~0.25
    small_d = cohens_d(group3, group4)
    assert interpret_cohens_d(small_d) == "small"