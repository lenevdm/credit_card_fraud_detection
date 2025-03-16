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
    np.random.seed(42)
    group1 = np.random.normal(10, 2, 100)  # mean=10, sd=2
    group2 = np.random.normal(12, 2, 100)  # mean=12, sd=2
    
    # Expected d = (12-10)/2 = 1.0 (large effect)
    # Note: Cohen's d is negative when group1 has lower mean than group2
    d = cohens_d(group1, group2)
    
    # Allow for wider sampling variation
    assert -1.3 < d < -0.9
    
    # Test with order reversed to get positive d
    d_positive = cohens_d(group2, group1)
    assert 0.9 < d_positive < 1.3
    
    # Test interpretation (should use absolute value)
    interpretation = interpret_cohens_d(d)
    assert interpretation == "large"
    
    # Test with smaller effect
    group3 = np.random.normal(10, 2, 100)
    group4 = np.random.normal(10.5, 2, 100)  # mean difference of 0.5, d should be ~0.25
    small_d = cohens_d(group3, group4)
    assert interpret_cohens_d(small_d) == "small"

def test_paired_t_test(mock_metrics_list):
    """Test paired t-test calculation with controlled data"""
    technique1_metrics, technique2_metrics = mock_metrics_list
    
    # Create a new copy of the metrics to avoid modifying the fixture
    technique1_mod = [m.copy() for m in technique1_metrics]
    technique2_mod = [m.copy() for m in technique2_metrics]
    
    # Modify with variable differences to avoid the special case
    np.random.seed(42)
    for i in range(len(technique1_mod)):
        # Add some noise to the difference to avoid identical differences
        noise = np.random.normal(0, 0.01)  # Small noise
        technique2_mod[i]['precision'] = technique1_mod[i]['precision'] - (0.05 + noise)
        
        noise = np.random.normal(0, 0.01)  # Different noise for recall
        technique2_mod[i]['recall'] = technique1_mod[i]['recall'] + (0.15 + noise)
    
    # Test on precision (expected to be better for technique1)
    precision_test = paired_t_test(
        technique1_mod,
        technique2_mod,
        "precision"
    )
    
    print(f"Debug - Precision test results: {precision_test}")
    
    # We expect technique1 to have higher precision
    assert precision_test['mean_difference'] > 0
    
    # With variable differences, p-value should be significant
    assert precision_test['p_value'] < 0.05
    
    # Test on recall (expected to be better for technique2)
    recall_test = paired_t_test(
        technique1_mod,
        technique2_mod,
        "recall"
    )
    
    print(f"Debug - Recall test results: {recall_test}")
    
    # We expect technique2 to have higher recall
    assert recall_test['mean_difference'] < 0
    
    # P-value should be significant
    assert recall_test['p_value'] < 0.05
    
    # Check confidence intervals make sense
    assert recall_test['ci_lower'] < recall_test['mean_difference'] < recall_test['ci_upper']

def test_multiple_comparison_correction():
    """Test p-value adjustment for multiple comparisons"""
    # Create mock comparison results with know p-values
    comparisons = {
        'precision': {'p_value': 0.01, 'mean_difference': 0.05},
        'recall': {'p_value': 0.02, 'mean_difference': -0.1},
        'f1_score': {'p_value': 0.03, 'mean_difference': 0.02},
        'roc_auc': {'p_value': 0.04, 'mean_difference': 0.01},
        'g_mean': {'p_value': 0.05, 'mean_difference': 0.03}
    }

    # Adjust p-values using BH
    adjusted = adjust_pvalues(comparisons)

    # Check all metrics are preserved
    assert set(adjusted.keys()) == set(comparisons.keys())

    # Check adjusted p-values are present
    for metric in adjusted:
        assert 'p_value_adjusted' in adjusted[metric]

    # BH correct maintain order but increase p-values
    metrics = list(comparisons.keys())
    for i in range(len(metrics) - 1):
        current_metric = metrics[i]
        next_metric = metrics[i+1]

        # Original p-values should be in ascending order for this test
        assert comparisons[current_metric]['p_value'] <= comparisons[next_metric]['p_value']
        
        # Adjusted p-values should maintain the same order
        assert adjusted[current_metric]['p_value_adjusted'] <= adjusted[next_metric]['p_value_adjusted']
        
        # Adjusted p-values should be >= original p-values
        assert adjusted[current_metric]['p_value_adjusted'] >= comparisons[current_metric]['p_value']

def test_compare_techniques(mock_metrics_list):
    """Test the comprehensive technique comparison function"""
    technique1_metrics, technique2_metrics = mock_metrics_list
    
    # Define metrics to compare
    metrics_of_interest = ['precision', 'recall', 'f1_score', 'roc_auc', 'g_mean']
    
    # Compare techniques
    comparison_results = compare_techniques(
        'Technique1',
        technique1_metrics,
        'Technique2',
        technique2_metrics,
        metrics_of_interest
    )
    
    # Check structure of results
    assert 'technique_names' in comparison_results
    assert 'comparisons' in comparison_results
    
    # Check technique names are correctly stored
    assert comparison_results['technique_names']['technique1'] == 'Technique1'
    assert comparison_results['technique_names']['technique2'] == 'Technique2'
    
    # Check all metrics are compared
    assert set(comparison_results['comparisons'].keys()) == set(metrics_of_interest)
    
    # Check each metric comparison has the expected fields
    for metric in metrics_of_interest:
        metric_comparison = comparison_results['comparisons'][metric]
        assert 'mean_difference' in metric_comparison
        assert 'p_value' in metric_comparison
        assert 'p_value_adjusted' in metric_comparison
        assert 'ci_lower' in metric_comparison
        assert 'ci_upper' in metric_comparison
        assert 'cohens_d' in metric_comparison
        assert 'effect_size' in metric_comparison
        assert 'is_significant' in metric_comparison

def test_confidence_interval_calculation():
    """Test confidence interval calculation with known distribution"""
    # Create sample with known distribution
    np.random.seed(42)
    sample = np.random.normal(loc=10, scale=2, size=100)
    
    # Calculate mean and std dev
    mean = np.mean(sample)
    std = np.std(sample, ddof=1)  # Using sample standard deviation
    
    # Calculate 95% confidence interval
    n = len(sample)
    ci = stats.t.interval(
        0.95,  # 95% confidence level
        df=n-1,  # degrees of freedom
        loc=mean,
        scale=std/np.sqrt(n)
    )
    
    # Expected 95% CI for normal distribution with mean=10, std=2, n=100  is approximately mean ± 0.4
    expected_lower = mean - 0.4
    expected_upper = mean + 0.4
    
    # Allow for some random variation
    tolerance = 0.1
    assert abs(ci[0] - expected_lower) < tolerance
    assert abs(ci[1] - expected_upper) < tolerance
    
    # Verify that confidence interval contains true mean (10)
    assert ci[0] < 10 < ci[1]

class MockBaselineExperiment(BaseExperiment):
    """Mock experiment class for testing aggregate metrics calculation"""
    def __init__(self, metrics_list):
        # Skip parent initialization
        self.metrics_list = metrics_list
        self.n_runs = len(metrics_list)
    
    def preprocess_data(self, data):
        """Mock implementation of required abstract method"""
        pass

def test_aggregate_metrics():
    """Test the aggregate metrics calculation in BaseExperiment"""
    # Create mock metrics list
    np.random.seed(42)
    
    metrics_list = []
    for i in range(30):
        metrics_list.append({
            'accuracy': 0.99 + np.random.normal(0, 0.001),
            'precision': 0.8 + np.random.normal(0, 0.01),
            'recall': 0.75 + np.random.normal(0, 0.01),
            'f1_score': 0.77 + np.random.normal(0, 0.01),
            'roc_auc': 0.85 + np.random.normal(0, 0.01),
            'auprc': 0.6 + np.random.normal(0, 0.01),
            'g_mean': 0.85 + np.random.normal(0, 0.01),
            'mcc': 0.7 + np.random.normal(0, 0.01),
            'training_time': 1.0 + np.random.normal(0, 0.1),
            'peak_memory_usage': 10.0 + np.random.normal(0, 1.0),
            'curves': {
                'pr': {'precision': [0.9, 0.8], 'recall': [0.1, 0.2]},
                'roc': {'fpr': [0, 0.1], 'tpr': [0, 0.8]}
            }
        })
    
    # Create mock experiment
    experiment = MockBaselineExperiment(metrics_list)
    
    # Compute aggregate metrics
    agg_metrics = experiment._aggregate_metrics()
    
    # Check that all metrics have mean, std, and confidence intervals
    expected_metrics = [
        'accuracy', 'precision', 'recall', 'f1_score', 
        'roc_auc', 'auprc', 'g_mean', 'mcc',
        'training_time', 'peak_memory_usage'
    ]
    
    for metric in expected_metrics:
        assert f"{metric}_mean" in agg_metrics
        assert f"{metric}_std" in agg_metrics
        assert f"{metric}_ci_lower" in agg_metrics
        assert f"{metric}_ci_upper" in agg_metrics
        
        # Check that confidence interval contains mean
        assert agg_metrics[f"{metric}_ci_lower"] <= agg_metrics[f"{metric}_mean"]
        assert agg_metrics[f"{metric}_mean"] <= agg_metrics[f"{metric}_ci_upper"]
        
        # Calculate mean manually to verify
        expected_mean = np.mean([m[metric] for m in metrics_list])
        assert abs(agg_metrics[f"{metric}_mean"] - expected_mean) < 1e-10
