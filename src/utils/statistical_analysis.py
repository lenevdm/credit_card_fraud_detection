""" Statistical analysis utilities for comparing detection techniques"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from statsmodels.stats.multitest import multipletests 

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Coohen's d effect size

    Args: 
        group1: First group's values
        group2: Second group's values

    Returns"
        float: Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(goup2, ddof=1)

    # Pooled standard deviation
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    return (np.mean(group1) - np.mean(group2)) / pooled_se

def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d effect size

    Args:
        d: Cohen's d value

    Returns:
        str: Interpretation of effect size
    """
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def paired_t_test(
        technique1_metrics: List[Dict[str, float]],
        technique2_metrics: List[Dict[str, float]],
        metric_name: str,
        alpha: float = 0.05
) -> Dict[str, float]:
    """
    Perform paired t-test between two techniques' performance metrics.

    Args:
        technique1_metrics: List of metric dictionaries from first technique
        technique2_metrics: List of metric dictionaries from second technique
        metric_name: Name of metric to compare (e.g., 'f1_score', 'auprc')
        alpha: Significance level (default: 0.05)

    Returns:
        Dictionary with:
        - t_statistic: t-test statistic
        - p_value: p-value from t-test
        - mean_difference: mean difference between techniques
        - ci_lower: lower bound of confidence interval
        - ci_upper: upper bound of confidence interval
        - is_significant: boolean indicating if difference is significant
    """

    # Extract metric values
    values1 = [m[metric_name] for m in technique1_metrics]
    values2 = [m[metric_name] for m in technique2_metrics]


    # Verify equal lengths
    if len(values1) != len(values2):
        raise ValueError("Both techniques must have the same number of runs")


    # Calculate differences
    differences = np.array(values1) - np.array(values2)

    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(values1, values2)

    # Calculate mean difference and confidence interval
    mean_diff = np.mean(differences)
    ci = stats.t.interval(
        1 - alpha,
        len(differences) - 1,
        loc=mean_diff,
        scale=stats.sem(differences)
    )


def compare_techniques(
        technique1_name: str,
        technique1_metrics: List[Dict[str, float]],
        technique2_name: str,
        technique2_metrics: List[Dict[str, float]],
        metrics_of_interest: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Compare two techniques across multiple metrics

    Args:
        technique1_name: Name of first technique
        technique1_metrics: List of metric dictionaries from technique 1
        technique2_name: Name of second technique
        technique2_metrics: List of metric dictionaries from technique 2
        metrics_of_interest: List of metrics to compare

    Returns:
        Dictionary of statistical comparisons for each metric
    """

    comparisons = {}

    for metric in metrics_of_interest:
        try:
            comparison = paired_t_test(
                technique1_metrics,
                technique2_metrics,
                metric
            )
            comparison[metric] = comparison
        except KeyError:
            print(f"Warning: Metric '{metric}' not found in both techniques")
            continue

    return comparisons
    

