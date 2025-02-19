""" Statistical analysis utilities for comparing detection techniques"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd

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


def compare_techniques():
    return

