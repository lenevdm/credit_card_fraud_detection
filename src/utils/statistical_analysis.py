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
    Perform paired t-test between two techniques' performance metrics. Enhanced with effect size.

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
    #differences = np.array(values1) - np.array(values2)
    differences = values1 - values2

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

    # Calculate effect size
    d = cohens_d(values1, values2)
    effect_size_interp = interpret_cohens_d(d)

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'mean_difference': mean_diff,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'cohens_d': d,
        'effect_size': effect_size_interp,
        'is_significant': p_value < alpha
    }


def adjust_pvalues(comparisons: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Apply multiple comparison correction (Benjamini-Hochberg)

    Args:
        comparisons: Dictionary of comparison results

    Returns: 
        Dictionary with adjusted p-values
    """
    # Extract p-values
    metrics = [m for m in comparisons.keys()]
    pvalues = [comparisons[m]['p_value'] for m in metrics]

    # Apply correction
    rejected, pvals_corrected, _, _ = multipletests(
        pvalues,
        alpha=0.05,
        method='fdr_bh' 
    )

    # Update results
    adjusted_comparisons = comparisons.copy()
    for i, metric in enumerate(metrics):
        adjusted_comparisons[metric]['p_value_adjusted'] = pvals_corrected[i]
        adjusted_comparisons[metric]['is_significant'] = rejected[i]

    return adjusted_comparisons

def compare_techniques(
        technique1_name: str,
        technique1_metrics: List[Dict[str, float]],
        technique2_name: str,
        technique2_metrics: List[Dict[str, float]],
        metrics_of_interest: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Compare two techniques across multiple metrics. Enhanced technique comparison with 
    effect sizes and multiple comparison correction

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
    

