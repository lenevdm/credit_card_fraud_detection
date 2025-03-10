""" Statistical analysis utilities for comparing detection techniques"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
from statsmodels.stats.multitest import multipletests 
from src.utils.visualization_utils import plot_metric_curves
from src.utils.visualization_utils import (
    plot_technique_comparison_pr,
    plot_metric_distributions_by_technique,
    plot_performance_radar
)

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Coohen's d effect size

    Args: 
        group1: First group's values
        group2: Second group's values

    Returns"
        float: Cohen's d effect size
    """
    differences = group1 - group2

    mean_diff = np.mean(differences)

    std_diff = np.std(differences, ddof=1) + 1e-10  # Add epsilon to avoid division by zero
    
    return mean_diff / std_diff

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

    # Validation
    if not technique1_metrics or not technique2_metrics:
        raise ValueError("Empty metrics list provided")
    
    if len(technique1_metrics) != len(technique2_metrics):
        raise ValueError(f"Unequal number of runs: {len(technique1_metrics)} vs {len(technique2_metrics)}")
    
    # Extract metric values
    try:
        values1 = [m[metric_name] for m in technique1_metrics]
        values2 = [m[metric_name] for m in technique2_metrics]
    except KeyError:
        raise KeyError(f"Metric '{metric_name}' not found in both techniques' results")
    except ValueError:
        raise ValueError(f"Non-numeric values found for metric '{metric_name}'")


    # Calculate differences
    #differences = np.array(values1) - np.array(values2)
    differences = values1 - values2

    # Check for near-identical values
    if np.std(differences) < 1e-10:
        return {
            't_statistic': 0.0,
            'p_value': 1.0,
            'mean_difference': np.mean(differences),
            'ci_lower': np.mean(differences),
            'ci_upper': np.mean(differences),
            'cohens_d': 0.0,
            'effect_size': 'negligible (identical values)',
            'is_significant': False
        }
        
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
    if not comparisons:
        raise ValueError("Empty comparisons dictionary provided")
    
    # Extract p-values
    metrics = list(comparisons.keys())
    if not metrics:
        raise ValueError("No metrics found in comparisons")

    pvalues = [comparisons[m]['p_value'] for m in metrics]
    if not pvalues:
        raise ValueError("No p-values found in comparisons")
    
    # Add debug prints
    print(f"\nDebug: Adjusting p-values")
    print(f"Debug: Number of metrics: {len(metrics)}")
    print(f"Debug: Number of p-values: {len(pvalues)}")

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

    # Add debug prints
    print(f"\nDebug: Comparing {technique1_name} vs {technique2_name}")
    print(f"Debug: Number of metrics for technique 1: {len(technique1_metrics)}")
    print(f"Debug: Number of metrics for technique 2: {len(technique2_metrics)}")
    print(f"Debug: Metrics of interest: {metrics_of_interest}")

    for metric in metrics_of_interest:
        try:
            comparison = paired_t_test(
                technique1_metrics,
                technique2_metrics,
                metric
            )
            comparisons[metric] = comparison
        except KeyError:
            print(f"Warning: Metric '{metric}' not found in both techniques")
            continue

    # Add debug print
    print(f"Debug: Number of successful comparisons: {len(comparisons)}")
    
    # Only adjust p-values if we have comparisons
    if not comparisons:
        raise ValueError("No valid comparisons were made between techniques")
    
    # Apply multiple comparison correction
    adjusted_comparisons = adjust_pvalues(comparisons)

    # Add visualizations
    metrics_by_technique = {
        technique1_name: technique1_metrics,
        technique2_name: technique2_metrics
    }
    
    comparison_plots = {
        'pr_trade_off': plot_technique_comparison_pr(metrics_by_technique),
        'f1_distribution': plot_metric_distributions_by_technique(metrics_by_technique, 'f1_score'),
        'performance_radar': plot_performance_radar(metrics_by_technique)
    }

    return {
        'technique_names': {
            'technique1': technique1_name,
            'technique2': technique2_name
        },
        'comparisons': adjusted_comparisons,
        'plots': comparison_plots
    }

def format_comparison_results(comparison_results: Dict[str, Any]) -> str:
    """
    Format comparison results for output
    
    Args:
        comparison_results: Results from compare_techniques
        
    Returns:
        str: Formatted results string
    """
    technique1 = comparison_results['technique_names']['technique1']
    technique2 = comparison_results['technique_names']['technique2']
    
    output = [
        f"\nComparison Results: {technique1} vs {technique2}",
        "=" * 50
    ]
    
    for metric, results in comparison_results['comparisons'].items():
        output.extend([
            f"\n{metric.upper()}:",
            f"Mean difference ({technique1} - {technique2}): {results['mean_difference']:.4f}",
            f"95% CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]",
            f"Effect size: {results['effect_size']} (d = {results['cohens_d']:.3f})",
            f"p-value: {results['p_value']:.4f} (adjusted: {results['p_value_adjusted']:.4f})",
            "* Statistically significant (after correction)" if results['is_significant'] else ""
        ])
    
    # Add debug print
    result = "\n".join(output)
    print(f"\nDebug - format_comparison_results output length: {len(result)}")
    return result   

