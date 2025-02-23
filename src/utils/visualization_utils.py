"""Utility functions for visualizing model performance metrics"""

from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_pr_curve(precision: np.ndarray, recall: np.ndarray, auprc: float) -> plt.Figure:
    """Plot Precision-Recall curve"""
    try:
        plt.figure(figsize=(10, 6))
        
        # Calculate the no-skill baseline (proportion of positive class)
        no_skill_baseline = len([x for x in recall if x > 0]) / len(recall)
        
        plt.plot(recall, precision, color='blue', lw=2)
        plt.axhline(y=no_skill_baseline, color='r', linestyle='--', label='No Skill')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve (AUPRC = {auprc:.3f})')
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    except Exception as e:
        plt.close('all')
        raise e


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float) -> plt.Figure:
    """Plot ROC curve"""
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='r', linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(True)
        return plt.gcf()
    except Exception as e:
        plt.close('all')
        raise e

def plot_confusion_matrix(tn: int, fp: int, fn: int, tp: int) -> plt.Figure:
    """Plot confusion matrix heatmap"""
    try:
        plt.figure(figsize=(8, 6))
        cm = np.array([[tn, fp], [fn, tp]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        return plt.gcf()
    except Exception as e:
        plt.close('all')
        raise e

def plot_performance_resources(metrics_list: List[Dict]) -> plt.Figure:
    """Plot relationship between model performance and resource usage across runs"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract metrics
        memory_usage = [m['peak_memory_usage'] for m in metrics_list]
        training_time = [m['training_time'] for m in metrics_list]
        f1_scores = [m['f1_score'] for m in metrics_list]
        g_means = [m['g_mean'] for m in metrics_list]
        
        # Memory usage vs performance
        ax1.scatter(memory_usage, f1_scores, label='F1 Score', alpha=0.6)
        ax1.scatter(memory_usage, g_means, label='G-Mean', alpha=0.6)
        ax1.set_xlabel('Peak Memory Usage (MB)')
        ax1.set_ylabel('Performance Metric')
        ax1.set_title('Performance vs Memory Usage')
        ax1.legend()
        ax1.grid(True)
        
        # Training time vs performance
        ax2.scatter(training_time, f1_scores, label='F1 Score', alpha=0.6)
        ax2.scatter(training_time, g_means, label='G-Mean', alpha=0.6)
        ax2.set_xlabel('Training Time (seconds)')
        ax2.set_ylabel('Performance Metric')
        ax2.set_title('Performance vs Training Time')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        plt.close('all')
        raise e

def plot_metric_correlations(metrics_list: List[Dict]) -> plt.Figure:
    """Create a correlation heatmap of all metrics"""
    try:
        metrics_df = pd.DataFrame(metrics_list)
        numerical_metrics = metrics_df.select_dtypes(include=[np.number]).columns
        correlations = metrics_df[numerical_metrics].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True)
        plt.title('Metric Correlations')
        return plt.gcf()
    except Exception as e:
        plt.close('all')
        raise e

def plot_metric_distributions(metrics_list: List[Dict]) -> plt.Figure:
    """Plot distribution of performance metrics across runs"""
    try:
        metrics_of_interest = ['g_mean', 'mcc', 'f1_score', 'auprc']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics_of_interest):
            values = [m[metric] for m in metrics_list]
            sns.histplot(values, kde=True, ax=axes[idx])
            axes[idx].set_title(f'{metric.replace("_", " ").title()} Distribution')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Count')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        plt.close('all')
        raise e

def plot_resource_timeline(metrics_list: List[Dict]) -> plt.Figure:
    """Plot timeline of resource usage during training"""
    try:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Create x-axis values
        times = np.arange(len(metrics_list))
        memory_usage = [m['peak_memory_usage'] for m in metrics_list]
        
        # Plot memory usage
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Memory Usage (MB)', color='tab:blue')
        ax1.plot(times, memory_usage, color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        # Create second y-axis for training time
        ax2 = ax1.twinx()
        training_times = [m['training_time'] for m in metrics_list]
        ax2.set_ylabel('Training Time (s)', color='tab:orange')
        ax2.plot(times, training_times, color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        
        plt.title('Resource Usage Timeline')
        return fig
    except Exception as e:
        plt.close('all')
        raise e

def plot_metric_curves(metrics_dict: Dict, metrics_list: Optional[List[Dict]] = None) -> Tuple[plt.Figure, plt.Figure, plt.Figure, Dict[str, plt.Figure]]:
    """Plot all relevant metric curves
    
    Args:
        metrics_dict: Dictionary containing current run metrics
        metrics_list: Optional list of metrics from all runs
        
    Returns:
        Tuple containing:
        - PR curve figure
        - ROC curve figure
        - Confusion matrix figure
        - Dictionary of additional figures
    """
    # Initialize additional_figs dictionary
    additional_figs = {}

    # Ensure all existing figures are closed before starting
    plt.close('all')

    try:
        # Debugging info
        print("\nPlot Metric Curves Debug Info:")
        print("-" * 50)
        print(f"Metrics list status: {'Provided' if metrics_list else 'Not provided'}")
        if metrics_list:
            print(f"Metrics list length: {len(metrics_list)}")
            if len(metrics_list) > 0:
                required_metrics = ['peak_memory_usage', 'training_time', 'f1_score', 'g_mean', 
                                'mcc', 'auprc']
                available_metrics = list(metrics_list[0].keys())
                print("Required metrics:", required_metrics)
                print("Available metrics:", available_metrics)
                missing_metrics = [m for m in required_metrics if m not in available_metrics]
                if missing_metrics:
                    print("Missing required metrics:", missing_metrics)
        print("-" * 50)
        
        # Create PR curve
        pr_fig = plot_pr_curve(
            metrics_dict['curves']['pr']['precision'],
            metrics_dict['curves']['pr']['recall'],
            metrics_dict['auprc']
        )
        
        # Create ROC curve
        roc_fig = plot_roc_curve(
            metrics_dict['curves']['roc']['fpr'],
            metrics_dict['curves']['roc']['tpr'],
            metrics_dict['roc_auc']
        )
        
        # Create confusion matrix
        cm_fig = plot_confusion_matrix(
            metrics_dict['true_negatives'],
            metrics_dict['false_positives'],
            metrics_dict['false_negatives'],
            metrics_dict['true_positives']
        )
        
        # Only create additional plots if metrics_list is provided and not empty
        if metrics_list and len(metrics_list) > 0:
            try:
                print("\nAttempting to create additional plots...")
                additional_figs['performance_resources'] = plot_performance_resources(metrics_list)
                print("Created performance_resources plot")
                additional_figs['metric_correlations'] = plot_metric_correlations(metrics_list)
                print("Created metric_correlations plot")
                additional_figs['metric_distributions'] = plot_metric_distributions(metrics_list)
                print("Created metric_distributions plot")
                additional_figs['resource_timeline'] = plot_resource_timeline(metrics_list)
                print("Created resource_timeline plot")
            except Exception as e:
                print(f"\nError creating additional plots:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print(f"Error occurred in: {e.__traceback__.tb_frame.f_code.co_name}")
                # Clean up any figures that were created before the error
                for fig in additional_figs.values():
                    plt.close(fig)
                additional_figs = {}

        return pr_fig, roc_fig, cm_fig, additional_figs
    
    except Exception as e:
        # Clean up all figures if an error occurs
        plt.close('all')
        raise e
    
def plot_technique_comparison_pr(metrics_by_technique: Dict[str, List[Dict[str, float]]]) -> plt.Figure:
    """Plot precision and recall comparison across techniques"""
    try:
        fig, ax = plt.subplots(figsize=(10,6))

        techniques = list(metrics_by_technique.keys())
        x = np.arange(len(techniques))
        width = 0.35

        precisions = [np.mean([m['precision'] for m in metrics_by_technique[t]]) for t in techniques]
        recalls = [np.mean([m['recall'] for m in metrics_by_technique[t]]) for t in techniques]
        
        prec_err = [np.std([m['precision'] for m in metrics_by_technique[t]]) for t in techniques]
        recall_err = [np.std([m['recall'] for m in metrics_by_technique[t]]) for t in techniques]
        
        ax.bar(x - width/2, precisions, width, label='Precision', yerr=prec_err, capsize=5)
        ax.bar(x + width/2, recalls, width, label='Recall', yerr=recall_err, capsize=5)
        
        ax.set_ylabel('Score')
        ax.set_title('Precision-Recall Trade-off by Technique')
        ax.set_xticks(x)
        ax.set_xticklabels(techniques)
        ax.legend()
        
        return fig
    except Exception as e:
        plt.close('all')
        raise e
    
def plot_metric_distributions_by_technique(
    metrics_by_technique: Dict[str, List[Dict[str, float]]],
    metric_name: str
) -> plt.Figure:
    """Create violin plots showing metric distribution across techniques"""
    try:
        plt.figure(figsize=(10, 6))
        
        data = []
        labels = []
        for technique, metrics in metrics_by_technique.items():
            values = [m[metric_name] for m in metrics]
            data.append(values)
            labels.append(technique)
        
        plt.violinplot(data, showmeans=True)
        plt.xticks(range(1, len(labels) + 1), labels)
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} Distribution by Technique')
        
        return plt.gcf()
    except Exception as e:
        plt.close('all')
        raise e
    
def plot_performance_radar(metrics_by_technique: Dict[str, List[Dict[str, float]]]) -> plt.Figure:
    """Create radar plot comparing key metrics across techniques"""
    try:
        metrics = ['precision', 'recall', 'f1_score', 'g_mean', 'roc_auc']
        num_metrics = len(metrics)
        
        # Compute means for each metric and technique
        technique_means = {}
        for technique, technique_metrics in metrics_by_technique.items():
            technique_means[technique] = [
                np.mean([m[metric] for m in technique_metrics])
                for metric in metrics
            ]
        
        # Create radar plot
        angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False)
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for technique, means in technique_means.items():
            means = np.concatenate((means, [means[0]]))  # complete the circle
            angles_plot = np.concatenate((angles, [angles[0]]))  # complete the circle
            ax.plot(angles_plot, means, 'o-', linewidth=2, label=technique)
            ax.fill(angles_plot, means, alpha=0.25)
        
        ax.set_xticks(angles)
        ax.set_xticklabels(metrics)
        ax.set_title('Performance Metrics Comparison')
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        return fig
    except Exception as e:
        plt.close('all')
        raise e
    
    