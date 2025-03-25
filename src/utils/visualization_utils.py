"""Utility functions for visualizing model performance metrics"""

from typing import List, Dict, Tuple, Optional, Any
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
        
        plt.plot(recall, precision, color='#cd34b5', lw=2)
        plt.axhline(y=no_skill_baseline, color='#ffb14e', linestyle='--', label='No Skill')
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
        plt.plot(fpr, tpr, color='#9d02d7', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='#ffb14e', linestyle='--', label='Random')
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
        sns.heatmap(cm, annot=True, fmt='d', cmap='BuPu',
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
        sns.heatmap(correlations, annot=True, cmap='plasma', center=0, fmt='.2f', square=True)
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
        fig, ax = plt.subplots(figsize=(10, 6))

        techniques = list(metrics_by_technique.keys())
        x = np.arange(len(techniques))
        width = 0.35

        precisions = [np.mean([m['precision'] for m in metrics_by_technique[t]]) for t in techniques]
        recalls = [np.mean([m['recall'] for m in metrics_by_technique[t]]) for t in techniques]
        
        prec_err = [np.std([m['precision'] for m in metrics_by_technique[t]]) for t in techniques]
        recall_err = [np.std([m['recall'] for m in metrics_by_technique[t]]) for t in techniques]

        # Define custom colors
        precision_color = '#9d02d7'  # Purple
        recall_color = '#ffb14e'     # Orange

        # Plot bars with custom colors
        ax.bar(
            x - width/2,
            precisions,
            width,
            label='Precision',
            yerr=prec_err,
            capsize=5,
            color=precision_color
        )
        ax.bar(
            x + width/2,
            recalls,
            width,
            label='Recall',
            yerr=recall_err,
            capsize=5,
            color=recall_color
        )

        ax.set_ylabel('Score')
        ax.set_title('Precision-Recall Trade-off by Technique')
        ax.set_xticks(x)
        ax.set_xticklabels(techniques)
        ax.legend()

        plt.tight_layout()
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
        
        # Create violin plot and capture the parts
        vp = plt.violinplot(data, showmeans=True)

        # Set custom color for each violin body
        custom_color = '#ea5f94'
        for body in vp['bodies']:
            body.set_facecolor(custom_color)
            body.set_edgecolor(custom_color)
            body.set_alpha(0.8)  # Optional transparency

        # Set colors for mean lines, min/max lines if desired
        vp['cbars'].set_color(custom_color)
        vp['cmins'].set_color(custom_color)
        vp['cmaxes'].set_color(custom_color)
        vp['cmeans'].set_color(custom_color)

        plt.xticks(range(1, len(labels) + 1), labels)
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} Distribution by Technique')

        plt.tight_layout()
        return plt.gcf()

    except Exception as e:
        plt.close('all')
        raise e

    
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List

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
        
        # Define colors for techniques
        colors = ['#17208D', '#ea5f94']  # Blue and Pink
        
        # Plot each technique with assigned color
        for i, (technique, means) in enumerate(technique_means.items()):
            color = colors[i % len(colors)]  # Cycle through colors if more than 2 techniques
            means = np.concatenate((means, [means[0]]))  # complete the circle
            angles_plot = np.concatenate((angles, [angles[0]]))  # complete the circle
            ax.plot(angles_plot, means, 'o-', linewidth=2, label=technique, color=color)
            ax.fill(angles_plot, means, alpha=0.25, color=color)
        
        ax.set_xticks(angles)
        ax.set_xticklabels(metrics)
        ax.set_title('Performance Metrics Comparison')
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.tight_layout()
        return fig
    except Exception as e:
        plt.close('all')
        raise e
    
def plot_multiple_techniques_radar(metrics_by_technique: Dict[str, List[Dict[str, float]]]) -> plt.Figure:
    """
    Create radar plot comparing key metrics across all techniques
    
    Args:
        metrics_by_technique: Dictionary with technique names as keys and lists of metric dictionaries as values
        
    Returns:
        Matplotlib figure object
    """
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
        
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        
        # Use custom color palette
        colors = [
            '#17208D',   # deep blue
            '#9d02d7',  # purple
            '#cd34b5',  # magenta
            '#ea5f94',  # pink
            '#fa8775',  # salmon
            '#ffb14e',  # orange
            '#ffd700',  # gold
        ]
        
        # Plot each technique with assigned color
        for i, (technique, means) in enumerate(technique_means.items()):
            color = colors[i % len(colors)]  # Cycle through colors
            means = np.concatenate((means, [means[0]]))  # complete the circle
            angles_plot = np.concatenate((angles, [angles[0]]))  # complete the circle
            ax.plot(angles_plot, means, 'o-', linewidth=2, label=technique, color=color)
            ax.fill(angles_plot, means, alpha=0.1, color=color)
        
        # Set labels and ticks
        ax.set_xticks(angles)
        ax.set_xticklabels(metrics)
        
        # Add labels in bold and slightly larger font
        # for label, angle in zip(metrics, angles):
            #  # Adjust label position for better alignment
            #  # 1.3 is a factor to push labels slightly outward from the circle
         #     x = 1.3 * np.cos(angle)
         #     y = 1.3 * np.sin(angle)
         #  #     ax.text(angle, 1.3, label, 
          #           ha='center' if 0 <= angle < np.pi else 'right' if angle == 0 else 'left',
          #           va='center', 
           #          fontweight='bold', 
           #          fontsize=12)
            
        # Set y-limits slightly beyond the data range for better visualization
        max_value = max([max(means) for means in technique_means.values()])
        ax.set_ylim(0, max_value * 1.1)
        
        # Add grid lines
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set background color for aesthetic appeal (very light)
        #ax.set_facecolor('#FADDD8')
        
        # Add title
        ax.set_title('Performance Comparison Across All Techniques', fontsize=15, fontweight='bold')
        
        # Add legend with better positioning
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        plt.close('all')
        raise e
    
def plot_fp_fn_comparison(metrics_by_technique: Dict[str, List[Dict[str, float]]]) -> plt.Figure:
    """
    Create grouped bar chart comparing false positives and false negatives across techniques
    
    Args:
        metrics_by_technique: Dictionary with technique names as keys and lists of metric dictionaries as values
        
    Returns:
        Matplotlib figure object
    """
    try:
        # Calculate mean FP and FN for each technique
        fp_means = {}
        fn_means = {}
        tp_means = {}
        
        for technique, metrics_list in metrics_by_technique.items():
            fp_values = [m.get('false_positives', 0) for m in metrics_list]
            fn_values = [m.get('false_negatives', 0) for m in metrics_list]
            tp_values = [m.get('true_positives', 0) for m in metrics_list]
            
            fp_means[technique] = np.mean(fp_values)
            fn_means[technique] = np.mean(fn_values)
            tp_means[technique] = np.mean(tp_values)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set up x positions
        techniques = list(fp_means.keys())
        x = np.arange(len(techniques))
        width = 0.35
        
        # Use custom colors from the palette
        fp_color = '#fa8775'  # salmon
        fn_color = '#ea5f94'  # pink
        tp_color = '#17208D'  # deep blue
        
        # Create bars
        fp_bars = ax.bar(x - width, list(fp_means.values()), width, label='False Positives', color=fp_color)
        fn_bars = ax.bar(x, list(fn_means.values()), width, label='False Negatives', color=fn_color)
        tp_bars = ax.bar(x + width, list(tp_means.values()), width, label='True Positives', color=tp_color)
        
        # Add labels and title
        ax.set_xlabel('Technique', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('False Positives vs False Negatives vs True Positives by Technique', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(techniques, rotation=45, ha='right')
        
        # Add value labels on top of bars
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontweight='bold')
        
        add_labels(fp_bars)
        add_labels(fn_bars)
        add_labels(tp_bars)
        
        # Add legend
        ax.legend()
        
        # Add grid for readability
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Set background color
        #ax.set_facecolor('#FADDD8')
        
        # Calculate total error rate and add as text
        # for i, technique in enumerate(techniques):
        #    total_errors = fp_means[technique] + fn_means[technique]
        #    ax.text(i, max(fp_means[technique], fn_means[technique]) * 1.1, 
        #           f'Total: {total_errors:.1f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        plt.close('all')
        raise e
       
def plot_all_metrics_comparison(metrics_by_technique: Dict[str, List[Dict[str, float]]]) -> plt.Figure:
    """
    Create a grid of bar charts comparing all metrics across techniques
    
    Args:
        metrics_by_technique: Dictionary with technique names as keys and lists of metric dictionaries as values
        
    Returns:
        Matplotlib figure with all metric comparisons
    """
    try:
        # Metrics to plot
        metrics_of_interest = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 
                              'auprc', 'g_mean', 'mcc']
        
        # Calculate grid layout
        n_metrics = len(metrics_of_interest)
        n_cols = 3  # 3 columns in the grid
        n_rows = (n_metrics + n_cols - 1) // n_cols  # Calculate needed rows
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        axes = axes.flatten()  # Flatten to easily iterate
        
        # Custom color palette
        colors = [
            '#17208D',   # deep blue
            '#9d02d7',   # purple
            '#cd34b5',   # magenta
            '#ea5f94',   # pink
            '#fa8775',   # salmon
            '#ffb14e',   # orange
            '#ffd700'    # gold
        ]
        
        # Map techniques to consistent colors
        techniques = list(metrics_by_technique.keys())
        technique_colors = {technique: colors[i % len(colors)] for i, technique in enumerate(techniques)}
        
        # For debugging: check first metrics entry to understand structure
        if techniques and metrics_by_technique[techniques[0]]:
            sample_metric = metrics_by_technique[techniques[0]][0]
            print(f"Debug - Sample metric keys: {list(sample_metric.keys())}")
        
        # Create a subplot for each metric
        for idx, metric in enumerate(metrics_of_interest):
            ax = axes[idx]
            print(f"\nProcessing metric: {metric} for subplot {idx}")
            
            # Calculate mean and std for each technique
            means = {}
            std_devs = {}
            
            for technique, metrics_list in metrics_by_technique.items():
                # Extract the raw values for this specific metric
                raw_values = []
                for m in metrics_list:
                    if metric in m:
                        raw_values.append(float(m[metric]))
                
                if raw_values:
                    means[technique] = np.mean(raw_values)
                    std_devs[technique] = np.std(raw_values)
                    print(f"  {technique}: Found {len(raw_values)} values, mean={means[technique]:.4f}, std={std_devs[technique]:.4f}")
                else:
                    print(f"  {technique}: No values found for {metric}")
                    means[technique] = 0
                    std_devs[technique] = 0
            
            # Set up x positions
            x = np.arange(len(techniques))
            
            # Create bars with error bars
            bars = ax.bar(x, list(means.values()), 
                         yerr=list(std_devs.values()),
                         capsize=5, 
                         color=[technique_colors[t] for t in techniques])
            
            # Add labels and title
            ax.set_xlabel('Technique', fontsize=10, fontweight='bold')
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10, fontweight='bold')
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([t.capitalize() for t in techniques], rotation=45, ha='right', fontsize=8)
            
            # Add value labels on top of bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
            
            # Add grid for readability
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Set y-axis limits slightly beyond data range
            max_value = max(means.values()) + max(std_devs.values()) if means.values() else 1
            ax.set_ylim(0, max_value * 1.1)
        
        # Remove any unused subplots
        for idx in range(len(metrics_of_interest), len(axes)):
            fig.delaxes(axes[idx])
        
        # Add a single legend for all subplots at the bottom of the figure
        handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors[:len(techniques)]]
        fig.legend(handles, [t.capitalize() for t in techniques], 
                  loc='lower center', ncol=len(techniques), bbox_to_anchor=(0.5, 0), 
                  fontsize=12)
            
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)  # Make room for the legend
            
        return fig
    except Exception as e:
        plt.close('all')
        print(f"Error in plot_all_metrics_comparison: {e}")
        import traceback
        print(traceback.format_exc())
        raise e