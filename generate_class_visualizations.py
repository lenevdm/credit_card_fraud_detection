# generate_class_visualizations.py
import os
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
import pandas as pd

def plot_class_distribution(class_counts: Dict[int, int], title: str = "Class Distribution") -> plt.Figure:
    """
    Create a bar chart showing class distribution
    
    Args:
        class_counts: Dictionary with class labels as keys and counts as values
        title: Title for the plot
        
    Returns:
        Matplotlib figure object
    """
    plt.figure(figsize=(8, 6))
    
    # Extract classes and counts
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    # Calculate percentages
    total = sum(counts)
    percentages = [count/total*100 for count in counts]
    
    # Create bar chart
    bars = plt.bar(classes, counts, color=['#3498db', '#e74c3c'])
    
    # Add count and percentage labels on top of bars
    for i, (count, percentage) in enumerate(zip(counts, percentages)):
        plt.text(i, count + (max(counts) * 0.01), f"{count:,}", 
                 ha='center', va='bottom', fontweight='bold')
        plt.text(i, count/2, f"{percentage:.2f}%", 
                 ha='center', va='center', color='white', fontweight='bold')
    
    # Add labels and title
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(title)
    plt.xticks(classes, ['Non-Fraud (0)', 'Fraud (1)'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    return plt.gcf()

def plot_class_distributions_comparison(
    distributions: Dict[str, Dict[int, int]]
) -> plt.Figure:
    """
    Create a comparison of class distributions across techniques
    
    Args:
        distributions: Dictionary with technique names as keys and class count dicts as values
        
    Returns:
        Matplotlib figure object
    """
    techniques = list(distributions.keys())
    n_techniques = len(techniques)
    
    # Calculate class ratios for each technique
    ratios = {}
    for technique, counts in distributions.items():
        if 1 in counts and 0 in counts and counts[0] > 0:  # Ensure we don't divide by zero
            ratios[technique] = counts[0] / counts[1]
        else:
            ratios[technique] = float('inf')  # Handle case where class 1 has zero samples
    
    # Create subplots: one row of pie charts, and a bar chart for ratios
    fig, axes = plt.subplots(2, n_techniques, figsize=(n_techniques*4, 8))
    
    # Create pie chart for each technique
    for i, technique in enumerate(techniques):
        counts = distributions[technique]
        labels = ['Non-Fraud', 'Fraud']
        sizes = [counts.get(0, 0), counts.get(1, 0)]
        
        # Calculate percentages
        total = sum(sizes)
        percentages = [size/total*100 for size in sizes]
        
        # Create labels with percentages
        labels = [f'{label}\n{percentage:.1f}%' for label, percentage in zip(labels, percentages)]
        
        # Plot pie chart
        axes[0, i].pie(sizes, labels=labels, autopct='', startangle=90, colors=['#3498db', '#e74c3c'])
        axes[0, i].set_title(f'{technique}')
    
    # Create bar chart of class ratios
    technique_names = list(ratios.keys())
    ratio_values = list(ratios.values())
    
    # Plot horizontal bar chart across the bottom row
    bars = axes[1, 0].barh(technique_names, ratio_values, color='#2ecc71')
    axes[1, 0].set_title('Class Ratio (Non-Fraud:Fraud)')
    axes[1, 0].set_xlabel('Ratio')
    
    # Add ratio labels
    for i, ratio in enumerate(ratio_values):
        if ratio == float('inf'):
            label = "∞"
        else:
            label = f"{ratio:.2f}:1"
        axes[1, 0].text(ratio + max(ratio_values) * 0.02, i, label, va='center')
    
    # Hide empty subplots in the ratio row
    for i in range(1, n_techniques):
        axes[1, i].axis('off')
    
    plt.tight_layout()
    return fig

def plot_before_after_comparison(
    technique_name: str,
    before_counts: Dict[int, int],
    after_counts: Dict[int, int]
) -> plt.Figure:
    """
    Create a before/after comparison for a specific technique
    
    Args:
        technique_name: Name of the balancing technique
        before_counts: Dictionary with class counts before applying technique
        after_counts: Dictionary with class counts after applying technique
        
    Returns:
        Matplotlib figure object
    """
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # For both subplots
    for ax, counts, title in zip([ax1, ax2], 
                              [before_counts, after_counts],
                              ['Before', 'After']):
        # Extract classes and counts
        classes = list(counts.keys())
        count_values = list(counts.values())
        
        # Calculate percentages
        total = sum(count_values)
        percentages = [count/total*100 for count in count_values]
        
        # Create bar chart
        bars = ax.bar(classes, count_values, color=['#3498db', '#e74c3c'])
        
        # Add count and percentage labels
        for i, (count, percentage) in enumerate(zip(count_values, percentages)):
            ax.text(i, count + (max(count_values) * 0.01), f"{count:,}", 
                     ha='center', va='bottom', fontweight='bold')
            ax.text(i, count/2, f"{percentage:.2f}%", 
                     ha='center', va='center', color='white', fontweight='bold')
        
        # Add labels
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title(f'{title} {technique_name}')
        ax.set_xticks(classes)
        ax.set_xticklabels(['Non-Fraud (0)', 'Fraud (1)'])
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a title for the entire figure
    fig.suptitle(f'Class Distribution: {technique_name}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust to make room for the suptitle
    
    return fig

def get_class_distributions():
    """
    Extract class distributions from each experiment type
    
    Returns:
        Dictionary with technique distributions
    """
    # Define distribution values from your existing experiments
    distributions = {
        "Original": {
            0: 181961,  # Non-fraudulent
            1: 315      # Fraudulent
        },
        "SMOTE": {
            0: 181961,  # Non-fraudulent (unchanged)
            1: 181961   # Fraudulent (matched to non-fraudulent)
        },
        "Random Undersampling": {
            0: 315,     # Non-fraudulent (reduced to match fraudulent)
            1: 315      # Fraudulent (unchanged)
        },
        "SMOTE-ENN": {
            0: 181961 * 0.9,  # Approximate after cleaning
            1: 181961 * 0.9   # Approximate after cleaning
        },
        "Class Weight": {
            0: 181961,  # Non-fraudulent (unchanged)
            1: 315      # Fraudulent (unchanged but weighted)
        }
    }
    
    return distributions

def main():
    """Generate and save class distribution visualizations"""
    # Get class distributions
    distributions = get_class_distributions()
    
    # Create output directory
    os.makedirs("class_visualizations", exist_ok=True)
    
    # Initialize MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    experiment_name = "class_distribution_visualizations"
    
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="class_visualizations"):
        # Original distribution
        print("Generating original distribution visualization...")
        fig1 = plot_class_distribution(distributions["Original"], "Original Class Distribution")
        fig1.savefig("class_visualizations/original_distribution.png")
        mlflow.log_artifact("class_visualizations/original_distribution.png")
        plt.close(fig1)
        
        # Individual technique distributions
        for technique, dist in distributions.items():
            if technique != "Original":
                print(f"Generating {technique} distribution visualization...")
                fig = plot_class_distribution(dist, f"Class Distribution: {technique}")
                fig.savefig(f"class_visualizations/{technique.lower().replace(' ', '_')}_distribution.png")
                mlflow.log_artifact(f"class_visualizations/{technique.lower().replace(' ', '_')}_distribution.png")
                plt.close(fig)
                
                # Before/After comparison
                print(f"Generating {technique} before/after comparison...")
                fig_comp = plot_before_after_comparison(technique, distributions["Original"], dist)
                fig_comp.savefig(f"class_visualizations/{technique.lower().replace(' ', '_')}_comparison.png")
                mlflow.log_artifact(f"class_visualizations/{technique.lower().replace(' ', '_')}_comparison.png")
                plt.close(fig_comp)
        
        # All techniques comparison
        print("Generating all techniques comparison...")
        fig_all = plot_class_distributions_comparison(distributions)
        fig_all.savefig("class_visualizations/all_techniques_comparison.png")
        mlflow.log_artifact("class_visualizations/all_techniques_comparison.png")
        plt.close(fig_all)
        
        print("All visualizations generated and logged to MLflow.")

if __name__ == "__main__":
    main()