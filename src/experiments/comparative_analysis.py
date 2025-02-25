"""Script for running and comparing different fraud detection techniques"""

import os
from typing import Dict, Any, List
import mlflow
import pandas as pd
import matplotlib.pyplot as plt 
from src.experiments.base_runs_final import BaselineExperimentFinal
from src.utils.statistical_analysis import format_comparison_results
from config.experiment_config import ExperimentConfig
from src.experiments.smote_experiment import SMOTEExperiment
from src.experiments.random_undersampling_experiment import RandomUndersamplingExperiment
from src.experiments.smoteenn_experiment import SMOTEENNExperiment

def run_multiple_techniques(data_path: str = "data/creditcard.csv") -> Dict[str, Any]:
    """
    Run multiple fraud detection techniques
    
    Args:
        data_path: Path to the credit card fraud dataset
        
    Returns:
        Dictionary containing results for each technique
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    
    # Initialize experiments
    experiments = {
        'baseline': BaselineExperimentFinal(),
        'smote': SMOTEExperiment(),
        'random_undersampling': RandomUndersamplingExperiment(),
        'smoteenn': SMOTEENNExperiment()
    }
    
    results = {}
    
    for name, experiment in experiments.items():
        print(f"\nRunning {name.upper()} Experiment...")
        try:
            experiment_results = experiment.run_experiment(data_path)
            results[name] = {
                'experiment': experiment,
                'results': experiment_results
            }
            print(f"\n{name.upper()} Results:")
            experiment.print_results(experiment_results)
        except Exception as e:
            print(f"Error running {name} experiment: {str(e)}")
            raise
            
    return results

def analyze_technique_comparisons(
    experiment_results: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Perform statistical analysis comparing techniques

    Args:
        experiment_results: Dictionary of results from different techniques
        
    Returns:
        Dictionary containing comparison analyses
    """
    comparisons = {}
    techniques = list(experiment_results.keys())
    
    # Add debug prints
    print(f"\nDebug: Analyzing techniques: {techniques}")
    
    # Start a new MLflow run for comparisons
    with mlflow.start_run(run_name="technique_comparison"):
        # Log the techniques being compared
        mlflow.log_param("techniques_compared", ", ".join(techniques))
        
        # Compare each pair of techniques
        for i in range(len(techniques)):
            for j in range(i + 1, len(techniques)):
                technique1 = techniques[i]
                technique2 = techniques[j]
                
                print(f"\nComparing {technique1} with {technique2}")
                
                # Get experiments
                exp1 = experiment_results[technique1]['experiment']
                exp2 = experiment_results[technique2]['experiment']
                
                try:
                    # Perform comparison
                    comparison = exp1.compare_with(exp2)
                    
                    # Store comparison results
                    comparison_key = f"{technique1}_vs_{technique2}"
                    comparisons[comparison_key] = comparison
                    
                    # Log comparison metrics to MLflow
                    for metric, results in comparison['comparisons'].items():
                        mlflow.log_metrics({
                            f"{comparison_key}_{metric}_mean_diff": results['mean_difference'],
                            f"{comparison_key}_{metric}_p_value": results['p_value'],
                            f"{comparison_key}_{metric}_cohens_d": results['cohens_d']
                        })
                    
                    # Log all the comparison plots
                    if 'plots' in comparison:
                        print("\nDebug: Logging comparison plots")
                        print(f"Available plots: {list(comparison['plots'].keys())}")
                        for plot_name, fig in comparison['plots'].items():
                            try:
                                plot_path = f"{comparison_key}_{plot_name}.png"
                                print(f"Saving plot: {plot_path}")
                                # Save the figure
                                mlflow.log_figure(fig, plot_path)
                                # Close the specific figure
                                plt.close(fig)
                            except Exception as e:
                                print(f"Error saving plot {plot_name}: {str(e)}")
                                # If there's an error, try to close all figures to clean up
                                plt.close('all')
                    
                    # Print formatted results
                    formatted_results = format_comparison_results(comparison)
                    print(formatted_results)
                    
                    # Save formatted results as text artifact
                    with open("comparison_results.txt", "w") as f:
                        f.write(formatted_results)
                    mlflow.log_artifact("comparison_results.txt")
                    
                except Exception as e:
                    print(f"Error comparing {technique1} vs {technique2}: {str(e)}")
                    print("Continuing with next comparison...")
                    continue
        
        # Log the summary table
        summary_table = generate_summary_table(experiment_results)
        
        # Save summary table as CSV
        summary_table.to_csv("technique_summary.csv", index=False)
        mlflow.log_artifact("technique_summary.csv")
        
        # Save as HTML table with basic styling
        html_content = """
        <html>
        <head>
            <style>
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid black; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
        """
        html_content += summary_table.to_html(index=False)
        html_content += "</body></html>"
        
        with open("technique_summary.html", "w") as f:
            f.write(html_content)
        mlflow.log_artifact("technique_summary.html")
            
    return comparisons

def generate_summary_table(
    experiment_results: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Generate summary table of all techniques' performance
    
    Args:
        experiment_results: Dictionary of results from different techniques
        
    Returns:
        DataFrame containing summary statistics
    """
    summary_data = []
    
    for technique, data in experiment_results.items():
        results = data['results']
        
        # Extract key metrics
        metrics = {
            'Technique': technique.upper(),
            'Accuracy': f"{results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}",
            'Precision': f"{results['precision_mean']:.4f} ± {results['precision_std']:.4f}",
            'Recall': f"{results['recall_mean']:.4f} ± {results['recall_std']:.4f}",
            'F1 Score': f"{results['f1_score_mean']:.4f} ± {results['f1_score_std']:.4f}",
            'ROC AUC': f"{results['roc_auc_mean']:.4f} ± {results['roc_auc_std']:.4f}",
            'PR AUC': f"{results['auprc_mean']:.4f} ± {results['auprc_std']:.4f}",
            'G-Mean': f"{results['g_mean_mean']:.4f} ± {results['g_mean_std']:.4f}",
            'MCC': f"{results['mcc_mean']:.4f} ± {results['mcc_std']:.4f}",
            'Training Time (s)': f"{results['training_time_mean']:.2f}",
            'Memory Usage (MB)': f"{results['peak_memory_usage_mean']:.2f}"
        }
        
        summary_data.append(metrics)
    
    return pd.DataFrame(summary_data)

def run_comparative_analysis(data_path: str = "data/creditcard.csv") -> Dict[str, Any]:
    """
    Run complete comparative analysis of fraud detection techniques
    
    Args:
        data_path: Path to the credit card fraud dataset
        
    Returns:
        Dictionary containing all analysis results
    """
    print("\nStarting Comparative Analysis")
    print("=" * 50)
    
    try:
        # Run all techniques
        results = run_multiple_techniques(data_path)
        
        # Perform statistical comparisons
        print("\nPerforming Statistical Analysis...")
        comparisons = analyze_technique_comparisons(results)
        
        # Generate summary table
        print("\nGenerating Summary Table...")
        summary_table = generate_summary_table(results)
        print("\nSummary of All Techniques:")
        print(summary_table.to_string(index=False))
        
        return {
            'results': results,
            'comparisons': comparisons,
            'summary_table': summary_table
        }
        
    except Exception as e:
        print(f"\nError during comparative analysis: {str(e)}")
        raise

def main():
    """Main entry point for comparative analysis"""
    try:
        # End any existing MLflow runs
        mlflow.end_run()
        
        # Run complete analysis
        analysis_results = run_comparative_analysis()
        
        print("\nComparative analysis completed successfully!")
        
    except Exception as e:
        print(f"\nError during comparative analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()