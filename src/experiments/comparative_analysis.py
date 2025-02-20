"""Script for running and comparing different fraud detection techniques"""

import os
from typing import Dict, Any, List
import mlflow
import pandas as pd
from src.experiments.base_runs_final import BaselineExperimentFinal
from src.experiments.smote_experiment import SMOTEExperiment
from config.experiment_config import ExperimentConfig
from src.utils.statistical_analysis import format_comparison_results

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
        'smote': SMOTEExperiment()
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
    
    # Compare each pair of techniques
    for i in range(len(techniques)):
        for j in range(i + 1, len(techniques)):
            technique1 = techniques[i]
            technique2 = techniques[j]
            
            # Get experiments
            exp1 = experiment_results[technique1]['experiment']
            exp2 = experiment_results[technique2]['experiment']
            
            # Perform comparison
            comparison = exp1.compare_with(exp2)
            
            # Store comparison results
            comparison_key = f"{technique1}_vs_{technique2}"
            comparisons[comparison_key] = comparison
            
            # Print formatted results
            print(format_comparison_results(comparison))
            
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