"""Script for running and comparing different fraud detection techniques"""

import os
from typing import Dict, Any
import mlflow
from src.experiments.base_runs_final import BaselineExperimentFinal
from src.experiments.smote_experiment import SMOTEExperiment
from config.experiment_config import ExperimentConfig

def run_comparative_analysis(data_path: str = "data/creditcard.csv") -> Dict[str, Any]:
    """
    Run and compare baseline and SMOTE experiments
    
    Args:
        data_path: Path to the credit card fraud dataset
        
    Returns:
        Dictionary containing comparative analysis results
    """
    # Verify data file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    
    print("\nStarting Comparative Analysis")
    print("=" * 50)
    
    # Initialize experiments
    baseline_exp = BaselineExperimentFinal()
    smote_exp = SMOTEExperiment()
    
    # Run baseline experiment
    print("\nRunning Baseline Experiment...")
    baseline_results = baseline_exp.run_experiment(data_path)
    print("\nBaseline Results:")
    baseline_exp.print_results(baseline_results)
    
    # Run SMOTE experiment
    print("\nRunning SMOTE Experiment...")
    smote_results = smote_exp.run_experiment(data_path)
    print("\nSMOTE Results:")
    smote_exp.print_results(smote_results)
    
    # Perform statistical comparison
    print("\nPerforming Statistical Analysis...")
    comparison = baseline_exp.compare_with(smote_exp)
    
    # Print comparative results
    print("\nComparative Analysis Results")
    print("=" * 50)
    
    metrics_to_display = [
        'accuracy', 'precision', 'recall', 'f1_score',
        'roc_auc', 'auprc', 'g_mean', 'mcc'
    ]
    
    for metric in metrics_to_display:
        if metric in comparison:
            stats = comparison[metric]
            print(f"\n{metric.upper()}:")
            print(f"Mean difference (SMOTE - Baseline): {stats['mean_difference']:.4f}")
            print(f"95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
            print(f"p-value: {stats['p_value']:.4f}")
            if stats['is_significant']:
                print("* Difference is statistically significant")
    
    return {
        'baseline_results': baseline_results,
        'smote_results': smote_results,
        'comparison': comparison
    }

def main():
    """Main entry point for comparative analysis"""
    try:
        # End any existing MLflow runs
        mlflow.end_run()
        
        # Run comparative analysis
        results = run_comparative_analysis()
        
        print("\nComparative analysis completed successfully!")
        
    except Exception as e:
        print(f"\nError during comparative analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()