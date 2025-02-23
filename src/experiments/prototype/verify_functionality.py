"""Script to verify MLflow tracking and statistical analysis functionality"""

import os
from src.utils.mlflow_utils import ExperimentTracker
from src.utils.statistical_analysis import compare_techniques, format_comparison_results
import pandas as pd

def verify_mlflow_storage():
    """Verify MLflow result storage and retrieval"""
    print("\nVerifying MLflow Result Storage:")
    print("=" * 50)
    
    tracker = ExperimentTracker("comparative_analysis")
    
    try:
        # Get results for both techniques
        baseline_results = tracker.get_results_for_technique("fraud_detection")
        smote_results = tracker.get_results_for_technique("fraud_detection_smote")
        
        print(f"Found {len(baseline_results)} baseline runs")
        print(f"Found {len(smote_results)} SMOTE runs")
        
        # Verify result structure
        if baseline_results:
            print("\nBaseline Results Structure:")
            print(f"Available metrics: {list(baseline_results[0].keys())}")
            
        if smote_results:
            print("\nSMOTE Results Structure:")
            print(f"Available metrics: {list(smote_results[0].keys())}")
            
        return baseline_results, smote_results
        
    except Exception as e:
        print(f"Error verifying MLflow storage: {str(e)}")
        return None, None

def verify_statistical_analysis(baseline_results, smote_results):
    """Verify statistical analysis functionality"""
    print("\nVerifying Statistical Analysis:")
    print("=" * 50)
    
    try:
        metrics_of_interest = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'roc_auc', 'auprc', 'g_mean', 'mcc'
        ]
        
        comparison_results = compare_techniques(
            "Baseline",
            baseline_results,
            "SMOTE",
            smote_results,
            metrics_of_interest
        )
        
        # Print formatted results
        print("\nStatistical Comparison Results:")
        print(format_comparison_results(comparison_results))
        
        # Verify effect sizes and corrections were calculated
        for metric, results in comparison_results['comparisons'].items():
            print(f"\nVerifying {metric} calculations:")
            print(f"Effect size calculated: {'cohens_d' in results}")
            print(f"P-value adjusted: {'p_value_adjusted' in results}")
            
        return comparison_results
        
    except Exception as e:
        print(f"Error in statistical analysis: {str(e)}")
        return None

def create_summary_table(baseline_results, smote_results):
    """Create summary table comparing techniques"""
    print("\nGenerating Summary Table:")
    print("=" * 50)
    
    try:
        # Combine results
        all_results = []
        
        # Process baseline results
        if baseline_results:
            baseline_metrics = {
                'Technique': 'Baseline',
                'Runs': len(baseline_results),
                'Mean F1': np.mean([r['f1_score'] for r in baseline_results]),
                'Mean ROC-AUC': np.mean([r['roc_auc'] for r in baseline_results]),
                'Mean G-Mean': np.mean([r['g_mean'] for r in baseline_results])
            }
            all_results.append(baseline_metrics)
            
        # Process SMOTE results
        if smote_results:
            smote_metrics = {
                'Technique': 'SMOTE',
                'Runs': len(smote_results),
                'Mean F1': np.mean([r['f1_score'] for r in smote_results]),
                'Mean ROC-AUC': np.mean([r['roc_auc'] for r in smote_results]),
                'Mean G-Mean': np.mean([r['g_mean'] for r in smote_results])
            }
            all_results.append(smote_metrics)
            
        # Create DataFrame
        summary_df = pd.DataFrame(all_results)
        print("\nSummary Table:")
        print(summary_df.to_string(index=False))
        
        return summary_df
        
    except Exception as e:
        print(f"Error creating summary table: {str(e)}")
        return None

def main():
    """Main verification function"""
    print("Starting Functionality Verification")
    print("=" * 50)
    
    # Verify MLflow
    baseline_results, smote_results = verify_mlflow_storage()
    
    if baseline_results and smote_results:
        # Verify statistical analysis
        comparison_results = verify_statistical_analysis(baseline_results, smote_results)
        
        # Create summary table
        summary_table = create_summary_table(baseline_results, smote_results)
        
        print("\nVerification Complete!")
        print("=" * 50)
        if comparison_results and summary_table is not None:
            print("All functionality verified successfully!")
        else:
            print("Some verifications failed - check the output above for details.")
    else:
        print("\nVerification failed - could not retrieve results from MLflow")

if __name__ == "__main__":
    main()