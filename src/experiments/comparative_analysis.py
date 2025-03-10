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
from src.experiments.class_weight_experiment import ClassWeightExperiment

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
        'smoteenn': SMOTEENNExperiment(),
        'class_weight': ClassWeightExperiment()
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

    # Create lists to store all results
    all_formatted_results = []
    all_comparison_data = []
    
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

                     # Store this comparison data for HTML formatting
                    all_comparison_data.append(comparison)
                    print(f"Debug - Added comparison data. Current length: {len(all_comparison_data)}")
                    
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
                    
                    # Format and store this comparison's results
                    formatted_results = format_comparison_results(comparison)
                    all_formatted_results.append(formatted_results)
                    #print(formatted_results)
                    print(f"Debug - Added formatted results. Current length: {len(all_formatted_results)}")
                    
                except Exception as e:
                    print(f"Error comparing {technique1} vs {technique2}: {str(e)}")
                    print("Exception details:", e.__class__.__name__)
                    import traceback
                    print(traceback.format_exc())
                    print("Continuing with next comparison...")
                    continue

        print(f"\nDebug - Final counts:")
        print(f"All comparison data length: {len(all_comparison_data)}")
        print(f"All formatted results length: {len(all_formatted_results)}")

        if not all_formatted_results:
            print("Warning: No formatted results collected!")
            return comparisons
            
        if not all_comparison_data:
            print("Warning: No comparison data collected!")
            return comparisons
        
         # Combine all formatted results with clear separation
        combined_results = (
            "COMPREHENSIVE STATISTICAL ANALYSIS OF FRAUD DETECTION TECHNIQUES\n" +
            "=" * 80 + "\n\n" +
            "Number of runs per technique: 2\n" +  # Update this to use actual n_runs
            "Metrics analyzed: " + ", ".join(ExperimentConfig.METRICS_OF_INTEREST) + "\n\n" +
            "=" * 80 + "\n\n" +
            "\n\n" + "=" * 80 + "\n\n".join(all_formatted_results) +
            "\n\n" + "=" * 80 + "\n\n" +
            "Analysis completed successfully.\n"
        )

        print(f"\nDebug - Combined results length: {len(combined_results)}")
        
        # Save all formatted results as text artifact
        with open("comparison_results.txt", "w") as f:
            f.write(combined_results)
        mlflow.log_artifact("comparison_results.txt")

        # Generate and save HTML comparison results
        try:
            html_content = format_comparison_results_html(all_comparison_data)
            print(f"\nDebug - HTML content length: {len(html_content)}")
            
            with open("comparison_results.html", "w") as f:
                f.write(html_content)
            mlflow.log_artifact("comparison_results.html")
        except Exception as e:
            print(f"Error generating HTML content: {str(e)}")
            print("Exception details:", e.__class__.__name__)
            import traceback
            print(traceback.format_exc())
        
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
                body { font-family: Arial, sans-serif;}
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid black; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
        """
        html_content += summary_table.to_html(index=False)
        html_content += "</body></html>"
        
        # Write with UTF-8 encoding explicitly specified
        with open("technique_summary.html", "w", encoding='utf-8') as f:
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

def format_comparison_results_html(comparisons: List[Dict[str, Any]]) -> str:
    """
    Format comparison results as HTML with styling

    Args: 
        comparisons: List of comparison result dictionaries

    Returns:
        str: Formatted HTML content
    """
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>ML Experiment Comparison Results</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f9f9f9;
                margin: 40px;
                color: #333;
            }}

            h1, h2 {{
                color: #444;
                text-align: center;
            }}

            .container {{
                max-width: 1000px;
                margin: 0 auto;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 30px;
                background-color: #fff;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }}

            th, td {{
                padding: 12px 16px;
                border-bottom: 1px solid #ddd;
                text-align: center;
            }}

            th {{
                background-color: #2c7be5;
                color: #fff;
                font-weight: 600;
            }}

            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}

            .highlight {{
                font-weight: bold;
                color: #2c7be5;
            }}

            .stat-significant {{
                color: #28a745;
                font-weight: bold;
            }}

            .stat-not-significant {{
                color: #dc3545;
                font-weight: bold;
            }}

            .footer {{
                font-size: 0.85em;
                color: #888;
                text-align: center;
                margin-top: 40px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Comparison Results</h1>
            {0}
            <div class="footer">Generated on: {1}</div>
        </div>
    </body>
    </html>
    """

    comparison_tables = []

    for comparison in comparisons:
        technique1 = comparison['technique_names']['technique1']
        technique2 = comparison['technique_names']['technique2']

        table_html = f"""
        <h2>{technique1} vs {technique2}</h2>
        <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Mean Difference</th>
                        <th>95% CI</th>
                        <th>Effect Size</th>
                        <th>p-value</th>
                        <th>Adjusted p-value</th>
                        <th>Significance</th>
                    </tr>
                </thead>
                <tbody>
        """

        for metric, results in comparison['comparisons'].items():
            significant = results['is_significant']
            significance_class = "stat-significant" if significant else "stat-not-significant"
            significance_text = "Significant" if significant else "Not Significant"
            
            table_html += f"""
                <tr>
                    <td>{metric.upper()}</td>
                    <td class="highlight">{results['mean_difference']:.4f}</td>
                    <td>[{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]</td>
                    <td>{results['effect_size']} (d = {results['cohens_d']:.3f})</td>
                    <td>{results['p_value']:.4f}</td>
                    <td>{results['p_value_adjusted']:.4f}</td>
                    <td class="{significance_class}">{significance_text}</td>
                </tr>
            """
        
        table_html += """
                </tbody>
            </table>
        """
        comparison_tables.append(table_html)
    
    from datetime import datetime
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    return html_template.format("\n".join(comparison_tables), current_date)

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