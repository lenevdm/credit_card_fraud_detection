"""Final baseline experiment implementation for credit card fraud detection"""

from typing import Dict, Any, List
from src.experiments.base_experiment import BaseExperiment
from config.experiment_config import ExperimentConfig

class BaselineExperimentFinal(BaseExperiment):
    """
    Final baseline experiment implementation.
    Inherits from BaseExperiment and implements required abstract methods.
    """
    
    def __init__(self, n_runs: int = None):
        """Initialize baseline experiment"""
        super().__init__(
            experiment_name=ExperimentConfig.BASE_EXPERIMENT_NAME,
            n_runs=n_runs
        )
        
    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implements required preprocessing for baseline experiment.
        For baseline, this simply returns the data unchanged.
        
        Args:
            data: Dictionary containing train/val/test splits
            
        Returns:
            Same data structure, unmodified for baseline
        """
         # Make sure we're returning data in the same structure that other techniques return
        print("Debug - baseline preprocess_data called")
        print("Debug - input data keys:", data.keys())
        
        # Return the data with consistent structure
        processed_data = {k: v for k, v in data.items()}
        print("Debug - returning processed data with keys:", processed_data.keys())
        return processed_data
    
    def log_experiment_params(self, tracker: Any) -> None:
        """
        Log experiment-specific parameters
        
        Args:
            tracker: ExperimentTracker instance
        """
        super().log_experiment_params(tracker)
        
        tracker.log_parameters({
            "experiment_type": "baseline",
            "data_modification": "none",
            "class_balancing": "none"
        })

    def _validate_metrics_list(self) -> bool:
        """
        Validate metrics_list data
        
        Returns:
            bool: True if metrics_list is valid, False otherwise
        """
        if not hasattr(self, 'metrics_list'):
            print("Warning: metrics_list not found")
            return False
            
        if not isinstance(self.metrics_list, list):
            print("Warning: metrics_list is not a list")
            return False
            
        if len(self.metrics_list) == 0:
            print("Warning: metrics_list is empty")
            return False
            
        return True

    def print_results(self, results: Dict[str, float]) -> None:
        """
        Print formatted results from the experiment
        
        Args:
            results: Dictionary containing aggregated metrics
        """
        print("\nFinal Baseline Results:")
        print("-" * 40)
        
        # Print core metrics with confidence intervals
        metrics_to_display = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'roc_auc', 'auprc', 'g_mean', 'mcc'
        ]
        
        for metric in metrics_to_display:
            print(f"\n{metric.upper()}:")
            print(f"Mean: {results[f'{metric}_mean']:.4f}")
            print(f"Std Dev: {results[f'{metric}_std']:.4f}")
            print(f"95% CI: [{results[f'{metric}_ci_lower']:.4f}, "
                  f"{results[f'{metric}_ci_upper']:.4f}]")
        
        # Print resource usage
        print("\nRESOURCE USAGE:")
        print(f"Avg. Training Time: {results['training_time_mean']:.2f}s")
        print(f"Avg. Peak Memory: {results['peak_memory_usage_mean']:.2f}MB")
        
        # Print run statistics
        if self._validate_metrics_list():
            print(f"\nRUN STATISTICS:")
            print(f"Completed Runs: {len(self.metrics_list)}")
            print(f"Expected Runs: {self.n_runs}")

    def _get_results_header(self) -> str:
        """Override header for baseline results"""
        return "Final Baseline Results"

def main():
    """Run the final baseline experiment"""
    
    experiment = BaselineExperimentFinal()
    
    try:
        results = experiment.run_experiment("data/creditcard.csv")
        experiment.print_results(results)
        
    except Exception as e:
        print(f"Experiment failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()