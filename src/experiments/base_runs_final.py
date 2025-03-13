"""Final baseline experiment implementation for credit card fraud detection"""

from typing import Dict, Any, List
import time
import psutil
import numpy as np
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
        For baseline, this tracks memory usage but returns data unchanged.
        
        Args:
            data: Dictionary containing train/val/test splits
            
        Returns:
            Same data structure, unmodified for baseline
        """
        print("\nPreparing baseline data...")
        start_time = time.time()
        
        # Add validation of input data
        if np.isnan(data['X_train']).any():
            raise ValueError("Input data contains NaN values")
        
        # Store initial memory usage
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        peak_memory_seen = initial_memory

        # Get original class distribution
        original_dist = np.bincount(data['y_train'].ravel())

        # Update peak memory
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        peak_memory_seen = max(peak_memory_seen, current_memory)
        
        # Calculate actual memory increase
        peak_memory_usage = max(0, peak_memory_seen - initial_memory)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Print baseline info
        print("\nBaseline Data Preparation:")
        print("-" * 40)
        print(f"Original class distribution:")
        print(f"Non-fraudulent: {original_dist[0]} ({original_dist[0]/sum(original_dist)*100:.2f}%)")
        print(f"Fraudulent: {original_dist[1]} ({original_dist[1]/sum(original_dist)*100:.2f}%)")
        print(f"Class ratio: {original_dist[0]/original_dist[1]:.2f}:1")
        print(f"\nPerformance metrics:")
        print(f"Memory used: {peak_memory_usage:.2f} MB")
        print(f"Time taken: {processing_time:.2f} seconds")
        
        # Return processed data with metadata
        processed_data = data.copy()
        
        # Add metadata
        processed_data['baseline_metadata'] = {
            'original_distribution': original_dist.tolist(),
            'processing_time': processing_time,
            'peak_memory_usage': peak_memory_usage,
            'class_ratio': original_dist[0] / original_dist[1]
        }

        # Store current data for logging
        self.current_data = processed_data
        
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

        # Log baseline metadata if available
        if hasattr(self, 'current_data') and 'baseline_metadata' in self.current_data:
            metadata = self.current_data['baseline_metadata']
            tracker.log_parameters({
                'original_class_ratio': metadata['class_ratio'],
                'processing_memory_mb': f"{metadata['peak_memory_usage']:.2f}",
                'processing_time_seconds': f"{metadata['processing_time']:.2f}"
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