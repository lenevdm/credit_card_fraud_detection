"""SMOTE=based experiment implementation for credit card fraud detection. Inherits from base_experiment"""

from typing import Dict, Any
import time
from imblearn.over_sampling import SMOTE
import numpy as np
import psutil

from src.experiments.base_experiment import BaseExperiment
from config.experiment_config import ExperimentConfig

class SMOTEExperiment(BaseExperiment):
    """
    SMOTE-based experiment implemenation
    Applies SMOTE oversampling to training data only.
    """

    def __init__(self, n_runs: int = None):
        """Initialize SMOTE experiment"""
        super().__init__(
            experiment_name=ExperimentConfig.SMOTE.NAME,
            n_runs=n_runs
        )

    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply SMOTE oversampling to training data

        Args:
            data: Dictionary containing train/val/test splits

        Returns:
            Dictionary with resampled training data and original val/test data
        
        """
        print("\nApplying SMOTE oversampling...")
        start_time = time.time()

        # Add validation of input data
        if np.isnan(data['X_train']).any():
            raise ValueError("Input data contains NaN values")
        
        # Store initial memory usage
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Initialize SMOTE
        smote = SMOTE(
            k_neighbors=ExperimentConfig.SMOTE.K_NEIGHBORS, 
            random_state=ExperimentConfig.SMOTE.RANDOM_STATE,
            n_jobs=-1  # Add parallel processing
        )

        # Get original class distribution
        original_dist = np.bincount(data['y_train'].ravel())

        # Apply SMOTE to training data only
        X_train_resampled, y_train_resampled = smote.fit_resample(
            data['X_train'],
            data['y_train'].ravel()
        )

        # Validate resampled data
        if np.isnan(X_train_resampled).any():
            raise ValueError("SMOTE produced NaN values")

        # Get resampled class distribution
        resampled_dist = np.bincount(y_train_resampled)

         # Calculate detailed metadata
        resampling_time = time.time() - start_time
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024 - initial_memory
        synthetic_samples = len(y_train_resampled) - len(data['y_train'])
        
        metadata = {
            'original_distribution': original_dist.tolist(),
            'resampled_distribution': resampled_dist.tolist(),
            'resampling_time': resampling_time,
            'peak_memory_usage': peak_memory,
            'synthetic_samples_generated': synthetic_samples,
            'final_ratio': resampled_dist[0] / resampled_dist[1]
        }

        # Print resampling info
        print("\nResampling Results:")
        print("-" * 40)
        print(f"Original ratio: {original_dist[0]/original_dist[1]:.2f}:1")
        print(f"Final ratio: {metadata['final_ratio']:.2f}:1")
        print(f"Synthetic samples generated: {synthetic_samples}")
        print(f"Memory used: {peak_memory:.2f} MB")
        print(f"Time taken: {resampling_time:.2f} seconds")

        print(f"\nResampling completed in {resampling_time:.2f} seconds")

        # Reshape target variables
        y_train_resampled = np.expand_dims(y_train_resampled, axis=1)

        # Return processed data with resampling metadata
        processed_data = {
            'X_train': X_train_resampled,
            'y_train': y_train_resampled,
            'X_val': data['X_val'],
            'y_val': data['y_val'],
            'X_test': data['X_test'],
            'y_test': data['y_test'],
            'resampling_metadata': {
                'original_distribution': original_dist.tolist(),
                'resampled_distribution': resampled_dist.tolist(),
                'resampling_time': resampling_time,
                'peak_memory_usage': peak_memory,
                'synthetic_samples_generated': synthetic_samples,
                'final_ratio': resampled_dist[0] / resampled_dist[1]
            }
        }

        # Store current data for logging
        self.current_data = processed_data

        return processed_data
    
    def log_experiment_params(self, tracker: Any) -> None:
        """
        Log SMOTE-specific parameters and metadata
        """
        super().log_experiment_params(tracker)

        # Log basic SMOTE config
        tracker.log_parameters({
            'experiment_type': 'smote',
            'smote_k_neighbors': ExperimentConfig.SMOTE.K_NEIGHBORS,
            'data_modification': 'oversampling',
            'class_balancing': 'smote'
        })

        # Log resampling metadata
        if hasattr(self, 'current_data') and 'resampling_metadata' in self.current_data:
            metadata = self.current_data['resampling_metadata']
            tracker.log_parameters({
                'original_class_ratio': metadata['original_distribution'][0] / metadata['original_distribution'][1],
                'final_class_ratio': metadata['final_ratio'],
                'synthetic_samples_count': metadata['synthetic_samples_generated'],
                'resampling_memory_mb': f"{metadata['peak_memory_usage']:.2f}",
                'resampling_time_seconds': f"{metadata['resampling_time']:.2f}"
            })

    def _get_results_header(self) -> str:
        """Override header for SMOTE results"""
        return "SMOTE Experiment Results"

def main():
    """Run the SMOTE experiment"""

    experiment = SMOTEExperiment()

    try:
        results = experiment.run_experiment("data/creditcard.csv")
        experiment.print_results(results)

    except Exception as e:
        print(f"Experiment failed: {str(e)}")
        raise 

if __name__ == "__main__":
    main()
