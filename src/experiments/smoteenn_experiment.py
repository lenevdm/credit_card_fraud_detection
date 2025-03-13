"""SMOTE-ENN experiment implementation for credit card fraud detection"""

from typing import Dict, Any
import time
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE
import numpy as np
import psutil
import warnings

from src.experiments.base_experiment import BaseExperiment
from config.experiment_config import ExperimentConfig

warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='imblearn')


class SMOTEENNExperiment(BaseExperiment):
    """
    SMOTE-ENN experiment implementation.
    Applies SMOTE-ENN to training data only.
    """

    def __init__(self, n_runs: int = None):
        """Initialize SMOTE-ENN experiment"""
        super().__init__(
            experiment_name=ExperimentConfig.SMOTEENN.NAME,
            n_runs=n_runs
        )

    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply SMOTE-ENN to training data

        Args:
            data: Dictionary containing train/val/test splits

        Returns:
            Dictionary with resampled training data and original val/test data
        """
        print("\nApplying SMOTE-ENN...")
        start_time = time.time()

        # Add validation of input data
        if np.isnan(data['X_train']).any():
            raise ValueError("Input data contains NaN values")
        
        # Store initial memory usage
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        peak_memory_seen = initial_memory

        try:
            # Create SMOTE instance
            smote = SMOTE(
                k_neighbors=ExperimentConfig.SMOTEENN.K_NEIGHBORS,
                sampling_strategy=ExperimentConfig.SMOTEENN.SAMPLING_STRATEGY,
                random_state=ExperimentConfig.SMOTEENN.RANDOM_STATE,
                n_jobs=ExperimentConfig.SMOTEENN.N_JOBS
            )
            
            # Create ENN instance
            enn = EditedNearestNeighbours(
                n_neighbors=ExperimentConfig.SMOTEENN.ENN_K_NEIGHBORS,
                sampling_strategy='all'  # Clean from both classes
            )
            
            # Initialize SMOTE-ENN with components
            smote_enn = SMOTEENN(
                sampling_strategy='auto',
                smote=smote,  # Pass SMOTE instance
                enn=enn,  # Pass ENN instance
                random_state=ExperimentConfig.SMOTEENN.RANDOM_STATE
            )
            
            # Get original class distribution
            original_dist = np.bincount(data['y_train'].ravel())
            
            print("\nStarting SMOTE-ENN processing...")
            print(f"Original data shape: {data['X_train'].shape}")
            print(f"Original class distribution: {original_dist}")
            
            # Apply SMOTE-ENN to training data only
            X_train_resampled, y_train_resampled = smote_enn.fit_resample(
                data['X_train'],
                data['y_train'].ravel()
            )

            # Update peak memory after processing
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            peak_memory_seen = max(peak_memory_seen, current_memory)
            
            # Calculate actual memory increase
            peak_memory_usage = max(0, peak_memory_seen - initial_memory)

        except Exception as e:
            print(f"\nError during SMOTE-ENN processing:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            raise

        # Print processing details
        print("\nSMOTE-ENN Processing Details:")
        print(f"Original shape: {data['X_train'].shape}")
        print(f"Resampled shape: {X_train_resampled.shape}")
        print(f"SMOTE Parameters used:")
        print(f"- k_neighbors: {ExperimentConfig.SMOTEENN.K_NEIGHBORS}")
        print(f"- sampling_strategy: {ExperimentConfig.SMOTEENN.SAMPLING_STRATEGY}")
        print(f"ENN Parameters used:")
        print(f"- n_neighbors: {ExperimentConfig.SMOTEENN.ENN_K_NEIGHBORS}")
        
        original_count = len(data['y_train'])
        resampled_count = len(y_train_resampled)
        samples_removed = original_count - resampled_count

        # Validate resampled data
        if np.isnan(X_train_resampled).any():
            raise ValueError("SMOTE-ENN produced NaN values")

        # Get resampled class distribution
        resampled_dist = np.bincount(y_train_resampled)

        # Calculate detailed metadata
        resampling_time = time.time() - start_time
        
        # Print resampling info
        print("\nResampling Results:")
        print("-" * 40)
        print(f"Original class distribution:")
        print(f"Non-fraudulent: {original_dist[0]} ({original_dist[0]/sum(original_dist)*100:.2f}%)")
        print(f"Fraudulent: {original_dist[1]} ({original_dist[1]/sum(original_dist)*100:.2f}%)")
        print(f"\nResampled class distribution:")
        print(f"Non-fraudulent: {resampled_dist[0]} ({resampled_dist[0]/sum(resampled_dist)*100:.2f}%)")
        print(f"Fraudulent: {resampled_dist[1]} ({resampled_dist[1]/sum(resampled_dist)*100:.2f}%)")
        print(f"\nPerformance metrics:")
        print(f"Samples removed by ENN: {samples_removed:,}")
        print(f"Memory used: {peak_memory_usage:.2f} MB")
        print(f"Time taken: {resampling_time:.2f} seconds")

        # Reshape target variables
        y_train_resampled = np.expand_dims(y_train_resampled, axis=1)

        # Return processed data with metadata
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
                'peak_memory_usage': peak_memory_usage,
                'samples_removed': samples_removed,
                'final_ratio': resampled_dist[0] / resampled_dist[1]
            }
        }

        # Store current data for logging
        self.current_data = processed_data

        return processed_data

    def log_experiment_params(self, tracker: Any) -> None:
        """
        Log SMOTE-ENN specific parameters and metadata
        """
        super().log_experiment_params(tracker)

        tracker.log_parameters({
            'experiment_type': 'smoteenn',
            'data_modification': 'hybrid',
            'class_balancing': 'smoteenn',
            'smote_k_neighbors': ExperimentConfig.SMOTEENN.K_NEIGHBORS,
            'enn_k_neighbors': ExperimentConfig.SMOTEENN.ENN_K_NEIGHBORS,
            'sampling_strategy': ExperimentConfig.SMOTEENN.SAMPLING_STRATEGY
        })

        if hasattr(self, 'current_data') and 'resampling_metadata' in self.current_data:
            metadata = self.current_data['resampling_metadata']
            tracker.log_parameters({
                'original_class_ratio': metadata['original_distribution'][0] / metadata['original_distribution'][1],
                'final_class_ratio': metadata['final_ratio'],
                'samples_removed': metadata['samples_removed'],
                'resampling_memory_mb': f"{metadata['peak_memory_usage']:.2f}",
                'resampling_time_seconds': f"{metadata['resampling_time']:.2f}"
            })

    def _get_results_header(self) -> str:
        """Override header for SMOTE-ENN results"""
        return "SMOTE-ENN Experiment Results"

def main():
    """Run the SMOTE-ENN experiment"""
    experiment = SMOTEENNExperiment()
    try:
        results = experiment.run_experiment("data/creditcard.csv")
        experiment.print_results(results)
    except Exception as e:
        print(f"Experiment failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()