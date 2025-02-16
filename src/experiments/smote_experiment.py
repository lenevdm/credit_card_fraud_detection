"""SMOTE=based experiment implementation for credit card fraud detection. Inherits from base_experiment"""

from typing import Dict, Any
import time
from imblearn.over_sampling import SMOTE
import numpy as np

from src.experiments.base_experiment import BaseExperiment
from config.experiment_config import ExperimentConfig

class SMOTEExperiment(BaseExperiment):
    """
    SMOTE-based experiment implemenation
    Applies SMOTE oversampling to training data only.
    """

    def __init__(self, n_runs: int = None)
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

        # Initialize SMOTE
        smote = SMOTE(
            k_neighbors=ExperimentConfig.SMOTE.K_NEIGHBORS,
            random_state=ExperimentConfig.SMOTE.RANDOM_STATE
        )

        # Get original class distribution
        original_dist = np.bincount(data['y_train'].ravel())

        # Apply SMOTE to training data only
        X_train_resampled, y_train_resampled = smote.fit_resample(
            data['X_train'],
            data['y_train'].ravel()
        )

        # Get resampled class distribution
        resampled_dist = np.bincount(y_train_resampled)

        # Calculate resampling time
        resampling_time = time.time() - start_time

        # Print resampling info
        print("\Reampling Results:")
        print("-" * 40)
        print("Original class distribution:")
        print(f"Class 0 (Non-fraud): {original_dist[0]}")
        print(f"Class 1 (Fraud): {original_dist[1]}")
        print(f"Imbalance ratio: {original_dist[0]/original_dist[1]:.2f}:1")

        print("\nResampled class distribution:")
        print(f"Class 0 (Non-fraud): {resampled_dist[0]}")
        print(f"Class 1 (Fraud): {resampled_dist[1]} ")
        print(f"Imbalanced ratio: {resampled_dist[0]/resampled_dist[1]:.2f}:1")

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
                'resampling_time': resampling_time
            }
        }

        return processed_data
    
    def log_experiment_params(self, tracker: Any) -> None:
        """
        Log SMOTE-specific parameters
        """
        super().log_experiment_params(tracker)

        tracker.log_parameters({
            
        })
