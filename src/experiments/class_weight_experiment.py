"""Class weight experiment implementation for credit card fraud detection"""

from typing import Dict, Any
import time
import numpy as np
import psutil
import warnings
from sklearn.utils.class_weight import compute_class_weight

from src.experiments.base_experiment import BaseExperiment
from config.experiment_config import ExperimentConfig

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

class ClassWeightExperiment(BaseExperiment):
    """
    Class Weight experiment implementation.
    Applies class weights during model training to address class imbalance.
    """

    def __init__(self, n_runs: int = None):
        """Initialize Class Weight experiment"""
        super().__init__(
            experiment_name=ExperimentConfig.ClassWeight.NAME,
            n_runs=n_runs
        )

    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate class weights and add them to the data dictionary. 
        The actual data remains unchanged.

        Args:
            data: Dictionary containing train/val/test splits

        Returns:
            Dictionary with original data and added class weights
        """
        print("\nCalculating class weights for imbalanced data...")
        start_time = time.time()

        # Add validation of input data
        if np.isnan(data['X_train']).any():
            raise ValueError("Input data contains NaN values")
        
        # Store initial memory usage
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        peak_memory_seen = initial_memory

        # Get original class distribution
        y_train_flat = data['y_train'].ravel()
        original_dist = np.bincount(y_train_flat)

        # Calculate class weights using sklearn's compute_class_weight
        classes = np.unique(y_train_flat)
        class_weights = compute_class_weight(
            class_weight=ExperimentConfig.ClassWeight.WEIGHT_METHOD,
            classes=classes,
            y=y_train_flat
        )

        # Create class weight dictionary for Keras
        class_weight_dict = {i: weight for i, weight in zip(classes, class_weights)}

        # Update peak memory after weight calculation
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        peak_memory_seen = max(peak_memory_seen, current_memory)
        
        # Calculate actual memory increase
        peak_memory_usage = max(0, peak_memory_seen - initial_memory)

        # Calculate detailed metadata
        calculation_time = time.time() - start_time

        # Print weight info
        print("\nClass Weight Results:")
        print("-" * 40)
        print(f"Original class distribution:")
        print(f"Non-fraudulent: {original_dist[0]} ({original_dist[0]/sum(original_dist)*100:.2f}%)")
        print(f"Fraudulent: {original_dist[1]} ({original_dist[1]/sum(original_dist)*100:.2f}%)")
        print(f"Original ratio: {original_dist[0]/original_dist[1]:.2f}:1")
        print(f"\nCalculated class weights:")
        print(f"Non-fraudulent (Class 0): {class_weight_dict[0]:.4f}")
        print(f"Fraudulent (Class 1): {class_weight_dict[1]:.4f}")
        print(f"Weight ratio: {class_weight_dict[1]/class_weight_dict[0]:.2f}:1")
        print(f"\nPerformance metrics:")
        print(f"Memory used: {peak_memory_usage:.2f} MB")
        print(f"Time taken: {calculation_time:.2f} seconds")

        # Return processed data with class weights
        processed_data = data.copy()
        processed_data['class_weights'] = class_weight_dict

        # Store metadata for logging
        processed_data['weight_metadata'] = {
            'original_distribution': original_dist.tolist(),
            'class_weights': class_weight_dict,
            'calculation_time': calculation_time,
            'peak_memory_usage': peak_memory_usage,
            'weight_ratio': class_weight_dict[1] / class_weight_dict[0]
        }

        # Store current data for logging
        self.current_data = processed_data

        return processed_data

    def log_experiment_params(self, tracker: Any) -> None:
        """Log class weight-specific parameters and metadata"""
        super().log_experiment_params(tracker)

        # Log basic config
        tracker.log_parameters({
            'experiment_type': 'class_weight',
            'weight_method': ExperimentConfig.ClassWeight.WEIGHT_METHOD,
            'data_modification': 'none',
            'class_balancing': 'weighted_loss'
        })

        # Log weight metadata if available
        if hasattr(self, 'current_data') and 'weight_metadata' in self.current_data:
            metadata = self.current_data['weight_metadata']
            tracker.log_parameters({
                'original_class_ratio': metadata['original_distribution'][0] / metadata['original_distribution'][1],
                'class_weight_ratio': metadata['weight_ratio'],
                'class_0_weight': metadata['class_weights'][0],
                'class_1_weight': metadata['class_weights'][1],
                'calculation_memory_mb': f"{metadata['peak_memory_usage']:.2f}",
                'calculation_time_seconds': f"{metadata['calculation_time']:.2f}"
            })

    def _get_results_header(self) -> str:
        """Override header for class weight results"""
        return "Class Weight Experiment Results"

def main():
    """Run the Class Weight experiment"""
    experiment = ClassWeightExperiment()
    try:
        results = experiment.run_experiment("data/creditcard.csv")
        experiment.print_results(results)
    except Exception as e:
        print(f"Experiment failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()