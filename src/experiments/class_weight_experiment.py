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

        # Calculate detailed metadata
        calculation_time = time.time() - start_time
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024 - initial_memory

        


    #def log_experiment_params(self, tracker: Any) -> None:

    #def _get_results_header(self) -> str:


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