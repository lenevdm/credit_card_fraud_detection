"""Ensemble experiment implementation using probability averaging from multiple techniques"""

from typing import Dict, Any, List
import time
import numpy as np
import psutil
from sklearn.metrics import precision_recall_curve

from src.experiments.base_experiment import BaseExperiment
from src.experiments.base_runs_final import BaselineExperimentFinal
from src.experiments.smoteenn_experiment import SMOTEENNExperiment
from src.experiments.random_undersampling_experiment import RandomUndersamplingExperiment
from src.experiments.class_weight_experiment import ClassWeightExperiment
from src.models.baseline_model import FraudDetectionModel
from config.experiment_config import ExperimentConfig

class EnsembleExperiment(BaseExperiment):
    """
    Ensemble experiment implementation that combines predictions from
    multiple class balancing techniques using probability averaging.
    """
    
    def __init__(self, n_runs: int = None):
        """Initialize Ensemble experiment"""
        super().__init__(
            experiment_name=ExperimentConfig.Ensemble.NAME,
            n_runs=n_runs
        )
        
        # Initialize component experiments
        self.technique_experiments = {
            'baseline': BaselineExperimentFinal(),
            'smoteenn': SMOTEENNExperiment(),
            'random_undersampling': RandomUndersamplingExperiment(),
            'class_weight': ClassWeightExperiment()
        }
        
        # Filter to include only specified techniques
        self.technique_experiments = {
            k: v for k, v in self.technique_experiments.items()
            if k in ExperimentConfig.Ensemble.TECHNIQUES
        }
        
        self.technique_weights = ExperimentConfig.Ensemble.TECHNIQUE_WEIGHTS
        self.decision_threshold = ExperimentConfig.Ensemble.DECISION_THRESHOLD
        self.optimize_threshold = ExperimentConfig.Ensemble.OPTIMIZE_THRESHOLD
        
        # Storage for trained models and metadata
        self.models = {}
        self.threshold_optimization_results = None
        
    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess data for all component techniques
        
        Args:
            data: Original data dictionary
            
        Returns:
            Dictionary containing preprocessed data for each technique
        """
        print("\nPreparing ensemble components...")
        start_time = time.time()
        
        # Store initial memory usage
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Store original data
        processed_data = {
            'original': data.copy(),
            'technique_data': {}
        }
        
        # Process data for each technique
        for name, experiment in self.technique_experiments.items():
            print(f"\nProcessing data for {name}...")
            technique_data = experiment.preprocess_data(data.copy())
            processed_data['technique_data'][name] = technique_data
        
        # Calculate and store metadata
        processing_time = time.time() - start_time
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024 - initial_memory
        
        # Print processing info
        print("\nEnsemble preparation complete:")
        print(f"Time taken: {processing_time:.2f} seconds")
        print(f"Memory used: {peak_memory:.2f} MB")
        
        # Store metadata
        processed_data['ensemble_metadata'] = {
            'processing_time': processing_time,
            'peak_memory_usage': peak_memory,
            'techniques': list(self.technique_experiments.keys()),
            'technique_weights': self.technique_weights
        }
        
        # Store current data for logging
        self.current_data = processed_data
        
        return processed_data

    def train_models(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train models for all component techniques
        
        Args:
            data: Processed data dictionary
            
        Returns:
            Dictionary of trained models and their metadata
        """
        training_metadata = {}
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        for name, technique_data in data['technique_data'].items():
            print(f"\nTraining model for {name}...")
            technique_start = time.time()
            
            # Initialize model for this technique
            model = self.technique_experiments[name].model if hasattr(self.technique_experiments[name], 'model') else FraudDetectionModel()
            
            # Train model with technique-specific data
            history = model.train(
                technique_data['X_train'],
                technique_data['y_train'],
                technique_data['X_val'],
                technique_data['y_val'],
                class_weight=technique_data.get('class_weights')
            )
            
            # Calculate technique-specific metadata
            technique_time = time.time() - technique_start
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Store model and metadata
            self.models[name] = model
            training_metadata[name] = {
                'training_time': technique_time,
                'memory_usage': current_memory - initial_memory,
                'history': history.history if history else None
            }
            
            print(f"Completed training {name}:")
            print(f"Time taken: {technique_time:.2f} seconds")
            print(f"Memory used: {current_memory - initial_memory:.2f} MB")
        
        # Calculate overall training metadata
        total_time = time.time() - start_time
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024 - initial_memory
        
        print("\nEnsemble training complete:")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Peak memory: {peak_memory:.2f} MB")
        
        # Store overall metadata
        training_metadata['ensemble'] = {
            'total_training_time': total_time,
            'peak_memory_usage': peak_memory,
            'techniques_trained': list(self.models.keys())
        }
        
        return training_metadata

    def _get_results_header(self) -> str:
        """Override header for ensemble results"""
        return "Ensemble Experiment Results"

def main():
    """Run the Ensemble experiment"""
    
    experiment = EnsembleExperiment()
    
    try:
        results = experiment.run_experiment("data/creditcard.csv")
        experiment.print_results(results)
        
    except Exception as e:
        print(f"Experiment failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()