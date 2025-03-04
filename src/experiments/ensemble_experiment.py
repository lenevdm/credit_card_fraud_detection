""" Ensemble experiment implementation using probability averaging from multiple techniques"""

from typing import Dict, Any, List, Optional, Tuple
import time
import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix

from src.experiments.base_experiment import BaseExperiment
from src.experiments.base_runs_final import BaselineExperimentFinal
from src.experiments.smote_experiment import SMOTEExperiment
from src.experiments.random_undersampling_experiment import RandomUndersamplingExperiment
from src.experiments.smoteenn_experiment import SMOTEENNExperiment
from src.experiments.class_weight_experiment import ClassWeightExperiment
from src.models.baseline_model import FraudDetectionModel
from config.experiment_config import ExperimentConfig

class EnsembleExperiment(BaseExperiment):
    """
    Ensemble experiment implementation that combines predictions from multiple class balancing techniques using weighted probability averaging.
    """

    def __init__(self, n_runs: int = None):
        """Initialize Ensemble experiment"""
        super().__init__(
            experiment_name=ExperimentConfig.Ensemble.NAME,
            n_runs=n_runs
        )

        # Initialize component elements
        self.technique_experiments = {
            'baseline': BaselineExperimentFinal(),
            'smote': SMOTEExperiment(),
            'random_undersampling': RandomUndersamplingExperiment(),
            'smoteenn': SMOTEENNExperiment(),
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
        self.threshold_metric = ExperimentConfig.Ensemble.THRESHOLD_METRIC

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
        print(f"Total techniques included: {len(self.technique_experiments)}")
        print(f"Techniques: {', '.join(self.technique_experiments.keys())}")
        print(f"Time taken: {processing_time:.2f} seconds")
        print(f"Memory used: {peak_memory:.2f} MB")

        # Store metadata
        processed_data['ensemble_metadata'] = {
            'processing_time': processing_time,
            'peak_memory_usage': peak_memory,
            'techniques': list(self.technique_experiments.keys()),
            'technique_weights': self.technique_weights.copy(),
            'weight_sum': sum(w for _, w in self.technique_weights.items() 
                            if _ in self.technique_experiments)
        }

        # Store current data for logging
        self.current_data = processed_data

        return processed_data
    
    def train_models(self, data: Dict[str, Any]) -> Dict[str, FraudDetectionModel]:
        """
        Train models for all component techniques

        Args:
            data: Processed data dictionary

        Returns:
            Dictionary of trained models for each technique
        """
        models = {}
        training_stats = {}

        for name, technique_data in data['technique_data'].items():
            print(f"\nTraining model for {name}...")

            # Start timing
            start_time = time.time()

            # Initialize model
            model = FraudDetectionModel()

            # Train model with technique-specific data
            model.train(
                technique_data['X_train'],
                technique_data['y_train'],
                technique_data['X_val'],
                technique_data['y_val'],
                class_weight=technique_data.get('class_weights')
            )

            # Record training time
            training_time = time.time() - start_time

            models[name] = model
            training_stats[name] = {
                'training_time': training_time
            }

            print(f"Completed training {name} model in {training_time:.2f} seconds")

        # Store models
        self.models = models
        self.training_stats = training_stats

        print(f"\nCompleted training {len(models)} models")

        return models
    
    def optimize_threshold(self, models: Dict[str, FraudDetectionModel], data: Dict[str, Any]) -> float:
        """
        Find optimal decision threshold using validation data

        Args:
            models: Dictionary of trained models
            data: Processed data dictionary

        Returns:
            Optimal threshold value
        """
        print("\nOptimizing decision threshold...")

        # Get predictions from all models
        X_val = data['original']['X_val']
        y_val = data['original']['y_val'].ravel() # Flatten to 1D array

        # Get predictions from all models        
        val_probs = []
        for name, model in models.items():
            probs = model.model.predict(X_val, verbose=0)
            val_probs.append(probs)

        # Apply weighted averaging
        avg_probs = np.zeros_like(val_probs[0])
        weight_sum = 0

        for i, name in enumerate(models.key()):
            if name in self.technique_weights:
                weight = self.technique_weights[name]
                avg_probs += val_probs[i] *  weight
                weight_sum += weight
        
        # Normalize by sum of weights
        if weight_sum > 0:
            avg_probs /= weight_sum

        # Find threshold that optimizes the chosen metric
        thresholds = np.linspace(0.1, 0.9, 100) # Test 100 thresholds
        best_threshold = 0.5 # Defualt
        best_score = 0
        scores = []

        for threshold in thresholds:
            y_pred = (avg_probs > threshold).astype(int)

            # Calculate score based on selected metric
            if self.threshold_metric == 'f1':
                score = f1_score(y_val, y_pred)
            elif self.threshold_metric == 'gmean':
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = np.sqrt(sensitivity * specificity)
            elif self.threshold_metric == 'balanced_accuracy':
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = (sensitivity + specificity) / 2
            else:
                # Default to F1
                score = f1_score(y_val, y_pred)

            scores.append(score)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        # Also calculate precision-recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_val, avg_probs)

        # Store optimization results
        self.threshold_optimization_results = {
            'thresholds': thresholds,
            'scores': scores,
            'best_threshold': best_threshold,
            'best_score': best_score,
            'metric': self.threshold_metric,
            'pr_curve': {
                'precision': precision,
                'recall': recall,
                'thresholds': pr_thresholds
            }
        }

        print(f"Optimal {self.threshold_metric} threshold: {best_threshold:.4f} (Score: {best_score:.4f})")
    
        # Update the decision threshold
        self.decision_threshold = best_threshold
        
        return best_threshold
