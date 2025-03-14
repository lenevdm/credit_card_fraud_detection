"""Ensemble experiment implementation using probability averaging from multiple techniques"""

from typing import Dict, Any, List, Tuple, Optional
import time
import numpy as np
import psutil
from sklearn.metrics import precision_recall_curve, roc_curve

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

    def optimize_threshold(
        self, 
        data: Dict[str, Any]
    ) -> float:
        """
        Find optimal decision threshold using validation data
        
        Args:
            data: Processed data dictionary
            
        Returns:
            Optimal threshold value
        """
        print("\nOptimizing decision threshold...")
        start_time = time.time()
        
        # Get validation data from original dataset
        X_val = data['original']['X_val']
        y_val = data['original']['y_val']
        
        # Get predictions from all models
        val_probs = []
        
        # Get model predictions
        for name, model in self.models.items():
            print(f"Getting predictions from {name} model...")
            probs = model.model.predict(X_val, verbose=0)
            val_probs.append(probs)
        
        # Average probabilities with weights
        avg_probs = np.zeros_like(val_probs[0])
        for i, probs in enumerate(val_probs):
            technique_name = list(self.models.keys())[i]
            weight = self.technique_weights.get(technique_name, 1.0)
            avg_probs += probs * weight
        
        avg_probs /= sum(self.technique_weights.values())
        
        # Find optimal threshold using PR curve
        precision, recall, thresholds = precision_recall_curve(y_val, avg_probs)
        
        # Calculate F1 score for each threshold
        # Add a small epsilon to avoid division by zero
        epsilon = 1e-10
        f1_scores = 2 * precision * recall / (precision + recall + epsilon)
        
        # Find threshold with highest F1 score
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        
        # Calculate g-mean for the best threshold
        binary_preds = (avg_probs > best_threshold).astype(int)
        tp = np.sum((binary_preds == 1) & (y_val.ravel() == 1))
        tn = np.sum((binary_preds == 0) & (y_val.ravel() == 0))
        fp = np.sum((binary_preds == 1) & (y_val.ravel() == 0))
        fn = np.sum((binary_preds == 0) & (y_val.ravel() == 1))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        g_mean = np.sqrt(sensitivity * specificity)
        
        # Store optimization results
        self.threshold_optimization_results = {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'f1_scores': f1_scores,
            'best_threshold': best_threshold,
            'best_f1': f1_scores[best_idx],
            'g_mean': g_mean,
            'optimization_time': time.time() - start_time
        }
        
        print("\nThreshold Optimization Results:")
        print(f"Optimal threshold: {best_threshold:.4f}")
        print(f"Best F1 Score: {f1_scores[best_idx]:.4f}")
        print(f"G-Mean: {g_mean:.4f}")
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        
        return best_threshold

    def ensemble_predict(
        self, 
        X: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ensemble predictions by averaging probabilities
        
        Args:
            X: Input features
            threshold: Optional custom threshold (uses self.decision_threshold if None)
            
        Returns:
            Tuple of (binary predictions, averaged probabilities)
        """
        if not self.models:
            raise ValueError("No trained models available for prediction")
            
        # Get predictions from all models
        print("\nGenerating ensemble predictions...")
        predictions = []
        
        for name, model in self.models.items():
            print(f"Getting predictions from {name} model...")
            probs = model.model.predict(X, verbose=0)
            predictions.append(probs)
        
        # Average probabilities with weights
        avg_probs = np.zeros_like(predictions[0])
        weight_sum = 0
        
        for i, probs in enumerate(predictions):
            technique_name = list(self.models.keys())[i]
            weight = self.technique_weights.get(technique_name, 1.0)
            avg_probs += probs * weight
            weight_sum += weight
        
        avg_probs /= weight_sum
        
        # Apply threshold
        threshold_to_use = threshold if threshold is not None else self.decision_threshold
        binary_predictions = (avg_probs > threshold_to_use).astype(int)
        
        return binary_predictions, avg_probs

    def run_single_experiment(self, data_path: str) -> Dict[str, Any]:
        """
        Run a single experiment iteration
        
        Args:
            data_path: Path to the data file
            
        Returns:
            Dictionary of metrics
        """
        # Prepare data
        data = self.data_prep.prepare_data(data_path)
        
        # Preprocess data for all techniques
        processed_data = self.preprocess_data(data)
        
        # Train models
        training_metadata = self.train_models(processed_data)
        
        # Optimize threshold if enabled
        if self.optimize_threshold:
            self.decision_threshold = self.optimize_threshold(processed_data)
        
        # Get test data
        X_test = processed_data['original']['X_test']
        y_test = processed_data['original']['y_test']
        
        # Get ensemble predictions
        y_pred, y_proba = self.ensemble_predict(X_test)
        
        # Create model evaluation
        metrics = self.evaluate_ensemble(y_test, y_pred, y_proba, training_metadata)
        
        return metrics

    def evaluate_ensemble(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        training_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate ensemble performance
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            training_metadata: Dictionary containing training information
            
        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, matthews_corrcoef,
            confusion_matrix
        )
        
        # Calculate confusion matrix values
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate standard metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'auprc': average_precision_score(y_true, y_proba),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
        
        # Calculate G-mean
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['g_mean'] = np.sqrt(sensitivity * specificity)
        
        # Add resource usage metrics
        metrics['training_time'] = training_metadata['ensemble']['total_training_time']
        metrics['peak_memory_usage'] = training_metadata['ensemble']['peak_memory_usage']
        
        # Add prediction curves data
        metrics['curves'] = {}
        
        # Calculate PR curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        metrics['curves']['pr'] = {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds
        }
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        metrics['curves']['roc'] = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
        
        print("\nEnsemble Evaluation Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"PR AUC: {metrics['auprc']:.4f}")
        print(f"G-Mean: {metrics['g_mean']:.4f}")
        print(f"MCC: {metrics['mcc']:.4f}")
        
        return metrics

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