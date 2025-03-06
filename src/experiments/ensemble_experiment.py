"""Ensemble experiment implementation using probability averaging from multiple techniques"""

from typing import Dict, Any, List
import time
import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

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
        
        # Validate input data
        if not isinstance(data, dict) or not all(k in data for k in ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']):
            raise ValueError("Invalid data format provided to ensemble preprocessing")
        
        # Store original data
        processed_data = {
            'original': data.copy(),
            'technique_data': {}
        }
        
        # Process data for each technique
        for name, experiment in self.technique_experiments.items():
            try:
                print(f"\nProcessing data for {name}...")
                
                # Create a deep copy of data for each technique
                technique_data = {k: v.copy() for k, v in data.items()}
                
                # Apply technique-specific preprocessing
                technique_data = experiment.preprocess_data(technique_data)
                processed_data['technique_data'][name] = technique_data
                
                # Print basic info about processed data
                if 'resampling_metadata' in technique_data:
                    meta = technique_data['resampling_metadata']
                    print(f"- Processed with {name}")
                    print(f"- Training samples: {len(technique_data['y_train'])}")
                    if 'final_ratio' in meta:
                        print(f"- Final class ratio: {meta['final_ratio']:.2f}")
                    
            except Exception as e:
                print(f"Error processing data for {name}: {str(e)}")
                raise
        
        # Calculate and store metadata
        processing_time = time.time() - start_time
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024 - initial_memory
        
        # Print processing info
        print("\nEnsemble preparation complete:")
        print(f"Time taken: {processing_time:.2f} seconds")
        print(f"Memory used: {peak_memory:.2f} MB")
        print(f"Number of techniques: {len(processed_data['technique_data'])}")
        
        # Store metadata
        processed_data['ensemble_metadata'] = {
            'processing_time': processing_time,
            'peak_memory_usage': peak_memory,
            'techniques': list(self.technique_experiments.keys()),
            'technique_weights': self.technique_weights,
            'data_shapes': {
                'X_train': data['X_train'].shape,
                'X_val': data['X_val'].shape,
                'X_test': data['X_test'].shape
            }
        }
        
        # Store current data for logging
        self.current_data = processed_data
        
        return processed_data
    
    def train_models(self, data: Dict[str, Any]) -> Dict[str, FraudDetectionModel]:
        """
        Train models for all component techniques
        
        Args:
            data: Processed data dictionary containing data for each technique
            
        Returns:
            Dictionary of trained models for each technique
        """
        print("\nTraining ensemble models...")
        start_time = time.time()
        
        # Initialize storage for models and training metadata
        models = {}
        training_metadata = {}
        
        # Validate input data
        if 'technique_data' not in data:
            raise ValueError("Missing technique_data in processed data")
        
        # Train a model for each technique
        for name, technique_data in data['technique_data'].items():
            try:
                print(f"\nTraining model for {name}...")
                technique_start_time = time.time()
                
                # Initialize model
                model = FraudDetectionModel()
                
                # Create callback list
                callbacks = []
                if hasattr(self, 'mlflow_tracker'):
                    callbacks.append(self.mlflow_tracker.create_keras_callback())
                
                # Train model with technique-specific data
                history = model.train(
                    technique_data['X_train'],
                    technique_data['y_train'],
                    technique_data['X_val'],
                    technique_data['y_val'],
                    callbacks=callbacks,
                    class_weight=technique_data.get('class_weights')
                )
                
                # Store model and training metadata
                models[name] = model
                training_metadata[name] = {
                    'training_time': time.time() - technique_start_time,
                    'final_val_loss': history.history['val_loss'][-1],
                    'final_val_accuracy': history.history['val_accuracy'][-1],
                    'epochs_trained': len(history.history['loss'])
                }
                
                print(f"- Training completed in {training_metadata[name]['training_time']:.2f} seconds")
                print(f"- Final validation accuracy: {training_metadata[name]['final_val_accuracy']:.4f}")
                
            except Exception as e:
                print(f"Error training model for {name}: {str(e)}")
                raise
        
        # Calculate total training time
        total_training_time = time.time() - start_time
        
        # Print summary
        print("\nEnsemble training complete:")
        print(f"Total training time: {total_training_time:.2f} seconds")
        print(f"Models trained: {len(models)}/{len(data['technique_data'])}")
        
        # Store metadata
        self.training_metadata = {
            'total_training_time': total_training_time,
            'technique_metadata': training_metadata,
            'models_trained': len(models)
        }
        
        # Store models
        self.models = models
        
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
        
        # Get validation data
        X_val = data['original']['X_val']
        y_val = data['original']['y_val']
        
        # Get predictions from all models
        val_probs = []
        model_predictions = {}
        
        for name, model in models.items():
            print(f"Getting predictions from {name} model...")
            probs = model.model.predict(X_val, verbose=0)
            val_probs.append(probs)
            model_predictions[name] = probs
        
        # Average probabilities with weights
        avg_probs = np.zeros_like(val_probs[0])
        for i, (name, probs) in enumerate(model_predictions.items()):
            weight = self.technique_weights.get(name, 1.0)
            avg_probs += probs * weight
        
        avg_probs /= sum(self.technique_weights.values())
        
        # Find optimal threshold
        precision, recall, thresholds = precision_recall_curve(y_val, avg_probs)
        
        # Calculate F1 score for each threshold
        epsilon = 1e-10  # Small value to prevent division by zero
        f1_scores = 2 * precision * recall / (precision + recall + epsilon)
        
        # Find threshold with highest F1 score
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        
        # Store optimization results
        self.threshold_optimization_results = {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'f1_scores': f1_scores,
            'best_threshold': best_threshold,
            'best_f1': f1_scores[best_idx],
            'model_predictions': model_predictions
        }
        
        print(f"Optimal threshold: {best_threshold:.4f} (F1: {f1_scores[best_idx]:.4f})")
        
        return best_threshold
    
    def ensemble_predict(self, models: Dict[str, FraudDetectionModel], X: np.ndarray) -> tuple:
        """
        Generate ensemble predictions by averaging probabilities
        
        Args:
            models: Dictionary of trained models
            X: Input features
            
        Returns:
            Tuple containing:
            - Binary predictions based on averaged probabilities
            - Raw probability predictions
            - Dictionary of individual model predictions
        """
        # Get predictions from all models
        probs_list = []
        model_predictions = {}
        
        print("\nGenerating ensemble predictions...")
        for name, model in models.items():
            print(f"Getting predictions from {name} model...")
            probs = model.model.predict(X, verbose=0)
            probs_list.append(probs)
            model_predictions[name] = probs
        
        # Average probabilities with weights
        avg_probs = np.zeros_like(probs_list[0])
        for name, probs in model_predictions.items():
            weight = self.technique_weights.get(name, 1.0)
            avg_probs += probs * weight
        
        avg_probs /= sum(self.technique_weights.values())
        
        # Apply threshold for binary predictions
        threshold = self.decision_threshold
        binary_predictions = (avg_probs > threshold).astype(int)
        
        # Calculate prediction statistics
        prediction_stats = {
            'positive_predictions': np.sum(binary_predictions),
            'prediction_mean': np.mean(avg_probs),
            'prediction_std': np.std(avg_probs),
            'threshold_used': threshold
        }
        
        print("\nPrediction Statistics:")
        print(f"Positive predictions: {prediction_stats['positive_predictions']}")
        print(f"Mean probability: {prediction_stats['prediction_mean']:.4f}")
        print(f"Probability std: {prediction_stats['prediction_std']:.4f}")
        
        return binary_predictions, avg_probs, model_predictions, prediction_stats
    
    def run_single_experiment(self, data_path: str) -> Dict[str, Any]:
        """
        Run a single ensemble experiment iteration
        
        Args:
            data_path: Path to the data file
            
        Returns:
            Dictionary of metrics from the ensemble evaluation
        """
        try:
            print("\nStarting ensemble experiment iteration...")
            start_time = time.time()
            
            # Prepare initial data
            data = self.data_prep.prepare_data(data_path)
            
            # Preprocess data for all techniques
            processed_data = self.preprocess_data(data)
            
            # Train models
            models = self.train_models(processed_data)
            
            # Optimize threshold if enabled
            if self.optimize_threshold:
                self.decision_threshold = self.optimize_threshold(models, processed_data)
            
            # Get test data
            X_test = processed_data['original']['X_test']
            y_test = processed_data['original']['y_test']
            
            # Get ensemble predictions
            y_pred, y_proba, model_predictions, prediction_stats = self.ensemble_predict(
                models, X_test
            )
            
            # Calculate comprehensive metrics (adapting from baseline_model.py)
            from sklearn.metrics import (
                confusion_matrix, f1_score, 
                average_precision_score, roc_auc_score,
                precision_recall_curve, roc_curve,
                matthews_corrcoef
            )
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            # Calculate standard metrics
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = f1_score(y_test, y_pred)
            
            # Calculate ROC AUC and PR AUC
            roc_auc = roc_auc_score(y_test, y_proba)
            auprc = average_precision_score(y_test, y_proba)
            
            # Calculate PR and ROC curves
            pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_test, y_proba)
            fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
            
            # Calculate G-mean
            sensitivity = recall
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            g_mean = np.sqrt(sensitivity * specificity)
            
            # Calculate Matthews Correlation Coefficient
            mcc = matthews_corrcoef(y_test, y_pred)
            
            # Calculate execution metrics
            execution_time = time.time() - start_time
            peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Compile all metrics
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'auprc': auprc,
                'g_mean': g_mean,
                'mcc': mcc,
                'true_positives': tp,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn,
                'training_time': execution_time,
                'peak_memory_usage': peak_memory,
                'decision_threshold': self.decision_threshold,
                'curves': {
                    'pr': {
                        'precision': pr_precision,
                        'recall': pr_recall,
                        'thresholds': pr_thresholds
                    },
                    'roc': {
                        'fpr': fpr,
                        'tpr': tpr,
                        'thresholds': roc_thresholds
                    }
                },
                'ensemble_metadata': {
                    'techniques_used': list(self.technique_experiments.keys()),
                    'technique_weights': self.technique_weights,
                    'prediction_stats': prediction_stats
                }
            }
            
            # Print key metrics
            print("\nEnsemble Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"ROC AUC: {roc_auc:.4f}")
            print(f"G-Mean: {g_mean:.4f}")
            print(f"Training Time: {execution_time:.2f} seconds")
            
            return metrics
            
        except Exception as e:
            print(f"Error in ensemble experiment: {str(e)}")
            raise
    
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