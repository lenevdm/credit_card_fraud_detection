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

    def ensemble_predict(self, models: Dict[str, FraudDetectionModel], X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ensemble predictions by weighted averaging of probabilities

        Args:
            models: Dictionary of trained models
            X: Input features

        Returns:
            Tuple containing:
            - Binary predictions based on threshold averaged probabilities
            - Raw averaged probability values
        """

        # Get predictions from all models
        probs_list = []
        model_probs = []

        print("\nGenerating predictions from all models...")
        for name, model in models.items():
            probs = model.model.predict(X, verbose=0)
            probs_list.append(probs)
            model_probs[name] = probs
        
        # Apply weighted averaging
        avg_probs = np.zeros_like(probs_list[0])
        weight_sum = 0

        for i, name in enumerate(models.keys()):
            if name in self.technique_weights:
                weight = self.technique_weights[name]
                avg_probs += probs_list[i] * weight
                weight_sum += weight
                print(f"Added {name} predictions with weight {weight}")

        # Normalize by sum of weights
        if weight_sum > 0:
            avg_probs /= weight_sum

        # Apply threshold
        binary_predictions = (avg_probs > self.decision_threshold).astype(int)

        # Store individual model predictions for analysis
        self.model_predictions = model_probs
        self.ensemble_probabilities = avg_probs

        prediction_counts = np.sum(binary_predictions)
        total_samples = len(binary_predictions)

        print(f"Ensemble predicted {prediction_counts} positives out of {total_samples} samples")
        print(f"Positive rate: {prediction_counts/total_samples:.4f}")
        
        return binary_predictions, avg_probs
    
    def run_single_experiment(self, data_path: str) -> Dict[str, Any]:
        """
        Run a single ensemble experiment iteration
        
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
        models = self.train_models(processed_data)
        
        # Optimize threshold if enabled
        if self.optimize_threshold:
            self.optimize_threshold(models, processed_data)
        
        # Get test data
        X_test = processed_data['original']['X_test']
        y_test = processed_data['original']['y_test'].ravel()  # Flatten
        
        # Get ensemble predictions
        y_pred, y_proba = self.ensemble_predict(models, X_test)
        
        # Calculate metrics
        import time
        import psutil
        import math
        from sklearn.metrics import (
            confusion_matrix, precision_score, recall_score,
            f1_score, roc_auc_score, average_precision_score,
            roc_curve, matthews_corrcoef
        )
        
        # Start timing
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Calculate standard metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Calculate ROC AUC and AUPRC
        roc_auc = roc_auc_score(y_test, y_proba)
        auprc = average_precision_score(y_test, y_proba)
        
        # Calculate PR and ROC curves
        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_test, y_proba)
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
        
        # Calculate G-mean
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        g_mean = math.sqrt(sensitivity * specificity)
        
        # Calculate Matthews Correlation Coefficient
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # Resource usage
        execution_time = time.time() - start_time
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        peak_memory_usage = final_memory - initial_memory
        
        # Collect all metadata from training phases
        total_train_time = sum(stats['training_time'] for stats in self.training_stats.values())
        
        # Store all metrics
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'auprc': auprc,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'g_mean': g_mean,
            'mcc': mcc,
            'training_time': total_train_time,
            'peak_memory_usage': peak_memory_usage,
            'threshold_used': self.decision_threshold,
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
            }
        }
        
        # Print detailed results
        print("\nEnsemble Test Results:")
        print("-" * 40)
        print(f"Decision Threshold: {self.decision_threshold:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"PR AUC: {auprc:.4f}")
        print(f"G-Mean: {g_mean:.4f}")
        print(f"Matthews Correlation Coefficient: {mcc:.4f}")
        print(f"Training Time: {total_train_time:.2f} seconds")
        print(f"Peak Memory Usage: {peak_memory_usage:.2f} MB")
        print("-" * 40)
        print("\nConfusion Matrix:")
        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Positives: {tp}")
        
        return metrics
    
    def log_experiment_params(self, tracker: Any) -> None:
        """
        Log ensemble-specific parameters and metadata
        
        Args:
            tracker: ExperimentTracker instance
        """
        super().log_experiment_params(tracker)
        
        # Log basic ensemble config
        tracker.log_parameters({
            'experiment_type': 'ensemble',
            'techniques_used': ','.join(self.technique_experiments.keys()),
            'combination_method': 'weighted_probability_averaging',
            'decision_threshold': self.decision_threshold,
            'threshold_optimization': str(self.optimize_threshold),
            'threshold_metric': self.threshold_metric
        })
        
        # Log technique weights
        for technique, weight in self.technique_weights.items():
            if technique in self.technique_experiments:
                tracker.log_parameters({
                    f'weight_{technique}': weight
                })
        
        # Log ensemble metadata if available
        if hasattr(self, 'current_data') and 'ensemble_metadata' in self.current_data:
            metadata = self.current_data['ensemble_metadata']
            tracker.log_parameters({
                'processing_time': metadata['processing_time'],
                'peak_memory_usage': metadata['peak_memory_usage']
            })
        
        # Log threshold optimization results if available
        if self.threshold_optimization_results:
            tracker.log_parameters({
                'optimized_threshold': self.threshold_optimization_results['best_threshold'],
                'threshold_best_score': self.threshold_optimization_results['best_score']
            })

    def visualize_ensemble_results(self, tracker: Any) -> None:
        """
        Create and log visualizations of ensemble results
        
        Args:
            tracker: ExperimentTracker instance
        """
        if not self.threshold_optimization_results:
            return
        
        # Visualize threshold optimization
        plt.figure(figsize=(10, 6))
        
        results = self.threshold_optimization_results
        thresholds = results['thresholds']
        scores = results['scores']
        
        plt.plot(thresholds, scores, 'b-', linewidth=2)
        plt.axvline(x=results['best_threshold'], color='r', linestyle='--', 
                    label=f'Best Threshold: {results["best_threshold"]:.4f}')
        
        plt.xlabel('Threshold')
        plt.ylabel(f'{self.threshold_metric.upper()} Score')
        plt.title(f'{self.threshold_metric.upper()} Score vs. Threshold')
        plt.legend()
        plt.grid(True)
        
        tracker.log_figure(plt.gcf(), "threshold_optimization.png")
        plt.close()
        
        # Visualize PR curve
        if 'pr_curve' in results:
            plt.figure(figsize=(10, 6))
            
            precision = results['pr_curve']['precision']
            recall = results['pr_curve']['recall']
            
            plt.plot(recall, precision, 'b-', linewidth=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.grid(True)
            
            tracker.log_figure(plt.gcf(), "ensemble_pr_curve.png")
            plt.close()
        
        # If we have individual model predictions, visualize comparison
        if hasattr(self, 'model_predictions') and hasattr(self, 'ensemble_probabilities'):
            # Create a histogram of prediction probabilities
            plt.figure(figsize=(12, 6))
            
            for name, probs in self.model_predictions.items():
                plt.hist(probs.ravel(), alpha=0.3, bins=30, label=name)
                
            plt.hist(self.ensemble_probabilities.ravel(), alpha=0.5, bins=30, 
                    color='black', label='Ensemble')
            
            plt.axvline(x=self.decision_threshold, color='r', linestyle='--', 
                        label=f'Threshold: {self.decision_threshold:.4f}')
            plt.xlabel('Probability')
            plt.ylabel('Count')
            plt.title('Distribution of Prediction Probabilities')
            plt.legend()
            plt.grid(True)
            
            tracker.log_figure(plt.gcf(), "prediction_distributions.png")
            plt.close()

    def _get_results_header(self) -> str:
        """
        Override header for ensemble results
        
        Returns:
            String header for results output
        """
        return "Ensemble Experiment Results"

    def run_experiment(self, data_path: str) -> Dict[str, Any]:
        """
        Override run_experiment to customize for ensemble approach
        
        Args:
            data_path: Path to the data file
            
        Returns:
            Dictionary of aggregated metrics
        """
        # Use parent's implementation for most functionality
        # but add ensemble-specific visualizations
        results = super().run_experiment(data_path)
        
        # If the run was successful and we have a tracker
        if hasattr(self, 'metrics_list') and len(self.metrics_list) > 0:
            with mlflow.start_run(run_name=f"{self.experiment_name}_summary"):
                tracker = ExperimentTracker(self.experiment_name)
                
                # Log ensemble-specific visualizations
                self.visualize_ensemble_results(tracker)
                
                # Log final metrics
                tracker.log_metrics(results)
        
        return results

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
