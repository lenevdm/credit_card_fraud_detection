"""Ensemble experiment implementation using probability averaging from multiple techniques"""

from typing import Dict, Any, List, Tuple, Optional
import time
import os
import numpy as np
import psutil
import mlflow
from sklearn.metrics import precision_recall_curve, roc_curve

from src.experiments.base_experiment import BaseExperiment
from src.experiments.base_runs_final import BaselineExperimentFinal
from src.experiments.smoteenn_experiment import SMOTEENNExperiment
from src.experiments.random_undersampling_experiment import RandomUndersamplingExperiment
from src.experiments.class_weight_experiment import ClassWeightExperiment
from src.models.baseline_model import FraudDetectionModel
from config.experiment_config import ExperimentConfig
from src.utils.mlflow_utils import ExperimentTracker

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

        # Set up MLflow tracking
        mlflow.set_tracking_uri("file:./mlruns")
        
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

    def optimize_threshold(self, data: Dict[str, Any]) -> float:
        """
        Find optimal decision threshold using validation data with enhanced optimization
        that prioritizes precision.
        
        Args:
            data: Processed data dictionary
            
        Returns:
            Optimal threshold value
        """
        print("\nOptimizing decision threshold...")
        start_time = time.time()
        
        # Get validation data
        X_val = data['original']['X_val']
        y_val = data['original']['y_val']
        
        # Get predictions from all models
        val_probs = []
        for name, model in self.models.items():
            print(f"Getting predictions from {name} model...")
            probs = model.model.predict(X_val, verbose=0)
            val_probs.append(probs)
        
        # Average probabilities with weights
        avg_probs = np.zeros_like(val_probs[0])
        weight_sum = 0
        for i, probs in enumerate(val_probs):
            technique_name = list(self.models.keys())[i]
            weight = self.technique_weights.get(technique_name, 1.0)
            avg_probs += probs * weight
            weight_sum += weight
        
        avg_probs /= weight_sum
        
        # Calculate metrics for different thresholds
        precision, recall, thresholds = precision_recall_curve(y_val, avg_probs)
        
        # Calculate F1 scores
        epsilon = 1e-10  # Small value to prevent division by zero
        f1_scores = 2 * precision * recall / (precision + recall + epsilon)
        
        # Find threshold that maximizes F1
        f1_idx = np.argmax(f1_scores[:-1])  # Exclude last point which might not have a threshold
        f1_threshold = thresholds[f1_idx]
        best_f1 = f1_scores[f1_idx]
        
        # Calculate precision at each threshold
        # The precision from precision_recall_curve already gives us this,
        # but it's in reverse order of thresholds, so we need to match them up
        precisions_at_thresholds = precision[:-1]  # Exclude last point which has no threshold
        
        # Define precision target based on baseline performance
        precision_target = 0.80  # Target precision to match baseline-like performance
        
        # Find highest threshold that gives precision >= target
        # We need to be careful since precision typically increases with threshold
        precision_threshold_pairs = [(t, p) for t, p in zip(thresholds, precisions_at_thresholds) if p >= precision_target]
        
        if precision_threshold_pairs:
            # From thresholds meeting precision target, choose the one with lowest value
            # (which will give highest recall while maintaining precision target)
            # Sort by threshold ascending
            precision_threshold_pairs.sort(key=lambda x: x[0])
            best_threshold = precision_threshold_pairs[0][0]
            
            # Calculate corresponding metrics for chosen threshold
            binary_preds = (avg_probs > best_threshold).astype(int)
            tp = np.sum((binary_preds == 1) & (y_val.ravel() == 1))
            fp = np.sum((binary_preds == 1) & (y_val.ravel() == 0))
            fn = np.sum((binary_preds == 0) & (y_val.ravel() == 1))
            tn = np.sum((binary_preds == 0) & (y_val.ravel() == 0))
            
            precision_at_best = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_at_best = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_at_best = 2 * precision_at_best * recall_at_best / (precision_at_best + recall_at_best + epsilon)
            g_mean_at_best = np.sqrt((tp / (tp + fn)) * (tn / (tn + fp))) if (tp + fn) > 0 and (tn + fp) > 0 else 0
        else:
            # Fall back to F1-optimizing threshold if target precision can't be met
            best_threshold = f1_threshold
            precision_at_best = precision[f1_idx]
            recall_at_best = recall[f1_idx]
            f1_at_best = best_f1
            
            # Calculate G-mean for F1 threshold
            binary_preds = (avg_probs > best_threshold).astype(int)
            tp = np.sum((binary_preds == 1) & (y_val.ravel() == 1))
            fn = np.sum((binary_preds == 0) & (y_val.ravel() == 1))
            tn = np.sum((binary_preds == 0) & (y_val.ravel() == 0))
            fp = np.sum((binary_preds == 1) & (y_val.ravel() == 0))
            g_mean_at_best = np.sqrt((tp / (tp + fn)) * (tn / (tn + fp))) if (tp + fn) > 0 and (tn + fp) > 0 else 0
        
        # Store optimization results
        self.threshold_optimization_results = {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'f1_scores': f1_scores,
            'best_threshold': best_threshold,
            'precision_at_best': precision_at_best,
            'recall_at_best': recall_at_best,
            'f1_at_best': f1_at_best,
            'g_mean': g_mean_at_best,
            'optimization_time': time.time() - start_time,
            'f1_threshold': f1_threshold,
            'f1_best': best_f1,
            'precision_target': precision_target
        }
        
        # Print detailed results
        print("\nThreshold Optimization Results:")
        print(f"F1-optimized threshold: {f1_threshold:.4f}")
        print(f"Precision-priority threshold: {best_threshold:.4f}")
        print(f"Target precision: {precision_target:.4f}")
        print(f"Achieved precision: {precision_at_best:.4f}")
        print(f"Achieved recall: {recall_at_best:.4f}")
        print(f"Achieved F1 Score: {f1_at_best:.4f}")
        print(f"Achieved G-Mean: {g_mean_at_best:.4f}")
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
        if ExperimentConfig.Ensemble.OPTIMIZE_THRESHOLD:  # Changed from self.optimize_threshold
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

    def log_experiment_params(self, tracker: Any) -> None:
        """
        Log ensemble-specific parameters and metadata
        
        Args:
            tracker: ExperimentTracker instance
        """
        print("\nLogging base parameters...")
        super().log_experiment_params(tracker)
        
        print("Logging ensemble-specific parameters...")
        # Log basic ensemble config
        tracker.log_parameters({
            'experiment_type': 'ensemble',
            'combination_method': 'probability_averaging',
            'techniques_used': ','.join(self.technique_experiments.keys()),
            'initial_threshold': self.decision_threshold,
            'threshold_optimization': ExperimentConfig.Ensemble.OPTIMIZE_THRESHOLD
        })

        print("Logging technique weights...")
        
        # Log technique weights
        for technique, weight in self.technique_weights.items():
            if technique in self.technique_experiments:
                tracker.log_parameters({
                    f'weight_{technique}': weight
                })
        
        # Log preprocessing metadata if available
        if hasattr(self, 'current_data') and 'ensemble_metadata' in self.current_data:
            metadata = self.current_data['ensemble_metadata']
            tracker.log_parameters({
                'preprocessing_time': metadata['processing_time'],
                'preprocessing_memory': metadata['peak_memory_usage']
            })
        
        # Log threshold optimization results if available
        if self.threshold_optimization_results:
            tracker.log_parameters({
                'optimized_threshold': self.threshold_optimization_results['best_threshold'],
                'threshold_f1': self.threshold_optimization_results['best_f1'],
                'threshold_g_mean': self.threshold_optimization_results['g_mean'],
                'optimization_time': self.threshold_optimization_results['optimization_time']
            })

    def log_training_metadata(self, tracker: Any, training_metadata: Dict[str, Any]) -> None:
        """
        Log training-related metadata for all techniques
        
        Args:
            tracker: ExperimentTracker instance
            training_metadata: Dictionary containing training information
        """
        # Log individual technique metrics
        for technique, metadata in training_metadata.items():
            if technique != 'ensemble':
                tracker.log_metrics({
                    f'{technique}_training_time': metadata['training_time'],
                    f'{technique}_memory_usage': metadata['memory_usage']
                })
        
        # Log ensemble totals
        ensemble_metadata = training_metadata['ensemble']
        tracker.log_metrics({
            'total_training_time': ensemble_metadata['total_training_time'],
            'peak_memory_usage': ensemble_metadata['peak_memory_usage']
        })

    def run_experiment(self, data_path: str) -> Dict[str, Any]:
        """
        Run multiple training iterations and aggregate results
        
        Args:
            data_path: Path to the dataset
            
        Returns:
            Dictionary containing aggregated metrics
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at: {data_path}")
        
        print("\nStarting MLflow tracking setup...")
        
        with ExperimentTracker(self.experiment_name) as tracker:
            try:
                print("Logging experiment parameters...")
                # Log experiment parameters
                self.log_experiment_params(tracker)
                
                successful_runs = 0
                failed_runs = []
                self.metrics_list = []
                
                # Run multiple iterations
                for run in range(self.n_runs):
                    print(f"\nStarting Run {run + 1}/{self.n_runs}")
                    
                    try:
                        # Run single experiment iteration
                        results = self.run_single_experiment(data_path)
                        
                        # Store metrics for this run
                        self.metrics_list.append(results)
                        
                        # Log individual run metrics
                        run_metrics = {f"run_{run}_{k}": v for k, v in results.items()
                                     if k != 'curves'}
                        tracker.log_metrics(run_metrics)
                        
                        # Log visualizations for individual run
                        if ExperimentConfig.SAVE_PLOTS:
                            tracker.log_visualization_artifacts(
                                metrics=results,
                                metrics_list=None,
                                prefix=f"run_{run}_"
                            )
                        
                        successful_runs += 1
                        
                    except Exception as e:
                        failed_runs.append((run, str(e)))
                        print(f"Run {run + 1} failed: {str(e)}")
                        continue
                    
                # Check for enough successful runs
                if successful_runs < self.n_runs * 0.5:
                    raise RuntimeError(
                        f"Too many failed runs. Only {successful_runs}/{self.n_runs} "
                        f"completed successfully. Failed runs: {failed_runs}"
                    )
                
                # Calculate aggregate metrics
                if len(self.metrics_list) > 0:
                    agg_metrics = self._aggregate_metrics()
                    tracker.log_metrics(agg_metrics)
                    
                    # Create final visualizations
                    if ExperimentConfig.SAVE_PLOTS:
                        final_metrics = self.metrics_list[-1].copy()
                        tracker.log_visualization_artifacts(
                            metrics=final_metrics,
                            metrics_list=self.metrics_list,
                            prefix="final_"
                        )
                    
                    # Log failed runs info
                    if failed_runs:
                        tracker.log_parameters({
                            "failed_runs": str(failed_runs),
                            "successful_runs": successful_runs
                        })
                    
                    return agg_metrics
                
                raise RuntimeError("No successful runs completed")
                
            except Exception as e:
                print(f"Experiment failed: {str(e)}")
                raise

    def _get_results_header(self) -> str:
        """Override header for ensemble results"""
        return "Ensemble Experiment Results"

def main():
    """Run the Ensemble experiment"""

    # Check MLflow directory
    mlflow_dir = "./mlruns"
    if not os.path.exists(mlflow_dir):
        os.makedirs(mlflow_dir)
        print(f"Created MLflow directory at {mlflow_dir}")
    
    experiment = EnsembleExperiment()
    
    try:
        results = experiment.run_experiment("data/creditcard.csv")
        experiment.print_results(results)
        
    except Exception as e:
        print(f"Experiment failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()