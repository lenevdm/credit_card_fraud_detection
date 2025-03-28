"""Base class for credit card fraud detection experiments"""
import numpy as np
from scipy import stats
import mlflow
import gc
import os
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from src.models.baseline_model import FraudDetectionModel
from src.data.data_preparation import DataPreparation
from src.utils.mlflow_utils import ExperimentTracker
from config.model_config import ModelConfig
from config.experiment_config import ExperimentConfig
from src.utils.statistical_analysis import compare_techniques

class BaseExperiment(ABC):
    """Abstract base class for fraud detection experiments"""

    def __init__(self, experiment_name, n_runs=None):
        """
        Initialize experiment
        Args:
            experiment_name: Name for this experiment
            n_runs: Optional override for number of runs (default is 30 runs)
        """

        self.experiment_name = experiment_name
        self.n_runs = n_runs if n_runs is not None else ExperimentConfig.N_RUNS
        self.data_prep = DataPreparation()
        self.metrics_list = []  # Initialize empty metrics list

        # Clean up existing runs
        mlflow.end_run()

        # Set up MLflow tracking
        mlflow.set_tracking_uri("file:./mlruns")

        # Create experiment if it doesn't exist
        try: 
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        except:
            self.experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id

        mlflow.set_experiment(self.experiment_name)

        self.metrics_list = []

    @abstractmethod
    def preprocess_data(self, data):
        """
        Apply any technique-specific preprocessing
        Args:
            data: Dictionary containing train/val/test splits
        Returns:
            Preprocessed data dictionary
        """
        pass

    def run_experiment(self, data_path):
        """Run multiple training iterations and aggregate results"""

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at: {data_path}")
        
        with ExperimentTracker(self.experiment_name) as tracker:
            try:
                # Log experiment parameters
                self.log_experiment_params(tracker)

                successful_runs = 0
                failed_runs = []
                self.metrics_list = []  # Ensure metrics_list is initialized

                # Run multiple iterations
                for run in range(self.n_runs):
                    print(f"\nStarting Run {run + 1}/{self.n_runs}")

                    try:
                        # Prepare fresh data split
                        data = self.data_prep.prepare_data(data_path)

                        # Apply technique specific preprocessing
                        processed_data = self.preprocess_data(data)

                        # Initialize model
                        model = FraudDetectionModel()

                        # Create MLflow callback if plot saving is enabled
                        callbacks = []
                        if ExperimentConfig.SAVE_PLOTS:
                            callbacks.append(tracker.create_keras_callback())

                        # Train model
                        history = model.train(
                            processed_data['X_train'],
                            processed_data['y_train'],
                            processed_data['X_val'],
                            processed_data['y_val'],
                            callbacks=callbacks,
                            class_weight=processed_data.get('class_weights')
                        )

                        # Evaluate model
                        metrics = model.evaluate(
                            processed_data['X_test'],
                            processed_data['y_test']
                        )
                        self.metrics_list.append(metrics)

                        # Log individual run metrics and basic visualizations
                        run_metrics = {f"run_{run}_{k}": v for k, v in metrics.items()
                                    if k != 'curves'}
                        tracker.log_metrics(run_metrics)

                        # Log basic visualizations for individual run
                        if ExperimentConfig.SAVE_PLOTS:
                            tracker.log_visualization_artifacts(
                                metrics=metrics,
                                metrics_list=None,  # Don't pass metrics_list for individual runs
                                prefix=f"run_{run}_"  # Add prefix for individual run artifacts
                            )

                        successful_runs += 1

                    except Exception as e:
                        failed_runs.append((run, str(e)))
                        print(f"Run {run + 1} failed: {str(e)}")
                        continue

                    finally:
                        # Clear memory
                        if 'model' in locals():
                            del model
                        if 'processed_data' in locals():
                            del processed_data
                        gc.collect()

                # Check there are enough successful runs
                if successful_runs < self.n_runs * 0.5:
                    raise RuntimeError(
                        f"Too many failed runs. Only {successful_runs}/{self.n_runs} "
                        f"completed successfully. Failed runs: {failed_runs}"
                    )
                
                # Calculate aggregate metrics
                if len(self.metrics_list) > 0:
                    agg_metrics = self._aggregate_metrics()
                    tracker.log_metrics(agg_metrics)

                    # Create and log final visualizations using raw metrics_list
                    if ExperimentConfig.SAVE_PLOTS:
                        # Use the last run's metrics for curves, but full metrics_list for aggregated plots
                        final_metrics = self.metrics_list[-1].copy()
                        tracker.log_visualization_artifacts(
                            metrics=final_metrics,
                            metrics_list=self.metrics_list,
                            prefix="final_"  # Add prefix for final artifacts
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

    def log_experiment_params(self, tracker):
        """Log experiment-specific parameters with enhanced metadata"""
        tracker.log_parameters({
            "n_runs": self.n_runs,
            "model_architecture": "MLP",
            "input_dim": ModelConfig.INPUT_DIM,
            "hidden_layers": ModelConfig.HIDDEN_LAYERS,
            "confidence_level": ExperimentConfig.CONFIDENCE_LEVEL,
            "metrics_tracked": ExperimentConfig.METRICS_OF_INTEREST,
            "technique_name": self.__class__.__name__,
            "experiment_timestamp": pd.Timestamp.now().isoformat(),
            "data_split_ratio": {
                "train": 1 - ModelConfig.TEST_SIZE - ModelConfig.VAL_SIZE,
                "val": ModelConfig.VAL_SIZE,
                "test": ModelConfig.TEST_SIZE
            }
        })

    def _aggregate_metrics(self):
        """Calculate aggregate statistics across all runs"""
        # Extract scalar metrics
        scalar_metrics = {}
        for metrics in self.metrics_list:
            for k, v in metrics.items():
                if k != 'curves' and isinstance(v, (int, float)):
                    if k not in scalar_metrics:
                        scalar_metrics[k] =[]
                    scalar_metrics[k].append(v)

        # Calculate statistics
        aggregated = {}
        for metric, values in scalar_metrics.items():
            values = np.array(values)
            mean = np.mean(values)
            std = np.std(values)
            n = len(values)

            # Calculate confidence interval
            ci = stats.t.interval(
                ExperimentConfig.CONFIDENCE_LEVEL,
                df=n-1,  # degrees of freedom = sample size - 1
                loc=mean,
                scale=std/np.sqrt(len(values))
            )

            aggregated.update({
                f"{metric}_mean": mean,
                f"{metric}_std": std,
                f"{metric}_ci_lower": ci[0],
                f"{metric}_ci_upper": ci[1],
            })

        return aggregated

    def _calculate_confidence_interval(self, values, mean, std):
        """Calculate confidence interval with handling for zero std dev"""
        if std == 0:
            return mean, mean  # Return mean for both bounds when std dev is 0
        n = len(values)
        ci = stats.t.interval(
            ExperimentConfig.CONFIDENCE_LEVEL,
            df=n-1,
            loc=mean,
            scale=std/np.sqrt(n)
        )
        return ci
    
    def compare_with(
            self,
            other_experiment: 'BaseExperiment',
            metrics_of_interest: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Statistically compare this experiment with another experiment

        Args:
            other_experiment: Another BaseExperiment instance to compare against
            metrics_of_interest: Optional list of metrics to compare.
            If None, use ExperimentConfig.METRICS_OF_INTEREST

        Returns:
            Dictionary of statistical comparisons for each metric
        """
        if not metrics_of_interest:
            metrics_of_interest = ExperimentConfig.METRICS_OF_INTEREST

        if not hasattr(self, 'metrics_list') or not hasattr(other_experiment, 'metrics_list'):
            raise ValueError("Both experiments must have completed runs with metrics_list")
        
        return compare_techniques(
            self.__class__.__name__,
            self.metrics_list,
            other_experiment.__class__.__name__,
            other_experiment.metrics_list,
            metrics_of_interest
        )

    
    def print_results(self, results: Dict[str, float]) -> None:
        """
        Print formatted results from the experiment

        Args:
            results: Dictionary containing aggregated metrics
        """
        print(f"\n{self._get_results_header()}")
        print("-" * 40)

        # Print core metrics with confidence intervals
        metrics_to_display = ['accuracy', 'precision', 'recall', 'f1_score',
        'roc_auc', 'auprc', 'g_mean', 'mcc'
        ]

        for metric in metrics_to_display:
            print(f"\n{metric.upper()}:")
            print(f"Mean: {results[f'{metric}_mean']:.4f}")
            print(f"Std Dev: {results[f'{metric}_std']:.4f}")
            print(f"95% CI: [{results[f'{metric}_ci_lower']:.4f}, "
                f"{results[f'{metric}_ci_upper']:.4f}]")
            
        # Print resource usage
        print("\nRESOURCE USAGE:")
        print(f"Avg. Training Time: {results['training_time_mean']:.2f}s")
        print(f"Avg. Peak Memory: {results['peak_memory_usage_mean']:.2f}MB")
        
        # Print run statistics
        print("\nRUN STATISTICS:")
        print(f"Completed Runs: {len(self.metrics_list)}")
        print(f"Expected Runs: {self.n_runs}")

    def _get_results_header(self) -> str:
        """
        Get the header for results printing.
        Can be overridden by child classes.

        Returns:
            str: Header string for results
        """

        return "Experiment Results"

