import numpy as np
from scipy import stats
import mlflow
from src.models.baseline_model import FraudDetectionModel
from src.data.data_preparation import DataPreparation
from src.utils.mlflow_utils import ExperimentTracker
from config.model_config import ModelConfig
import os
import gc
import logging
import warnings

class BaselineExperiment:
    def __init__(self, n_runs=None):
        """
        Initialize baseline experiment
        
        Args:
            n_runs: Optional override for number of runs. If None, uses ModelConfig.N_RUNS
        """
        self.n_runs = n_runs if n_runs is not None else ModelConfig.N_RUNS
        self.data_prep = DataPreparation()
        self.metrics_list = []

    def run_experiment(self, data_path):
        """Run multiple training iterations and aggregate results"""
    
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at: {data_path}")
            
        with ExperimentTracker("baseline_multiple_runs") as tracker:
            try:
                # Log experiment parameters
                tracker.log_parameters({
                    "n_runs": self.n_runs,
                    "model_architecture": "MLP",
                    "input_dim": ModelConfig.INPUT_DIM,
                    "hidden_layers": ModelConfig.HIDDEN_LAYERS,
                    "confidence_level": ModelConfig.CONFIDENCE_LEVEL,
                    "metrics_tracked": ModelConfig.METRICS_OF_INTEREST
                })
                
                successful_runs = 0
                failed_runs = []
                
                # Run multiple iterations
                for run in range(self.n_runs):
                    print(f"\nStarting Run {run + 1}/{self.n_runs}")
                    
                    try:
                        # Prepare fresh data split for each run
                        data = self.data_prep.prepare_data(data_path)
                        
                        # Initialize and train model
                        model = FraudDetectionModel()
                        
                        # Create MLflow callback if plot saving is enabled
                        callbacks = []
                        if ModelConfig.SAVE_PLOTS:
                            callbacks.append(tracker.create_keras_callback())
                        
                        # Train model
                        history = model.train(
                            data['X_train'], data['y_train'],
                            data['X_val'], data['y_val'],
                            callbacks=callbacks
                        )
                        
                        # Evaluate model
                        metrics = model.evaluate(data['X_test'], data['y_test'])
                        self.metrics_list.append(metrics)
                        
                        # Log individual run metrics
                        run_metrics = {f"run_{run}_{k}": v for k, v in metrics.items() 
                                    if k != 'curves'}
                        tracker.log_metrics(run_metrics)
                        
                        if ModelConfig.SAVE_PLOTS:
                            tracker.log_visualization_artifacts(metrics)
                            
                        successful_runs += 1
                        
                    except Exception as e:
                        failed_runs.append((run, str(e)))
                        print(f"Run {run + 1} failed: {str(e)}")
                        continue
                    
                    finally:
                        # Clear memory after each run
                        if 'model' in locals():
                            del model
                        if 'data' in locals():
                            del data
                        import gc
                        gc.collect()
                
                # Check if we have enough successful runs
                if successful_runs < self.n_runs * 0.5:  # Less than 50% successful
                    raise RuntimeError(
                        f"Too many failed runs. Only {successful_runs}/{self.n_runs} "
                        f"completed successfully. Failed runs: {failed_runs}"
                    )
                
                if len(self.metrics_list) > 0:
                    # Calculate and log aggregate metrics
                    agg_metrics = self._aggregate_metrics()
                    tracker.log_metrics(agg_metrics)
                    
                    # Log information about failed runs if any
                    if failed_runs:
                        tracker.log_parameters({
                            "failed_runs": str(failed_runs),
                            "successful_runs": successful_runs
                        })
                    
                    return agg_metrics
                else:
                    raise RuntimeError("No successful runs completed")
                    
            except Exception as e:
                print(f"Experiment failed: {str(e)}")
                raise RuntimeError("Some error - to fix")
        
    def _aggregate_metrics(self):
        """Calculate aggregate statistics across all runs"""
        # Extract scalar metrics (excludes curves)
        scalar_metrics = {}
        for metrics in self.metrics_list:
            for k, v in metrics.items():
                if k != 'curves' and isinstance(v, (int, float)):
                    if k not in scalar_metrics:
                        scalar_metrics[k] = []
                    scalar_metrics[k].append(v)

        # Calculate statistics
        aggregated = {}
        for metric, values in scalar_metrics.items():
            values = np.array(values)
            mean = np.mean(values)
            std = np.std(values)
            
            # Calculate confidence interval
            ci = stats.t.interval(
                ModelConfig.CONFIDENCE_LEVEL, 
                len(values)-1, 
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

def main():
    """Run the baseline experiment"""
    experiment = BaselineExperiment()
    results = experiment.run_experiment("data/creditcard.csv")
    
    # Print aggregate results
    print("\nAggregate Results:")
    print("-" * 40)
    for metric in ModelConfig.METRICS_OF_INTEREST:
        print(f"\n{metric.upper()}:")
        print(f"Mean: {results[f'{metric}_mean']:.4f}")
        print(f"Std Dev: {results[f'{metric}_std']:.4f}")
        print(f"95% CI: [{results[f'{metric}_ci_lower']:.4f}, "
              f"{results[f'{metric}_ci_upper']:.4f}]")

if __name__ == "__main__":
    main()