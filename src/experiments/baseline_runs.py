import numpy as np
from scipy import stats
import mlflow
from src.models.baseline_model import FraudDetectionModel
from src.data.data_preparation import DataPreparation
from src.utils.mlflow_utils import ExperimentTracker
from config.model_config import ModelConfig

class BaselineExperiment:
    def __init__(self, n_runs=10):
        self.n_runs = n_runs
        self.data_prep = DataPreparation()
        self.metrics_list = []

    def run_experiment(self, data_path):
        """Run multiple training iterations and aggregate results"""

        with ExperimentTracker("baseline_multiple_runs") as tracker:
            # Log experiment parameters
            tracker.log_parameters({
                "n_runs": self.n_runs,
                "model_architecture": "MLP",
                "input_dim": ModelConfig.INPUT_DIM,
                "hidden_layers": ModelConfig.HIDDEN_LAYERS
            })

            # Run multiple iterations
            for run in range(self.n_runs):
                print(f"\nStarting Run {run + 1}/{self.n_runs}")

                # Prepare fresh data split for each run
                data = self.data_prep.prepare_data(data_path)

                # Initialize and train model
                model = FraudDetectionModel()
                history = model.train(
                    data['X_train'], 
                    data['y_train'],
                    data['X_val'], 
                    data['y_val']
                )

                # Evaluate model
                metrics = model.evaluate(data['X_test'], data['y_test'])
                self.metrics_list.append(metrics)

            # Calculate and log aggregate metrics
            agg_metrics = self._aggregate_metrics()
            tracker.log_metrics(agg_metrics)

            return agg_metrics
        
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
            ci = stats.t.interval(0.95, len(values)-1, loc=mean, scale=std/np.sqrt(len(values)))

            aggregated.update({
                f"{metric}_mean": mean,
                f"{metric}_std": std,
                f"{metric}_ci_lower": ci[0],
                f"{metric}_ci_upper": ci[1],
            })

        return aggregated

