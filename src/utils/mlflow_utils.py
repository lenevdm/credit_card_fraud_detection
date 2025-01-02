import mlflow
from typing import Dict, Any
import tensorflow as tf
import matplotlib.pyplot as plt
from src.utils.visualisation_utils import plot_metric_curves # our own visualisation utility

class ExperimentTracker:
    """MLflow experiment tracking wrapper"""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        
    def log_parameters(self, params: Dict[str, Any]):
        """Log multiple parameters to MLflow"""
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log multiple metrics to MLflow - exluding 'curves'"""

        # Log scalar metrics
        metrics_to_log = {k: v for k, v in metrics.items()
                          if k != 'curves' and isinstance(v, (int,float))}
        mlflow.log_metrics(metrics_to_log)

        # Create and log visualization artifacts
        if 'curves' in metrics:
            self._log_visualization_artifacts(metrics)

    # mlflow.log_metrics(metrics, step=step)
    
    def log_visualitzation_artifacts(self, metrics: Dict):
        """Create and log visualization plots as artifacts"""

        # Generate plots
        pr_fig, roc_fig, cm_fig = plot_metric_curves(metrics)

        # Log each figure
        mlflow.log_figure(pr_fig, "precision_recall_curve.png")
        mlflow.log_figure(roc_fig, "roc_curve.png")
        mlflow.log_figure(cm_fig, "confusion_matrix.png")

        # Close figures to free memory
        plt.close(pr_fig)
        plt.close(roc_fig)
        plt.close(cm_fig)

        # Log curbe data as JSON for later
        if 'curves' in metrics:
            mlflow.log_dict(metrics['curves'], "curve_data.json")
    
    def create_keras_callback(self):
        """Create a Keras callback for logging during training"""
        class MLFlowCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs:
                    # Log training metrics
                    mlflow.log_metrics(logs, step=epoch)

                    # Create and log learning curves periodically
                    if epoch % 5 == 0: # Log every 5 epochs
                        plt.figure(figsize=(10,6))
                        metrics = ['loss', 'val_loss']
                        for metric in metrics:
                            if metric in self.model.history.history:
                                plt.plot(
                                    self.model.history.history[metric],
                                    label=metric
                                )
                        plt.title('Learning Curves')
                        plt.xlabel('Epoch')
                        plt.ylable('Loss')
                        plt.legend()
                        plt.grid(True)

                        # Log the learning curve
                        mlflow.log_figure(
                            plt.gcf(),
                            f"learning_curbe_epoch_{epoch}.png"
                        )
                        plt.close()
        
        return MLFlowCallback()
    
    def __enter__(self):
        """Start MLflow run when entering context"""
        mlflow.start_run(run_name=self.experiment_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End MLflow run when exiting context"""
        mlflow.end_run()