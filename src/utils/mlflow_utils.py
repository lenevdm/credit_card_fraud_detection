"""MLflow experiment tracking wrapper"""
import mlflow
from typing import Dict, Any, Optional, List
import tensorflow as tf
import matplotlib.pyplot as plt
from src.utils.visualization_utils import plot_metric_curves

class ExperimentTracker:
    """MLflow experiment tracking wrapper"""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters to MLflow"""
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics to MLflow - excluding 'curves'"""
        try:
            # Log scalar metrics
            metrics_to_log = {k: v for k, v in metrics.items() 
                            if k != 'curves' and isinstance(v, (int, float))}
            mlflow.log_metrics(metrics_to_log, step=step)

        except Exception as e:
            print(f"Warning: Error during metric logging: {str(e)}")

    def log_visualization_artifacts(self, metrics: Dict[str, Any], metrics_list: Optional[List[Dict]] = None) -> None:
        """
        Create and log visualization plots as artifacts
        
        Args:
            metrics: Dictionary containing current run metrics
            metrics_list: Optional list of metrics from all runs
        """
        try:
            # Generate plots
            pr_fig, roc_fig, cm_fig, additional_figs = plot_metric_curves(
                metrics, 
                metrics_list if metrics_list is not None else []
            )

            # Log each figure
            mlflow.log_figure(pr_fig, "precision_recall_curve.png")
            mlflow.log_figure(roc_fig, "roc_curve.png")
            mlflow.log_figure(cm_fig, "confusion_matrix.png")
            
            if additional_figs:
                for name, fig in additional_figs.items():
                    mlflow.log_figure(fig, f"{name}.png")
                    plt.close(fig)

            # Close main figures
            plt.close(pr_fig)
            plt.close(roc_fig)
            plt.close(cm_fig)

            # Log curve data as JSON for later reference
            if 'curves' in metrics:
                mlflow.log_dict(metrics['curves'], "curve_data.json")
                
        except Exception as e:
            print(f"Warning: Error during visualization artifact logging: {str(e)}")
    
    def create_keras_callback(self) -> tf.keras.callbacks.Callback:
        """Create a Keras callback for logging during training"""
        class MLFlowCallback(tf.keras.callbacks.Callback):
            def __init__(self, parent):
                super().__init__()
                self.parent = parent

            def on_epoch_end(self, epoch, logs=None):
                if logs:
                    # Log training metrics
                    self.parent.log_metrics(logs, step=epoch)

                    # Create and log learning curves periodically
                    if epoch % 5 == 0:  # Log every 5 epochs
                        plt.figure(figsize=(10, 6))
                        metrics = ['loss', 'val_loss']
                        has_data = False
                        
                        for metric in metrics:
                            if metric in self.model.history.history:
                                plt.plot(
                                    self.model.history.history[metric],
                                    label=metric
                                )
                                has_data = True
                                
                        if has_data:
                            plt.title('Learning Curves')
                            plt.xlabel('Epoch')
                            plt.ylabel('Loss')
                            plt.legend()
                            plt.grid(True)
                            
                            # Log the learning curve
                            mlflow.log_figure(
                                plt.gcf(),
                                f"learning_curve_epoch_{epoch}.png"
                            )
                            plt.close()
        
        return MLFlowCallback(self)
    
    def __enter__(self):
        """Start MLflow run when entering context"""
        mlflow.start_run(run_name=self.experiment_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End MLflow run when exiting context"""
        mlflow.end_run()