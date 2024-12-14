import mlflow
from typing import Dict, Any
import tensorflow as tf

class ExperimentTracker:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        
    def log_parameters(self, params: Dict[str, Any]):
        """Log multiple parameters to MLflow"""
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log multiple metrics to MLflow"""
        mlflow.log_metrics(metrics, step=step)
    
    def create_keras_callback(self):
        """Create a Keras callback for logging during training"""
        class MLFlowCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs:
                    mlflow.log_metrics(logs, step=epoch)
        
        return MLFlowCallback()
    
    def __enter__(self):
        """Start MLflow run when entering context"""
        mlflow.start_run(run_name=self.experiment_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End MLflow run when exiting context"""
        mlflow.end_run()