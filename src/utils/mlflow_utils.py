"""MLflow experiment tracking wrapper with enhanced storage and retrieval"""
import mlflow
from typing import Dict, Any, Optional, List
import mlflow.tracking
import tensorflow as tf
import matplotlib.pyplot as plt
from src.utils.visualization_utils import plot_metric_curves

class ExperimentTracker:
    """MLflow experiment tracking wrapper"""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        
    def log_complete_results(self, results: Dict[str, Any], run_id: Optional[str] = None) -> None:
        """
        Log complete results including all metrics and metadata

        Args:
            results: Complete results dictionary
            run_id: Optional run identifier
        """
        # Store complete rusults as json artifacts
        results_for_storage = {
            k: v for k, v in results.items()
            if k != 'curves' # handle curves separately
        }

        mlflow.log_dict(results_for_storage, "complete_results.json")

        # Store curves data separately if present
        if 'curves' in results:
            mlflow.log_dict(results['curves'], "curves_data.json")

        # Log technique metadata
        if 'technique_metadata' in results:
            mlflow.log_dict(results['technique_metadata'], "technique_metadata.json")

    def get_results_for_techniques(self, technique_name: str) -> List[Dict[str, Any]]:
        """
        Retrieve all results for a specific technique

        Args:
            technique_name: Name of the specific technique

        Returns:
            List of result dictionaries for the technique
        """
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(technique_name)

        if not experiment:
            raise ValueError(f"No experiment found for technique: {technique_name}")
        
        all_results = []
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )

        for run in runs:
            try:
                # Get complete results from artifacts
                results_path = client.download_artifacts(run.info.run_id, "complete_results.json")
                with open(results_path, 'r') as f:
                    results = json.load(f)

                # Add run metadata
                results['run_id'] = run.info.run_id
                results['start_time'] = run.info.start_time

                all_results.append(results)
            except Exception as e:
                print(f"Warning: Could not load results for run {run.info.run_id}: {str(e)}")

        return all_results
    
    def get_run_metadata(self, run_id: str) -> Dict[str, Any]:
        """
        Get metadata for a specific run

        Args:
            run_id: MLflow run ID

        Returns: 
            Dictionary containing run metadata
        """
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)

        return {
            'run_id': run.info.run_id,
            'start_time': run.info.start_time,
            'end_time': run.info.end_time,
            'status': run.info.status,
            'parameters': run.data.params,
            'metrics': run.data.metrics
        }
    
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

    def log_visualization_artifacts(self, metrics: Dict[str, Any], metrics_list: Optional[List[Dict]] = None, prefix: str = "") -> None:
        """
        Create and log visualization plots as artifacts
        
        Args:
            metrics: Dictionary containing current run metrics
            metrics_list: Optional list of metrics from all runs
        """
        try:
            # Add debugging information
            print("\nVisualization Artifact Logging Debug Info:")
            print("-" * 50)
            print(f"Metrics list provided: {metrics_list is not None}")
            if metrics_list is not None:
                print(f"Metrics list length: {len(metrics_list)}")
                if len(metrics_list) > 0:
                    print("First metrics entry keys:", list(metrics_list[0].keys()))
            print(f"Current metrics keys:", list(metrics.keys()))
            print("-" * 50)

            # Generate plots
            pr_fig, roc_fig, cm_fig, additional_figs = plot_metric_curves(
                metrics, 
                metrics_list if metrics_list is not None else []
            )

            # Log each figure with prefix
            mlflow.log_figure(pr_fig, f"{prefix}precision_recall_curve.png")
            mlflow.log_figure(roc_fig, f"{prefix}roc_curve.png")
            mlflow.log_figure(cm_fig, f"{prefix}confusion_matrix.png")
            
            if additional_figs:
                for name, fig in additional_figs.items():
                    mlflow.log_figure(fig, f"{prefix}{name}.png")
                    plt.close(fig)

            # Close main figures
            plt.close(pr_fig)
            plt.close(roc_fig)
            plt.close(cm_fig)

            # Log curve data as JSON for later reference
            if 'curves' in metrics:
                mlflow.log_dict(metrics['curves'], f"{prefix}curve_data.json")
                
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