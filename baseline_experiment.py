"""Run baseline experiment for credit card fraud detection"""
# Library imports
import tensorflow as tf
import numpy as np
import mlflow

# Module imports
from src.data.data_preparation import DataPreparation
from src.models.baseline_model import FraudDetectionModel
from src.utils.mlflow_utils import ExperimentTracker
from config.model_config import ModelConfig

def run_baseline_experiment():
    """Run baseline experiment with enhanced logging"""
    # Clean up any existing runs
    mlflow.end_run()

    # Set up MLflow for local tracking
    mlflow.set_tracking_uri("file:./mlruns")
    experiment_name = "fraud_detection"
    
    # Create experiment if it doesn't exist
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    # Set the experiment
    mlflow.set_experiment(experiment_name)

    
    # Prepare data
    print("Loading and preparing data...")
    data_prep = DataPreparation()
    data = data_prep.prepare_data('data/creditcard.csv')
    
    # Initialize model
    print("Initializing model...")
    model = FraudDetectionModel()
    
    # Train model with experiment tracking
    print("Training model...")
    with ExperimentTracker("baseline_experiment") as tracker:
        # Log model parameters
        tracker.log_parameters({
            "model_type": "baseline_mlp",
            "learning_rate": ModelConfig.LEARNING_RATE,
            "batch_size": ModelConfig.BATCH_SIZE,
            "dropout_rate": ModelConfig.DROPOUT_RATE
        })
        
        # Create MLflow callback
        mlflow_callback = tracker.create_keras_callback()
        
        # Train model
        history = model.train(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            callbacks=[mlflow_callback]
        )
        
        # Evaluate and log results
        print("Evaluating model...")
        test_results = model.evaluate(data['X_test'], data['y_test'])
        tracker.log_metrics(test_results)
        
    return test_results

if __name__ == "__main__":
    # Set random seeds for reproducibility
    tf.random.set_seed(ModelConfig.RANDOM_SEED)
    np.random.seed(ModelConfig.RANDOM_SEED)
    
    run_baseline_experiment()