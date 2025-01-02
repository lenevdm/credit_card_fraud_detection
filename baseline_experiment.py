# Import project modules
from src.data.data_preparation import DataPreparation # Preprocessing pipeline to load and process data
from src.models.baseline_model import FraudDetectionModel
from config.model_config import ModelConfig
# Other imports
import pandas as pd
import mlflow

def run_baseline_experiment():
    """
    Run baseline experiment without any class balancing techniques
    """
    # Prepare data
    print("Loading and preparing data...")
    data_prep = DataPreparation() # initialises data processing pipeline from data_prepartion
    data = data_prep.prepare_data('data/creditcard.csv') # passes prepared data into a dataframe
    
    # Initialize model
    print("Initializing model...")
    model = FraudDetectionModel() # gets model from baseline_model.py: FraudDetectionModel
    
    # Train model with experiment tracking
    print("Training model...")
    with ExperimentTracker("baseline_experiment") as tracker:
        # Log model params
        tracker.log_parameters({
            "model_type": "baseline_mlp",
            "learning_rate": ModelConfig.LEARNING_RATE,
            "batch_size": ModelConfig.BATCH_SIZE,
            "dropout_rate": ModelConfig.DROPOUT_RATE
        })
        

        # Train model
        history = model.train(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            "baseline_no_balancing" # experiment name for MLflow
        )
        
        # Evaluate model
        print("Evaluating model...")
        test_results = model.evaluate(data['X_test'], data['y_test']) # from baseline_model
        tracker.log_metrics(test_results)
        
        # Print results
        print("\nTest Results:")
        for metric, value in test_results.items():
            print(f"{metric}: {value:.4f}")
    
    return test_results

if __name__ == "__main__":
    # Set random seeds for reproducibility
    import tensorflow as tf
    import numpy as np
    
    tf.random.set_seed(ModelConfig.RANDOM_SEED)
    np.random.seed(ModelConfig.RANDOM_SEED)
    
    run_baseline_experiment()