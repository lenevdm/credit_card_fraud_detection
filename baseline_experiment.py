from src.data.data_preparation import DataPreparation
from src.models.baseline_model import FraudDetectionModel
from config.model_config import ModelConfig
import pandas as pd
import mlflow

def run_baseline_experiment():
    """
    Run baseline experiment without any class balancing techniques
    """
    # Prepare data
    print("Loading and preparing data...")
    data_prep = DataPreparation()
    data = data_prep.prepare_data('data/creditcard.csv')
    
    # Initialize model
    print("Initializing model...")
    model = FraudDetectionModel()
    
    # Train model
    print("Training model...")
    history = model.train(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        "baseline_no_balancing"
    )
    
    # Evaluate model
    print("Evaluating model...")
    test_results = model.evaluate(data['X_test'], data['y_test'])
    
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