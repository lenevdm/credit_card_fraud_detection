class ModelConfig:
    """
    Configuration settings for the fraud detection model
    """
    # Network architecture
    INPUT_DIM = 29
    HIDDEN_LAYERS = [64, 32, 16]
    DROPOUT_RATE = 0.3
    
    # Training parameters
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    MAX_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 3
    
    # Data split ratios
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    # Experiment tracking
    EXPERIMENT_NAME = "fraud_detection"
    MODEL_VERSION = "v1.0"