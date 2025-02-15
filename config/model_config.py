class ModelConfig:
    """
    Configuration settings for the fraud detection model
    """
    # Network architecture
    INPUT_DIM = 10 # Updated from 29 through feature selection process
    HIDDEN_LAYERS = [64, 32, 16]
    DROPOUT_RATE = 0.3
    
    # Training parameters
    LEARNING_RATE = 0.0001
    USE_LR_SCHEDULER = True  # Add learning rate scheduling
    LR_PATIENCE = 5         # Patience for LR reduction
    LR_FACTOR = 0.5        # Factor to reduce LR by
    BATCH_SIZE = 64        # Larger batch size for more stable gradient
    MAX_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10  # More patience to find better minima
    EARLY_STOPPING_MIN_DELTA = 0.0001  # Minimum change to qualify as improvement
    
    # Data split ratios
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    
    # Random seed for reproducibility
    RANDOM_SEED = 42