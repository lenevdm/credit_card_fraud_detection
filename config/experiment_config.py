
class ExperimentConfig:
    """Configuration settings for experiments"""

    # Base experiment settings
    BASE_EXPERIMENT_NAME = "fraud_detection"
    N_RUNS = 10
    CONFIDENCE_LEVEL = 0.95

    # Metrics to track
    METRICS_OF_INTEREST = [
        'accuracy',
        'precision',
        'recall',
        'f1_score',
        'roc_auc',
        'auprc',
        'g_mean',
        'mcc',
        'training_time',
        'peak_memory_usage'
    ]

    # Visualization settings
    SAVE_PLOTS = True
    PLOT_FREQUENCY = 5

    # Method-specific configurations
    class SMOTE:
        NAME = "fraud_detection_smote"
        K_NEIGHBORS = 5
        RANDOM_STATE = 42

    class RandomUndersampling:
        NAME = "fraud_detection_rus"
        SAMPLING_STRATEGY = 0.5  # Make classes equal
        RANDOM_STATE = 42

    class SMOTEENN:
        NAME = "fraud_detection_smoteenn"
         # SMOTE parameters
        SAMPLING_STRATEGY = 0.6  # Try to maintain some imbalance (80% of majority class)
        K_NEIGHBORS = 6  # For SMOTE synthetic sample generation
        ENN_K_NEIGHBORS = 7  # For ENN cleaning
        N_JOBS = -1  # Parallel processing
        RANDOM_STATE = 42

    class ClassWeight:
        NAME = "fraud_detection_classweight"
        # Class weight calculator method: can be 'balanced', 'balanced_subsample', or a specific ratio
        WEIGHT_METHOD = 'balanced'
        # Or specify a fixed ratio (alternative approach)
        # WEIGHT_RATIO = 100  # Adjust based on empirical testing

    