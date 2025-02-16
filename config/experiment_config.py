
class ExperimentConfig:
    """Configuration settings for experiments"""

    # Base experiment settings
    BASE_EXPERIMENT_NAME = "fraud_detection"
    N_RUNS = 30
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
        RANDOM_STATE = 42

    class SMOTEENN:
        NAME = "fraud_detection_smoteenn"
        K_NEIGHBORS = 5
        RANDOM_STATE = 42

    class ClassWeight:
        NAME = "fraud_detection_classweight"

    