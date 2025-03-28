
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
    PLOT_FREQUENCY = 10

    # Method-specific configurations
    class SMOTE:
        NAME = "fraud_detection_smote"
        K_NEIGHBORS = 7 #increase from 5 to get more diverse synthetic samples
        RANDOM_STATE = 42

    class RandomUndersampling:
        NAME = "fraud_detection_rus"
        SAMPLING_STRATEGY = 0.2 
        RANDOM_STATE = 42

    class SMOTEENN:
        NAME = "fraud_detection_smoteenn"
         # SMOTE parameters
        SAMPLING_STRATEGY = 0.4  # Try to maintain some imbalance (80% of majority class)
        K_NEIGHBORS = 8  # For SMOTE synthetic sample generation
        ENN_K_NEIGHBORS = 5  # For ENN cleaning
        N_JOBS = -1  # Parallel processing
        RANDOM_STATE = 42

    class ClassWeight:
        NAME = "fraud_detection_classweight"
        # Class weight calculator method: can be 'balanced', 'balanced_subsample', or a specific ratio
        WEIGHT_METHOD = 'balanced'
        # Or specify a fixed ratio (alternative approach)
        # WEIGHT_RATIO = 100  # Adjust based on empirical testing
        RANDOM_STATE = 42

    class Ensemble:
        NAME = "fraud_detection_ensemble"
        # Define which techniques to include in the ensemble
        TECHNIQUES = ['baseline', 'random_undersampling', 'smoteenn', 'class_weight']
        # Define technique weights for the weighted averaging
        TECHNIQUE_WEIGHTS = {
            'baseline': 5.0,  # Higher weight for baseline (better precision)
            'random_undersampling': 1.0,
            'smoteenn': 0.5,
            'class_weight': 0.5
        }
        # Default threshold for final classification
        DECISION_THRESHOLD = 0.7 # Increase to favour precision
        # Whether to optimize threshold using validation data
        OPTIMIZE_THRESHOLD = True
        RANDOM_STATE = 42

    