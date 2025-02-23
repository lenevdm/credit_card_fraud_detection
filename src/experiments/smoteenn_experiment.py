"""SMOTE-ENN experiment implementation for credit card fraud detection"""


class SMOTEENNExperiment(BaseExperiment):
    """
    SMOTE-ENN experiment implementation.
    Applies SMOTE-ENN to training data only.
    """

    def __init__(self, n_runs: int = None):
        """Initialize SMOTE-ENN experiment"""
        super().__init__(
            experiment_name=ExperimentConfig.SMOTEENN.NAME,
            n_runs=n_runs
        )

    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply SMOTE-ENN to training data

        Args:
            data: Dictionary containing train/val/test splits

        Returns:
            Dictionary with resampled training data and original val/test data
        """