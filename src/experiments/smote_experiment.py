"""SMOTE=based experiment implementation for credit card fraud detection. Inherits from base_experiment"""

from typing import Dict, Any
import time
from imblearn.over_sampling import SMOTE
import numpy as np

from src.experiments.base_experiment import BaseExperiment
from config.experiment_config import ExperimentConfig

class SMOTEExperiment(BaseExperiment):
    """
    SMOTE-based experiment implemenation
    Applies SMOTE oversampling to training data only.
    """

    def __init__(self, n_runs: int = None)
        """Initialize SMOTE experiment"""
        super().__init__(
            experiment_name=ExperimentConfig.SMOTE.NAME,
            n_runs=n_runs
        )

    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply SMOTE oversampling to training data

        Args:
            data: Dictionary containing train/val/test splits

        Returns:
            Dictionary with resampled training data and original val/test data
        
        """
        