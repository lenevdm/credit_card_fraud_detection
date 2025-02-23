"""Random Undersampling experiment implementation for credit card fraud detection"""

from typing import Dict, Any
import time
import numpy as np
import psutil
import warnings
from imblearn.under_sampling import RandomUnderSampler

from src.experiments.base_experiment import BaseExperiment
from config.experiment_config import ExperimentConfig

class RandomUndersamplingExperiment(BaseExperiment):
    """Random Undersampling experiment implementation"""
    
    def __init__(self, n_runs: int = None):
        """Initialize Random Undersampling experiment"""
        super().__init__(
            experiment_name=ExperimentConfig.RandomUndersampling.NAME,
            n_runs=n_runs
        )

    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random undersampling to training data"""
        print("\nApplying Random Undersampling...")
        # Implementation here