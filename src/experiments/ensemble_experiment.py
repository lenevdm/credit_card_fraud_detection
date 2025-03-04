""" Ensemble experiment implementation using probability averaging from multiple techniques"""

from typing import Dict, Any, List, Optional, Tuple
import time
import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix

from src.experiments.base_experiment import BaseExperiment
from src.experiments.base_runs_final import BaselineExperimentFinal
from src.experiments.smote_experiment import SMOTEExperiment
from src.experiments.random_undersampling_experiment import RandomUndersamplingExperiment
from src.experiments.smoteenn_experiment import SMOTEENNExperiment
from src.experiments.class_weight_experiment import ClassWeightExperiment
from src.models.baseline_model import FraudDetectionModel
from config.experiment_config import ExperimentConfig

class EnsembleExperiment(BaseExperiment):
    """
    Ensemble experiment implementation that combines predictions from multiple class balancing techniques using weighted probability averaging.
    """

    def __init__(self, n_runs: int = None):
        """Initialize Ensemble experiment"""
        super().__init__(
            experiment_name=ExperimentConfig.Ensemble.NAME,
            n_runs=n_runs
        )

        # Initialize component elements
        self.technique_experiments = {
            'baseline': BaselineExperimentFinal(),
            'smote': SMOTEExperiment(),
            'random_undersampling': RandomUndersamplingExperiment(),
            'smoteenn': SMOTEENNExperiment(),
            'class_weight': ClassWeightExperiment()
        }

        # Filter to include only specified techniques
        self.technique_experiments = {
            k: v for k, v in self.technique_experiments.items()
            if k in ExperimentConfig.Ensemble.TECHNIQUES
        }

        self.technique_weights = ExperimentConfig.Ensemble.TECHNIQUE_WEIGHTS
        self.decision_threshold = ExperimentConfig.Ensemble.DECISION_THRESHOLD
        self.optimize_threshold = ExperimentConfig.Ensemble.OPTIMIZE_THRESHOLD
        self.threshold_metric = ExperimentConfig.Ensemble.THRESHOLD_METRIC

        # Storage for trained models and metadata
        self.models = {}
        self.threshold_optimization_results = None


    # def preprocess_data