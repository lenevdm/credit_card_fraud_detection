"""Base class for credit card fraud detection experiments"""
import numpy as np
from scipy import stats
import mlflow
import gc
import os
from abc import ABC, abstractmethod

from src.models.baseline_model import FraudDetectionModel
from src.data.data_preparation import DataPreparation
from src.utils.mlflow_utils import ExperimentTracker
from config.model_config import ModelConfig
from config.experiment_config import ExperimentConfig

class BaseExperiment(ABC):
    """Abstract base class for fraud detection experiments"""

    def __init__(self, experiment_name, n_runs=None):
        """
        Initialize experiment
        Args:
            experiment_name: Name for this experiment
            n_runs: Optional override for number of runs (default is 30 runs)
        """

        self.experiment_name = experiment_name
        self.n_runs = n_runs if n_runs is not None else ExperimentConfig.N_RUNS
        self.data_prep = DataPreparation()
        self.metrics_list = []

        # Clean up existing runs
        mlflow.end_run()

        # Set up MLflow tracking
        mlflow.set_tracking_uri("file:./mlruns")

        # Create experiment if it doesn't exist
        try: 
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        except:
            self.experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id

        mlflow.set_experiment(self.experiment_name)

    @abstractmethod
    def preprocess_data(self, data):
        """
        Apply any technique-specific preprocessing
        Args:
            data: Dictionary containing train/val/test splits
        Returns:
            Preprocessed data dictionary
        """
        pass

    