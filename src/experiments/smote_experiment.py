"""SMOTE=based experiment implementation for credit card fraud detection. Inherits from base_experiment"""

from typing import Dict, Any
import time
from imblearn.over_sampling import SMOTE
import numpy as np

from src.experiments.base_experiment import BaseExperiment
from config.experiment_config import ExperimentConfig