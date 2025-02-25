"""Class weight experiment implementation for credit card fraud detection"""

from typing import Dict, Any
import time
import numpy as np
import psutil
import warnings
from sklearn.utils.class_weight import compute_class_weight

from src.experiments.base_experiment import BaseExperiment
from config.experiment_config import ExperimentConfig

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

class ClassWeightExperiment(BaseExperiment):
    """
    Class Weight experiment implementation.
    Applies class weights during model training to address class imbalance.
    """

    def __init__(self, n_runs: int = None):
        """Initialize Class Weight experiment"""
        super().__init__(
            experiment_name=ExperimentConfig.ClassWeight.NAME,
            n_runs=n_runs
        )

    #def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:

    #def log_experiment_params(self, tracker: Any) -> None:

    #def _get_results_header(self) -> str:


def main():
    """Run the Class Weight experiment"""
    
    experiment = ClassWeightExperiment()
    
    try:
        results = experiment.run_experiment("data/creditcard.csv")
        experiment.print_results(results)
        
    except Exception as e:
        print(f"Experiment failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()