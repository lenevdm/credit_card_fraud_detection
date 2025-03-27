# Credit Card Fraud Detection with Deep Learning

This project compares techniques for handling class imbalanced data in deep learning models for credit card fraud detection. Using the European Credit Card dataset from 2013, we systematically compare different methods for addressing class imbalance and analyze their impact on a baseline MLP model's performance.

Metrics are tracked and logged in MLflow.

## Research Overview

The project aims to systematically evaluate five different class balancing techniques for credit card fraud detection:
- SMOTE (Synthetic Minority Oversampling Technique)
- Random Undersampling
- SMOTE-ENN
- Class Weight Minimization 
- Ensemble Method

## Dataset

We use the European Credit Card dataset from 2013, containing:
- 284,807 transactions
- 492 fraudulent cases (0.17%)
- 28 principal components from PCA
- Time and amount features
- Binary class labels

## Project Structure

```
credit_card_fraud_detection/
├── src/
│   ├── __init__.py
│   ├── data/
│   │    __init__.py
│   │   └── data_preparation.py            # Handles data preprocessing
|   |── experiments/
│   |   ├── __init__.py
│   |   └── base_runs_final.py                 # Baseline experiment with no balancing
│   |   └── base_experiment.py                 # BaseExperiment abstract class to inherit from
│   |   └── class_weight_experiment.py         # Class weight adjustment
|   |   └── comparative_analysis.py            # Runs all experiments and performs statistical analysis and comparison
│   |   └── ensemble_experiment.py             # Ensembel (SMOTE-ENN, RUS, Class weighting plus baseline model)
|   |   └── random_undersampling_experiment.py # RUS undersampling
|   |   └── smote_experiment.py                # SMOTE oversampling
|   |   └── smoteenn_experiment.py             # SMOTE with cleanup
│   ├── models/
│   │   ├── __init__.py
│   │   └── baseline_model.py           # The MLP model
│   └── utils/
│       ├── __init__.py
│       ├── mlflow_utils.py             # Controls metric and artefact logging to MLflow server
|       ├── statistical_analysis.py     # Statistical analysis functions like paired t-tests, Cohen's d
│       └── visualization_utils.py      # Defines visualizations for model evaluation metrics that are generated for each run and logged in MLflow
├── config/
│   ├── __init__.py
│   ├── model_config.py                 # Model-specific configurations (learning rate, epochs, metrics of interest)
│   └── experiment_config.py            # Experiment-specific configs for each method
└── data/
|    └── creditcard.csv                 # Raw data
|
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    # Shared fixtures
│   ├── test_data_preparation.py       # Data pipeline tests
│   ├── test_baseline_model.py         # Model tests
│   ├── test_balancing_techniques.py   # Tests for SMOTE, RUS, etc.
│   └── test_evaluation_metrics.py     # Metrics calculation tests
```

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
- Download from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Place `creditcard.csv` in the `data/` directory

## Usage

1. Start MLflow tracking server:
```bash
mlflow ui
```

2. Run comparative analysis experiment to complete a run of all listed experiments and create comparative analysis results output:
```bash
python -m src.experiments.baseline_runs
```

3. Or run indivudual experiments:
```bash
python -m src.experiments.<experiment_file_name>
```

View results in MLflow at http://localhost:5000

## Experiments

Experimental results and analysis will be tracked using MLflow. Each experiment includes:
- Model architecture details
- Training parameters
- Performance metrics
- Class balancing technique used

## Contributing

This is a research project for an undergraduate Computer Science degree. Contributions by others are not possible.

## Author

Lene van der Merwe <br>
University of London <br>
2025

## References

[\[Bibliography\]](https://drive.google.com/file/d/1kh3wAVQSVbrqF2b-L_-s3N0tbcz4NoDj/view?usp=sharing)