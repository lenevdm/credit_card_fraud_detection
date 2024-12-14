# Credit Card Fraud Detection with Deep Learning

This project explores techniques for handling class imbalanced data in deep learning models for credit card fraud detection. Using the European Credit Card dataset from 2013, we systematically compare different methods for addressing class imbalance and analyze their impact on model performance.

## Research Overview

The project aims to systematically evaluate five different class balancing techniques for credit card fraud detection:
- SMOTE (Synthetic Minority Oversampling Technique)
- Random Undersampling
- SMOTE-ENN
- Class Weight Minimization 
- Ensemble Method (combining SMOTE, class weights, and binary cross-entropy loss)

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
├── config/               # Model configuration
├── data/                # Dataset storage
├── notebooks/           # Jupyter notebooks for analysis
├── src/                 # Source code
│   ├── data/           # Data processing
│   ├── models/         # Model implementations
│   └── utils/          # Utility functions
└── tests/              # Unit tests
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

2. Run baseline model:
```bash
python main.py
```

View results at http://localhost:5000

## Experiments

Experimental results and analysis will be tracked using MLflow. Each experiment includes:
- Model architecture details
- Training parameters
- Performance metrics
- Class balancing technique used

## Contributing

This is a research project for an undergraduate Computer Science degree.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Lene van der Merwe
University of London
2024

## References

[Links to key papers and resources used in the project]