# Credit Card Fraud Detection with Deep Learning

This project explores techniques for handling class imbalanced data in deep learning models for credit card fraud detection. Using the European Credit Card dataset from 2013, we systematically compare different methods for addressing class imbalance and analyze their impact on model performance.

## Project Structure

```mermaid
graph TD
    A[credit_card_fraud_detection/] 
    A --> B[config/ <br> <i>Model configuration</i>]
    A --> C[data/ <br> <i>Dataset storage</i>]
    A --> D[notebooks/ <br> <i>Jupyter notebooks for analysis</i>]
    A --> E[src/ <br> <i>Source code</i>]
    E --> E1[data/ <br> <i>Data processing</i>]
    E --> E2[models/ <br> <i>Model implementations</i>]
    E --> E3[utils/ <br> <i>Utility functions</i>]
    A --> F[tests/ <br> <i>Unit tests</i>]

```

