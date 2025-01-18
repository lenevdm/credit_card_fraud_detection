import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config.model_config import ModelConfig

class DataPreparation:
    """
    Handles loading and preprocessing of credit card fraud data
    """
    def __init__(self, random_state=None):
        self.random_state = random_state or ModelConfig.RANDOM_SEED
        self.scaler = StandardScaler()

        # Define selected features based on dual analysis
        self.primary_features = [
            'V14', 'V17', 'V12', 'V10', 'V3', 
            'V7', 'V4', 'V16', 'V11'
        ]
        
    def prepare_data(self, data_path):
        """
        Load and prepare data for training
        
        Args:
            data_path: Path to the creditcard.csv file
            
        Returns:
            dict: Dictionary containing train, validation, and test splits
        """
        # Load the data
        print("Loading data...")
        df = pd.read_csv(data_path)
        
        # Select features and scale Amount separately if needed
        print("Selecting features...")
        X = df[self.primary_features]
        y = df['Class']
        # feature_columns = ['V%d' % i for i in range(1,29)] + ['Amount']
        
        # Scale the features
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y,
            test_size=ModelConfig.TEST_SIZE,
            random_state=self.random_state,
            stratify=y
        )
        
        # Second split: separate validation set from training set
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=ModelConfig.VAL_SIZE,
            random_state=self.random_state,
            stratify=y_temp
        )
        
        # Reshape target variables to match expected dimensions
        y_train = np.expand_dims(y_train.values, axis=1)
        y_val = np.expand_dims(y_val.values, axis=1)
        y_test = np.expand_dims(y_test.values, axis=1)
        
        # Print data split information
        print("\nData split information:")
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Validation set size: {X_val.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        print(f"\nTraining set fraud distribution:")
        print(f"Non-fraudulent: {(y_train == 0).sum()}")
        print(f"Fraudulent: {(y_train == 1).sum()}")
        print(f"\nFeature dimensionality: {X_train.shape[1]}")
        print(f"Target shape: {y_train.shape}") # remove this?
        print(f"Selected features: {', '.join(self.primary_features)}")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }