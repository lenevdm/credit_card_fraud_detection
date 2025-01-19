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
        
    def load_and_split_data(self, data_path):
        """
        Load data and perform train/val/test split before resampling
        
        Args:
            data_path: Path to the creditcard.csv file
            
        Returns:
            dict: Dictionary containing data splits and additional metadata
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

        # Calculate class distribution information
        class_dist = {
            'train': {
                'total': len(y_train),
                'fraud': (y_train == 1).sum(),
                'non_fraud': (y_train == 0).sum()
            },
            'val': {
                'total': len(y_val),
                'fraud': (y_val == 1).sum(),
                'non_fraud': (y_val == 0).sum()
            },
            'test': {
                'total': len(y_test),
                'fraud': (y_test == 1).sum(),
                'non_fraud': (y_test == 0).sum()
            }
        }

        # Print data split information
        self._print_split_info(class_dist)
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'class_distribution': class_dist,
            'feature_names': self.primary_features,
            'scaler': self.scaler # Include scaler for potential inverse transformations
        }
    
    def prepare_data(self, data_path):
        """
        Legacy method to maintain compatibility with existing code
        """

        return self.load_and_split_data(data_path)

        
    def _print_split_info(self, class_dist):    
        print("\nData split information:")
        print("-" * 40)
        for split_name, dist in class_dist.items():
            print(f"\n{split_name.captialize()} set:")
            print(f"Total samples: {dist['total']}")
            print(f"Non-fraudulent: {dist['non_fraud']}")
            print(f"Fraudulent: {dist['fraud']}")
            print(f"Fraud ratio: {dist['fraud']/dist['total']:.4&}")

        print(f"\nFeature dimentionality: {len(self.primary_features)}")
        print(f"Selected features: {','.join(self.primary_features)}")