import tensorflow as tf
from tensorflow.keras import layers, models
import mlflow
from src.utils.mlflow_utils import ExperimentTracker
from config.model_config import ModelConfig

class FraudDetectionModel:
    """
    Baseline deep learning model for credit card fraud detection.
    Architecture: MLP with 3 hidden layers (64 -> 32 -> 16 -> 1)
    """
    
    def __init__(self):
        self.config = ModelConfig()
        self.model = self._build_model()
        
    def _build_model(self):
        """Constructs the neural network architecture"""
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(self.config.INPUT_DIM,)),
            
            # Hidden layer 1
            layers.Dense(64, activation='relu'),
            layers.Dropout(self.config.DROPOUT_RATE),
            
            # Hidden layer 2
            layers.Dense(32, activation='relu'),
            layers.Dropout(self.config.DROPOUT_RATE),
            
            # Hidden layer 3
            layers.Dense(16, activation='relu'),
            layers.Dropout(self.config.DROPOUT_RATE),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc'),
                # Remove F1Score for now
            ]
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, experiment_name="baseline_model"):
        """
        Train the model with experiment tracking
        """
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True
            )
        ]
        
        with ExperimentTracker(experiment_name) as tracker:
            # Log model parameters
            tracker.log_parameters({
                "input_dim": self.config.INPUT_DIM,
                "hidden_layers": [64, 32, 16],
                "dropout_rate": self.config.DROPOUT_RATE,
                "learning_rate": self.config.LEARNING_RATE,
                "batch_size": self.config.BATCH_SIZE
            })
            
            # Add MLflow callback
            callbacks.append(tracker.create_keras_callback())
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=self.config.BATCH_SIZE,
                epochs=self.config.MAX_EPOCHS,
                callbacks=callbacks,
                verbose=1
            )
            
            return history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data and log detailed metrics
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Dictionary containing detailed evaluation metrics
        """
        import numpy as np
        from sklearn.metrics import classification_report, confusion_matrix, f1_score
        
        # Get model predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = f1_score(y_test, y_pred)
        
        # Store all metrics
        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
        
        # Print detailed results
        print("\nDetailed Test Results:")
        print("-" * 40)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nConfusion Matrix:")
        print("-" * 40)
        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Positives: {tp}")
        
        # Log to MLflow
        with ExperimentTracker("baseline_evaluation") as tracker:
            tracker.log_metrics(metrics_dict)
        
        return metrics_dict