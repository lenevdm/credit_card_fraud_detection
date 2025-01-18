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
        # Set model configuration in model_config.py
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

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE)
        
        model.compile(
            optimizer=optimizer,
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
    
    def train(self, X_train, y_train, X_val, y_val, callbacks=None):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            callbacks: Optional list of callbacks
        """
        if callbacks is None:
            callbacks = []
            
        # Add early stopping callback
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.EARLY_STOPPING_PATIENCE,
                min_delta=self.config.EARLY_STOPPING_MIN_DELTA,
                restore_best_weights=True
            )
        )

        # Add learning rate scheduler (if enabled)
        if self.config.USE_LR_SCHEDULER:
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlatue(
                    monitor='val_los',
                    factor=self.config.LR_FACTOR,
                    patience=self.config.LR_PATIENCE,
                    min_lr=1e-6,
                    verbose=1
                )
            )
        
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
        from sklearn.metrics import (classification_report, confusion_matrix, 
                               f1_score, average_precision_score, roc_auc_score,
                               precision_recall_curve, roc_curve)
        import matplotlib.pyplot as plt
    
        # Get model predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int)
    
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
        # Calculate standard metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision_score_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_score_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = f1_score(y_test, y_pred)
    
        # Calculate ROC AUC and AUPRC
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        auprc = average_precision_score(y_test, y_pred_proba)
    
        # Calculate PR and ROC curves
        pr_curve_precision, pr_curve_recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
    
        # Store all metrics
        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision_score_val,
            'recall': recall_score_val,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'auprc': auprc,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'curves': {
                'pr': {
                    'precision': pr_curve_precision,
                    'recall': pr_curve_recall,
                    'thresholds': pr_thresholds
                },
                'roc': {
                    'fpr': fpr,
                    'tpr': tpr,
                    'thresholds': roc_thresholds
                }
            }
        }
    
        # Print detailed results
        print("\nDetailed Test Results:")
        print("-" * 40)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision_score_val:.4f}")
        print(f"Recall: {recall_score_val:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"PR AUC: {auprc:.4f}")
        print("\nConfusion Matrix:")
        print("-" * 40)
        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Positives: {tp}")
            
        return metrics_dict