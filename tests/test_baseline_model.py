"""Tests for baseline fraud detection model"""

import pytest
import numpy as np
import tensorflow as tf
from src.models.baseline_model import FraudDetectionModel
from config.model_config import ModelConfig

@pytest.fixture
def model():
    """Create model instance for testing"""
    return FraudDetectionModel()

@pytest.fixture
def sample_batch():
    """Create sample batch of data for testing"""
    np.random.seed(42)
    batch_size = 32
    n_features = ModelConfig.INPUT_DIM
    
    X = np.random.randn(batch_size, n_features)
    y = np.random.randint(0, 2, size=(batch_size, 1))
    return X, y

def test_model_initialization(model):
    """Test model builds with correct architecture"""
    # Check model attributes
    assert hasattr(model, 'model')
    assert hasattr(model, 'config')
    
    # Check layer structure
    layers = model.model.layers
    assert len(layers) == 7  # Input + 3 Dense/Dropout pairs + Output
    
    # Create test input and run a forward pass to build the model
    sample_input = tf.random.normal((1, ModelConfig.INPUT_DIM))
    _ = model.model(sample_input)
    
    # Check first layer's input dimension through the layer's weights shape
    assert layers[0].kernel.shape[0] == ModelConfig.INPUT_DIM
    
    # Check hidden layer sizes
    assert layers[0].units == 64  # First hidden layer
    assert layers[2].units == 32  # Second hidden layer
    assert layers[4].units == 16  # Third hidden layer
    
    # Check output layer
    assert layers[-1].units == 1
    assert layers[-1].activation.__name__ == 'sigmoid'
    
    # Check dropout rates
    dropout_layers = [layer for layer in layers if isinstance(layer, tf.keras.layers.Dropout)]
    for dropout_layer in dropout_layers:
        assert dropout_layer.rate == model.config.DROPOUT_RATE

def test_model_forward_pass(model, sample_batch):
    """Test model can perform forward pass"""
    X, _ = sample_batch
    
    # Get predictions
    predictions = model.model.predict(X, verbose=0)
    
    # Check predictions shape and range
    assert predictions.shape == (len(X), 1)
    assert np.all((predictions >= 0) & (predictions <= 1))

def test_model_training(model, sample_batch):
    """Test model can be trained for one epoch"""
    X, y = sample_batch
    X_val, y_val = sample_batch  # Using same data for simplicity
    
    history = model.train(
        X, y,
        X_val, y_val,
        class_weight=None,
        callbacks=None
    )
    
    # Check training history contains expected metrics
    assert 'loss' in history.history
    assert 'val_loss' in history.history
    
    # Check metrics are being tracked
    assert 'accuracy' in history.history
    assert 'precision' in history.history
    assert 'recall' in history.history
    assert 'auc' in history.history

def test_model_evaluation(model, sample_batch):
    """Test model evaluation returns expected metrics"""
    X, y = sample_batch
    
    metrics = model.evaluate(X, y)
    
    # Check all expected metrics are present
    expected_metrics = [
        'accuracy', 'precision', 'recall', 'f1_score',
        'roc_auc', 'auprc', 'g_mean', 'mcc',
        'training_time', 'peak_memory_usage'
    ]
    
    for metric in expected_metrics:
        assert metric in metrics
    
    # Check curves data is present
    assert 'curves' in metrics
    assert 'pr' in metrics['curves']
    assert 'roc' in metrics['curves']

def test_class_weight_application(model, sample_batch):
    """Test model handles class weights correctly"""
    X, y = sample_batch
    X_val, y_val = sample_batch
    
    # Define class weights
    class_weight = {0: 1.0, 1: 10.0}
    
    # Train with class weights
    history = model.train(
        X, y,
        X_val, y_val,
        class_weight=class_weight,
        callbacks=None
    )
    
    # Verify training proceeded without errors
    assert 'loss' in history.history
    assert len(history.history['loss']) > 0

@pytest.mark.parametrize("invalid_shape", [
    (32, ModelConfig.INPUT_DIM + 1),  # Too many features
    (32, ModelConfig.INPUT_DIM - 1),  # Too few features
])
def test_model_input_validation(model, invalid_shape):
    """Test model handles invalid input shapes appropriately"""
    invalid_input = np.random.randn(*invalid_shape)
    invalid_target = np.random.randint(0, 2, size=(invalid_shape[0], 1))
    
    with pytest.raises(ValueError):
        model.model.predict(invalid_input, verbose=0)

def test_model_reproducibility(sample_batch):
    """Test model training is reproducible with same seed"""
    X, y = sample_batch
    X_val, y_val = sample_batch

    def train_model_with_seeds():
        # Set all relevant random seeds
        np.random.seed(42)
        tf.random.set_seed(42)
        tf.keras.utils.set_random_seed(42)
        
        model = FraudDetectionModel()
        history = model.train(
            X, y, 
            X_val, y_val,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=2,
                    restore_best_weights=True
                )
            ]
        )
        return history.history['loss'][0]  # Compare just first epoch loss

    # Train two models with same seeds
    loss1 = train_model_with_seeds()
    loss2 = train_model_with_seeds()
    
    # Check losses match with some tolerance
    assert np.allclose(loss1, loss2, rtol=1e-5)