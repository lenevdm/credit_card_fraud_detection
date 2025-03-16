import mlflow
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Set random seed for reproducibility
np.random.seed(42)

# Create some dummy data
X = np.random.randn(1000, 29)  # 29 features like our real dataset
y = np.random.randint(0, 2, 1000)  # Binary classification

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
with mlflow.start_run(run_name="mlflow_test"):
    # Log parameters
    mlflow.log_param("input_dim", 29)
    mlflow.log_param("hidden_units", 64)
    mlflow.log_param("learning_rate", 0.001)
    
    # Create a simple model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(29,)),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=5,
        verbose=1
    )
    
    # Log metrics
    for epoch in range(len(history.history['loss'])):
        mlflow.log_metrics({
            "train_loss": history.history['loss'][epoch],
            "train_accuracy": history.history['accuracy'][epoch],
            "val_loss": history.history['val_loss'][epoch],
            "val_accuracy": history.history['val_accuracy'][epoch]
        }, step=epoch)
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    
    # Log final test metrics
    mlflow.log_metrics({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    })