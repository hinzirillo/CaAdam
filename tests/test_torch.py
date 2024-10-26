import os
import pytest
import numpy as np
import tempfile
import json

BACKEND = 'torch'
os.environ['KERAS_BACKEND'] = BACKEND

from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Input
from caadam import CaAdam
from caadam import (
    AdditiveMinMaxMedianConnectionScaling,
    MultiplicativeMinMaxMedianConnectionScaling,
    DepthConnectionScaling
)

# Generate synthetic data for testing
@pytest.fixture
def dummy_data():
    np.random.seed(42)
    X = np.random.random((100, 100))
    y = np.random.randint(0, 2, (100, 1))
    return X, y

def create_model():
    """Helper function to create a consistent model architecture"""
    return Sequential([
        Input(shape=(100,)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

@pytest.mark.parametrize("scaling_strategy", [
    AdditiveMinMaxMedianConnectionScaling(),
    MultiplicativeMinMaxMedianConnectionScaling(),
    DepthConnectionScaling()
])
def test_optimizer_fit(scaling_strategy, dummy_data):
    """Test if model can fit with different scaling strategies"""
    X, y = dummy_data
    
    # Create and compile model
    model = create_model()
    optimizer = CaAdam(scaling_strategy=scaling_strategy)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        X, y,
        epochs=5,
        batch_size=32,
        verbose=0
    )
    
    # Basic assertions to verify training occurred
    assert len(history.history['loss']) == 5
    assert all(not np.isnan(loss) for loss in history.history['loss'])
    assert history.history['loss'][-1] < history.history['loss'][0]

@pytest.mark.parametrize("scaling_strategy", [
    AdditiveMinMaxMedianConnectionScaling(),
    MultiplicativeMinMaxMedianConnectionScaling(),
    DepthConnectionScaling()
])
def test_optimizer_predictions(scaling_strategy, dummy_data):
    """Test if model can make predictions after training"""
    X, y = dummy_data
    
    # Create and train model
    model = create_model()
    optimizer = CaAdam(scaling_strategy=scaling_strategy)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(X, y, epochs=2, batch_size=32, verbose=0)
    
    # Make predictions
    predictions = model.predict(X, verbose=0)
    
    # Verify predictions
    assert predictions.shape == (100, 1)
    assert np.all((predictions >= 0) & (predictions <= 1))

@pytest.mark.parametrize("scaling_strategy", [
    AdditiveMinMaxMedianConnectionScaling(),
    MultiplicativeMinMaxMedianConnectionScaling(),
    DepthConnectionScaling()
])
def test_optimizer_gradient_updates(scaling_strategy, dummy_data):
    """Test if optimizer updates weights properly"""
    X, y = dummy_data
    
    # Create model
    model = create_model()
    optimizer = CaAdam(scaling_strategy=scaling_strategy)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Get initial weights
    initial_weights = [layer.get_weights() for layer in model.layers]
    
    # Train for one epoch
    model.fit(X, y, epochs=1, batch_size=32, verbose=0)
    
    # Get updated weights
    final_weights = [layer.get_weights() for layer in model.layers]
    
    # Verify weights were updated
    for initial, final in zip(initial_weights, final_weights):
        for i_w, f_w in zip(initial, final):
            assert not np.allclose(i_w, f_w)

@pytest.mark.parametrize("scaling_strategy", [
    AdditiveMinMaxMedianConnectionScaling(),
    MultiplicativeMinMaxMedianConnectionScaling(),
    DepthConnectionScaling()
])
def test_optimizer_serialization(scaling_strategy, dummy_data):
    """Test if model with optimizer can be saved and loaded correctly"""
    X, y = dummy_data
    
    # Create and compile original model
    original_model = create_model()
    original_optimizer = CaAdam(
        scaling_strategy=scaling_strategy,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999
    )
    original_model.compile(
        optimizer=original_optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the original model
    original_history = original_model.fit(
        X, y,
        epochs=3,
        batch_size=32,
        verbose=0
    )
    
    # Get predictions from original model
    original_predictions = original_model.predict(X, verbose=0)
    
    # Save the model with optimizer state
    with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as temp_model:
        save_model(original_model, temp_model.name)
        
        # Load the model with optimizer state
        loaded_model = load_model(temp_model.name)
    
    # Clean up temporary file
    os.unlink(temp_model.name)
    
    # Verify the loaded model has the same architecture
    assert len(loaded_model.layers) == len(original_model.layers)
    
    # Verify optimizer configuration is preserved
    original_config = original_model.optimizer.get_config()
    loaded_config = loaded_model.optimizer.get_config()
    assert original_config == loaded_config
    
    # Get predictions from loaded model
    loaded_predictions = loaded_model.predict(X, verbose=0)
    
    # Verify predictions are the same
    np.testing.assert_allclose(original_predictions, loaded_predictions, rtol=1e-5)
    
    # Continue training the loaded model
    continued_history = loaded_model.fit(
        X, y,
        epochs=2,
        batch_size=32,
        verbose=0
    )
    
    # Verify the loaded model can continue training
    assert len(continued_history.history['loss']) == 2
    assert all(not np.isnan(loss) for loss in continued_history.history['loss'])

@pytest.mark.parametrize("scaling_strategy", [
    AdditiveMinMaxMedianConnectionScaling(),
    MultiplicativeMinMaxMedianConnectionScaling(),
    DepthConnectionScaling()
])
def test_optimizer_config_serialization(scaling_strategy):
    """Test if optimizer configuration can be serialized to JSON"""
    # Create optimizer with custom configuration
    optimizer = CaAdam(
        scaling_strategy=scaling_strategy,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        ema_momentum=0.99
    )
    
    # Get optimizer config
    config = optimizer.get_config()
    
    # Verify config can be serialized to JSON
    try:
        json_config = json.dumps(config)
        restored_config = json.loads(json_config)
    except Exception as e:
        pytest.fail(f"Failed to serialize optimizer config: {e}")
    
    # Create new optimizer from config
    restored_optimizer = CaAdam.from_config(restored_config)
    
    # Verify restored optimizer has the same configuration
    assert optimizer.get_config() == restored_optimizer.get_config()

def test_invalid_scaling_strategy():
    """Test if invalid scaling strategy raises appropriate error"""
    with pytest.raises(Exception):
        CaAdam(scaling_strategy="invalid_strategy")