"""Tests for the sequence transformer model."""

import torch

from nxtint.model import SequenceTransformer


def test_model_forward():
    """Test basic forward pass of the model."""
    # Create model
    model = SequenceTransformer()

    # Create sample batch
    batch_size = 3
    x = torch.randint(0, 256, (batch_size, 8))

    # Forward pass
    logits = model(x)

    # Check output shape
    assert logits.shape == (batch_size, 256)

    # Check output range (logits should be finite)
    assert torch.isfinite(logits).all()


def test_model_predict():
    """Test model prediction."""
    # Create model
    model = SequenceTransformer()

    # Create sample batch
    batch_size = 3
    x = torch.randint(0, 256, (batch_size, 8))

    # Get predictions
    predictions = model.predict(x)

    # Check output shape
    assert predictions.shape == (batch_size,)

    # Check output range
    assert torch.all(predictions >= 0) and torch.all(predictions <= 255)
