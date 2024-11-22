"""Tests for the sequence transformer model."""

import torch

from nxtint.model import SequenceTransformer
from nxtint.utils.config import GenConfig


def test_model_forward():
    """Test basic forward pass of the model."""
    # Create model
    model = SequenceTransformer()

    # Create sample batch
    batch_size = 3
    x = torch.randint(-GenConfig.max_int, GenConfig.max_int, (batch_size, 8))

    # Forward pass
    logits = model(x)

    # Check output shape
    assert logits.shape == (batch_size, GenConfig.max_int * 2)

    # Check output range (logits should be finite)
    assert torch.isfinite(logits).all()

    return


def test_model_predict():
    """Test model prediction."""
    # Create model
    model = SequenceTransformer()

    # Create sample batch
    batch_size = 3
    x = torch.randint(-GenConfig.max_int, GenConfig.max_int, (batch_size, 8))

    # Get predictions
    logits = model(x)
    predictions = logits.predict()

    # Check output shape
    assert predictions.shape == (batch_size,)

    # Check output range
    assert (predictions >= -GenConfig.max_int).all() and (predictions < GenConfig.max_int).all()

    return
