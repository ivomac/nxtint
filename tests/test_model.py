"""Tests for the sequence transformer model."""

import pytest
import torch

from nxtint.model import SequenceTransformer
from nxtint.utils.constants import MAX_INT


def test_model_forward():
    """Test basic forward pass of the model."""
    # Create model with default 64-dim embeddings
    model = SequenceTransformer(d_model=64, n_heads=4, d_ff=256)

    # Create sample batch
    batch_size = 3
    x = torch.randint(-MAX_INT, MAX_INT, (batch_size, 8))

    # Forward pass
    logits = model(x)

    # Check output shape
    assert logits.shape == (batch_size, 256)

    # Check output range (logits should be finite)
    assert torch.isfinite(logits).all()

    return


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
    assert (predictions >= -MAX_INT).all() and (predictions < MAX_INT).all()

    return


@pytest.mark.skipif(
    not (torch.cuda.is_available() or hasattr(torch.backends, "mps")),
    reason="No GPU (CUDA or ROCm) available",
)
def test_model_gpu():
    """Test model on GPU if available."""
    # Create model with GPU device
    model = SequenceTransformer(device="cuda")
    assert next(model.parameters()).device.type == "cuda"

    # Create sample batch
    batch_size = 3
    x = torch.randint(-MAX_INT, MAX_INT, (batch_size, 8))

    # Forward pass
    logits = model(x)
    assert logits.device.type == "cuda"

    # Check output shape
    assert logits.shape == (batch_size, 256)

    # Get predictions
    predictions = model.predict(x)
    assert predictions.device.type == "cuda"

    # Move to CPU for assertions
    predictions = predictions.cpu()
    assert (predictions >= -MAX_INT).all() and (predictions < MAX_INT).all()

    return
