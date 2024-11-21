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
    assert logits.shape == (batch_size, 256), f"Got: {logits.shape}"

    # Check output range (logits should be finite)
    assert torch.isfinite(logits).all(), f"Got: {logits}"


def test_model_parameters():
    """Test model parameter initialization."""
    model = SequenceTransformer()

    # All parameters should be initialized
    for p in model.parameters():
        assert p.requires_grad, f"Got: {p.requires_grad}"
        assert torch.isfinite(p).all(), f"Got: {p}"
