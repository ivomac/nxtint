"""Tests for sequence generation module."""

import torch

from nxtint.data.sequences import FOSequence
from nxtint.utils.config import Config


def test_sequence_validity(batch_size=50):
    """Test that generated sequences are valid."""
    generator = FOSequence()
    x, y = generator.generate_batch(batch_size=batch_size)

    assert x.shape == (batch_size, Config.model.x_len)
    assert (x >= -Config.gen.max_int).all() and (x < Config.gen.max_int).all()
    assert (y >= -Config.gen.max_int).all() and (y < Config.gen.max_int).all()


def test_batch_generation(batch_size=50):
    """Test that batch generation works correctly."""
    generator = FOSequence()

    # Generate multiple batches
    x1, _ = generator.generate_batch(batch_size=batch_size)
    x2, _ = generator.generate_batch(batch_size=batch_size)

    assert x1.shape == (batch_size, Config.model.x_len)
    assert x2.shape == (batch_size, Config.model.x_len)

    # Check that sequences are different between batches
    assert not torch.allclose(x1, x2)
