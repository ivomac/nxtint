"""Tests for sequence generation module."""

import pytest
import torch

from nxtint.data.sequences import Sequence
from nxtint.utils.config import Config

# test linear sequence generation for different parameter sets


@pytest.mark.parametrize(
    "initial, constant, vector",
    [
        (5, 5, [0]),
        # (5, 5, [2, 1]),
        # (2, 2, [2, 1, 1, 1]),
    ],
)
def test_sequence_validity(initial, constant, vector, batch_size=10):
    """Test that generated sequences are valid."""
    generator = Sequence.linear(initial, constant, vector)
    x, y = generator.generate_batch(batch_size=batch_size)

    assert x.shape == (batch_size, Config.model.x_len)
    assert (x >= -Config.gen.max_int).all() and (x < Config.gen.max_int).all()
    assert (y >= -Config.gen.max_int).all() and (y < Config.gen.max_int).all()


def test_batch_generation(batch_size=50):
    """Test that batch generation works correctly."""
    generator = Sequence.linear(5, 5, [2, 1])

    # Generate multiple batches
    x1, _ = generator.generate_batch(batch_size=batch_size)
    x2, _ = generator.generate_batch(batch_size=batch_size)

    assert x1.shape == (batch_size, Config.model.x_len)
    assert x2.shape == (batch_size, Config.model.x_len)

    # Check that sequences are different between batches
    assert not torch.allclose(x1, x2)
