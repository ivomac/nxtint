"""Tests for sequence generation module."""

import torch

from nxtint.data.sequences import SequenceGenerator
from nxtint.utils.constants import MAX_INT


def test_sequence_validity():
    """Test that generated sequences are valid."""
    generator = SequenceGenerator(seq_length=8)
    batch = generator.generate_batch(batch_size=10)

    assert batch.shape == (10, 8)
    assert (batch >= -MAX_INT).all() and (batch < MAX_INT).all()


def test_batch_generation():
    """Test that batch generation works correctly."""
    generator = SequenceGenerator(seq_length=8)

    # Generate multiple batches
    batch1 = generator.generate_batch(batch_size=50)
    batch2 = generator.generate_batch(batch_size=50)

    assert batch1.shape == (50, 8)
    assert batch2.shape == (50, 8)

    # Check that sequences are different between batches
    assert not torch.allclose(batch1, batch2)
