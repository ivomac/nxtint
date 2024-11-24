"""Tests for sequence generation module."""

import pytest
import torch

from nxtint.data.sequences import Sequence
from nxtint.utils.config import Config


@pytest.mark.parametrize(
    "initial, constant, vector",
    [
        (5, 5, [0]),
        (5, 5, [2, 1]),
        (2, 2, [2, 1, 1, 1]),
    ],
)
def test_sequence_validity(initial, constant, vector):
    """Test that generated sequences are valid."""
    sequence = Sequence.linear(initial, constant, vector)
    x, y = next(sequence)

    assert x.shape == (Config.train.batch_size, Config.model.x_len)
    assert (x >= -Config.gen.max_int).all() and (x < Config.gen.max_int).all()
    assert (y >= -Config.gen.max_int).all() and (y < Config.gen.max_int).all()


def test_sequence_iteration():
    """Test that sequence iteration works correctly."""
    sequence = Sequence.linear(10, 10, [3, 2, 1])

    # Get multiple batches via iteration
    x1, _ = next(sequence)
    x1 = x1.clone()
    x2, _ = next(sequence)

    assert x1.shape == (Config.train.batch_size, Config.model.x_len)
    assert x2.shape == (Config.train.batch_size, Config.model.x_len)

    # Check that sequences are different between iterations
    assert not torch.allclose(x1, x2)


@pytest.mark.parametrize(
    "initial, constant, matrix, shift",
    [
        ([1, 1], [1, 1], [[1, 1], [0, 2]], 0),
        ([6, 6], [4, 4], [[2, 3], [3, 2]], 10),
    ],
)
def test_sequence_coupled(initial, constant, matrix, shift):
    """Test that coupled sequences are generated correctly."""
    sequence = Sequence.coupled(initial, constant, matrix)
    x, y = next(sequence)

    assert x.shape == (Config.train.batch_size, Config.model.x_len)
    assert (x >= -Config.gen.max_int).all() and (x < Config.gen.max_int).all()
    assert (y >= -Config.gen.max_int).all() and (y < Config.gen.max_int).all()
