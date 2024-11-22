"""Tests for training utilities."""

import torch

from nxtint.model import SequenceTransformer
from nxtint.training import EarlyStopping, setup_training


def test_early_stopping():
    """Test early stopping behavior."""
    model = SequenceTransformer()
    stopper = EarlyStopping(patience=2, min_delta=0.1)

    # Test improvement case
    weights = stopper(model, 1.0)
    assert weights is None

    # Test no improvement but not stopping
    weights = stopper(model, 1.5)
    assert weights is None

    # Test stopping after patience exceeded
    weights = stopper(model, 1.5)
    assert weights is not None

    return


def test_optimizer_setup():
    """Test optimizer and scheduler setup."""
    model = SequenceTransformer()
    optimizer, scheduler = setup_training(
        model,
        lr=1e-3,
        warmup_steps=10,
        max_steps=100,
    )

    # Test optimizer type and parameters
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 1e-3
    assert optimizer.param_groups[0]["weight_decay"] == 0.01

    # Test scheduler type
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    return
