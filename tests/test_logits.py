"""Tests for loss functions."""

import torch

from nxtint.logits import Logits
from nxtint.utils.config import Config


def test_sequence_loss_correct_predictions():
    """Test loss calculation for correct predictions."""
    # Create logits that strongly predict specific classes
    logits = Logits(torch.zeros((2, 2 * Config.gen.max_int)))
    logits.tensor[0, Config.gen.max_int + 1] = 100.0  # Predict +1
    logits.tensor[1, Config.gen.max_int + 2] = 100.0  # Predict +2

    # Create matching targets
    targets = torch.tensor([1, 2])

    # Calculate loss with different alpha values
    losses = [logits.loss(targets, alpha=alpha).mean().item() for alpha in (0.0, 0.1, 0.5)]

    # Loss should be very small since predictions are confident and correct
    for loss in losses:
        assert loss < 1e-10

    # Distance penalty shouldn't affect correct predictions
    for loss in losses[1:]:
        assert losses[0] == loss


def test_sequence_loss_distance_penalty():
    """Test that distance affects loss magnitude."""
    # Create logits for different prediction scenarios
    logits = Logits(torch.zeros((3, 2 * Config.gen.max_int + 1)))

    # Case i: Predict +distance
    for i, distance in enumerate([2, 10, 100]):
        logits.tensor[i, Config.gen.max_int + distance] = 100.0

    # All targets are +1
    targets = torch.tensor([1, 1, 1])

    # Calculate losses
    loss = logits.loss(targets).tolist()

    # Verify distance penalty increases loss
    assert loss[0] < loss[1] < loss[2]


def test_sequence_loss_alpha_scaling():
    """Test that alpha parameter scales distance penalty correctly."""
    # Create logits predicting +10 when target is +2
    logits = Logits(torch.zeros((1, 2 * Config.gen.max_int + 1)))
    logits.tensor[0, Config.gen.max_int + 10] = 100.0
    targets = torch.tensor([2])

    # Calculate loss with different alpha values
    losses = [logits.loss(targets, alpha=alpha).mean().item() for alpha in (0.0, 0.1, 0.5)]

    # Verify alpha scales penalty appropriately
    assert losses[0] < losses[1] < losses[2]


def test_accuracy():
    """Test accuracy calculation."""
    # Create logits that strongly predict specific classes
    logits = Logits(torch.zeros((3, 2 * Config.gen.max_int + 1)))
    logits.tensor[0, Config.gen.max_int + 1] = 100.0  # Predict +1
    logits.tensor[1, Config.gen.max_int + 2] = 100.0  # Predict +2
    logits.tensor[2, Config.gen.max_int + 3] = 100.0  # Predict +3

    # Test perfect accuracy
    perfect_targets = torch.tensor([1, 2, 3])
    assert logits.inaccuracy(perfect_targets) == 0.0

    # Test partial accuracy
    partial_targets = torch.tensor([1, 0, 3])  # Second prediction wrong
    assert 0.333 <= logits.inaccuracy(partial_targets) <= 0.334

    # Test zero accuracy
    wrong_targets = torch.tensor([-1, 0, 2])  # All predictions wrong
    assert logits.inaccuracy(wrong_targets) == 1.0
