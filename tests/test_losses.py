"""Tests for loss functions."""

import torch

from nxtint.losses import sequence_loss


def test_sequence_loss():
    """Test cross entropy loss calculation."""
    # Create logits that strongly predict class 1 and class 2
    logits = torch.tensor(
        [
            [0.0, 10.0, 0.0, 0.0],  # Should predict class 1
            [0.0, 0.0, 10.0, 0.0],  # Should predict class 2
        ]
    )

    # Create matching targets
    targets = torch.tensor([1, 2])

    # Calculate loss
    loss = sequence_loss(logits, targets)

    # Loss should be very small since predictions are confident and correct
    assert loss.item() < 0.1

    # Test with incorrect targets
    wrong_targets = torch.tensor([0, 3])
    wrong_loss = sequence_loss(logits, wrong_targets)

    # Loss should be much larger for incorrect predictions
    assert wrong_loss.item() > 1.0
