"""Loss functions for sequence prediction."""

import torch
import torch.nn.functional as functional

from nxtint.utils.logging import DEBUG, log, setup_logger

logger = setup_logger(__name__, level=DEBUG)


@log(logger, level=DEBUG)
def sequence_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute cross entropy loss for sequence prediction.

    Args:
        logits: Predicted logits of shape (batch_size, num_classes)
        targets: Target class indices of shape (batch_size,)

    Returns:
        torch.Tensor: Mean cross entropy loss
    """
    return functional.cross_entropy(logits, targets)
