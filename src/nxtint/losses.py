"""Loss functions for sequence prediction."""

import torch
from torch.nn import functional

from nxtint.utils.constants import MAX_INT
from nxtint.utils.logging import DEBUG, log, setup_logger

logger = setup_logger(__name__, level=DEBUG)


@log(logger, level=DEBUG)
def sequence_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.1,
    reduce: bool = True,
) -> torch.Tensor:
    """Calculate distance-weighted cross entropy loss.

    Args:
        logits: Raw model outputs of shape (batch_size, INT_N)
        targets: Target integers of shape (batch_size,)
        alpha: Weight factor for distance penalty (default: 0.1)
        reduce: Whether to return mean loss (default: True)

    Returns:
        torch.Tensor: loss values OR mean loss value
    """
    # Convert targets to class indices (shift from [-MAX_INT, MAX_INT] to [0, INT_N])
    target_classes = targets + MAX_INT

    # Calculate standard cross-entropy loss
    base_loss = functional.cross_entropy(logits, target_classes.long(), reduction="none")

    # Calculate predicted class indices
    pred_classes = torch.argmax(logits, dim=-1)

    # Calculate absolute distance between predicted and target values
    distance = torch.abs(pred_classes - target_classes)

    # Apply distance weighting
    weighted_loss = base_loss * (1 + alpha * distance)

    if reduce:
        # Return mean loss
        return weighted_loss.mean()
    return weighted_loss
