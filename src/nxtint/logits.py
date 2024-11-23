"""Handle logits output from the transformer model.

This module provides the Logits class which extends torch.Tensor to add
prediction and loss calculation functionality specifically for integer
sequence prediction.
"""

import torch
import torch.nn as nn

from .utils.config import Config
from .utils.logging import log_io, setup_logger

logger = setup_logger(__name__)


class Logits:
    """Handle model logits output.

    Inherits from torch.Tensor to provide specialized functionality for
    converting logits to predictions and calculating loss values for
    integer sequence prediction.

    Methods:
        predict: Convert logits to integer predictions
        accuracy: Calculate percentage of correct predictions
        loss: Calculate distance-weighted cross entropy loss
    """

    def __init__(self, logits: torch.Tensor):
        """Initialize the Logits object.

        Args:
            logits: Logits tensor of shape (batch_size, seq_len, num_classes)
        """
        self.tensor = logits
        return

    @log_io(logger)
    def predict(self) -> torch.Tensor:
        """Get the most likely next number prediction.

        Returns:
            torch.Tensor: Predicted next integers of shape (batch_size,)
        """
        # Get logits and return argmax
        return self.tensor.argmax(dim=-1) - Config.gen.max_int

    @log_io(logger)
    def accuracy(self, targets: torch.Tensor) -> float:
        """Calculate percentage of correct predictions.

        Args:
            targets: Target integers of shape (batch_size,)

        Returns:
            float: Percentage of correct predictions (0-100)
        """
        predictions = self.predict()
        correct = (predictions == targets).sum().item()
        return 100 * correct / targets.size(0)

    @log_io(logger)
    def loss(self, targets: torch.Tensor, alpha: float | None = None) -> torch.Tensor:
        """Calculate distance-weighted cross entropy loss.

        Args:
            targets: Target integers of shape (batch_size,)
            alpha: Distance weighting factor (default: Config.loss.alpha)

        Returns:
            torch.Tensor: loss values OR mean loss value
        """
        alpha = alpha if alpha is not None else Config.loss.alpha

        # Convert targets to class indices (shift from [-MAX_INT, MAX_INT] to [0, 2 * MAX_INT])
        target_classes = targets + Config.gen.max_int

        # Calculate standard cross-entropy loss
        base_loss = nn.functional.cross_entropy(
            self.tensor, target_classes.long(), reduction="none"
        )

        # Calculate predicted class indices
        pred_classes = self.tensor.argmax(dim=-1)

        # Calculate absolute distance between predicted and target values
        distance = torch.abs(pred_classes - target_classes)

        # Apply distance weighting
        return base_loss * (1 + alpha * distance)
