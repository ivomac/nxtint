import torch
import torch.nn as nn

from nxtint.utils.config import Config


class Logits(torch.Tensor):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def predict(self) -> torch.Tensor:
        """Get the most likely next number prediction.

        Returns:
            torch.Tensor: Predicted next integers of shape (batch_size,)
        """
        # Get logits and return argmax
        return self.argmax(dim=-1) - Config.gen.max_int

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
        base_loss = nn.functional.cross_entropy(self, target_classes.long(), reduction="none")

        # Calculate predicted class indices
        pred_classes = self.argmax(dim=-1)

        # Calculate absolute distance between predicted and target values
        distance = torch.abs(pred_classes - target_classes)

        # Apply distance weighting
        return base_loss * (1 + alpha * distance)
