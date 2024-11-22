"""Training utilities for sequence prediction model."""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from nxtint.utils.constants import INF
from nxtint.utils.logging import DEBUG, log, setup_logger

logger = setup_logger(__name__, level=DEBUG)


class EarlyStopping:
    """Early stopping handler based on validation loss.

    Attributes:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as an improvement
        best_loss: Best validation loss seen so far
        counter: Number of epochs without improvement
        best_weights: Copy of model weights with lowest validation loss
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.001) -> None:
        """Initialize early stopping handler.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = INF
        self.counter = 0
        self.best_weights = None
        return

    def __call__(self, model: torch.nn.Module, val_loss: float) -> dict[str, torch.Tensor] | None:
        """Check if training should stop and save best weights.

        Args:
            model: Current model
            val_loss: Current validation loss

        Returns:
            dict: Best model state dict if stopping, None otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            # New best loss found
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return None

        # No improvement
        self.counter += 1
        if self.counter >= self.patience:
            # Return best weights when stopping
            return self.best_weights
        return None


@log(logger, level=DEBUG)
def setup_training(
    model: torch.nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    warmup_steps: int = 5000,
    max_steps: int = 50000,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.CosineAnnealingLR]:
    """Set up optimizer and learning rate scheduler.

    Args:
        model: Model to train
        lr: Maximum learning rate after warmup
        weight_decay: AdamW weight decay
        warmup_steps: Number of warmup steps
        max_steps: Total number of training steps

    Returns:
        tuple: (optimizer, scheduler)
    """
    # Create AdamW optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
    )

    # Create cosine scheduler with linear warmup
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max_steps - warmup_steps,
        eta_min=1e-6,
    )

    return optimizer, scheduler
