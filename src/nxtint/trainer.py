"""Training loop for sequence prediction model."""

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from nxtint.data.sequences import FOSequenceGenerator
from nxtint.model import SequenceTransformer
from nxtint.utils.config import EarlyStoppingConfig, TrainConfig
from nxtint.utils.constants import INF


class Trainer:
    """Trainer for sequence prediction model.

    Attributes:
        model: Model to train
    """

    def __init__(self, model: SequenceTransformer):
        """Initialize trainer.

        Args:
            model: Model to train
        """
        self.model = model

        # Setup training components
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=TrainConfig.lr,
            betas=TrainConfig.betas,
            eps=TrainConfig.eps,
            weight_decay=TrainConfig.weight_decay,
        )

        # Create cosine scheduler with linear warmup
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=TrainConfig.max_steps - TrainConfig.warmup_steps,
            eta_min=TrainConfig.eta_min,
        )

        self.early_stopping = EarlyStopping()

        # Setup data generators
        self.train_gen = FOSequenceGenerator()
        self.val_gen = FOSequenceGenerator()
        return

    def validate(self, num_batches: int = 10) -> float:
        """Run validation and return mean loss.

        Args:
            num_batches: Number of validation batches

        Returns:
            float: Mean validation loss
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for _ in range(num_batches):
                # Get batch of sequences
                x, y = self.val_gen.generate_batch(TrainConfig.batch_size)

                # Get predictions and loss
                logits = self.model(x)
                loss = logits.loss(y).mean()
                total_loss += loss.item()

        self.model.train()
        return total_loss / num_batches

    def train(self):
        """Train the model."""
        self.model.train()
        step = 0
        best_weights = None

        while step < TrainConfig.max_steps:
            # Get batch of sequences
            x, y = self.train_gen.generate_batch(TrainConfig.batch_size)

            # Forward pass
            logits = self.model(x)
            loss = logits.loss(y).mean()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            clip_grad_norm_(self.model.parameters(), TrainConfig.clip_norm)

            # Update weights
            self.optimizer.step()
            self.scheduler.step()

            # Log training loss
            # if step % 100 == 0:
            #     logger.debug(f"Step {step}, Loss: {loss.item():.4f}")

            # Validate and check early stopping
            if step % TrainConfig.validate_every == 0:
                val_loss = self.validate()
                # logger.debug(f"Step {step}, Validation Loss: {val_loss:.4f}")

                # Check early stopping
                best_weights = self.early_stopping(self.model, val_loss)
                if best_weights is not None:
                    # logger.debug("Early stopping triggered")
                    # Restore best weights
                    self.model.load_state_dict(best_weights)
                    break

            step += 1

        return


class EarlyStopping:
    """Early stopping handler based on validation loss.

    Attributes:
        best_loss: Best validation loss seen so far
        counter: Number of epochs without improvement
        best_weights: Copy of model weights with lowest validation loss
    """

    def __init__(self):
        """Initialize early stopping handler."""
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
        if val_loss < self.best_loss - EarlyStoppingConfig.min_delta:
            # New best loss found
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return None

        # No improvement
        self.counter += 1
        if self.counter >= EarlyStoppingConfig.patience:
            # Return best weights when stopping
            return self.best_weights
        return None
