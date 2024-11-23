"""Training loop for sequence prediction model."""

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data.sequences import FOSequenceGenerator
from .model import SequenceTransformer
from .utils.config import Config
from .utils.constants import INF
from .utils.logging import log_io, setup_logger

logger = setup_logger(__name__)


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
        self.init_components()
        return

    def init_components(self):
        """Set training components."""
        # Setup training components
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=Config.train.lr,
            betas=Config.train.betas,
            eps=Config.train.eps,
            weight_decay=Config.train.weight_decay,
        )

        # Create cosine scheduler with linear warmup
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=Config.train.max_steps - Config.train.warmup_steps,
            eta_min=Config.train.eta_min,
        )

        self.early_stopping = EarlyStopping()

        # Setup data generators
        self.train_gen = FOSequenceGenerator()
        self.val_gen = FOSequenceGenerator()
        return

    @log_io(logger)
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
                x, y = self.val_gen.generate_batch(Config.train.batch_size)

                # Get predictions and loss
                logits = self.model(x)
                loss = logits.loss(y).mean()
                total_loss += loss.item()

        self.model.train()
        return total_loss / num_batches

    @log_io(logger)
    def train(self):
        """Train the model."""
        self.model.train()
        step = 0
        best_weights = None

        while step < Config.train.max_steps:
            # Get batch of sequences
            x, y = self.train_gen.generate_batch(Config.train.batch_size)

            # Forward pass
            logits = self.model(x)
            loss = logits.loss(y).mean()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            clip_grad_norm_(self.model.parameters(), Config.train.clip_norm)

            # Update weights
            self.optimizer.step()
            self.scheduler.step()

            # Log training loss
            if step % 100 == 0:
                logger.info(f"Step {step}, Loss: {loss.item():.3g}")

            # Validate and check early stopping
            if step % Config.train.validate_every == 0:
                val_loss = self.validate()
                logger.info(f"Step {step}, Validation Loss: {val_loss:.3g}")

                # Check early stopping
                best_weights = self.early_stopping(self.model, val_loss)
                if best_weights is not None:
                    logger.info("Early stopping triggered")
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

    @log_io(logger)
    def __call__(self, model: torch.nn.Module, val_loss: float) -> dict[str, torch.Tensor] | None:
        """Check if training should stop and save best weights.

        Args:
            model: Current model
            val_loss: Current validation loss

        Returns:
            dict: Best model state dict if stopping, None otherwise
        """
        if val_loss < self.best_loss - Config.early.min_delta:
            # New best loss found
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return None

        # No improvement
        self.counter += 1
        if self.counter >= Config.early.patience:
            # Return best weights when stopping
            return self.best_weights
        return None
