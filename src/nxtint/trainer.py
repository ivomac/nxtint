"""Training loop for sequence prediction model."""

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data.sequences import FOSequence
from .model import SequenceTransformer
from .utils.config import Config
from .utils.constants import INF
from .utils.logging import log_io, setup_logger

logger = setup_logger(__name__)


class Trainer:
    """Trainer for sequence prediction model.

    Attributes:
        model: Model to train
        train_logger: Logger for training progress
        optimizer: AdamW optimizer instance
        scheduler: Learning rate scheduler
        early_stopping: Early stopping handler
        train_gen: Training data generator
        val_gen: Validation data generator

    Methods:
        init_components: Set training components
        validate: Run validation and return mean loss
        train: Train the model
    """

    def __init__(self, model: SequenceTransformer):
        """Initialize trainer.

        Args:
            model: Model to train
        """
        self.model = model
        # Setup training-specific logger
        train_log = model.save_dir / Config.save.log_file
        self.train_logger = setup_logger(
            f"{__name__}.{model.model_id[:8]}",
            log_file=train_log,
            propagate=False,
        )
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
        self.train_gen = FOSequence()
        self.val_gen = FOSequence()
        return

    @log_io(logger)
    def validate(self, num_batches: int = 10) -> tuple[float, float]:
        """Run validation and return mean loss and accuracy.

        Args:
            num_batches: Number of validation batches

        Returns:
            float: Mean validation loss
            float: Mean validation accuracy
        """
        self.model.eval()
        total_loss = 0.0
        accuracy = 0.0

        with torch.no_grad():
            for _ in range(num_batches):
                # Get batch of sequences
                x, y = self.val_gen.generate_batch(Config.train.batch_size)

                # Get predictions, loss and accuracy
                logits = self.model(x)
                total_loss += logits.loss(y).mean().item()
                accuracy += logits.accuracy(y)

        mean_loss = total_loss / num_batches
        mean_accuracy = accuracy / num_batches

        self.model.train()
        return mean_loss, mean_accuracy

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
            loss = self.model(x).loss(y).mean()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            clip_grad_norm_(self.model.parameters(), Config.train.clip_norm)

            # Update weights
            self.optimizer.step()
            self.scheduler.step()

            # Log training loss
            log_training_step = step % 100 == 0
            validation_step = step % Config.train.validate_every == 0
            if log_training_step or validation_step:
                self.train_logger.info(f"Step {step}")
                if log_training_step:
                    self.train_logger.info(f"Training Loss:       {loss.item():.3g}")
                if validation_step:
                    loss, accuracy = self.validate()
                    self.train_logger.info(f"Validation Loss:     {loss:.3g}")
                    self.train_logger.info(f"Validation Accuracy: {accuracy:.1f}%")

                    # Check early stopping
                    best_weights = self.early_stopping(self.model, loss)
                    if best_weights is not None:
                        self.train_logger.info("Early stopping triggered")
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
