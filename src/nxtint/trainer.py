"""Training loop for sequence prediction model."""

import torch
from torch.nn.utils import clip_grad_norm_

from nxtint.data.sequences import SequenceGenerator
from nxtint.losses import sequence_loss
from nxtint.model import SequenceTransformer
from nxtint.training import EarlyStopping, setup_training
from nxtint.utils.logging import DEBUG, log, setup_logger

logger = setup_logger(__name__, level=DEBUG)


class Trainer:
    """Trainer for sequence prediction model.

    Attributes:
        model: Model to train
        device: Device to train on
        batch_size: Training batch size
        max_steps: Maximum number of training steps
        clip_norm: Maximum gradient norm
    """

    def __init__(
        self,
        model: SequenceTransformer,
        device: str = "cpu",
        batch_size: int = 32,
        max_steps: int = 50000,
        clip_norm: float = 1.0,
    ) -> None:
        """Initialize trainer.

        Args:
            model: Model to train
            device: Device to train on
            batch_size: Training batch size
            max_steps: Maximum number of training steps
            clip_norm: Maximum gradient norm
        """
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.clip_norm = clip_norm

        # Setup training components
        self.optimizer, self.scheduler = setup_training(
            model,
            max_steps=max_steps,
        )
        self.early_stopping = EarlyStopping()

        # Setup data generators
        self.train_gen = SequenceGenerator(
            seq_length=model.seq_length + 1,
            device=device,
        )
        self.val_gen = SequenceGenerator(
            seq_length=model.seq_length + 1,
            device=device,
        )
        return

    @log(logger, level=DEBUG)
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
                seqs = self.val_gen.generate_batch(self.batch_size)

                # Split into inputs and targets (last sequence element is target)
                x, y = seqs[:, :-1], seqs[:, -1].long()

                # Get predictions and loss
                logits = self.model(x)
                loss = sequence_loss(logits, y)
                total_loss += loss.item()

        self.model.train()
        return total_loss / num_batches

    @log(logger, level=DEBUG)
    def train(self, validate_every: int = 1000) -> None:
        """Train the model.

        Args:
            validate_every: Steps between validation
        """
        self.model.train()
        step = 0
        best_weights = None

        while step < self.max_steps:
            # Get batch of sequences
            seqs = self.train_gen.generate_batch(self.batch_size)

            # Split into inputs and targets
            x, y = seqs[:, :-1], seqs[:, -1].long()

            # Forward pass
            logits = self.model(x)
            loss = sequence_loss(logits, y)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            clip_grad_norm_(self.model.parameters(), self.clip_norm)

            # Update weights
            self.optimizer.step()
            self.scheduler.step()

            # Log training loss
            if step % 100 == 0:
                logger.debug(f"Step {step}, Loss: {loss.item():.4f}")

            # Validate and check early stopping
            if step % validate_every == 0:
                val_loss = self.validate()
                logger.debug(f"Step {step}, Validation Loss: {val_loss:.4f}")

                # Check early stopping
                best_weights = self.early_stopping(self.model, val_loss)
                if best_weights is not None:
                    logger.debug("Early stopping triggered")
                    # Restore best weights
                    self.model.load_state_dict(best_weights)
                    break

            step += 1

        return
