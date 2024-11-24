"""Training loop and phased training for sequence prediction model."""

from collections.abc import Iterator
from itertools import cycle

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data.sequences import Sequence
from .model import SequenceTransformer
from .utils.config import C, Config
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
        generator: Data generator

    Methods:
        init_components: Set training components
        validate: Run validation and return mean loss
        train: Train the model
    """

    @classmethod
    @log_io(logger)
    def train_from_phases(cls, model: SequenceTransformer, phases: list[list[Sequence]]):
        """Train the model through multiple phases of sequences.

        Each phase consists of a different set of sequence types.

        Args:
            model: Model to train
            phases: List of lists of sequences for each training phase
        """
        for i, sequences in enumerate(phases):
            logger.info(f"Starting phase {i+1}/{len(phases)}")

            # Create cycling iterator over sequences and trainer for this phase
            trainer = cls(model, sequences)

            # Train on this phase
            trainer.train()

            logger.info(f"Completed phase {i+1}")
        return

    def __init__(
        self,
        model: SequenceTransformer,
        sequence_gen: Iterator[Sequence] | list[Sequence] | Sequence,
    ):
        """Initialize trainer.

        Args:
            model: Model to train
            sequence_gen: Iterator over sequences
        """
        self.model = model
        # Setup sequence iterator
        if isinstance(sequence_gen, Sequence):
            sequence_gen = cycle([sequence_gen])
        elif isinstance(sequence_gen, list):
            sequence_gen = cycle(sequence_gen)
        self.sequence_gen = sequence_gen

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

        # Setup training-specific logger
        train_log = model.save_dir / Config.save.log_file
        self.train_logger = setup_logger(
            f"{__name__}.{model.model_id[:8]}",
            log_file=train_log,
            propagate=False,
        )
        return

    @log_io(logger)
    def validate(self) -> tuple[float, float]:
        """Run validation and return mean loss and accuracy.

        Returns:
            float: Mean validation loss
            float: Mean validation accuracy
        """
        self.model.eval()
        total_loss = 0.0
        accuracy = 0.0

        with torch.no_grad():
            for _ in range(Config.train.val_batches):
                # Get batch of sequences
                x, y = next(next(self.sequence_gen))

                # Get predictions, loss and accuracy
                logits = self.model(x)
                total_loss += logits.loss(y).mean().item()
                accuracy += logits.accuracy(y)

        mean_loss = total_loss / Config.train.val_batches
        mean_accuracy = accuracy / Config.train.val_batches

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
            x, y = next(next(self.sequence_gen))

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
            log_training_step = step % Config.log.train_steps == 0
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
                    best_weights = self.early_stopping(self.model, loss, accuracy)
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
        self.best_loss = C.INF
        self.best_accuracy = 0.0
        self.counter = 0
        self.best_weights = None
        return

    @log_io(logger)
    def __call__(
        self, model: torch.nn.Module, val_loss: float, val_accuracy: float
    ) -> dict[str, torch.Tensor] | None:
        """Check if training should stop and save best weights.

        Args:
            model: Current model
            val_loss: Current validation loss
            val_accuracy: Current validation accuracy

        Returns:
            dict: Best model state dict if stopping, None otherwise
        """
        ce = Config.early
        if (not ce.use_loss or (val_loss < self.best_loss - ce.min_loss_delta)) and (
            not ce.use_accuracy or (val_accuracy > self.best_accuracy + ce.min_accuracy_delta)
        ):
            # New best found
            self.best_loss = val_loss
            self.best_accuracy = val_accuracy
            self.counter = 0
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return None

        # No improvement
        self.counter += 1
        if self.counter >= ce.patience:
            # Return best weights when stopping
            return self.best_weights
        return None
