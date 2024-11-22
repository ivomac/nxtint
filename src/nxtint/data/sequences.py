"""First-order recurrence sequence generation module."""

import torch

from nxtint.utils.config import Config
from nxtint.utils.logging import setup_logger

logger = setup_logger(__name__)


class FOSequenceGenerator:
    """Generate first-order recurrence sequences with integer parameters.

    Attributes:
        buffer: Tensor of generated sequences
        current_idx: Index of last returned sequence from buffer
        total_sequences: Total number of generated sequences
        invalid_sequences: Number of sequences with invalid values
    """

    def __init__(self):
        """Initialize the sequence generator."""
        # Initialize empty buffer and position
        self.buffer = torch.empty(0)
        self.current_idx = 0

        # Initialize counters
        self.total_sequences = 0
        self.invalid_sequences = 0
        return

    def is_valid(self, sequence: torch.Tensor) -> torch.Tensor:
        """Check if a sequence is valid (no negatives or values > max_int).

        Args:
            sequence: Sequence to validate

        Returns:
            bool: True if sequence is valid, False otherwise
        """
        return (sequence >= -Config.gen.max_int).all() and (sequence < Config.gen.max_int).all()

    def _generate_buffer(self):
        """Generate a new buffer of sequences using first-order recurrence relations."""
        # Preallocate the sequence buffer
        self.buffer = torch.zeros(
            (Config.gen.buffer_size, Config.model.x_length + 1),
            dtype=Config.dtype.int,
        )

        # Generate random parameters for each sequence
        a0 = torch.randint(
            -5,
            6,
            (Config.gen.buffer_size,),
            dtype=Config.dtype.int,
        )
        mult = torch.randint(
            -2,
            3,
            (Config.gen.buffer_size,),
            dtype=Config.dtype.int,
        )
        add = torch.randint(
            -6,
            7,
            (Config.gen.buffer_size,),
            dtype=Config.dtype.int,
        )

        # Fill first column with initial values
        self.buffer[:, 0] = a0

        # Generate each sequence using the recurrence relation
        for i in range(1, Config.model.x_length + 1):
            self.buffer[:, i] = self.buffer[:, i - 1] * mult + add

        # Reset position counter
        self.current_idx = 0

        return

    def generate_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of valid sequences.

        Args:
            batch_size: Number of sequences to generate

        Returns:
            torch.Tensor: Batch of valid sequences of shape (batch_size, x_length)
        """
        # Preallocate output tensor
        batch = torch.empty(
            (batch_size, Config.model.x_length),
            dtype=Config.dtype.int,
        )
        y = torch.empty(
            (batch_size,),
            dtype=torch.int,
        )
        valid_count = 0

        while valid_count < batch_size:
            # Generate new buffer if needed
            if self.current_idx >= len(self.buffer):
                self._generate_buffer()

            # Get remaining sequences needed
            remaining = batch_size - valid_count

            # Get candidates from current buffer position
            candidates = self.buffer[self.current_idx : self.current_idx + remaining]

            # Find valid sequences
            for seq in candidates:
                self.total_sequences += 1
                if self.is_valid(seq):
                    batch[valid_count] = seq[:-1]
                    y[valid_count] = seq[-1]
                    valid_count += 1
                    if valid_count == batch_size:
                        break
                else:
                    self.invalid_sequences += 1

            # Update position
            self.current_idx += len(candidates)

        invalid_percent = 100 * self.invalid_sequences / self.total_sequences
        logger.debug(
            f"Invalid sequences: {self.invalid_sequences}/{self.total_sequences} "
            f"({invalid_percent:.1f}%)"
        )

        return batch, y
