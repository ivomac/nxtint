"""First-order recurrence sequence generation module."""

import torch

from nxtint.utils.constants import INT_TYPE, MAX_INT
from nxtint.utils.logging import DEBUG, log, setup_logger

logger = setup_logger(__name__, level=DEBUG)


class SequenceGenerator:
    """Generate first-order recurrence sequences with integer parameters.

    Attributes:
        seq_length: Length of sequences to generate
        buffer_size: Number of sequences to generate at once
        device: Torch device to use for computations
        buffer: Tensor of generated sequences
        current_idx: Index of last returned sequence from buffer
    """

    def __init__(
        self,
        seq_length: int = 8,
        buffer_size: int = 1024,
        device: str = "cpu",
    ) -> None:
        """Initialize the sequence generator.

        Args:
            seq_length: Length of sequences to generate
            buffer_size: Number of sequences to generate at once
            device: Torch device to use for computations
        """
        self.seq_length = seq_length
        self.buffer_size = buffer_size
        self.device = device

        # Initialize empty buffer and position
        self.buffer = torch.empty(0, device=self.device)
        self.current_idx = 0
        return

    def is_valid(self, sequence: torch.Tensor) -> torch.Tensor:
        """Check if a sequence is valid (no negatives or values > max_int).

        Args:
            sequence: Sequence to validate

        Returns:
            bool: True if sequence is valid, False otherwise
        """
        return (sequence >= -MAX_INT).all() and (sequence < MAX_INT).all()

    def _generate_buffer(self) -> None:
        """Generate a new buffer of sequences using first-order recurrence relations."""
        # Preallocate the sequence buffer
        self.buffer = torch.zeros(
            (self.buffer_size, self.seq_length),
            dtype=INT_TYPE,
            device=self.device,
        )

        # Generate random parameters for each sequence
        a0 = torch.randint(
            -5,
            6,
            (self.buffer_size,),
            dtype=INT_TYPE,
            device=self.device,
        )
        mult = torch.randint(
            -4,
            5,
            (self.buffer_size,),
            dtype=INT_TYPE,
            device=self.device,
        )
        add = torch.randint(
            -4,
            5,
            (self.buffer_size,),
            dtype=INT_TYPE,
            device=self.device,
        )

        # Fill first column with initial values
        self.buffer[:, 0] = a0

        # Generate each sequence using the recurrence relation
        for i in range(1, self.seq_length):
            self.buffer[:, i] = self.buffer[:, i - 1] * mult + add

        # Reset position counter
        self.current_idx = 0
        return

    @log(logger, level=DEBUG)
    def generate_batch(self, batch_size: int) -> torch.Tensor:
        """Generate a batch of valid sequences.

        Args:
            batch_size: Number of sequences to generate

        Returns:
            torch.Tensor: Batch of valid sequences of shape (batch_size, seq_length)
        """
        # Preallocate output tensor
        batch = torch.empty(
            (batch_size, self.seq_length),
            dtype=INT_TYPE,
            device=self.device,
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
                if self.is_valid(seq):
                    batch[valid_count] = seq
                    valid_count += 1
                    if valid_count == batch_size:
                        break

            # Update position
            self.current_idx += len(candidates)

        return batch
