"""First-order recurrence sequence generation module."""

from collections.abc import Callable
from functools import cached_property

import torch

from nxtint.utils.config import Config
from nxtint.utils.logging import log_io, setup_logger

logger = setup_logger(__name__)


class Sequence:
    """Generate recurrence sequences with integer parameters.

    Attributes:
        buffer: Tensor of generated sequences
        current_idx: Index of last returned sequence from buffer
        valid_indices: Indices of valid sequences in the buffer
        total_sequences: Total number of generated sequences
        valid_sequences: Total number of valid generated sequences

    Methods:
        generate_buffer: Generate a new buffer of sequences
        generate_batch: Generate a batch of valid sequences
    """

    @classmethod
    @log_io(logger)
    def linear(
        cls,
        initial: int,
        constant: int,
        vector: list[int],
        modlims: tuple[int, int] | None = None,
        shift: int | None = None,
    ) -> "Sequence":
        """Create a linear recurrence sequence generator.

        Recurrence relation: x_i = const + sum_{j=1}^{i} vec_j * x_{i-j}

        Args:
            initial: Initial max value for the sequence
            constant: Max value for the constant term
            vector: Vector of max multiplicative integers for parameter generation
            modlims: Modulus to apply to the final sequence
            shift: std of random shift to the final sequence
                Generated from the normal distribution

        Returns:
            Callable: Function to generate a buffer of sequences
        """
        t_vector = torch.tensor(vector, dtype=Config.C.INT)

        @log_io(logger)
        def linmodshift() -> torch.Tensor:
            """Generate a new buffer of sequences using linear/mod/shift recurrence relations."""
            # Preallocate the sequence buffer
            buffer = torch.zeros(
                (Config.gen.buffer_size, Config.model.x_len + 1), dtype=Config.C.INT
            )

            # Fill first column with initial values
            buffer[:, 0] = gen_scalar(initial)
            const = gen_scalar(constant)
            vec = gen_vector(t_vector)
            mod = gen_positive(*modlims) if modlims is not None else None

            # Generate each sequence using the recurrence relation
            for i in range(1, Config.model.x_len + 1):
                lim = min(i, vec.shape[1])
                if mod is None:
                    buffer[:, i] = const + torch.sum(vec[:, :lim] * buffer[:, i - lim : i], dim=1)
                else:
                    buffer[:, i] = (
                        const + torch.sum(vec[:, :lim] * buffer[:, i - lim : i], dim=1)
                    ) % mod

            if shift:
                buffer += gen_normal(shift).view(-1, 1)

            return buffer

        return cls(linmodshift)

    @classmethod
    @log_io(logger)
    def coupled(
        cls,
        initial: list[int],
        constant: list[int],
        matrix: list[list[int]],
        shift: int | None = None,
    ) -> "Sequence":
        """Create a coupled recurrence sequence generator.

        Recurrence relations:
            (x_i, y_i) = const + mat * x_{i-1}

        Args:
            initial: Vector of max initial values for the sequence
            constant: Vector of max constants for the sequence
            matrix: Matrix of max mult integers for parameter generation
            shift: std of random shift to the final sequence
                Generated from the normal distribution

        Returns:
            Callable: Function to generate a buffer of sequences
        """
        t_initial = torch.tensor(initial, dtype=Config.C.INT)
        t_constant = torch.tensor(constant, dtype=Config.C.INT)
        t_matrix = torch.tensor(matrix, dtype=Config.C.INT)

        @log_io(logger)
        def coupshift() -> torch.Tensor:
            """Generate a new buffer of sequences using coupled recurrence relations."""
            # Preallocate the sequence buffers
            buffer = torch.zeros(
                (Config.gen.buffer_size, Config.model.x_len + 1, 2), dtype=Config.C.INT
            )

            buffer[:, 0, :] = gen_vector(t_initial)
            const = gen_vector(t_constant)
            mat = gen_matrix(t_matrix)

            # Generate each sequence using the recurrence relation
            for i in range(1, Config.model.x_len + 1):
                # TODO(ivo): integer matrix multiplication on GPU is not supported:
                # https://github.com/pytorch/pytorch/issues/44428
                for j in range(2):
                    buffer[:, i, j] = const[:, j] + (mat[:, j] * buffer[:, i - 1, :]).sum(dim=-1)

            if shift:
                buffer += gen_normal(shift).view(-1, 1, 1)
            return buffer[:, :, 0]

        return cls(coupshift)

    def __init__(self, generate_buffer: Callable):
        """Initialize the sequence generator."""
        self._generate_buffer = generate_buffer

        # Initialize empty buffer
        self.buffer = torch.empty(0)

        self.x = torch.empty((Config.train.batch_size, Config.model.x_len), dtype=Config.C.INT)
        self.y = torch.empty((Config.train.batch_size,), dtype=Config.C.INT)

        # Initialize counters
        self.current_idx = 0
        self.total_sequences = 0
        self.valid_sequences = 0
        return

    @log_io(logger)
    def __iter__(self):
        """Create iterator for batches of sequences."""
        return self

    @log_io(logger)
    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate next batch of valid sequences.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Batch of sequences and targets
        """
        # Preallocate output tensor
        valid_count = 0

        while valid_count < Config.train.batch_size:
            # Generate new buffer if needed
            if self.current_idx >= self.valid_indices.shape[0]:
                self.generate_buffer()

            # Get remaining sequences needed
            remaining = Config.train.batch_size - valid_count

            # Get available from current buffer
            indices = self.valid_indices[self.current_idx : self.current_idx + remaining]
            available = len(indices)

            self.x[valid_count : valid_count + available] = self.buffer[indices, :-1]
            self.y[valid_count : valid_count + available] = self.buffer[indices, -1]

            # Update position
            self.current_idx += available
            valid_count += available

        return self.x, self.y

    def generate_buffer(self):
        """Generate a new buffer of sequences."""
        self.buffer = self._generate_buffer()
        if hasattr(self, "valid_indices"):
            del self.valid_indices
        self.current_idx = 0

        self.total_sequences += len(self.buffer)
        self.valid_sequences += len(self.valid_indices)

        valid_percent = 100 * self.valid_sequences / self.total_sequences
        logger.debug(
            f"Valid sequences: {self.valid_sequences}/{self.total_sequences} "
            f"({valid_percent:.1f}%)"
        )
        return

    @cached_property
    def valid_indices(self) -> torch.Tensor:
        """Check which sequences in the buffer are valid.

        Returns:
            torch.Tensor: Boolean tensor of valid sequences
        """
        return (
            torch.all(
                (self.buffer >= -Config.gen.max_int) & (self.buffer < Config.gen.max_int),
                dim=-1,
            )
            .nonzero(as_tuple=False)
            .squeeze()
        )


def gen_normal(std):
    """Generate a random integer from a normal distribution."""
    return (
        torch.normal(0, std, (Config.gen.buffer_size,), dtype=Config.C.FLOAT)
        .round()
        .to(dtype=Config.C.INT, device=Config.C.DEVICE)
    )


def gen_scalar(val):
    """Generate a random integer parameter scalar."""
    return torch.randint(-val, val + 1, (Config.gen.buffer_size,), dtype=Config.C.INT)


def gen_positive(min, max):
    """Generate a random integer parameter scalar."""
    return torch.randint(min, max + 1, (Config.gen.buffer_size,), dtype=Config.C.INT)


def gen_vector(val_vector: torch.Tensor):
    """Generate a random integer parameter vector."""
    vector = torch.zeros((Config.gen.buffer_size, *val_vector.shape), dtype=Config.C.INT)
    for i, val in enumerate(val_vector):
        vector[:, i] = gen_scalar(val.item())
    return vector


def gen_matrix(val_matrix: torch.Tensor):
    """Generate a random integer parameter matrix."""
    vector = gen_vector(val_matrix.view((-1,)))
    return vector.view((Config.gen.buffer_size, *val_matrix.shape))
