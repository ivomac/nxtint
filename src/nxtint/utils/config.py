"""Configuration classes for nxtint."""

import logging
from contextlib import contextmanager
from pathlib import Path

import torch

INFO = logging.INFO
DEBUG = logging.DEBUG

if torch.cuda.is_available():
    torch.set_default_device("cuda")


class BaseConfig(type):
    """Metaclass for configuration classes.

    Prevents instantiation and direct modification of class attributes.
    Provides a context manager to temporarily override class attributes.
    """

    _block_assign = True

    def __call__(cls):
        """Prevent instantiation of the configuration classes."""
        raise TypeError(f"{cls.__name__} cannot be instantiated.")

    def __setattr__(cls, name, value):
        """Prevent direct modification of class attributes."""
        if name != "_block_assign" and cls._block_assign:
            raise AttributeError(f"Cannot modify {name} directly. Use the 'override' method.")
        super().__setattr__(name, value)

    @contextmanager
    def override(cls, **kwargs):
        """Context manager to temporarily override class attributes."""
        # Save original values of the attributes to be overridden
        cls._block_assign = False
        original = {k: getattr(cls, k) for k in kwargs}
        try:
            # Override the class attributes
            for k, v in kwargs.items():
                if not hasattr(cls, k):
                    raise AttributeError(f"{k} is not a valid attribute of {cls.__name__}.")
                setattr(cls, k, v)
            yield
        finally:
            # Restore the original values
            for k, v in original.items():
                setattr(cls, k, v)
            cls._block_assign = True


class ModelConfig(metaclass=BaseConfig):
    """Model architecture configuration.

    Attributes:
        x_len: Length of input sequences
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        d_model: Model dimension
        d_ff: Feed-forward network dimension
        dropout: Dropout rate
        activation: Activation function
        batch_first: Whether input is batch-first
        norm_first: Whether normalization is applied first
    """

    x_len: int = 8
    n_layers: int = 2
    n_heads: int = 4
    d_model: int = 64
    d_ff: int = 256
    dropout: float = 0.0
    activation: str = "gelu"
    batch_first: bool = True
    norm_first: bool = False


class TrainConfig(metaclass=BaseConfig):
    """Training configuration.

    Attributes:
        batch_size: Training batch size
        max_steps: Maximum number of training steps
        clip_norm: Maximum gradient norm
        lr: Learning rate
        weight_decay: Weight decay factor
        warmup_steps: Number of warmup steps
        validate_every: Steps between validation
    """

    batch_size: int = 32
    max_steps: int = 50000
    clip_norm: float = 1.0
    lr: float = 1e-3
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    warmup_steps: int = 5000
    validate_every: int = 1000
    eta_min: float = 1e-6


class EarlyStoppingConfig(metaclass=BaseConfig):
    """Early stopping configuration.

    Attributes:
        patience: Number of steps without improvement before stopping
        min_delta: Minimum improvement to qualify as an improvement
    """

    patience: int = 10
    min_delta: float = 0.001


class GenConfig(metaclass=BaseConfig):
    """Data generation configuration.

    Attributes:
        max_int: Maximum integer value
        buffer_size: Number of sequences to generate at once
    """

    max_int: int = 512
    buffer_size: int = 1024


class LossConfig(metaclass=BaseConfig):
    """Loss function configuration.

    Attributes:
        alpha: Distance weighting factor
    """

    alpha: float = 0.1


class TypeConfig(metaclass=BaseConfig):
    """Data types configuration.

    Attributes:
        int_type: Integer type
        float_type: Float type
    """

    int: torch.dtype = torch.int32
    float: torch.dtype = torch.float32


class SaveConfig(metaclass=BaseConfig):
    """Model saving configuration.

    Attributes:
        base_dir: Base directory for saving models
        model_dir: Directory for specific model instance
        weights_file: Model weights filename
        config_file: Model config filename
        log_file: Training log filename
    """

    base_dir: Path = Path("./cache/models")
    weights_file: str = "weights.pt"
    config_file: str = "config.json"
    log_file: str = "training.log"


class LogConfig(metaclass=BaseConfig):
    """Logging configuration.

    Attributes:
        dir: Directory for log files
        file: Log filename
        level: Logging level
    """

    dir: Path = Path("./cache")
    file: str = "nxtint.log"
    level: int = INFO


class Config(metaclass=BaseConfig):
    """Global configuration container.

    Attributes:
        model: Model configuration
        training: Training configuration
        early: Early stopping configuration
        gen: Data generation configuration
        loss: Loss function configuration
        dtype: Data types configuration
        save: Save configuration
        log: Logging configuration
    """

    model = ModelConfig
    train = TrainConfig
    early = EarlyStoppingConfig
    gen = GenConfig
    loss = LossConfig
    dtype = TypeConfig
    save = SaveConfig
    log = LogConfig

    @classmethod
    @contextmanager
    def override(cls, **kwargs):
        """Context manager to set configuration options.

        Args:
            kwargs: Config attributes (e.g., "model", "gen")
                to kwargs dictionaries for the corresponding Config class's
                override method.
        """
        config_managers = []
        try:
            # Create context managers for each config class
            for config_name, subkwargs in kwargs.items():
                if not hasattr(cls, config_name):
                    raise AttributeError(
                        f"{config_name} is not a valid attribute of {cls.__name__}."
                    )
                config_class = getattr(cls, config_name)
                config_managers.append(config_class.override(**subkwargs))

            # Enter all context managers
            for manager in config_managers:
                manager.__enter__()
            yield
        finally:
            # Exit all context managers in reverse order
            for manager in reversed(config_managers):
                manager.__exit__(None, None, None)
        return
