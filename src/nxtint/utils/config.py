"""Configuration classes for nxtint."""

import json
import logging
from contextlib import contextmanager
from pathlib import Path

import torch


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

    def to_dict(cls) -> dict:
        """Return class attributes as a dictionary."""
        return {k: v for k, v in vars(cls).items() if not k.startswith("_")}


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
        val_batches: Number of validation batches
        eta_min: Minimum learning rate
    """

    batch_size: int = 32
    max_steps: int = 10000
    clip_norm: float = 1.0
    lr: float = 1e-3
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    warmup_steps: int = 1000
    validate_every: int = 200
    val_batches: int = 20
    eta_min: float = 1e-6


class EarlyStoppingConfig(metaclass=BaseConfig):
    """Early stopping configuration.

    Attributes:
        patience: Number of steps without improvement before stopping
        min_loss_delta: Minimum improvement to qualify as an improvement
        threshold_inaccuracy: Threshold inaccuracy for early stopping
        use_loss: Whether to use loss for early stopping
        use_inaccuracy: Whether to use accuracy for early stopping
    """

    patience: int = 5
    min_loss_delta: float = 1e-6
    threshold_inaccuracy: float = 1e-2
    use_loss: bool = True
    use_inaccuracy: bool = True


class GenConfig(metaclass=BaseConfig):
    """Data generation configuration.

    Attributes:
        max_int: Maximum integer value
        buffer_size: Number of sequences to generate at once
    """

    max_int: int = 512
    buffer_size: int = 2048


class LossConfig(metaclass=BaseConfig):
    """Loss function configuration.

    Attributes:
        alpha: Distance weighting factor
    """

    alpha: float = 0.1


class SaveConfig(metaclass=BaseConfig):
    """Model saving configuration.

    Attributes:
        dir: Base directory for saving models
        model_dir: Directory for specific model instance
        weights_file: Model weights filename
        config_file: Model config filename
        log_file: Training log filename
    """

    dir: str = "./cache/models"
    weights_file: str = "weights.pt"
    config_file: str = "config.json"
    log_file: str = "training.log"


class C(metaclass=BaseConfig):
    """Global constants.

    Attributes:
        DEVICE: Default device
        NAN: NaN tensor
        INF: Infinity tensor
        INFO: Logging level for informational messages
        DEBUG: Logging level for debug messages
        LOGFMT: Log message format
        LOGDATEFMT: Log date format
        ADJECTIVES: List of adjectives for naming
        NOUNS: List of nouns for naming
    """

    DEVICE: "str" = "cuda" if torch.cuda.is_available() else "cpu"
    INT: torch.dtype = torch.int32
    FLOAT: torch.dtype = torch.float32

    NAN = torch.tensor(float("nan"), dtype=FLOAT)

    INF = torch.tensor(float("inf"), dtype=FLOAT)

    INFO = logging.INFO
    DEBUG = logging.DEBUG

    LOGFMT = "%(asctime)s - %(levelname)s - %(module)s: %(message)s"
    LOGDATEFMT = "%y-%m-%d %H:%M:%S"

    ADJECTIVES = [
        "swift",
        "bright",
        "clever",
        "deep",
        "eager",
        "fierce",
        "gentle",
        "happy",
        "keen",
        "lively",
        "mighty",
        "noble",
        "proud",
        "quiet",
        "rapid",
        "sharp",
        "smart",
        "strong",
        "wise",
        "bold",
        "brave",
        "calm",
        "deft",
        "fair",
        "grand",
        "quick",
        "wild",
        "young",
        "agile",
        "alert",
        "ancient",
        "astute",
        "broad",
        "clear",
        "cosmic",
        "divine",
        "exact",
        "fleet",
        "fluid",
        "fresh",
        "golden",
        "great",
        "high",
        "just",
        "kind",
        "light",
        "lucid",
        "prime",
        "pure",
        "rare",
        "royal",
        "sage",
        "sure",
        "true",
        "vast",
    ]

    NOUNS = [
        "falcon",
        "tiger",
        "wolf",
        "eagle",
        "bear",
        "dragon",
        "phoenix",
        "lion",
        "hawk",
        "owl",
        "panther",
        "dolphin",
        "raven",
        "lynx",
        "fox",
        "cobra",
        "jaguar",
        "whale",
        "shark",
        "seal",
        "griffin",
        "unicorn",
        "sphinx",
        "hydra",
        "kraken",
        "serpent",
        "leopard",
        "falcon",
        "stag",
        "horse",
        "turtle",
        "salmon",
        "crane",
        "heron",
        "swan",
        "falcon",
        "kestrel",
        "osprey",
        "condor",
        "albatross",
        "gazelle",
        "cheetah",
        "puma",
        "orca",
        "manta",
        "octopus",
        "dragon",
        "phoenix",
        "sphinx",
        "hydra",
        "pegasus",
        "chimera",
        "manticore",
        "basilisk",
        "wyvern",
    ]


class LogConfig(metaclass=BaseConfig):
    """Logging configuration.

    Attributes:
        dir: Directory for log files
        file: Log filename
        level: Logging level
        train_steps: Number of training steps between log messages
    """

    dir: str = "./cache"
    file: str = "nxtint.log"
    level: int = C.DEBUG
    train_steps: int = TrainConfig.validate_every


class Config(metaclass=BaseConfig):
    """Global configuration container.

    Attributes:
        model: Model configuration
        training: Training configuration
        early: Early stopping configuration
        gen: Data generation configuration
        loss: Loss function configuration
        save: Save configuration
        log: Logging configuration
    """

    _objects = ["model", "train", "early", "gen", "loss"]
    model = ModelConfig
    train = TrainConfig
    early = EarlyStoppingConfig
    gen = GenConfig
    loss = LossConfig
    save = SaveConfig
    log = LogConfig
    C = C

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

    @classmethod
    def init_dirs(cls):
        """Initialize directories for saving models and logs."""
        Path(cls.log.dir).mkdir(parents=True, exist_ok=True)
        Path(cls.save.dir).mkdir(parents=True, exist_ok=True)
        return

    @classmethod
    def to_dict(cls):
        """Return the configuration as a dictionary."""
        return {k: getattr(cls, k).to_dict() for k in cls._objects}

    @classmethod
    def to_json(cls, filename: str | Path):
        """Save the configuration to a JSON file.

        Args:
            filename: Output filename
        """
        with open(filename, "w") as f:
            json.dump(cls.to_dict(), f, indent=2)
        return


if torch.cuda.is_available():
    torch.set_default_device(Config.C.DEVICE)
