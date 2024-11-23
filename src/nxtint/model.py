"""Transformer model for integer sequence prediction."""

import json
import uuid

import torch
import torch.nn as nn

from .logits import Logits
from .utils.config import Config
from .utils.logging import log_io, setup_logger

logger = setup_logger(__name__)


class SequenceTransformer(nn.Module):
    """Transformer model for predicting the next integer in a sequence."""

    @classmethod
    @log_io(logger)
    def model_list(cls) -> list[str]:
        """List available models.

        Returns:
            list[str]: List of model identifiers
        """
        return [p.name for p in Config.save.base_dir.iterdir() if p.is_dir()]

    @classmethod
    @log_io(logger)
    def load_config(cls, model_id: str | None) -> dict:
        """Load configuration from saved model.

        Args:
            model_id: Model identifier

        Returns:
            dict: Saved configuration parameters
        """
        if model_id is None:
            logger.info("No model ID provided")
            return {}

        config_path = Config.save.base_dir / model_id / Config.save.config_file
        if not config_path.is_file():
            logger.info(f"No config file found with ID {model_id}")
            return {}

        with open(config_path) as f:
            logger.info(f"Found configuration for model {model_id}")
            return json.load(f)

    def __init__(self, model_id: str | None = None):
        """Initialize the model.

        Args:
            model_id: Unique identifier for the model. If None, generates new ID.
                    Attempts to load existing model weights with this ID.
        """
        super().__init__()

        # Set or generate model ID
        self.model_id = str(uuid.uuid4()) if model_id is None else model_id
        self.save_dir = Config.save.base_dir / self.model_id
        self.weights_file = self.save_dir / Config.save.weights_file
        self.config_file = self.save_dir / Config.save.config_file

        # Try to load existing model
        self.init_layers()
        self.load_weights()
        return

    @log_io(logger)
    def init_layers(self):
        """Initialize model layers."""
        # Integer embedding layer
        self.int_shift = Config.gen.max_int
        self.int_embedding = nn.Embedding(
            2 * Config.gen.max_int,
            Config.model.d_model,
            dtype=Config.dtype.float,
        )

        # Positional embedding layer
        self.positions = torch.arange(
            Config.model.x_len,
            dtype=Config.dtype.int,
        )

        self.pos_embedding = nn.Embedding(
            Config.model.x_len,
            Config.model.d_model,
            dtype=Config.dtype.float,
        )

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.model.d_model,
            nhead=Config.model.n_heads,
            dim_feedforward=Config.model.d_ff,
            dropout=Config.model.dropout,
            activation=Config.model.activation,
            batch_first=Config.model.batch_first,
            norm_first=Config.model.norm_first,
            dtype=Config.dtype.float,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=Config.model.n_layers,
        )

        # Output projection
        self.output = nn.Linear(
            Config.model.d_model,
            2 * Config.gen.max_int,
            dtype=Config.dtype.float,
        )
        return

    @log_io(logger)
    def save(self):
        """Save model weights and configuration."""
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save(self.state_dict(), self.weights_file)

        # Save configuration
        config = {
            "model": {k: v for k, v in vars(Config.model).items() if not k.startswith("_")},
            "gen": {k: v for k, v in vars(Config.gen).items() if not k.startswith("_")},
        }
        with self.config_file.open("w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved model to {self.save_dir}")
        return

    @log_io(logger)
    def load_weights(self):
        """Load model weights from saved file."""
        if self.weights_file.is_file():
            self.load_state_dict(torch.load(self.weights_file, weights_only=True))
            logger.info(f"Loaded existing model {self.model_id}")
        else:
            logger.info(f"No existing model found with ID {self.model_id}")
        return

    @log_io(logger)
    def forward(self, x: torch.Tensor) -> Logits:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, x_len) containing integers

        Returns:
            Logits: Logits object for next integer prediction (batch_size, 2 * MAX_INT)
        """
        # Get integer embeddings
        int_embeddings = self.int_embedding(x + self.int_shift)

        # Create position indices and get embeddings
        pos_embeddings = self.pos_embedding(self.positions)

        # Add positional embeddings to integer embeddings
        embeddings = int_embeddings + pos_embeddings.unsqueeze(0)

        # Pass through transformer
        transformed = self.transformer(embeddings)

        # Use final sequence position for prediction
        final = transformed[:, -1]

        # Project to output logits
        return Logits(self.output(final))
