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

    def __init__(self, model_id: str | None = None):
        """Initialize the model.

        Args:
            model_id: Unique identifier for the model. If None, generates new ID.
                     If provided, attempts to load existing model with this ID.
        """
        super().__init__()

        # Set or generate model ID
        self.model_id = model_id or str(uuid.uuid4())
        self.save_dir = Config.save.base_dir / self.model_id

        # Integer embedding layer
        self.int_embedding = nn.Embedding(
            2 * Config.gen.max_int,
            Config.model.d_model,
            dtype=Config.dtype.float,
        )

        # Positional embedding layer
        self.pos_embedding = nn.Embedding(
            Config.model.x_length,
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

        # Try to load existing model
        if model_id is not None:
            self.load()
        return

    @log_io(logger)
    def save(self):
        """Save model weights and configuration."""
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        weights_path = self.save_dir / Config.save.weights_file
        torch.save(self.state_dict(), weights_path)

        # Save configuration
        config_path = self.save_dir / Config.save.config_file
        config = {
            "model_id": self.model_id,
            "model": {k: v for k, v in vars(Config.model).items() if not k.startswith("_")},
            "gen": {k: v for k, v in vars(Config.gen).items() if not k.startswith("_")},
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved model to {self.save_dir}")
        return

    @log_io(logger)
    def load(self):
        """Load model weights from saved file."""
        weights_path = self.save_dir / Config.save.weights_file
        if weights_path.is_file():
            self.load_state_dict(torch.load(weights_path))
            logger.info(f"Loaded existing model {self.model_id}")
        else:
            logger.warning(f"No existing model found with ID {self.model_id}")
        return

    @log_io(logger)
    def forward(self, x: torch.Tensor) -> Logits:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, x_length) containing integers

        Returns:
            Logits: Logits object for next integer prediction (batch_size, 2 * MAX_INT)
        """
        # Shift input to be 0 to 2*MAX_INT for embedding lookup
        x_shifted = x + Config.gen.max_int

        # Get integer embeddings
        int_embeddings = self.int_embedding(x_shifted)

        # Create position indices and get embeddings
        positions = torch.arange(
            Config.model.x_length,
            dtype=Config.dtype.int,
        )
        pos_embeddings = self.pos_embedding(positions)

        # Add positional embeddings to integer embeddings
        embeddings = int_embeddings + pos_embeddings.unsqueeze(0)

        # Pass through transformer
        transformed = self.transformer(embeddings)

        # Use final sequence position for prediction
        final = transformed[:, -1]

        # Project to output logits
        return Logits(self.output(final))
