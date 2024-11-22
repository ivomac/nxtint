"""Transformer model for integer sequence prediction."""

import torch
import torch.nn as nn

from .logits import Logits
from .utils.config import Config


class SequenceTransformer(nn.Module):
    """Transformer model for predicting the next integer in a sequence."""

    def __init__(self):
        """Initialize the model."""
        # Initialize parent class first
        super().__init__()

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
        return

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
