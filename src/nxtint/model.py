"""Transformer model for integer sequence prediction."""

import torch
import torch.nn as nn

from nxtint.utils.logging import DEBUG, log, setup_logger

logger = setup_logger(__name__, level=DEBUG)


class SequenceTransformer(nn.Module):
    """Transformer model for predicting the next integer in a sequence.

    Attributes:
        seq_length: Length of input sequences
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        d_model: Model dimension (2 for number+position encoding)
        d_ff: Feed-forward network dimension
        max_int: Maximum integer value (255 for byte sequences)
    """

    def __init__(
        self,
        seq_length: int = 8,
        n_layers: int = 2,
        n_heads: int = 2,
        d_model: int = 2,
        d_ff: int = 32,
        max_int: int = 255,
    ) -> None:
        """Initialize the model.

        Args:
            seq_length: Length of input sequences
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            d_model: Model dimension
            d_ff: Feed-forward network dimension
            max_int: Maximum integer value
        """
        # Initialize parent class first
        super().__init__()
        self.seq_length = seq_length
        self.max_int = max_int

        # Create position indices tensor once
        self.register_buffer(
            "positions",
            torch.arange(seq_length).float() / (seq_length - 1),
        )

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Output projection
        self.output = nn.Linear(d_model, max_int + 1)
        return

    @log(logger, level=DEBUG)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, seq_length) containing integers
                in range [0, max_int]

        Returns:
            torch.Tensor: Logits for next integer prediction (batch_size, max_int + 1)
        """
        # Create embeddings (batch_size, seq_length, 2)
        numbers = x.float() / self.max_int
        positions = self.positions.expand(x.size(0), -1)

        # Combine into embedding
        embeddings = torch.stack([numbers, positions], dim=-1)

        # Pass through transformer
        transformed = self.transformer(embeddings)

        # Use final sequence position for prediction
        final = transformed[:, -1]

        # Project to output logits
        return self.output(final)

    @log(logger, level=DEBUG)
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get the most likely next number prediction.

        Args:
            x: Input tensor of shape (batch_size, seq_length) containing integers
                in range [0, max_int]

        Returns:
            torch.Tensor: Predicted next integers of shape (batch_size,)
        """
        # Get logits and return argmax
        return torch.argmax(self(x), dim=-1)
