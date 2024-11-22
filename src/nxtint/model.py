"""Transformer model for integer sequence prediction."""

import torch
import torch.nn as nn

from nxtint.utils.constants import INT_N, MAX_INT
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
    """

    def __init__(
        self,
        seq_length: int = 8,
        n_layers: int = 2,
        n_heads: int = 2,
        d_model: int = 2,
        d_ff: int = 32,
        device: str = "cpu",
    ) -> None:
        """Initialize the model.

        Args:
            seq_length: Length of input sequences
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            d_model: Model dimension
            d_ff: Feed-forward network dimension
            device: Device to run the model on
        """
        # Initialize parent class first
        super().__init__()
        self.seq_length = seq_length
        self.device = device
        self.to(device)

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
            device=device,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Final layer norm (post-norm)
        self.norm = nn.LayerNorm(d_model, device=device)

        # Output projection
        self.output = nn.Linear(d_model, INT_N, device=device)
        return

    @log(logger, level=DEBUG)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, seq_length) containing integers

        Returns:
            torch.Tensor: Logits for next integer prediction (batch_size, INT_N)
        """
        # Move input to device and create embeddings (batch_size, seq_length, 2)
        x = x.to(self.device)

        numbers = x.float() / MAX_INT
        positions = self.positions.expand(x.size(0), -1).to(self.device)

        # Combine into embedding
        embeddings = torch.stack([numbers, positions], dim=-1)

        # Pass through transformer
        transformed = self.transformer(embeddings)

        # Use final sequence position for prediction
        final = transformed[:, -1]

        # Apply final layer norm (post-norm)
        normalized = self.norm(final)

        # Project to output logits
        return self.output(normalized)

    @log(logger, level=DEBUG)
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get the most likely next number prediction.

        Args:
            x: Input tensor of shape (batch_size, seq_length) containing integers

        Returns:
            torch.Tensor: Predicted next integers of shape (batch_size,)
        """
        # Get logits and return argmax
        return torch.argmax(self(x), dim=-1) - MAX_INT
