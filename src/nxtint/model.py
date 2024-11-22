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
        n_heads: int = 4,
        d_model: int = 64,
        d_ff: int = 256,
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

        # Integer embedding layer
        self.int_embedding = nn.Embedding(
            2 * MAX_INT + 1,  # -MAX_INT to +MAX_INT
            d_model,
            device=device,
        )

        # Positional embedding layer
        self.pos_embedding = nn.Embedding(
            seq_length,
            d_model,
            device=device,
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
        # Move input to device
        x = x.to(self.device)

        # Shift input to be 0 to 2*MAX_INT for embedding lookup
        x_shifted = x + MAX_INT

        # Get integer embeddings
        int_embeddings = self.int_embedding(x_shifted)

        # Create position indices and get embeddings
        positions = torch.arange(self.seq_length, device=self.device)
        pos_embeddings = self.pos_embedding(positions)

        # Add positional embeddings to integer embeddings
        embeddings = int_embeddings + pos_embeddings.unsqueeze(0)

        # Normalize input embeddings
        normalized_input = self.input_norm(embeddings)

        # Pass through transformer
        transformed = self.transformer(normalized_input)

        # Use final sequence position for prediction
        final = transformed[:, -1]

        # Apply final layer norm (post-norm)
        normalized = self.final_norm(final)

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
