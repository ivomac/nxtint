"""Tests for the sequence transformer model."""

import torch

from nxtint.model import SequenceTransformer
from nxtint.utils.config import Config


def test_model_forward(batch_size=3):
    """Test basic forward pass of the model."""
    # Create model
    model = SequenceTransformer()

    # Create sample batch
    x = torch.randint(-Config.gen.max_int, Config.gen.max_int, (batch_size, Config.model.x_len))

    # Forward pass
    logits = model(x)

    # Check output shape
    assert logits.shape == (batch_size, Config.gen.max_int * 2)

    # Check output range (logits should be finite)
    assert torch.isfinite(logits).all()


def test_model_predict():
    """Test model prediction."""
    # Create model
    model = SequenceTransformer()

    # Create sample batch
    batch_size = 3
    x = torch.randint(-Config.gen.max_int, Config.gen.max_int, (batch_size, Config.model.x_len))

    # Get predictions
    logits = model(x)
    predictions = logits.predict()

    # Check output shape
    assert predictions.shape == (batch_size,)

    # Check output range
    assert (predictions >= -Config.gen.max_int).all() and (predictions < Config.gen.max_int).all()


def test_model_save_load():
    """Test saving and loading model with custom parameters."""
    custom_config = {
        "model": {"n_layers": 3, "n_heads": 8, "d_model": 128},
        "gen": {"max_int": 256},
    }

    # Create and save model with custom config
    with Config.override(**custom_config):
        model1 = SequenceTransformer()
        model1.save()
    model_id = model1.model_id

    # Load the model and verify parameters
    loaded_config = SequenceTransformer.load_config(model_id)
    with Config.override(**loaded_config):
        model2 = SequenceTransformer(model_id=model_id)

    # Check if parameters match
    assert model2.transformer.num_layers == 3
    assert model2.transformer.layers[0].self_attn.num_heads == 8
    assert model2.transformer.layers[0].self_attn.embed_dim == 128
    assert model2.int_embedding.num_embeddings == 512  # 2 * max_int

    # Clean up
    model1.delete()
