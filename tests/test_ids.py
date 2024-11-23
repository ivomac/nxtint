"""Tests for model ID generation."""

from pathlib import Path
from tempfile import TemporaryDirectory

from nxtint.utils.ids import ModelID


def test_id_generation():
    """Test that generated IDs follow the correct format."""
    model_id = ModelID._gen_id()

    # Check format
    assert isinstance(model_id, str)
    adj, noun = model_id.split("-")
    assert adj in ModelID.ADJECTIVES
    assert noun in ModelID.NOUNS


def test_unique_ids():
    """Test that generated IDs are unique."""
    # Generate multiple IDs
    ids = {ModelID.new() for _ in range(10)}

    # Check uniqueness
    assert len(ids) == 10


def test_used_ids():
    """Test tracking of used IDs."""
    from nxtint.utils.config import Config

    # Temporarily override save directory
    with TemporaryDirectory() as tmp_dir, Config.save.override(dir=Path(tmp_dir)):
        # Create some fake model directories
        test_ids = ["test-model1", "test-model2"]
        for id in test_ids:
            (Path(tmp_dir) / id).mkdir()

        # Check used IDs
        used = ModelID.used()
        assert used == set(test_ids)
