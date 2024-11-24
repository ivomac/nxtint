"""Tests for training loop."""

from nxtint.data.sequences import Sequence
from nxtint.model import SequenceTransformer
from nxtint.trainer import EarlyStopping, Trainer
from nxtint.utils.config import Config


def test_training_loop():
    """Test basic training loop for a few steps."""
    # Create sequence and model
    sequence = Sequence.linear(5, 5, [2, 1])
    model = SequenceTransformer()

    # Run training with temporarily reduced max_steps
    with Config.train.override(max_steps=10):
        trainer = Trainer(model, [sequence])
        trainer.train()
    assert True


def test_validation():
    """Test validation step."""
    # Create sequence and model
    sequence = Sequence.linear(5, 5, [2, 1])
    model = SequenceTransformer()
    trainer = Trainer(model, sequence)

    loss, inaccuracy = trainer.validate()
    assert isinstance(loss, float)
    assert loss > 0
    assert isinstance(inaccuracy, float)
    assert inaccuracy >= 0


def test_early_stopping():
    """Test early stopping behavior."""
    with Config.early.override(patience=2):
        model = SequenceTransformer()
        early = EarlyStopping()

        # Test improvement case
        weights = early(model, 1.0, 0.5)
        assert weights is None

        # Test no improvement but not stopping
        weights = early(model, 1.5, 0.5)
        assert weights is None

        # Test stopping after patience exceeded
        weights = early(model, 1.5, 0.5)
        assert weights is not None
