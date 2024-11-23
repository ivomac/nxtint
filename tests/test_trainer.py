"""Tests for training loop."""

from nxtint.model import SequenceTransformer
from nxtint.trainer import Trainer
from nxtint.utils.config import Config


def test_training_loop():
    """Test basic training loop for a few steps."""
    # Run training with temporarily reduced max_steps
    with Config.train.override(max_steps=10):
        model = SequenceTransformer()
        trainer = Trainer(model)
        trainer.train()
    assert True


def test_validation():
    """Test validation step."""
    model = SequenceTransformer()
    trainer = Trainer(model)

    val_loss = trainer.validate(num_batches=2)
    assert isinstance(val_loss, float)
    assert val_loss > 0


def test_early_stopping():
    """Test early stopping behavior."""
    with Config.early.override(patience=2):
        model = SequenceTransformer()
        trainer = Trainer(model)
        # Test improvement case
        weights = trainer.early_stopping(model, 1.0)
        assert weights is None

        # Test no improvement but not stopping
        weights = trainer.early_stopping(model, 1.5)
        assert weights is None

        # Test stopping after patience exceeded
        weights = trainer.early_stopping(model, 1.5)
        assert weights is not None
