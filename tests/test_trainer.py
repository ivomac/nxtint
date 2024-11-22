"""Tests for training loop."""

from nxtint.model import SequenceTransformer
from nxtint.trainer import Trainer


def test_trainer_init():
    """Test trainer initialization."""
    model = SequenceTransformer()
    trainer = Trainer(model)

    assert trainer.batch_size == 32
    assert trainer.max_steps == 50000
    assert trainer.clip_norm == 1.0
    return


def test_validation():
    """Test validation step."""
    model = SequenceTransformer()
    trainer = Trainer(model)

    val_loss = trainer.validate(num_batches=2)
    assert isinstance(val_loss, float)
    assert val_loss > 0
    return


def test_training_loop():
    """Test basic training loop for a few steps."""
    model = SequenceTransformer()
    trainer = Trainer(model, max_steps=10)

    # Should run without errors
    trainer.train(validate_every=5)
    assert True
    return
