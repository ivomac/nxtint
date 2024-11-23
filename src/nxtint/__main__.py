"""Command-line interface for training sequence prediction models."""

import argparse
import sys

import torch

from nxtint.model import SequenceTransformer
from nxtint.trainer import Trainer
from nxtint.utils.logging import log_io, setup_logger

logger = setup_logger(__name__)


@log_io(logger)
def main():
    """Run model training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model-id", type=str, help="Load existing model ID")
    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)

    # Initialize model and trainer
    model = SequenceTransformer(model_id=args.model_id)
    trainer = Trainer(model)

    # Train
    logger.info(f"Starting training for model {model.model_id}")
    trainer.train()

    # Save model
    model.save()
    logger.info("Training complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
