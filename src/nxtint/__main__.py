"""Command-line interface for training sequence prediction models."""

import argparse

import torch

from nxtint.model import SequenceTransformer
from nxtint.trainer import Trainer
from nxtint.utils.config import Config
from nxtint.utils.logging import log_io, setup_logger

logger = setup_logger(__name__)


@log_io(logger)
def main():
    """Run model training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model-id", type=str, default=None, help="Load existing model ID")
    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)

    config = SequenceTransformer.load_config(args.model_id)
    with Config.override(**config):
        model = SequenceTransformer(args.model_id)
        trainer = Trainer(model)
        trainer.train()
        model.save()

    logger.info("Training complete")
    return 0


if __name__ == "__main__":
    main()
