"""Command-line interface to get a prediction from a model."""

import argparse

from .model import SequenceTransformer
from .utils.config import Config
from .utils.ids import ModelID
from .utils.logging import log_io, setup_logger

logger = setup_logger(__name__)


@log_io(logger)
def main():
    """Run model prediction."""
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--model-id", type=str, default=None, help="Load existing model ID")
    parser.add_argument("input", type=str, help="Comma-separated input sequence")
    args = parser.parse_args()

    # If no model ID is provided, list the available models
    if args.model_id is None:
        print("Available models:")
        for model in ModelID.used():
            print(f"  {model}")
        return 0

    config = SequenceTransformer.load_config(args.model_id)
    with Config.override(**config):
        sequence = [int(x) for x in args.input.split(",")]
        model = SequenceTransformer(args.model_id)
        prediction = model.predict(sequence)
        print(f"Prediction: {prediction}")

    return 0
