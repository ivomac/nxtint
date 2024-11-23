"""Package for an integer sequence prediction transformer model."""

from dotenv import find_dotenv, load_dotenv

from .data.sequences import FOSequence
from .model import SequenceTransformer
from .trainer import Trainer
from .utils.config import Config
from .utils.ids import ModelID

load_dotenv(find_dotenv(raise_error_if_not_found=True, usecwd=True), verbose=True)

__all__ = ["FOSequence", "SequenceTransformer", "Trainer", "Config", "ModelID"]
