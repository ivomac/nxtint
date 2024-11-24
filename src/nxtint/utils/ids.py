"""Generate human-readable unique model identifiers."""

import torch

from .config import Config
from .constants import ADJECTIVES, NOUNS


class ModelID:
    """Generate human-readable unique model identifiers.

    Model IDs are generated as 'adjective-noun' pairs.

    Attributes:
        ADJECTIVES: List of adjectives for ID generation
        NOUNS: List of nouns for ID generation

    Methods:
        gen_id: Generate a human-readable unique model identifier
        new: Generate and verify uniqueness of a new model ID
        used: List currently used model identifiers
    """

    ADJECTIVES = ADJECTIVES
    NOUNS = NOUNS

    @classmethod
    def gen_id(cls) -> str:
        """Generate a human-readable unique model identifier.

        Returns:
            str: Model ID in format 'adjective_noun'
        """
        adj, noun = torch.randint(0, len(cls.NOUNS), (2,))
        return f"{cls.ADJECTIVES[adj]}-{cls.NOUNS[noun]}"

    @classmethod
    def new(cls) -> str:
        """Generate a human-readable unique model identifier.

        Verify that the generated ID is not already in use.

        Returns:
            str: Model ID in format 'adjective_noun'
        """
        used_ids = cls.used()

        while (id := cls.gen_id()) in used_ids:
            pass
        return id

    @classmethod
    def used(cls) -> set:
        """List used model identifiers.

        Returns:
            set: Set of used model identifiers
        """
        return set(dir.name for dir in Config.save.dir.iterdir() if dir.is_dir())
