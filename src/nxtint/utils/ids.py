"""Generate human-readable unique model identifiers."""

import torch

from .config import Config


class ModelID:
    """Generate human-readable unique model identifiers.

    Model IDs are generated as 'adjective_noun' pairs.

    Attributes:
        ADJECTIVES: List of adjectives
        NOUNS: List of nouns
        USED_IDS: Set of used model identifiers
    """

    ADJECTIVES = [
        "swift",
        "bright",
        "clever",
        "deep",
        "eager",
        "fierce",
        "gentle",
        "happy",
        "keen",
        "lively",
        "mighty",
        "noble",
        "proud",
        "quiet",
        "rapid",
        "sharp",
        "smart",
        "strong",
        "wise",
        "bold",
        "brave",
        "calm",
        "deft",
        "fair",
        "grand",
        "quick",
        "wild",
        "young",
        "agile",
        "alert",
        "ancient",
        "astute",
        "broad",
        "clear",
        "cosmic",
        "divine",
        "exact",
        "fleet",
        "fluid",
        "fresh",
        "golden",
        "great",
        "high",
        "just",
        "kind",
        "light",
        "lucid",
        "prime",
        "pure",
        "rare",
        "royal",
        "sage",
        "sure",
        "true",
        "vast",
    ]

    NOUNS = [
        "falcon",
        "tiger",
        "wolf",
        "eagle",
        "bear",
        "dragon",
        "phoenix",
        "lion",
        "hawk",
        "owl",
        "panther",
        "dolphin",
        "raven",
        "lynx",
        "fox",
        "cobra",
        "jaguar",
        "whale",
        "shark",
        "seal",
        "griffin",
        "unicorn",
        "sphinx",
        "hydra",
        "kraken",
        "serpent",
        "leopard",
        "falcon",
        "stag",
        "horse",
        "turtle",
        "salmon",
        "crane",
        "heron",
        "swan",
        "falcon",
        "kestrel",
        "osprey",
        "condor",
        "albatross",
        "gazelle",
        "cheetah",
        "puma",
        "orca",
        "manta",
        "octopus",
        "dragon",
        "phoenix",
        "sphinx",
        "hydra",
        "pegasus",
        "chimera",
        "manticore",
        "basilisk",
        "wyvern",
    ]

    @classmethod
    def _gen_id(cls) -> str:
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

        while (id := cls._gen_id()) in used_ids:
            pass
        return id

    @classmethod
    def used(cls) -> set:
        """List used model identifiers.

        Returns:
            set: Set of used model identifiers
        """
        return set(dir.name for dir in Config.save.dir.iterdir() if dir.is_dir())
