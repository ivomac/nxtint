"""Constants for the nxtint package."""

import torch

from .config import TypeConfig

NAN = torch.tensor(float("nan"), dtype=TypeConfig.float)

INF = torch.tensor(float("inf"), dtype=TypeConfig.float)
