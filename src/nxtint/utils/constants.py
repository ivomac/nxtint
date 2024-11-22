"""Constants for the nxtint package."""

import torch

INT_TYPE = torch.int16

INT_N = 256

MAX_INT = 128

NAN = torch.tensor(float("nan"), dtype=torch.float32)
