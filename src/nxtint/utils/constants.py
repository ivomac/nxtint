"""Constants for the nxtint package."""

import torch

INT_TYPE = torch.int32

MAX_INT = 512

NAN = torch.tensor(float("nan"), dtype=torch.float32)

INF = torch.tensor(float("inf"), dtype=torch.float32)
