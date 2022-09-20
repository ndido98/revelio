from typing import Any

import torch

from .optimizer import Optimizer


class SGD(Optimizer):
    def get(self, **kwargs: Any) -> torch.optim.Optimizer:
        return torch.optim.SGD(**kwargs)
