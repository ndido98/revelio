from abc import abstractmethod
from typing import Any

import torch.optim

from revelio.registry.registry import Registrable


class Optimizer(Registrable):
    @abstractmethod
    def get(self, **kwargs: Any) -> torch.optim.Optimizer:
        raise NotImplementedError  # pragma: no cover
