from abc import abstractmethod
from typing import Any

import torch

from revelio.registry.registry import Registrable


class Loss(Registrable):
    @abstractmethod
    def get(self, **kwargs: Any) -> torch.nn.Module:
        raise NotImplementedError  # pragma: no cover
