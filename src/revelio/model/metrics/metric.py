from abc import abstractmethod

import torch

from revelio.registry.registry import Registrable


class Metric(Registrable):
    @abstractmethod
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def compute(self) -> float:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError  # pragma: no cover
