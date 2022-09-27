from abc import abstractmethod

import torch

from revelio.registry.registry import Registrable


class Metric(Registrable):
    def __init__(self, *, _device: str) -> None:
        self._device = _device

    @property
    def device(self) -> str:
        return self._device

    @property
    @abstractmethod
    def name(self) -> str | list[str]:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def compute(self) -> torch.Tensor:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError  # pragma: no cover
