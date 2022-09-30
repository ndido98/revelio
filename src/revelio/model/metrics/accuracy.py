from typing import Any

import torch

from .metric import Metric


class Accuracy(Metric):
    @property
    def name(self) -> str:
        return "accuracy"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reset()

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        pred_classes = torch.round(y_pred)
        self.total += y_true.shape[0]
        self.correct += (pred_classes == y_true).sum()

    def compute(self) -> torch.Tensor:
        return self.correct / self.total

    def reset(self) -> None:
        self.total = torch.zeros((), device=self.device)
        self.correct = torch.zeros((), device=self.device)
