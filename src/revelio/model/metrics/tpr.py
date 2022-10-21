from typing import Any

import torch

from .metric import Metric


class TPR(Metric):
    @property
    def name(self) -> str:
        return "tpr"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reset()

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        pred_classes = torch.round(y_pred)
        self.true_positives += (pred_classes * y_true).sum()
        self.positives += y_true.sum()

    def compute(self) -> torch.Tensor:
        return self.true_positives / self.positives

    def reset(self) -> None:
        self.true_positives = torch.zeros((), device=self.device)
        self.positives = torch.zeros((), device=self.device)
