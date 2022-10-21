from typing import Any

import torch

from .metric import Metric


class TNR(Metric):
    @property
    def name(self) -> str:
        return "tnr"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reset()

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        pred_classes = torch.round(y_pred)
        self.true_negatives += ((1 - pred_classes) * (1 - y_true)).sum()
        self.negatives += (1 - y_true).sum()

    def compute(self) -> torch.Tensor:
        return self.true_negatives / self.negatives

    def reset(self) -> None:
        self.true_negatives = torch.zeros((), device=self.device)
        self.negatives = torch.zeros((), device=self.device)
