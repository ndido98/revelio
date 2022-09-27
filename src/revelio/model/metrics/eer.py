from typing import Any

import torch

from .metric import Metric
from .utils import _compute_roc


class EqualErrorRate(Metric):
    @property
    def name(self) -> str:
        return "eer"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reset()

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        self._preds.append(y_pred)
        self._trues.append(y_true)

    def compute(self) -> torch.Tensor:
        all_pred = torch.cat(self._preds)
        all_true = torch.cat(self._trues)
        Pfa, Pmiss = _compute_roc(all_pred, all_true, self.device)  # noqa: N806
        idx = torch.nonzero(Pfa <= Pmiss)[0]
        return (Pmiss[idx - 1] + Pfa[idx]) / 2

    def reset(self) -> None:
        self._preds: list[torch.Tensor] = []
        self._trues: list[torch.Tensor] = []
