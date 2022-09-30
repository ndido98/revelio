from typing import Any

import torch

from .metric import Metric
from .utils import _compute_roc


class BPCERAtAPCER(Metric):
    @property
    def name(self) -> list[str]:
        return [f"bpcer@apcer {t}" for t in self._thresholds]

    def __init__(self, thresholds: list[float], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._thresholds = thresholds
        self._thresholds_tensor = torch.tensor(thresholds, device=self.device)
        self.reset()

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        self._preds.append(y_pred)
        self._trues.append(y_true)

    def compute(self) -> torch.Tensor:
        all_pred = torch.cat(self._preds)
        all_true = torch.cat(self._trues)
        Pfa, Pmiss = _compute_roc(all_pred, all_true, self.device)  # noqa: N806
        idx = Pfa.shape[0] - torch.searchsorted(
            torch.flip(Pfa, dims=(0,)), self._thresholds_tensor
        )
        # Linearly interpolate between the two closest points
        d1 = torch.abs(self._thresholds_tensor - Pmiss[idx - 1])
        d2 = torch.abs(self._thresholds_tensor - Pmiss[idx])
        w1 = d1 / (d1 + d2)
        w2 = d2 / (d1 + d2)
        return Pmiss[idx - 1] * w1 + Pmiss[idx] * w2

    def reset(self) -> None:
        self._preds: list[torch.Tensor] = []
        self._trues: list[torch.Tensor] = []
