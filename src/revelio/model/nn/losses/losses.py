from typing import Any

import torch

from .loss import Loss


class BCEWithLogitsLoss(Loss):
    def get(self, **kwargs: Any) -> torch.nn.Module:
        return torch.nn.BCEWithLogitsLoss(**kwargs)
