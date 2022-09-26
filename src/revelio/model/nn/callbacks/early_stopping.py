from typing import Any, Literal, Optional

import torch

from .callback import Callback


class EarlyStopping(Callback):
    def __init__(
        self,
        monitor: str = "val_loss",
        min_delta: float = 0.0,
        patience: int = 0,
        direction: Literal["min", "max"] = "min",
        restore_best_weights: bool = False,
    ) -> None:
        self._monitor = monitor
        self._min_delta = min_delta
        self._patience = patience
        self._direction = direction
        self._restore_best_weights = restore_best_weights
        self._best_metric_value = (
            torch.tensor(float("inf"))
            if direction == "min"
            else torch.tensor(float("-inf"))
        )
        self._best_weights: Optional[dict[str, Any]] = None
        self._wait = 0

    def after_validation_epoch(
        self, epoch: int, steps_count: int, metrics: dict[str, torch.Tensor]
    ) -> None:
        if self._monitor not in metrics:
            raise ValueError(f"{self._monitor} is not a valid metric")
        metric_value = metrics[self._monitor]
        if self._direction == "min":
            has_improved = metric_value < self._best_metric_value - self._min_delta
        else:
            has_improved = metric_value > self._best_metric_value + self._min_delta
        if has_improved:
            self._best_metric_value = metric_value
            self._wait = 0
            if self._restore_best_weights:
                self._best_weights = self.model.get_state_dict()
        else:
            self._wait += 1

        if self._wait >= self._patience:
            self.model.should_stop = True

    def after_training(self, metrics: dict[str, torch.Tensor]) -> None:
        if self._restore_best_weights and self._best_weights is not None:
            self.model.load_state_dict(self._best_weights)
