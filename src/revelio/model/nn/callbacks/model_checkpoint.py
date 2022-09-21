from typing import Literal

import torch

from .callback import Callback


class ModelCheckpoint(Callback):
    def __init__(
        self,
        file_path: str,
        monitor: str = "val_loss",
        min_delta: float = 0.0,
        direction: Literal["min"] | Literal["max"] = "min",
        save_best_only: bool = False,
    ) -> None:
        self._file_path = file_path
        self._monitor = monitor
        self._min_delta = min_delta
        self._direction = direction
        self._save_best_only = save_best_only
        self._best_metric_value = (
            torch.tensor(float("inf"))
            if direction == "min"
            else torch.tensor(float("-inf"))
        )

    def after_validation_epoch(
        self, epoch: int, metrics: dict[str, torch.Tensor]
    ) -> None:
        if self._monitor not in metrics:
            raise ValueError(f"{self._monitor} is not a valid metric")
        metric_value = metrics[self._monitor]
        if self._direction == "min":
            has_improved = metric_value < self._best_metric_value - self._min_delta
        else:
            has_improved = metric_value > self._best_metric_value + self._min_delta
        if has_improved or not self._save_best_only:
            self._best_metric_value = metric_value
            torch.save(
                self.model.get_state_dict(),
                self._parse_file_path(epoch, metrics),
            )

    def _parse_file_path(self, epoch: int, metrics: dict[str, torch.Tensor]) -> str:
        return self._file_path.format(epoch=epoch, **metrics)
