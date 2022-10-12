from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import torch

from .callback import Callback


class ModelCheckpoint(Callback):
    def __init__(
        self,
        file_path: str,
        monitor: str = "val_loss",
        min_delta: float = 0.0,
        direction: Literal["min", "max"] = "min",
        save_best_only: bool = False,
    ) -> None:
        self._file_path = Path(file_path)
        self._monitor = monitor
        self._min_delta = min_delta
        self._direction = direction
        self._save_best_only = save_best_only
        self._epoch_metric_value = torch.tensor(0.0, dtype=torch.float32)
        self._best_metric_value = (
            torch.tensor(float("inf"))
            if direction == "min"
            else torch.tensor(float("-inf"))
        )

    def after_validation_step(
        self,
        epoch: int,
        step: int,
        batch: dict[str, Any],
        metrics: dict[str, torch.Tensor],
    ) -> None:
        if self._monitor not in metrics:
            raise ValueError(f"{self._monitor} is not a valid metric")
        self._epoch_metric_value += metrics[self._monitor]

    def after_validation_epoch(
        self, epoch: int, steps_count: int, metrics: dict[str, torch.Tensor]
    ) -> None:
        metric_value = self._epoch_metric_value / steps_count
        self._epoch_metric_value = torch.tensor(0.0, dtype=torch.float32)
        if self._direction == "min":
            has_improved = metric_value < self._best_metric_value - self._min_delta
        else:
            has_improved = metric_value > self._best_metric_value + self._min_delta
        if has_improved or not self._save_best_only:
            self._best_metric_value = metric_value
            dest_file = self._parse_file_path(epoch, metrics)
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                self.model.get_state_dict(),
                self._parse_file_path(epoch, metrics),
            )

    def _parse_file_path(self, epoch: int, metrics: dict[str, torch.Tensor]) -> Path:
        return self._file_path.with_name(
            self._file_path.name.format(
                now=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                epoch=epoch + 1,
                **metrics,
            )
        )
