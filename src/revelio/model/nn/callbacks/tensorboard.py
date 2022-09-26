from typing import Optional

import torch
import torch.utils.tensorboard as tb

from .callback import Callback


class TensorBoard(Callback):
    def __init__(self, log_dir: Optional[str] = None) -> None:
        self._writer = tb.SummaryWriter(log_dir=log_dir)

    def after_training(self, metrics: dict[str, torch.Tensor]) -> None:
        self._writer.close()

    def after_validation_epoch(
        self, epoch: int, steps_count: int, metrics: dict[str, torch.Tensor]
    ) -> None:
        train_metrics = {k: v for k, v in metrics.items() if not k.startswith("val_")}
        val_metrics = {k: v for k, v in metrics.items() if k.startswith("val_")}
        for metric_name in train_metrics.keys():
            self._writer.add_scalars(
                metric_name,
                {
                    "train": train_metrics[metric_name],
                    "val": val_metrics[f"val_{metric_name}"],
                },
                epoch,
            )
