from typing import Optional

import torch
import torch.utils.tensorboard as tb
from torchvision.utils import make_grid

from .callback import Callback


class TensorBoard(Callback):
    def __init__(self, log_dir: Optional[str] = None) -> None:
        self._writer = tb.SummaryWriter(log_dir=log_dir)

    def before_training(self) -> None:
        train_batch = next(iter(self.model.train_dataloader))
        train_grid = make_grid(train_batch, normalize=True)
        self._writer.add_image("train", train_grid)
        self._writer.add_graph(self.model, train_batch)

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
