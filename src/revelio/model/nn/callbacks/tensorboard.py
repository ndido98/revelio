from typing import Any, Optional

import matplotlib.pyplot as plt
import torch
import torch.utils.tensorboard as tb
from torch.utils.data import DataLoader

from ..utils import _dict_to_device
from .callback import Callback


class TensorBoard(Callback):
    def __init__(self, log_dir: Optional[str] = None) -> None:
        self._writer = tb.SummaryWriter(log_dir=log_dir)

    def before_training(self) -> None:
        self._add_images_view(self.model.train_dataloader, "train")
        self._add_images_view(self.model.val_dataloader, "val")

    def _get_image(
        self, imgs: torch.Tensor, labels: torch.Tensor, nrow: int = 8
    ) -> Any:
        n_imgs = imgs.shape[0] if imgs.dim() == 4 else 1
        rows = n_imgs // nrow + 1
        fig, axs = plt.subplots(rows, nrow, figsize=(4 * nrow, 4 * rows))
        for i in range(rows):
            for j in range(nrow):
                idx = i * nrow + j
                if idx < n_imgs:
                    # Convert the image from CHW TO HWC
                    img = imgs[idx].permute(1, 2, 0)
                    # Normalize the image to be in the [0, 1] range
                    img = (img - img.min()) / (img.max() - img.min())
                    axs[i, j].imshow(img.cpu().numpy())
                    axs[i, j].set_title(f"Label: {labels[idx]}")
                    axs[i, j].axis("off")
                else:
                    axs[i, j].remove()
        return fig

    def _add_images_view(self, data_loader: DataLoader, phase: str) -> None:
        batch = _dict_to_device(next(iter(data_loader)), self.model.device)
        if len(batch["x"]) == 1:
            probe_grid = self._get_image(batch["x"][0]["image"], batch["y"])
            self._writer.add_figure(f"{phase}/probe", probe_grid, 0)
        elif len(batch["x"]) == 2:
            probe_grid = self._get_image(batch["x"][0]["image"], batch["y"])
            self._writer.add_figure(f"{phase}/probe", probe_grid, 0)
            bona_fide_grid = self._get_image(batch["x"][1]["image"], batch["y"])
            self._writer.add_figure(f"{phase}/bona fide", bona_fide_grid, 0)
        else:
            for i, x in enumerate(batch["x"]):
                probe_grid = self._get_image(x["image"], batch["y"])
                self._writer.add_image(f"{phase}/image {i}", probe_grid, 0)
        self._writer.add_graph(self.model.classifier, [batch["x"]])

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
