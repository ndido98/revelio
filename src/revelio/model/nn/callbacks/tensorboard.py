from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import torch
import torch.profiler as profiler
import torch.utils.tensorboard as tb

from .callback import Callback


class TensorBoard(Callback):
    _writer: tb.SummaryWriter
    _profiler: Optional[profiler.profile]

    def __init__(self, log_dir: str, profile: bool = False) -> None:
        self._run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir_with_run = Path(log_dir) / self._run_name
        self._writer = tb.SummaryWriter(log_dir=str(log_dir_with_run))
        if profile:
            self._profiler = profiler.profile(
                activities=[
                    profiler.ProfilerActivity.CPU,
                    profiler.ProfilerActivity.CUDA,
                ],
                on_trace_ready=profiler.tensorboard_trace_handler(self._writer.log_dir),
                record_shapes=True,
                profile_memory=True,
                # with_stack=True,  # FIXME: this is bugged on Windows (shocker)
            )
        else:
            self._profiler = None

    def after_training(self, metrics: dict[str, torch.Tensor]) -> None:
        self._writer.close()

    def before_training_epoch(self, epoch: int, steps_count: int) -> None:
        if self._profiler is not None:
            self._profiler.start()

    def after_training_epoch(
        self, epoch: int, steps_count: int, metrics: dict[str, torch.Tensor]
    ) -> None:
        if self._profiler is not None:
            self._profiler.stop()

    def before_training_step(
        self, epoch: int, step: int, batch: dict[str, Any]
    ) -> None:
        if epoch == 0 and step == 0:
            self._add_images_view(batch, "train")

    def after_training_step(
        self,
        epoch: int,
        step: int,
        batch: dict[str, Any],
        metrics: dict[str, torch.Tensor],
    ) -> None:
        if self._profiler is not None:
            self._profiler.step()

    def before_validation_step(
        self, epoch: int, step: int, batch: dict[str, Any]
    ) -> None:
        if epoch == 0 and step == 0:
            self._add_images_view(batch, "val")

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

    def _add_images_view(self, batch: dict[str, Any], phase: str) -> None:
        if len(batch["x"]) == 1:
            probe_grid = self._get_image(batch["x"][0]["image"], batch["y"])
            self._writer.add_figure(f"{phase}/probe", probe_grid, 0)
        elif len(batch["x"]) == 2:
            probe_grid = self._get_image(batch["x"][0]["image"], batch["y"])
            self._writer.add_figure(f"{phase}/probe", probe_grid, 0)
            live_grid = self._get_image(batch["x"][1]["image"], batch["y"])
            self._writer.add_figure(f"{phase}/live", live_grid, 0)
        else:
            for i, x in enumerate(batch["x"]):
                probe_grid = self._get_image(x["image"], batch["y"])
                self._writer.add_image(f"{phase}/image {i}", probe_grid, 0)
        self._writer.add_graph(self.model.classifier, [batch["x"]])

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
