from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import torch
import torch.profiler as profiler
import torch.utils.tensorboard as tb

from .callback import Callback


class TensorBoard(Callback):  # pragma: no cover
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
        self._accumulated_metrics: dict[str, torch.Tensor] = {}

    def before_training(self) -> None:
        if self._profiler is not None:
            self._profiler.start()

    def after_training(self, metrics: dict[str, torch.Tensor]) -> None:
        self._writer.close()

    def after_training_epoch(
        self, epoch: int, metrics: dict[str, torch.Tensor]
    ) -> None:
        if self._profiler is not None and epoch == self.model.initial_epoch:
            self._profiler.stop()
        # Write the average metrics for the epoch
        # If we iterate through metrics we won't have any validation metric, because
        # they are not yet computed
        for metric_name in metrics.keys():
            self._writer.add_scalar(
                f"{metric_name}/epoch_train",
                self._accumulated_metrics[metric_name]
                / self.model.train_steps_per_epoch[-1],
                epoch,
            )

    def before_training_step(
        self, epoch: int, step: int, batch: dict[str, Any]
    ) -> None:
        if epoch == self.model.initial_epoch and step == 0:
            self._add_images_view(batch, "train")

    def after_training_step(
        self,
        epoch: int,
        step: int,
        batch: dict[str, Any],
        metrics: dict[str, torch.Tensor],
    ) -> None:
        global_step = sum(self.model.train_steps_per_epoch) + step
        for metric_name in metrics.keys():
            self._writer.add_scalar(
                f"{metric_name}/train", metrics[metric_name], global_step
            )
            # Accumulate the metrics to compute the average at the end of the epoch
            if metric_name not in self._accumulated_metrics:
                self._accumulated_metrics[metric_name] = metrics[metric_name]
            else:
                self._accumulated_metrics[metric_name] += metrics[metric_name]
        if self._profiler is not None:
            self._profiler.step()

    def after_validation_epoch(
        self, epoch: int, metrics: dict[str, torch.Tensor]
    ) -> None:
        # Write the average val metrics for the epoch
        for metric_name in metrics.keys():
            if metric_name.startswith("val_"):
                original_metric_name = metric_name[4:]
                self._writer.add_scalar(
                    f"{original_metric_name}/epoch_val",
                    self._accumulated_metrics[metric_name]
                    / self.model.val_steps_per_epoch[-1],
                    epoch,
                )
        # Reset the accumulated metrics
        self._accumulated_metrics = {}

    def before_validation_step(
        self, epoch: int, step: int, batch: dict[str, Any]
    ) -> None:
        if epoch == 0 and step == 0:
            self._add_images_view(batch, "val")

    def after_validation_step(
        self,
        epoch: int,
        step: int,
        batch: dict[str, Any],
        metrics: dict[str, torch.Tensor],
    ) -> None:
        val_metrics = {k: v for k, v in metrics.items() if k.startswith("val_")}
        global_step = sum(self.model.val_steps_per_epoch) + step
        for metric_name in val_metrics.keys():
            original_metric_name = metric_name[4:]
            self._writer.add_scalar(
                f"{original_metric_name}/val", val_metrics[metric_name], global_step
            )
            # Accumulate the metrics to compute the average at the end of the epoch
            if metric_name not in self._accumulated_metrics:
                self._accumulated_metrics[metric_name] = val_metrics[metric_name]
            else:
                self._accumulated_metrics[metric_name] += val_metrics[metric_name]
        if self._profiler is not None:
            self._profiler.step()

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
            self._writer.add_figure(f"{phase}_images/probe", probe_grid, 0)
        elif len(batch["x"]) == 2:
            probe_grid = self._get_image(batch["x"][0]["image"], batch["y"])
            self._writer.add_figure(f"{phase}_images/probe", probe_grid, 0)
            live_grid = self._get_image(batch["x"][1]["image"], batch["y"])
            self._writer.add_figure(f"{phase}_images/live", live_grid, 0)
        else:
            for i, x in enumerate(batch["x"]):
                probe_grid = self._get_image(x["image"], batch["y"])
                self._writer.add_image(f"{phase}_images/image {i}", probe_grid, 0)
