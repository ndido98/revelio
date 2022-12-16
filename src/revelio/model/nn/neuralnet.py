import copy
from abc import abstractmethod
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm

from revelio.model.model import Model

from .callbacks.callback import Callback
from .losses.loss import Loss
from .optimizers.optimizer import Optimizer
from .utils import _dict_to_device


class NeuralNetwork(Model):

    transparent: bool = True

    initial_epoch: int
    epochs: int
    train_steps_per_epoch: list[int] = []
    val_steps_per_epoch: list[int] = []
    optimizer: torch.optim.Optimizer
    loss_function: torch.nn.Module
    callbacks: list[Callback]
    should_stop: bool

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.classifier = self.get_classifier(**self.config.experiment.model.args)
        # Load the training configuration
        if "epochs" not in self.config.experiment.training.args:
            raise ValueError("Missing epochs in training configuration")
        self.epochs = int(self.config.experiment.training.args["epochs"])
        self._load_optimizer()
        self._load_loss()
        self._load_callbacks()
        self.should_stop = False
        # Load a checkpoint if present
        if self.config.experiment.model.checkpoint is not None:
            checkpoint = torch.load(
                self.config.experiment.model.checkpoint,
                map_location="cpu",
            )
            self.load_state_dict(checkpoint)
        else:
            self.initial_epoch = 0
        self._train_batches_count: Optional[int] = None
        self._val_batches_count: Optional[int] = None
        # Move the model to the device
        self.loss_function = self.loss_function.to(self.device, non_blocking=True)
        self.classifier = self.classifier.to(self.device, non_blocking=True)
        # Once the model is fully initialized, pass a reference to it to the callbacks
        for callback in self.callbacks:
            callback.model = self

    def _load_optimizer(self) -> None:
        if "optimizer" not in self.config.experiment.training.args:
            raise ValueError("Missing optimizer in training configuration")
        if "name" not in self.config.experiment.training.args["optimizer"]:
            raise ValueError("Missing optimizer name in training configuration")
        found_optimizer = Optimizer.find(
            self.config.experiment.training.args["optimizer"]["name"],
        )
        self.optimizer = found_optimizer.get(
            params=self.classifier.parameters(),
            **self.config.experiment.training.args["optimizer"].get("args", {}),
        )

    def _load_loss(self) -> None:
        if "loss" not in self.config.experiment.training.args:
            raise ValueError("Missing loss in training configuration")
        if "name" not in self.config.experiment.training.args["loss"]:
            raise ValueError("Missing loss name in training configuration")
        found_loss = Loss.find(
            self.config.experiment.training.args["loss"]["name"],
        )
        self.loss_function = found_loss.get(
            **self.config.experiment.training.args["loss"].get("args", {}),
        )

    def _load_callbacks(self) -> None:
        # Load the callbacks if training is enabled
        self.callbacks = []
        if self.config.experiment.training.enabled:
            for callback in self.config.experiment.training.args.get("callbacks", []):
                if "name" not in callback:
                    raise ValueError("Missing callback name in training configuration")
                found_callback = Callback.find(
                    callback["name"],
                    **callback.get("args", {}),
                )
                self.callbacks.append(found_callback)

    @abstractmethod
    def get_classifier(self, **kwargs: Any) -> torch.nn.Module:
        raise NotImplementedError  # pragma: no cover

    def fit(self) -> None:
        self.should_stop = False
        for callback in self.callbacks:
            callback.before_training()

        for epoch in range(self.initial_epoch, self.epochs):
            if self._train_batches_count is None or self._val_batches_count is None:
                # We don't know how many batches are there yet
                pbar_len = float("inf")
            else:
                pbar_len = self._train_batches_count + self._val_batches_count
            with tqdm(total=pbar_len) as pbar:
                if self.should_stop:
                    break
                pbar.set_description(f"Epoch {epoch + 1}/{self.epochs}")
                for callback in self.callbacks:
                    callback.before_training_epoch(epoch)
                train_metrics = self._train(epoch, pbar)
                for callback in self.callbacks:
                    callback.after_training_epoch(epoch, train_metrics)

                pbar.set_description(f"Epoch {epoch + 1}/{self.epochs} (validation)")
                for callback in self.callbacks:
                    callback.before_validation_epoch(epoch)
                with torch.no_grad():
                    val_metrics = self._validate(epoch, pbar, train_metrics)
                for callback in self.callbacks:
                    callback.after_validation_epoch(epoch, train_metrics | val_metrics)

                pbar.set_description(f"Epoch {epoch + 1}/{self.epochs} (done)")
                display_metrics = {
                    k: v.item() for k, v in (train_metrics | val_metrics).items()
                }
                pbar.set_postfix(display_metrics)

        for callback in self.callbacks:
            callback.after_training(train_metrics | val_metrics)

    def predict(self, batch: dict[str, Any]) -> npt.NDArray[np.float32]:
        self.classifier.eval()
        with torch.no_grad():
            device_batch = _dict_to_device(batch, self.device)
            # Use the classifier to get the batch classes
            prediction = self.classifier(device_batch["x"])
            prediction = torch.sigmoid(prediction)
            return torch.squeeze(prediction).cpu().numpy()  # type: ignore

    def get_state_dict(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "model_state_dict": self.classifier.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_steps_per_epoch": self.train_steps_per_epoch,
                "val_steps_per_epoch": self.val_steps_per_epoch,
            }
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.classifier.load_state_dict(state_dict["model_state_dict"])
        # HACK: we need to move the model to the desired device *before* loading the
        # optimizer's state dict, otherwise the optimizer will complain that the model
        # is on the wrong device
        self.classifier = self.classifier.to(self.device, non_blocking=True)
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        self.train_steps_per_epoch = state_dict["train_steps_per_epoch"]
        self.val_steps_per_epoch = state_dict["val_steps_per_epoch"]
        if len(self.train_steps_per_epoch) != len(self.val_steps_per_epoch):
            raise ValueError(
                "The number of training and validation steps per epoch must be the same"
            )
        last_epoch = len(self.train_steps_per_epoch)
        self.initial_epoch = last_epoch

    def _train(self, epoch: int, pbar: tqdm) -> dict[str, torch.Tensor]:
        self.classifier.train()
        metrics, batches_count = self._run_epoch(True, epoch, pbar, {})
        if self._train_batches_count is None:
            self._train_batches_count = batches_count
        else:
            assert self._train_batches_count == batches_count
        self.train_steps_per_epoch.append(batches_count)
        return metrics

    def _validate(
        self,
        epoch: int,
        pbar: tqdm,
        train_metrics: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        self.classifier.eval()
        metrics, batches_count = self._run_epoch(False, epoch, pbar, train_metrics)
        if self._val_batches_count is None:
            self._val_batches_count = batches_count
        else:
            assert self._val_batches_count == batches_count
        self.val_steps_per_epoch.append(batches_count)
        return metrics

    def _run_epoch(
        self,
        training: bool,
        epoch: int,
        pbar: tqdm,
        initial_metrics: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], int]:
        metrics = {}
        dl = self.train_dataloader if training else self.val_dataloader
        self._reset_metrics()
        batches_count = 0
        for step, batch in enumerate(dl):
            batches_count += 1
            device_batch = _dict_to_device(batch, self.device)
            for callback in self.callbacks:
                if training:
                    callback.before_training_step(epoch, step, device_batch)
                else:
                    callback.before_validation_step(epoch, step, device_batch)
            # Use the classifier to get the batch classes
            if training:
                self.optimizer.zero_grad()
            prediction = self.classifier(device_batch["x"])
            prediction = torch.squeeze(prediction)
            ground_truth = torch.squeeze(device_batch["y"])
            # Compute the loss
            loss = self.loss_function(prediction, ground_truth)
            # Before updating the metrics, apply a sigmoid to the prediction
            # (the output of the classifier is not a probability)
            prediction = torch.sigmoid(prediction)
            self._update_metrics(prediction, ground_truth)
            # Do backpropagation and optimize weights
            if training:
                loss.backward()
                self.optimizer.step()
            # Compute the metrics up until this point
            metrics = self._compute_metrics_dict(loss, "val" if not training else "")
            metrics = initial_metrics | metrics
            # Call .item() so we don't have tensor() around each number
            display_metrics = {k: v.item() for k, v in metrics.items()}
            for callback in self.callbacks:
                if training:
                    callback.after_training_step(epoch, step, device_batch, metrics)
                else:
                    callback.after_validation_step(epoch, step, device_batch, metrics)
            pbar.update(1)
            pbar.set_postfix(display_metrics)
        # At the end the metrics dictionary will contain the metrics for the whole epoch
        return metrics, batches_count

    def _reset_metrics(self) -> None:
        with torch.no_grad():
            for metric in self.metrics:
                metric.reset()

    def _update_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        with torch.no_grad():
            for metric in self.metrics:
                metric.update(y_pred, y_true)

    def _compute_metrics_dict(
        self, loss: torch.Tensor, metric_prefix: str = ""
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            loss_name = "loss" if metric_prefix == "" else f"{metric_prefix}_loss"
            metrics = {loss_name: loss.cpu()}
            for metric in self.metrics:
                metric_dict = metric.compute_to_dict()
                for k, v in metric_dict.items():
                    if metric_prefix != "":
                        metrics[f"{metric_prefix}_{k}"] = v
                    else:
                        metrics[k] = v
            return metrics
