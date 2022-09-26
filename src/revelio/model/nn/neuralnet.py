from abc import abstractmethod
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm

from revelio.model.model import Model
from revelio.registry.registry import Registrable

from .callbacks.callback import Callback
from .losses.loss import Loss
from .optimizers.optimizer import Optimizer


def _dict_to_device(data: dict[str, Any], device: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        elif isinstance(v, dict):
            result[k] = _dict_to_device(v, device)
        else:
            raise TypeError(f"Unexpected type {type(v)} when trying to move to device")
    return result


class NeuralNetwork(Model):

    transparent: bool = True

    epochs: int
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
        if "optimizer" not in self.config.experiment.training.args:
            raise ValueError("Missing optimizer in training configuration")
        if "name" not in self.config.experiment.training.args["optimizer"]:
            raise ValueError("Missing optimizer name in training configuration")
        found_optimizer: Optimizer = Registrable.find(
            Optimizer,
            self.config.experiment.training.args["optimizer"]["name"],
        )
        self.optimizer = found_optimizer.get(
            params=self.classifier.parameters(),
            **self.config.experiment.training.args["optimizer"].get("args", {}),
        )
        if "loss" not in self.config.experiment.training.args:
            raise ValueError("Missing loss in training configuration")
        if "name" not in self.config.experiment.training.args["loss"]:
            raise ValueError("Missing loss name in training configuration")
        found_loss: Loss = Registrable.find(
            Loss,
            self.config.experiment.training.args["loss"]["name"],
        )
        self.loss_function = found_loss.get(
            **self.config.experiment.training.args["loss"].get("args", {}),
        )
        self.loss_function = self.loss_function.to(self.device)
        # Load the callbacks
        self.callbacks = []
        for callback in self.config.experiment.training.args.get("callbacks", []):
            if "name" not in callback:
                raise ValueError("Missing callback name in training configuration")
            found_callback: Callback = Registrable.find(
                Callback,
                callback["name"],
                **callback.get("args", {}),
            )
            self.callbacks.append(found_callback)
        self.should_stop = False
        # Load a checkpoint if present
        if self.config.experiment.model.checkpoint is not None:
            checkpoint = torch.load(
                self.config.experiment.model.checkpoint,
                map_location="cpu",
            )
            self.load_state_dict(checkpoint)
        else:
            self._initial_epoch = 0
        self._last_epoch: Optional[int] = None
        # Move the model to the device
        self.classifier = self.classifier.to(self.device)
        # Once the model is fully initialized, pass a reference to it to the callbacks
        for callback in self.callbacks:
            callback.model = self

    @abstractmethod
    def get_classifier(self, **kwargs: Any) -> torch.nn.Module:
        raise NotImplementedError  # pragma: no cover

    def fit(self) -> None:
        self.should_stop = False
        for callback in self.callbacks:
            callback.before_training()

        pbar_len = len(self.train_dataloader) + len(self.val_dataloader)
        for epoch in range(self._initial_epoch, self.epochs):
            with tqdm(total=pbar_len) as pbar:
                if self.should_stop:
                    break
                self._last_epoch = epoch
                pbar.set_description(f"Epoch {epoch + 1}/{self.epochs}")
                for callback in self.callbacks:
                    callback.before_training_epoch(epoch, len(self.train_dataloader))
                train_metrics = self._train(epoch, pbar)
                for callback in self.callbacks:
                    callback.after_training_epoch(
                        epoch, len(self.train_dataloader), train_metrics
                    )

                pbar.set_description(f"Epoch {epoch + 1}/{self.epochs} (validation)")
                for callback in self.callbacks:
                    callback.before_validation_epoch(epoch, len(self.val_dataloader))
                with torch.no_grad():
                    val_metrics = self._validate(epoch, pbar, train_metrics)
                for callback in self.callbacks:
                    callback.after_validation_epoch(
                        epoch, len(self.val_dataloader), train_metrics | val_metrics
                    )

                pbar.set_description(f"Epoch {epoch + 1}/{self.epochs} (done)")
                display_metrics = {
                    k: v.item() for k, v in (train_metrics | val_metrics).items()
                }
                pbar.set_postfix(display_metrics)

        for callback in self.callbacks:
            callback.after_training(train_metrics | val_metrics)

    def predict(self) -> npt.NDArray[np.double]:
        scores = []
        for batch in self.test_dataloader:
            device_batch = _dict_to_device(batch, self.device)
            # Use the classifier to get the batch classes
            prediction = self.classifier(device_batch["x"])
            prediction = torch.squeeze(prediction)
            ground_truth = torch.squeeze(device_batch["y"])
            scores.append(np.array([prediction, ground_truth]))
        return np.stack(scores)

    def get_state_dict(self) -> dict[str, Any]:
        return {
            "model_state_dict": self.classifier.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self._last_epoch,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.classifier.load_state_dict(state_dict["model_state_dict"])
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        last_epoch = state_dict["epoch"]
        self._initial_epoch = last_epoch + 1 if last_epoch is not None else 0

    def _train(self, epoch: int, pbar: tqdm) -> dict[str, torch.Tensor]:
        self.classifier.train()
        return self._run_epoch(True, epoch, pbar, {})

    def _validate(
        self,
        epoch: int,
        pbar: tqdm,
        train_metrics: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        self.classifier.eval()
        return self._run_epoch(False, epoch, pbar, train_metrics)

    def _run_epoch(
        self,
        training: bool,
        epoch: int,
        pbar: tqdm,
        initial_metrics: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        cumulative_loss = torch.tensor(0.0)
        metrics = {}
        dl = self.train_dataloader if training else self.val_dataloader
        for step, batch in enumerate(dl):
            device_batch = _dict_to_device(batch, self.device)
            self._reset_metrics()
            for callback in self.callbacks:
                if training:
                    callback.before_training_step(epoch, step)
                else:
                    callback.before_validation_step(epoch, step)
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
            cumulative_loss += loss.item()
            # Compute the metrics up until this point
            metrics = self._compute_metrics_dict(
                cumulative_loss / (step + 1), "val" if not training else ""
            )
            metrics = initial_metrics | metrics
            # Call .item() so we don't have tensor() around each number
            display_metrics = {k: v.item() for k, v in metrics.items()}
            for callback in self.callbacks:
                if training:
                    callback.after_training_step(epoch, step, metrics)
                else:
                    callback.after_validation_step(epoch, step, metrics)
            pbar.update(1)
            pbar.set_postfix(display_metrics)
        # At the end the metrics dictionary will contain the metrics for the whole epoch
        return metrics

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
            metrics = {}
            for metric in self.metrics:
                metric_name = metric.name
                metric_result = metric.compute()
                if isinstance(metric_name, list):
                    if len(metric_name) != len(metric_result):
                        raise ValueError(
                            f"The metric {type(metric).__name__} returned "
                            f"{len(metric_name)} metric names, "
                            f"but {len(metric_result)} metric results"
                        )
                    for name, value in zip(metric_name, metric_result):
                        if name.startswith("val_"):
                            raise ValueError(
                                f"The metric {type(metric).__name__} contains a value "
                                "which starts with the reserved prefix 'val_'"
                            )
                        if metric_prefix != "":
                            name = f"{metric_prefix}_{name}"
                        metrics[name] = value
                else:
                    if metric_name.startswith("val_"):
                        raise ValueError(
                            f"The metric {type(metric).__name__} contains a value "
                            "which starts with the reserved prefix 'val_'"
                        )
                    if metric_prefix != "":
                        metric_name = f"{metric_prefix}_{metric_name}"
                    metrics[metric_name] = metric_result
            loss_name = "loss" if metric_prefix == "" else f"{metric_prefix}_loss"
            metrics[loss_name] = loss
            return metrics
