from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from revelio.config import Config
from revelio.config.model import Experiment
from revelio.config.model import Metric as ConfigMetric
from revelio.config.model import Model, Training
from revelio.model.metrics import Metric
from revelio.model.nn import NeuralNetwork
from revelio.model.nn.callbacks import Callback


class DebugCallback(Callback):
    def __init__(self) -> None:
        self.first_loss = float("inf")
        self.last_loss = float("inf")

    def after_training(self, metrics: dict[str, torch.Tensor]) -> None:
        assert "loss" in metrics
        assert "val_loss" in metrics
        assert "accuracy" in metrics
        assert "val_accuracy" in metrics
        self.last_loss = metrics["loss"].item()
        # The last loss must be less than the first loss
        assert self.last_loss < self.first_loss

    def after_training_epoch(
        self, epoch: int, steps_count: int, metrics: dict[str, torch.Tensor]
    ) -> None:
        assert "loss" in metrics
        assert "val_loss" not in metrics
        assert "accuracy" in metrics
        assert "val_accuracy" not in metrics
        if epoch == 0:
            self.first_loss = metrics["loss"].item()

    def after_training_step(
        self, epoch: int, step: int, metrics: dict[str, torch.Tensor]
    ) -> None:
        assert "loss" in metrics
        assert "val_loss" not in metrics
        assert "accuracy" in metrics
        assert "val_accuracy" not in metrics

    def after_validation_epoch(
        self, epoch: int, steps_count: int, metrics: dict[str, torch.Tensor]
    ) -> None:
        assert "loss" in metrics
        assert "val_loss" in metrics
        assert "accuracy" in metrics
        assert "val_accuracy" in metrics

    def after_validation_step(
        self, epoch: int, step: int, metrics: dict[str, torch.Tensor]
    ) -> None:
        assert "loss" in metrics
        assert "val_loss" in metrics
        assert "accuracy" in metrics
        assert "val_accuracy" in metrics


class DummyAccuracy2(Metric):
    name: str = "accuracy"

    def __init__(self) -> None:
        self.reset()

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        assert y_pred.shape == y_true.shape
        self.total += y_true.shape[0]
        y_pred_class = (y_pred >= 0.5).float()
        self.correct += (y_pred_class == y_true).sum()

    def compute(self) -> torch.Tensor:
        return self.correct / self.total

    def reset(self) -> None:
        self.correct = torch.tensor(0.0)
        self.total = torch.tensor(0.0)


class XORModel(NeuralNetwork):
    class Classifier(torch.nn.Module):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__()
            assert "testme" in kwargs
            assert kwargs["testme"] == "test"
            self.net = torch.nn.Sequential(
                torch.nn.Linear(2, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 1),
            )

        def forward(self, x: Any) -> Any:
            return self.net(x)

    def get_classifier(self, **kwargs: Any) -> torch.nn.Module:
        return self.Classifier(**kwargs)


@pytest.fixture
def config() -> Config:
    return Config.construct(
        experiment=Experiment.construct(
            model=Model.construct(
                name="XORModel",
                args={"testme": "test"},
            ),
            training=Training.construct(
                args={
                    "epochs": 10,
                    "optimizer": {
                        "name": "SGD",
                        "args": {"lr": 0.01},
                    },
                    "loss": {"name": "BCEWithLogitsLoss"},
                    "callbacks": [
                        {"name": "DebugCallback"},
                    ],
                },
            ),
            metrics=[
                ConfigMetric.construct(name="dummyaccuracy2"),
            ],
        )
    )


class XORDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.y = y

    def __getitem__(self, index: int) -> dict:
        return {"x": self.x[index], "y": self.y[index]}

    def __len__(self) -> int:
        return len(self.x)


def _generate_points(count: int) -> Dataset:
    c00 = np.random.rand(count, 2) * 0.5
    c01 = np.random.rand(count, 2) * 0.5 + np.array([0.5, 0.0])
    c10 = np.random.rand(count, 2) * 0.5 + np.array([0.0, 0.5])
    c11 = np.random.rand(count, 2) * 0.5 + 0.5
    x = np.concatenate([c00, c01, c10, c11])
    y = np.concatenate(
        [
            np.zeros(count),
            np.ones(count),
            np.ones(count),
            np.zeros(count),
        ]
    )
    return XORDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())


@pytest.fixture
def train_dataset() -> Dataset:
    return _generate_points(1000)


@pytest.fixture
def train_dataloader(train_dataset: Dataset) -> DataLoader:
    return DataLoader(train_dataset, batch_size=32, shuffle=True)


@pytest.fixture
def val_dataset() -> Dataset:
    return _generate_points(500)


@pytest.fixture
def val_dataloader(val_dataset: Dataset) -> DataLoader:
    return DataLoader(val_dataset, batch_size=32, shuffle=True)


@pytest.fixture
def test_dataset() -> Dataset:
    return _generate_points(500)


@pytest.fixture
def test_dataloader(test_dataset: Dataset) -> DataLoader:
    return DataLoader(test_dataset, batch_size=32, shuffle=True)


def test_model_training(
    config: Config,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
) -> None:
    model = XORModel(
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        device="cpu",
    )
    model.fit()
