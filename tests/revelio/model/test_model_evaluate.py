import unittest.mock as mock
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import numpy.typing as npt
import pytest
import torch

from revelio.config import Config
from revelio.config.model import Experiment
from revelio.config.model import Metric as ConfigMetric
from revelio.config.model import Scores
from revelio.model import Model
from revelio.model.metrics import Metric


class DummyAccuracy(Metric):
    name: str = "accuracy"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
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


class DummyListMetric(Metric):
    @property
    def name(self) -> list[str]:
        return ["name1", self.test_arg]

    def __init__(self, test_arg: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.test_arg = test_arg

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        pass

    def compute(self) -> torch.Tensor:
        return torch.tensor([0.125, 0.25])

    def reset(self) -> None:
        pass


class CorrectDummyModel(Model):
    def fit(self) -> None:
        pass

    def predict(self, batch: dict[str, Any]) -> npt.NDArray[np.double]:
        return np.array([0.9, 0.1, 0.8, 0.2])


class WrongDummyModel(Model):
    def fit(self) -> None:
        pass

    def predict(self, batch: dict[str, Any]) -> npt.NDArray[np.double]:
        return np.array([[0.9, 1], [0.1, 0], [0.8, 1], [0.2, 0]])


@pytest.fixture
def config() -> Config:
    return Config.construct(
        experiment=Experiment.construct(
            scores=Scores.construct(
                bona_fide=Path("bona_fide_scores.txt"),
                morphed=Path("morphed_scores.txt"),
            ),
            metrics=[
                ConfigMetric.construct(
                    name="dummyaccuracy",
                ),
                ConfigMetric.construct(
                    name="dummy_list_metric",
                    args={"test_arg": "test"},
                ),
            ],
        )
    )


@pytest.fixture
def test_dataloader() -> Iterable[dict[str, Any]]:
    return iter(
        [
            {
                "x": None,
                "y": torch.tensor([1, 0, 1, 0]),
                "dataset": ["ds1", "ds1", "ds2", "ds2"],
            }
        ]
    )


def test_model_evaluate(
    config: Config, test_dataloader: Iterable[dict[str, Any]]
) -> None:
    model = CorrectDummyModel(
        config=config,
        train_dataloader=None,  # type: ignore
        val_dataloader=None,  # type: ignore
        test_dataloader=test_dataloader,  # type: ignore
        device="cpu",
    )
    with (
        mock.patch("pathlib.Path.mkdir", return_value=None),
        mock.patch("numpy.savetxt") as mock_save,
    ):
        metrics = model.evaluate()
        # We have to manually check the call args because Numpy arrays override __eq__
        for call in mock_save.call_args_list:
            args, _ = call
            if args[0] == "bona_fide_scores.txt":
                assert np.array_equal(args[1], np.array([0.1, 0.2]))
            elif args[0] == "morphed_scores.txt":
                assert np.array_equal(args[1], np.array([0.9, 0.8]))
        assert metrics == {
            "all": {"accuracy": 1.0, "name1": 0.125, "test": 0.25},
            "ds1": {"accuracy": 1.0, "name1": 0.125, "test": 0.25},
            "ds2": {"accuracy": 1.0, "name1": 0.125, "test": 0.25},
        }


def test_model_evaluate_shape_error(
    config: Config, test_dataloader: Iterable[dict[str, Any]]
) -> None:
    model = WrongDummyModel(
        config=config,
        train_dataloader=None,  # type: ignore
        val_dataloader=None,  # type: ignore
        test_dataloader=test_dataloader,  # type: ignore
        device="cpu",
    )
    with pytest.raises(ValueError):
        model.evaluate()
