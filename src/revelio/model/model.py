from abc import abstractmethod
from typing import Mapping

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader

from revelio.config.config import Config
from revelio.registry.registry import Registrable

from .metrics.metric import Metric


class Model(Registrable):
    def __init__(
        self,
        *,
        config: Config,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
        device: str,
    ):
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.metrics: list[Metric] = [
            Registrable.find(Metric, m.name, _device=device, **m.args)
            for m in config.experiment.metrics
        ]
        self.device = device

    @abstractmethod
    def fit(self) -> None:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def predict(self) -> npt.NDArray[np.double]:
        raise NotImplementedError  # pragma: no cover

    def evaluate(self) -> Mapping[str, npt.ArrayLike]:
        scores_labels = self.predict()
        if scores_labels.ndim != 2 or scores_labels.shape[1] != 2:
            raise ValueError(
                "The predict() method must return a 2D array, "
                "with scores in the left column and labels in the right column"
            )
        scores = scores_labels[:, 0]
        labels = scores_labels[:, 1]
        computed_metrics = {}
        for metric in self.metrics:
            metric.reset()
            metric.update(
                torch.from_numpy(scores).to(self.device),
                torch.from_numpy(labels).to(self.device),
            )
            metric_name = metric.name
            metric_result: torch.Tensor = metric.compute().cpu()
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
                    np_value = value.numpy()
                    computed_metrics[name] = (
                        np_value if np_value.size > 1 else np_value.item()
                    )
            else:
                if metric_name.startswith("val_"):
                    raise ValueError(
                        f"The metric {type(metric).__name__} contains a value "
                        "which starts with the reserved prefix 'val_'"
                    )
                np_result = metric_result.numpy()
                computed_metrics[metric_name] = (
                    np_result if np_result.size > 1 else np_result.item()
                )
        bona_fide_scores = scores[labels == 0]
        morphed_scores = scores[labels == 1]
        self.config.experiment.scores.bona_fide.parent.mkdir(
            parents=True, exist_ok=True
        )
        self.config.experiment.scores.morphed.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(self.config.experiment.scores.bona_fide, bona_fide_scores)
        np.savetxt(self.config.experiment.scores.morphed, morphed_scores)
        return computed_metrics
