from abc import abstractmethod

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
            Registrable.find(Metric, m) for m in config.experiment.metrics
        ]
        self.device = device

    @abstractmethod
    def fit(self) -> None:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def predict(self) -> npt.NDArray[np.double]:
        raise NotImplementedError  # pragma: no cover

    def evaluate(self) -> dict[str, npt.ArrayLike]:
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
            metric.update(torch.tensor(scores), torch.tensor(labels))
            computed_metrics[type(metric).name] = metric.compute().numpy()
        bona_fide_scores = scores[labels == 0]
        morphed_scores = scores[labels == 1]
        np.savetxt(self.config.experiment.scores.bona_fide, bona_fide_scores)
        np.savetxt(self.config.experiment.scores.morphed, morphed_scores)
        return computed_metrics
