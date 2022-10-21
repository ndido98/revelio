import logging
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader

from revelio.config.config import Config
from revelio.registry.registry import Registrable
from revelio.utils.random import set_seed

from .metrics.metric import Metric

log = logging.getLogger(__name__)


def _format_path(path: Path, **format: Any) -> Path:
    return Path(str(path).format(**format))


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
        self.metrics = [
            Metric.find(m.name, _device=device, **m.args)
            for m in config.experiment.metrics
        ]
        self.device = device

    @abstractmethod
    def fit(self) -> None:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def predict(self, batch: dict[str, Any]) -> npt.NDArray[np.float32]:
        raise NotImplementedError  # pragma: no cover

    def evaluate(self) -> Mapping[str, Mapping[str, npt.ArrayLike]]:
        # Reset the seed so we are sure to get the same results
        if self.config.seed is not None:
            set_seed(self.config.seed)
        scores_list: list[npt.NDArray[np.float32]] = []
        labels_list: list[int] = []
        original_datasets_list: list[str] = []
        testing_groups = self._get_testing_groups_datasets()
        log.debug("Found testing groups: %s", testing_groups)
        # Predict on each element of the test set
        for elem in self.test_dataloader:
            batch_scores = self.predict(elem)
            batch_scores = np.atleast_1d(batch_scores)
            if batch_scores.ndim != 1:
                raise ValueError("predict() must return a 1D-array of scores")
            batch_gt = elem["y"].cpu().numpy()
            batch_dataset = elem["dataset"]
            scores_list.append(np.atleast_1d(batch_scores))
            labels_list.append(np.atleast_1d(batch_gt))
            if isinstance(batch_dataset, list):
                original_datasets_list.extend(batch_dataset)
            else:
                original_datasets_list.append(batch_dataset)
        scores = np.concatenate(scores_list)
        labels = np.concatenate(labels_list)
        original_datasets = np.array(original_datasets_list)
        # Create the directories for the scores
        self.config.experiment.scores.bona_fide.parent.mkdir(
            parents=True, exist_ok=True
        )
        self.config.experiment.scores.morphed.parent.mkdir(parents=True, exist_ok=True)
        # Compute metrics for each testing group and save the scores to file
        metrics: dict[str, Mapping[str, npt.ArrayLike]] = {}
        for group, datasets in testing_groups.items():
            mask = np.zeros_like(original_datasets, dtype=bool)
            for dataset in datasets:
                mask |= original_datasets == dataset
            try:
                metrics[group] = self._compute_metrics(scores, labels, mask=mask)
            except Exception as e:
                raise RuntimeError(
                    f"Error while computing metrics for testing group {group}"
                ) from e
            bona_fide_scores = scores[mask][labels[mask] == 0]
            morphed_scores = scores[mask][labels[mask] == 1]
            formatted_bona_fide = _format_path(
                self.config.experiment.scores.bona_fide,
                group=group,
                now=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                timestamp=datetime.now().timestamp(),
                today=datetime.now().strftime("%Y-%m-%d"),
            )
            formatted_morphed = _format_path(
                self.config.experiment.scores.morphed,
                group=group,
                now=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                timestamp=datetime.now().timestamp(),
                today=datetime.now().strftime("%Y-%m-%d"),
            )
            np.savetxt(formatted_bona_fide, bona_fide_scores, "%.5f")
            np.savetxt(formatted_morphed, morphed_scores, "%.5f")
        return metrics

    def _compute_metrics(
        self,
        scores: np.ndarray[int, np.dtype[np.float32]],
        labels: np.ndarray[int, np.dtype[np.uint8]],
        mask: np.ndarray[int, np.dtype[np.bool_]],
    ) -> Mapping[str, npt.ArrayLike]:
        computed_metrics = {}
        for metric in self.metrics:
            metric.reset()
            metric.update(
                torch.from_numpy(scores[mask]).to(self.device),
                torch.from_numpy(labels[mask]).to(self.device),
            )
            metric_dict = metric.compute_to_dict()
            for key, value in metric_dict.items():
                np_value = value.numpy()
                metric_dict[key] = np_value if np_value.size > 1 else np_value.item()
            computed_metrics.update(metric_dict)
        return computed_metrics

    def _get_testing_groups_datasets(self) -> dict[str, set[str]]:
        testing_groups = {}
        for dataset in self.config.datasets:
            for testing_group in dataset.testing_groups:
                if testing_group not in testing_groups:
                    testing_groups[testing_group] = {dataset.name}
                else:
                    testing_groups[testing_group].add(dataset.name)
        testing_groups["all"] = {ds.name for ds in self.config.datasets}
        return testing_groups
