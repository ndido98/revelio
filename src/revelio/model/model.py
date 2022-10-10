from abc import abstractmethod
from typing import Any, Mapping

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
        scores_list: list[npt.NDArray[np.float32]] = []
        labels_list: list[int] = []
        original_datasets_list: list[str] = []
        for elem in self.test_dataloader:
            batch_scores = self.predict(elem)
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
        metrics: dict[str, Mapping[str, npt.ArrayLike]] = {}
        metrics["all"] = self._compute_metrics(
            scores, labels, mask=np.ones_like(scores, dtype=bool)
        )
        for dataset in np.unique(original_datasets):
            dataset_mask = original_datasets == dataset
            dataset_metrics = self._compute_metrics(scores, labels, mask=dataset_mask)
            metrics[dataset] = dataset_metrics
        bona_fide_scores = scores[labels == 0]
        morphed_scores = scores[labels == 1]
        self.config.experiment.scores.bona_fide.parent.mkdir(
            parents=True, exist_ok=True
        )
        self.config.experiment.scores.morphed.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(self.config.experiment.scores.bona_fide, bona_fide_scores, "%.5f")
        np.savetxt(self.config.experiment.scores.morphed, morphed_scores, "%.5f")
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
        return computed_metrics
