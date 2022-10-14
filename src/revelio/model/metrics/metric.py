from abc import abstractmethod

import torch

from revelio.registry.registry import Registrable


class Metric(Registrable):
    def __init__(self, *, _device: str) -> None:
        self._device = _device

    @property
    def device(self) -> str:
        return self._device

    @property
    @abstractmethod
    def name(self) -> str | list[str]:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def compute(self) -> torch.Tensor:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError  # pragma: no cover

    def compute_to_dict(self) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            name = self.name
            try:
                result = self.compute().cpu()
            except Exception as e:
                raise RuntimeError(
                    f"Error while computing metric {type(self).__name__}"
                ) from e
            if isinstance(name, list):
                if len(name) != len(result):
                    raise ValueError(
                        f"The metric {type(self).__name__} returned "
                        f"{len(name)} metric names, "
                        f"but {len(result)} metric results"
                    )
                # Make sure the metric names don't start with a reserved prefix
                for n in name:
                    self._check_metric_name_valid(n)
                return {n: v for n, v in zip(name, result)}
            else:
                self._check_metric_name_valid(name)
                return {name: result}

    def _check_metric_name_valid(self, name: str) -> None:
        reserved_prefixes = ("val_", "epoch_val_", "epoch_")
        for prefix in reserved_prefixes:
            if name.startswith(prefix):
                raise ValueError(
                    f"The metric {type(self).__name__} contains a value "
                    f"which starts with the reserved prefix '{prefix}'"
                )
