from abc import abstractmethod
from pathlib import Path

from revelio.dataset.element import DatasetElementDescriptor
from revelio.registry.registry import Registrable

__all__ = ("DatasetLoader",)


class DatasetLoader(Registrable):

    suffix = "Loader"

    @abstractmethod
    def load(self, path: Path) -> list[DatasetElementDescriptor]:
        raise NotImplementedError  # pragma: no cover
