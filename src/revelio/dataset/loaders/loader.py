from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Type

from revelio.dataset.element import DatasetElement

__all__ = ("DatasetLoader",)


_loaders: dict[str, Type["DatasetLoader"]] = {}


def _find_loader(name: str) -> "DatasetLoader":
    # The loader's name for the XYZ dataset should be XYZLoader
    lowercase_loaders = [k.lower() for k in _loaders.keys()]
    wanted_loader = f"{name.lower()}loader"
    if wanted_loader not in lowercase_loaders:
        raise ValueError(f"Could not find a loader for the dataset {name}")
    # Get the correct loader name from the lowercase list
    loader_index = lowercase_loaders.index(wanted_loader)
    loader_name = list(_loaders.keys())[loader_index]
    return _loaders[loader_name]()


class DatasetLoader(ABC):
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Make sure there is no other loader with the same case-insensitive name
        lowercase_loaders = [k.lower() for k in _loaders.keys()]
        if cls.__name__.lower() in lowercase_loaders:
            raise TypeError(f"Loader {cls.__name__} already exists")
        _loaders[cls.__name__] = cls

    @abstractmethod
    def load(self, path: Path) -> list[DatasetElement]:
        raise NotImplementedError  # pragma: no cover
