from pathlib import Path

import pytest

from revelio.dataset.element import DatasetElement
from revelio.dataset.loaders.loader import DatasetLoader, _find_loader


class ALoader(DatasetLoader):
    def load(self, path: Path) -> list[DatasetElement[Path]]:
        return []


class BLoader(DatasetLoader):
    def load(self, path: Path) -> list[DatasetElement[Path]]:
        return []


def test_find_loader() -> None:
    assert isinstance(_find_loader("a"), ALoader)
    assert isinstance(_find_loader("b"), BLoader)
    with pytest.raises(ValueError):
        _find_loader("c")


def test_find_loader_multiple_loaders_same_name() -> None:
    with pytest.raises(TypeError):
        # Add a class which already exists
        class ALoader(DatasetLoader):
            def load(self, path: Path) -> list[DatasetElement[Path]]:
                return []
