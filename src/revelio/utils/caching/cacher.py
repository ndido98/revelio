from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy.typing as npt


class Cacher(ABC):
    @abstractmethod
    def save(self, filename: Path, **data: npt.ArrayLike) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, filename: Path) -> dict[str, Any]:
        raise NotImplementedError
