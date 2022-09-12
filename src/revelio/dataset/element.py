from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from PIL.Image import Image


class ElementClass(Enum):
    BONA_FIDE = 0.0
    MORPHED = 1.0


class ElementImage:
    _path: Path
    _image: Optional[Image]
    _landmarks: Optional[np.ndarray]
    _features: dict[str, np.ndarray]

    def __init__(
        self,
        path: Path,
        image: Optional[Image] = None,
        landmarks: Optional[np.ndarray] = None,
        features: Optional[dict[str, np.ndarray]] = None,
    ) -> None:
        self._path = path
        self._image = image
        self._landmarks = landmarks
        self._features = features or {}

    @property
    def path(self) -> Path:
        return self._path

    @property
    def image(self) -> Optional[Image]:
        return self._image

    @property
    def landmarks(self) -> Optional[np.ndarray]:
        return self._landmarks

    @property
    def features(self) -> dict[str, np.ndarray]:
        return self._features


class DatasetElement:
    _original_dataset: str
    _x: tuple[ElementImage, ...]
    _y: ElementClass

    def __init__(
        self,
        x: tuple[ElementImage, ...],
        y: ElementClass,
        original_dataset: str,
    ) -> None:
        self._x = x
        self._y = y
        self._original_dataset = original_dataset

    @property
    def original_dataset(self) -> str:
        return self._original_dataset

    @property
    def x(self) -> tuple[ElementImage, ...]:
        return self._x

    @property
    def y(self) -> ElementClass:
        return self._y