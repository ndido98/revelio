from enum import Enum
from pathlib import Path
from typing import Optional, TypeAlias

import numpy as np

Image: TypeAlias = np.ndarray[int, np.dtype[np.uint8]]


class ElementClass(Enum):
    BONA_FIDE = 0.0
    MORPHED = 1.0


class DatasetElementDescriptor:
    _x: tuple[Path, ...]
    _y: ElementClass
    _root_path: Path
    _dataset_name: str

    def __init__(self, x: tuple[Path, ...], y: ElementClass) -> None:
        self._x = x
        self._y = y

    @property
    def x(self) -> tuple[Path, ...]:
        return self._x

    @property
    def y(self) -> ElementClass:
        return self._y

    def __repr__(self) -> str:
        return f"DatasetElementDescriptor(x={self.x}, y={self.y})"


class ElementImage:
    _path: Path
    _image: Image
    _landmarks: Optional[np.ndarray]
    _features: dict[str, np.ndarray]

    def __init__(
        self,
        path: Path,
        image: Image,
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
    def image(self) -> Image:
        return self._image

    @property
    def landmarks(self) -> Optional[np.ndarray]:
        return self._landmarks

    @property
    def features(self) -> dict[str, np.ndarray]:
        return self._features

    def __repr__(self) -> str:
        return f"ElementImage(path={self._path})"


class DatasetElement:
    _dataset_root_path: Path
    _original_dataset: str
    _x: tuple[ElementImage, ...]
    _y: ElementClass

    def __init__(
        self,
        x: tuple[ElementImage, ...],
        y: ElementClass,
        *,
        dataset_root_path: Path,
        original_dataset: Optional[str] = None,
    ) -> None:
        self._x = x
        self._y = y
        self._dataset_root_path = dataset_root_path
        if original_dataset is not None:
            self._original_dataset = original_dataset
        else:
            self._original_dataset = self._dataset_root_path.stem

    @property
    def dataset_root_path(self) -> Path:
        return self._dataset_root_path

    @property
    def original_dataset(self) -> str:
        return self._original_dataset

    @property
    def x(self) -> tuple[ElementImage, ...]:
        return self._x

    @property
    def y(self) -> ElementClass:
        return self._y

    def __repr__(self) -> str:
        return f"DatasetElement(x={self.x}, y={self.y})"
