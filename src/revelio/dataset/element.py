from enum import Enum
from pathlib import Path
from typing import Any, Optional, TypeAlias

import numpy as np
import numpy.typing as npt

Image: TypeAlias = npt.NDArray[np.uint8 | np.float32]
Landmarks: TypeAlias = npt.NDArray[np.int32]


class ElementClass(Enum):
    """
    The class of a dataset element, which can be either bona fide or morphed.
    """

    BONA_FIDE = 0.0
    MORPHED = 1.0


class DatasetElementDescriptor:
    """
    A descriptor of a dataset element before it is loaded into memory.

    Attributes:
        x: The path to the image file(s).
        y: The class of the dataset element.
    """

    _x: tuple[Path, ...]
    _y: ElementClass
    _root_path: Path
    _dataset_name: str

    def __init__(self, x: tuple[Path, ...], y: ElementClass) -> None:
        """
        Creates a new dataset element descriptor.

        Args:
            x: The path to the image file(s).
            y: The class of the dataset element.
        """
        self._x = x
        self._y = y

    @property
    def x(self) -> tuple[Path, ...]:
        """
        Gets the path to the image file(s).
        """
        return self._x

    @property
    def y(self) -> ElementClass:
        """
        Gets the class of the dataset element.
        """
        return self._y

    def __repr__(self) -> str:
        return f"DatasetElementDescriptor(x={self.x}, y={self.y})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DatasetElementDescriptor):
            return NotImplemented
        return (
            self.x == other.x
            and self.y == other.y
            and self._root_path == other._root_path
            and self._dataset_name == other._dataset_name
        )

    def __hash__(self) -> int:
        return hash((self.x, self.y, self._root_path, self._dataset_name))


class ElementImage:
    """
    An image that is part of a dataset element.

    An image is represented with a Numpy array with shape (height, width, channels)
    and dtype uint8 or float32. The channels are ordered as BGR, following the OpenCV
    convention.

    Attributes:
        path: The path to the image file.
        image: The image.
        landmarks: The facial landmarks of the image (if present).
        features: The features of the image produced by each feature extractor
            (if present).
    """

    _path: Path
    _image: Image
    _landmarks: Optional[Landmarks]
    _features: dict[str, np.ndarray]

    def __init__(
        self,
        path: Path,
        image: Image,
        landmarks: Optional[Landmarks] = None,
        features: Optional[dict[str, np.ndarray]] = None,
    ) -> None:
        """
        Creates a new image of a dataset element.

        Args:
            path: The path to the image file.
            image: The image.
            landmarks: The facial landmarks of the image (if present).
            features: The features of the image produced by each feature extractor
                (if present).
        """
        self._path = path
        self._image = image
        self._landmarks = landmarks
        self._features = features or {}

    @property
    def path(self) -> Path:
        """
        Gets the path to the image file.
        """
        return self._path

    @property
    def image(self) -> Image:
        """
        Gets the image.
        """
        return self._image

    @property
    def landmarks(self) -> Optional[Landmarks]:
        """
        Gets the facial landmarks of the image (if present).
        """
        return self._landmarks

    @property
    def features(self) -> dict[str, np.ndarray]:
        """
        Gets the features of the image produced by each feature extractor (if present).
        """
        return self._features

    def __repr__(self) -> str:
        return f"ElementImage(path={self._path})"


class DatasetElement:
    """
    An element of the dataset.

    Attributes:
        x: The image(s) of the dataset element.
        y: The class of the dataset element.
        original_dataset: The name of the original dataset from which the element
            was taken. It is equal to the dataset name specified in the configuration
            file.
        dataset_root_path: The path to the root directory of the dataset.
    """

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
        original_dataset: str,
    ) -> None:
        """
        Creates a new dataset element.

        Args:
            x: The image(s) of the dataset element.
            y: The class of the dataset element.
            dataset_root_path: The path to the root directory of the dataset.
            original_dataset: The name of the original dataset from which the element
                was taken.
        """
        self._x = x
        self._y = y
        self._dataset_root_path = dataset_root_path
        self._original_dataset = original_dataset

    @property
    def dataset_root_path(self) -> Path:
        """
        Gets the path to the root directory of the dataset.
        """
        return self._dataset_root_path

    @property
    def original_dataset(self) -> str:
        """
        Gets the name of the original dataset from which the element was taken.
        """
        return self._original_dataset

    @property
    def x(self) -> tuple[ElementImage, ...]:
        """
        Gets the image(s) of the dataset element.
        """
        return self._x

    @property
    def y(self) -> ElementClass:
        """
        Gets the class of the dataset element.
        """
        return self._y

    def __repr__(self) -> str:
        return f"DatasetElement(x={self.x}, y={self.y})"
