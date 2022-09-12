import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Type, TypeAlias

import numpy as np
from PIL import Image as ImageModule
from PIL.Image import Image

from revelio.config.config import Config
from revelio.dataset.element import DatasetElement, ElementImage

__all__ = ("FaceDetector",)


_algorithms: dict[str, Type["FaceDetector"]] = {}


BoundingBox: TypeAlias = tuple[int, int, int, int]
Landmarks: TypeAlias = np.ndarray


def _find_face_detector(name: str, config: Config, **kwargs: Any) -> "FaceDetector":
    lowercase_algorithms = [k.lower() for k in _algorithms.keys()]
    wanted_algorithm = name.lower()
    if wanted_algorithm not in lowercase_algorithms:
        raise ValueError(f"Could not find a face detection algorithm named {name}")
    # Get the correct face detection algorithm name from the lowercase list
    algorithm_index = lowercase_algorithms.index(wanted_algorithm)
    algorithm_name = list(_algorithms.keys())[algorithm_index]
    return _algorithms[algorithm_name](config=config, **kwargs)


class FaceDetector(ABC):
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Make sure there is no other algorithm with the same case-insensitive name
        lowercase_algorithms = [k.lower() for k in _algorithms.keys()]
        if cls.__name__.lower() in lowercase_algorithms:
            raise TypeError(f"Face detection algorithm {cls.__name__} already exists")
        _algorithms[cls.__name__] = cls

    def __init__(self, config: Config) -> None:
        self._config = config

    def _get_meta_path(self, elem: DatasetElement, x_idx: int) -> Path:
        output_path = Path(self._config.face_detection.output_path)
        algorithm_name = type(self).__name__.lower()
        return (
            output_path
            / algorithm_name
            / elem.original_dataset
            / (elem.x[x_idx].path.stem + ".meta.json")
        )

    @abstractmethod
    def process_element(self, elem: Image) -> tuple[BoundingBox, Optional[Landmarks]]:
        raise NotImplementedError

    def process(self, elem: DatasetElement) -> DatasetElement:
        new_xs = []
        for i, x in enumerate(elem.x):
            meta_path = self._get_meta_path(elem, i)
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text())
                landmarks = np.array(meta["landmarks"]) if "landmarks" in meta else None
                if "bb" in meta:
                    # We have the bounding boxes, skip loading a new image
                    # and instead crop the one we already have
                    if x.image is not None:
                        image = x.image.crop(meta["bb"])
                    else:
                        image = ImageModule.open(x.path).crop(meta["bb"])
                    new_x = ElementImage(
                        path=x.path,
                        image=image,
                        landmarks=landmarks,
                    )
                    new_xs.append(new_x)
                else:
                    raise ValueError(f"No bounding box found in {meta_path}")
            else:
                image = x.image if x.image is not None else ImageModule.open(x.path)
                bb, landmarks = self.process_element(image)
                new_x = ElementImage(
                    path=x.path,
                    image=image.crop(bb),
                    landmarks=landmarks,
                )
                meta = {
                    "bb": bb,
                    "landmarks": landmarks,
                }
                # Create the meta file
                meta_path.write_text(json.dumps(meta))
                new_xs.append(new_x)
        return DatasetElement(
            original_dataset=elem.original_dataset,
            x=tuple(new_xs),
            y=elem.y,
        )
