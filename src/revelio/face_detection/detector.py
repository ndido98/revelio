import json
from abc import abstractmethod
from pathlib import Path
from typing import Optional, TypeAlias

import numpy as np

from revelio.config.config import Config
from revelio.dataset.element import DatasetElement, ElementImage, Image
from revelio.registry.registry import Registrable

BoundingBox: TypeAlias = tuple[int, int, int, int]
Landmarks: TypeAlias = np.ndarray


class FaceDetector(Registrable):
    def __init__(self, *, _config: Config) -> None:
        self._config = _config

    def _get_meta_path(self, elem: DatasetElement, x_idx: int) -> Path:
        output_path = Path(self._config.face_detection.output_path)
        algorithm_name = type(self).__name__.lower()
        relative_img_path = elem.x[x_idx].path.relative_to(elem.dataset_root_path)
        return (
            output_path
            / algorithm_name
            / elem.original_dataset
            / relative_img_path.parent
            / f"{relative_img_path.stem}.meta.json"
        )

    @abstractmethod
    def process_element(self, elem: Image) -> tuple[BoundingBox, Optional[Landmarks]]:
        raise NotImplementedError  # pragma: no cover

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
                    x1, y1, x2, y2 = meta["bb"]
                    image = x.image[y1:y2, x1:x2]
                    new_x = ElementImage(
                        path=x.path,
                        image=image,
                        landmarks=landmarks,
                    )
                    new_xs.append(new_x)
                else:
                    raise ValueError(f"No bounding box found in {meta_path}")
            else:
                try:
                    bb, landmarks = self.process_element(x.image)
                except Exception as e:
                    raise RuntimeError(f"Failed to process {x.path}: {e}") from e
                x1, y1, x2, y2 = bb
                new_x = ElementImage(
                    path=x.path,
                    image=x.image[y1:y2, x1:x2],
                    landmarks=landmarks,
                )
                meta = {
                    "bb": bb,
                    "landmarks": landmarks.tolist() if landmarks is not None else None,
                }
                # Create the meta file
                meta_path.parent.mkdir(parents=True, exist_ok=True)
                meta_path.write_text(json.dumps(meta))
                new_xs.append(new_x)
        return DatasetElement(
            dataset_root_path=elem.dataset_root_path,
            original_dataset=elem.original_dataset,
            x=tuple(new_xs),
            y=elem.y,
        )
