from typing import Any, Iterator, Optional

import numpy as np
from PIL import Image as ImageModule
from torch.utils.data import IterableDataset, get_worker_info

from revelio.dataset.element import DatasetElement, ElementImage
from revelio.face_detection import FaceDetector


class Dataset(IterableDataset):
    def __init__(
        self,
        paths: list[DatasetElement],
        face_detector: Optional[FaceDetector],
        augmentation_steps: list,  # TODO: add type
        feature_extractor: None,  # TODO: add type
    ) -> None:
        self.paths = paths
        self.face_detector = face_detector
        self.augmentation_steps = augmentation_steps
        self.feature_extractor = feature_extractor

    def __iter__(self) -> Iterator[dict[str, Any]]:
        elems = self._get_elems_iterator()
        for elem in elems:
            # Load the images
            new_xs = []
            for x in elem.x:
                img = ImageModule.open(x.path)
                new_x = ElementImage(x.path, img, None, None)
                new_xs.append(new_x)
            elem = DatasetElement(
                original_dataset=elem.original_dataset,
                x=tuple(new_xs),
                y=elem.y,
            )
            if self.face_detector is not None:
                elem = self.face_detector.process(elem)
            for step in self.augmentation_steps:
                elem = step.process(elem)
            if self.feature_extractor is not None:
                elem = self.feature_extractor.process(elem)
            yield {
                "x": [
                    {
                        "image": np.array(x.image),
                        "landmarks": x.landmarks,
                        "features": x.features,
                    }
                    for x in elem.x
                ]
            }

    def _get_elems_iterator(self) -> Iterator[DatasetElement]:
        worker_info = get_worker_info()
        if worker_info is None:
            return iter(self.paths)
        else:
            for i, p in enumerate(self.paths):
                if i % worker_info.num_workers == worker_info.id:
                    yield p
