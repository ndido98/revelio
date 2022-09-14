from typing import Any, Iterator, Optional

import numpy as np
from PIL import Image as ImageModule
from torch.utils.data import IterableDataset, get_worker_info

from revelio.augmentation.step import AugmentationStep
from revelio.face_detection.detector import FaceDetector

from .element import DatasetElement, ElementImage


def _element_with_images(elem: DatasetElement) -> DatasetElement:
    new_xs = []
    for x in elem.x:
        img = ImageModule.open(x.path)
        new_x = ElementImage(x.path, img, None, None)
        new_xs.append(new_x)
    return DatasetElement(
        original_dataset=elem.original_dataset,
        x=tuple(new_xs),
        y=elem.y,
    )


class Dataset(IterableDataset):
    def __init__(
        self,
        paths: list[DatasetElement],
        face_detector: Optional[FaceDetector],
        augmentation_steps: list[AugmentationStep],
        feature_extractor: None,  # TODO: add type
    ) -> None:
        self._paths = paths
        self._face_detector = face_detector
        self._augmentation_steps = augmentation_steps
        self._feature_extractor = feature_extractor

    def __iter__(self) -> Iterator[dict[str, Any]]:
        # Offline preprocessing
        elems = self._get_elems_iterator()
        for elem in elems:
            elem = _element_with_images(elem)
            if self._face_detector is not None:
                elem = self._face_detector.process(elem)
            # Feature extraction is offline only if augmentation is disabled
            if (
                len(self._augmentation_steps) == 0
                and self._feature_extractor is not None
            ):
                elem = self._feature_extractor.process(elem)
            # Now we can close the images, as we don't need them anymore in the offline
            # preprocessing phase; this way we avoid opening them twice
            for x in elem.x:
                if x.image is not None:
                    x.image.close()
        # Online processing
        for elem in elems:
            elem = _element_with_images(elem)
            # We can safely call the face detector again, as this time it will load
            # the precomputed bounding boxes and landmarks
            if self._face_detector is not None:
                elem = self._face_detector.process(elem)
            # Augment the dataset; this is always done online
            for step in self._augmentation_steps:
                elem = step.process(elem)
            if self._feature_extractor is not None:
                # If augmentation is enabled, feature extraction is done online;
                # otherwise, we load the precomputed features
                elem = self._feature_extractor.process(
                    elem,
                    force_online=len(self._augmentation_steps) > 0,
                )
            yield {
                "x": [
                    {
                        "image": np.array(x.image),
                        "landmarks": x.landmarks,
                        "features": x.features,
                    }
                    for x in elem.x
                ],
                "y": elem.y.value,
            }

    def _get_elems_iterator(self) -> Iterator[DatasetElement]:
        worker_info = get_worker_info()
        if worker_info is None:
            for p in self._paths:
                yield p
        else:
            for i, p in enumerate(self._paths):
                if i % worker_info.num_workers == worker_info.id:
                    yield p
