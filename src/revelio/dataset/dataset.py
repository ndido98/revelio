from typing import Any, Generator, Iterator, Optional

import cv2 as cv
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from revelio.augmentation.step import AugmentationStep
from revelio.face_detection.detector import FaceDetector
from revelio.feature_extraction.extractor import FeatureExtractor
from revelio.utils.iterators import consume

from .element import DatasetElement, ElementImage


def _element_with_images(elem: DatasetElement) -> DatasetElement:
    new_xs = []
    for x in elem.x:
        if x.image is None:
            img = cv.imread(str(x.path))
            new_x = ElementImage(x.path, img, x.landmarks, x.features)
        else:
            new_x = x
        new_xs.append(new_x)
    return DatasetElement(
        dataset_root_path=elem.dataset_root_path,
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
        feature_extractors: list[FeatureExtractor],
    ) -> None:
        self._paths = paths
        self._face_detector = face_detector
        self._augmentation_steps = augmentation_steps
        self._feature_extractors = feature_extractors
        self.warmup = False

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.warmup:
            return self._offline_processing()
        else:
            consume(self._offline_processing())
            return self._online_processing()

    def __len__(self) -> int:
        return len(self._paths)

    def _get_elems_iterator(self) -> Iterator[DatasetElement]:
        worker_info = get_worker_info()
        if worker_info is None:
            for p in self._paths:
                yield p
        else:
            for i, p in enumerate(self._paths):
                if i % worker_info.num_workers == worker_info.id:
                    yield p

    def _offline_processing(self) -> Iterator:
        elems = self._get_elems_iterator()
        for elem in elems:
            elem = _element_with_images(elem)
            if self._face_detector is not None:
                elem = self._face_detector.process(elem)
            # Feature extraction is offline only if augmentation is disabled
            if len(self._augmentation_steps) == 0 and len(self._feature_extractors) > 0:
                for feature_extractor in self._feature_extractors:
                    elem = feature_extractor.process(elem)
            # HACK: the data loader expects something it can collate to a tensor,
            # so we return a dummy value
            yield 0

    def _online_processing(self) -> Generator[dict[str, Any], None, None]:
        elems = self._get_elems_iterator()
        for elem in elems:
            elem = _element_with_images(elem)
            # We can safely call the face detector again, as this time it will load
            # the precomputed bounding boxes and landmarks
            if self._face_detector is not None:
                elem = self._face_detector.process(elem)
            # Augment the dataset; this is always done online
            for augmentation_step in self._augmentation_steps:
                elem = augmentation_step.process(elem)
            # If augmentation is enabled, feature extraction is done online;
            # otherwise, we load the precomputed features
            for feature_extractor in self._feature_extractors:
                elem = feature_extractor.process(
                    elem,
                    force_online=len(self._augmentation_steps) > 0,
                )
            yield {
                "x": [
                    {
                        "image": x.image,
                        "landmarks": (
                            x.landmarks if x.landmarks is not None else np.array([])
                        ),
                        "features": (
                            x.features if x.features is not None else np.array([])
                        ),
                    }
                    for x in elem.x
                ],
                "y": elem.y.value,
            }
