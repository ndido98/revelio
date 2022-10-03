from typing import Any, Generator, Iterator, Optional

import cv2 as cv
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from revelio.augmentation.step import AugmentationStep
from revelio.face_detection.detector import FaceDetector
from revelio.feature_extraction.extractor import FeatureExtractor
from revelio.preprocessing.step import PreprocessingStep
from revelio.utils.iterators import consume

from .element import DatasetElement, DatasetElementDescriptor, ElementImage


def _element_with_images(elem: DatasetElementDescriptor) -> DatasetElement:
    return DatasetElement(
        dataset_root_path=elem._root_path,
        original_dataset=elem._dataset_name,
        x=tuple(ElementImage(path=x, image=cv.imread(str(x))) for x in elem.x),
        y=elem.y,
    )


class Dataset(IterableDataset):
    def __init__(
        self,
        paths: list[DatasetElementDescriptor],
        face_detector: Optional[FaceDetector],
        augmentation_steps: list[AugmentationStep],
        feature_extractors: list[FeatureExtractor],
        preprocessing_steps: list[PreprocessingStep],
    ) -> None:
        self._paths = paths
        self._face_detector = face_detector
        self._augmentation_steps = augmentation_steps
        self._feature_extractors = feature_extractors
        self._preprocessing_steps = preprocessing_steps
        self.warmup = False

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.warmup:
            return self._offline_processing()
        else:
            consume(self._offline_processing())
            return self._online_processing()

    def __len__(self) -> int:
        return len(self._paths)

    def _get_elems_iterator(self) -> Iterator[DatasetElementDescriptor]:
        worker_info = get_worker_info()
        if worker_info is None:
            for p in self._paths:
                yield p
        else:
            for i, p in enumerate(self._paths):
                if i % worker_info.num_workers == worker_info.id:
                    yield p

    def _offline_processing(self) -> Iterator:
        descriptors = self._get_elems_iterator()
        for descriptor in descriptors:
            elem = _element_with_images(descriptor)
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
        descriptors = self._get_elems_iterator()
        for descriptor in descriptors:
            elem = _element_with_images(descriptor)
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
            for preprocessing_step in self._preprocessing_steps:
                elem = preprocessing_step.process(elem)
            elem_xs = []
            for x in elem.x:
                rgb = cv.cvtColor(x.image, cv.COLOR_BGR2RGB)
                chw = np.transpose(rgb, (2, 0, 1))
                elem_x: dict[str, Any] = {"image": chw}
                if x.landmarks is not None:
                    elem_x["landmarks"] = x.landmarks
                if len(x.features.keys()) > 0:
                    elem_x["features"] = x.features
                elem_xs.append(elem_x)
            yield {
                "x": elem_xs,
                "y": elem.y.value,
            }
