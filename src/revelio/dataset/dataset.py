import logging
from math import ceil
from typing import Any, Generator, Iterator, Optional

import cv2 as cv
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from revelio.augmentation.step import AugmentationStep
from revelio.face_detection.detector import FaceDetector
from revelio.feature_extraction.extractor import FeatureExtractor
from revelio.preprocessing.step import PreprocessingStep
from revelio.utils.random import shuffled

from .element import DatasetElement, DatasetElementDescriptor, ElementImage

log = logging.getLogger(__name__)


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
        shuffle: bool,
    ) -> None:
        self._paths = paths
        self._face_detector = face_detector
        self._augmentation_steps = augmentation_steps
        self._feature_extractors = feature_extractors
        self._preprocessing_steps = preprocessing_steps
        self._shuffle = shuffle
        self.warmup = False

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.warmup:
            return self._offline_processing()
        else:
            return self._online_processing()

    def __len__(self) -> int:
        return len(self._paths)

    def _get_elems_list(self) -> list[DatasetElementDescriptor]:
        worker_info = get_worker_info()
        if worker_info is None:
            return shuffled(self._paths) if self._shuffle else self._paths
        else:
            # Split the dataset across workers
            per_worker = int(ceil(len(self._paths) / worker_info.num_workers))
            worker_id = worker_info.id
            from_idx = worker_id * per_worker
            to_idx = (worker_id + 1) * per_worker
            if self._shuffle:
                return shuffled(self._paths[from_idx:to_idx])
            else:
                return self._paths[from_idx:to_idx]

    def _apply_face_detection(
        self, elem: DatasetElement, silent: bool = False
    ) -> tuple[DatasetElement, bool]:
        success = True
        if self._face_detector is not None:
            try:
                elem = self._face_detector.process(elem)
            except RuntimeError:
                if not silent:
                    log.warning(
                        "Skipping %s due to face detection failure",
                        elem,
                        exc_info=True,
                    )
                success = False
        return elem, success

    def _apply_feature_extraction(
        self, elem: DatasetElement, force_online: bool, silent: bool = False
    ) -> tuple[DatasetElement, bool]:
        success = True
        for feature_extractor in self._feature_extractors:
            try:
                elem = feature_extractor.process(elem, force_online=force_online)
            except RuntimeError:
                if not silent:
                    log.warning(
                        "Skipping %s due to feature extraction failure",
                        elem,
                        exc_info=True,
                    )
                success = False
                break
        return elem, success

    def _offline_processing(self) -> Iterator:
        descriptors = self._get_elems_list()
        for descriptor in descriptors:
            elem = _element_with_images(descriptor)
            elem, _ = self._apply_face_detection(elem)
            # Feature extraction is offline only if augmentation is disabled
            if len(self._augmentation_steps) == 0 and len(self._feature_extractors) > 0:
                elem, _ = self._apply_feature_extraction(elem, False)
            # HACK: the data loader expects something it can collate to a tensor,
            # so we return a dummy value
            yield 0

    def _online_processing(self) -> Generator[dict[str, Any], None, None]:
        descriptors = self._get_elems_list()
        for descriptor in descriptors:
            elem = _element_with_images(descriptor)
            # We can safely call the face detector again, as this time it will load
            # the precomputed bounding boxes and landmarks
            elem, fd_success = self._apply_face_detection(elem, silent=True)
            if not fd_success:
                continue
            # Augment the dataset; this is always done online
            for augmentation_step in self._augmentation_steps:
                elem = augmentation_step.process(elem)
            # If augmentation is enabled, feature extraction is done online;
            # otherwise, we load the precomputed features
            elem, fe_success = self._apply_feature_extraction(
                elem, len(self._augmentation_steps) > 0, silent=True
            )
            if not fe_success:
                continue
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
                "dataset": elem.original_dataset,
            }
