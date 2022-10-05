from pathlib import Path
from typing import Any, Generator, Iterator, Optional

import cv2 as cv
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from revelio.augmentation.step import AugmentationStep
from revelio.face_detection.detector import FaceDetector
from revelio.feature_extraction.extractor import FeatureExtractor
from revelio.preprocessing.step import PreprocessingStep
from revelio.utils.iterators import consume

from .element import (
    DatasetElement,
    DatasetElementDescriptor,
    ElementClass,
    ElementImage,
)


def _element_with_images(elem: Any) -> DatasetElement:
    xs: tuple[ElementImage, ...]
    if len(elem["x"].shape) == 0:
        # We only have one x
        xs = (ElementImage(path=Path(elem["x"]), image=cv.imread(elem["x"])),)
    else:
        xs = tuple(ElementImage(path=Path(x), image=cv.imread(x)) for x in elem["x"])
    return DatasetElement(
        dataset_root_path=Path(elem["root_path"]),
        original_dataset=elem["dataset_name"],
        x=xs,
        y=ElementClass(elem["y"]),
    )


def _descriptors_to_numpy(
    paths: list[DatasetElementDescriptor],
) -> np.ndarray[int, np.dtype[np.generic]]:
    # We are sure that this number is always the same for all elements (it's checked
    # during data loading in the dataset factory)
    x_count = len(paths[0].x)
    longest_path_length = 0
    longest_root_path = 0
    longest_dataset_name = 0
    for elem in paths:
        for x in elem.x:
            longest_path_length = max(longest_path_length, len(str(x)))
        longest_root_path = max(longest_root_path, len(str(elem._root_path)))
        longest_dataset_name = max(longest_dataset_name, len(elem._dataset_name))
    dt = np.dtype(
        [
            ("x", f"U{longest_path_length}", x_count),
            ("y", np.uint8),
            ("root_path", f"U{longest_root_path}"),
            ("dataset_name", f"U{longest_dataset_name}"),
        ]
    )
    return np.array(
        [
            (
                *tuple(str(x) for x in elem.x),
                elem.y.value,
                str(elem._root_path),
                elem._dataset_name,
            )
            for elem in paths
        ],
        dtype=dt,
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
        self._paths = _descriptors_to_numpy(paths)
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

    def _get_elems_iterator(self) -> Iterator[Any]:
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
