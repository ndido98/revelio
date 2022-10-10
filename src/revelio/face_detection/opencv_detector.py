from pathlib import Path
from typing import Any, Optional

import cv2 as cv

from revelio.dataset.element import Image

from .detector import BoundingBox, FaceDetector, Landmarks


class OpenCVDetector(FaceDetector):  # pragma: no cover
    def __init__(
        self, *, classifier_path: Path, equalize_histogram: bool = True, **kwargs: Any
    ):
        super().__init__(**kwargs)
        self._classifier = cv.CascadeClassifier()
        self._classifier.load(str(classifier_path))
        self._equalize_histogram = equalize_histogram

    def process_element(self, elem: Image) -> tuple[BoundingBox, Optional[Landmarks]]:
        img = cv.cvtColor(elem, cv.COLOR_BGR2GRAY)
        if self._equalize_histogram:
            img = cv.equalizeHist(img)
        faces = self._classifier.detectMultiScale(img)
        if len(faces) != 1:
            raise ValueError(f"Expected 1 face, got {len(faces)}")
        face = faces[0]
        bb = (face[0], face[1], face[0] + face[2], face[1] + face[3])
        return bb, None
