from pathlib import Path
from typing import Any, Optional

import cv2 as cv

from revelio.dataset.element import Image, Landmarks

from .detector import BoundingBox, FaceDetector


class OpenCVDetector(FaceDetector):  # pragma: no cover
    def __init__(self, *, classifier_path: Path, **kwargs: Any):
        super().__init__(**kwargs)
        self._classifier_path = classifier_path

    def process_element(self, elem: Image) -> tuple[BoundingBox, Optional[Landmarks]]:
        classifier = cv.CascadeClassifier()
        classifier.load(str(self._classifier_path))
        faces = classifier.detectMultiScale(elem)
        if len(faces) == 0:
            raise ValueError("Expected 1 face, got 0")
        if len(faces) == 1:
            face = faces[0]
        else:
            # Choose the face with maximum area and hope for the best
            face = max(faces, key=lambda f: f[2] * f[3])  # type: ignore
        bb = (
            int(face[0]),
            int(face[1]),
            int(face[0] + face[2]),
            int(face[1] + face[3]),
        )
        return bb, None
