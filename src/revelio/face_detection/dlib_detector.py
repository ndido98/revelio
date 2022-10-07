from pathlib import Path
from typing import Any, Optional

import cv2 as cv
import dlib
import numpy as np

from revelio.dataset.element import Image

from .detector import BoundingBox, FaceDetector, Landmarks


class DLIBDetector(FaceDetector):
    def __init__(self, landmark_predictor_path: Optional[Path] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._face_detector = dlib.get_frontal_face_detector()
        self._landmark_predictor_path = landmark_predictor_path
        if self._landmark_predictor_path is not None:
            self._landmark_predictor = dlib.shape_predictor(
                str(self._landmark_predictor_path)
            )
        else:
            self._landmark_predictor = None

    def process_element(self, elem: Image) -> tuple[BoundingBox, Optional[Landmarks]]:
        rgb_elem = cv.cvtColor(elem, cv.COLOR_BGR2RGB)
        faces, scores, _ = self._face_detector.run(rgb_elem, 1)
        if len(faces) == 0:
            raise ValueError("Expected 1 face, got 0")

        max_score_idx = np.argmax(scores)
        face = faces[max_score_idx]
        bb = (face.left(), face.top(), face.right(), face.bottom())

        if self._landmark_predictor is not None:
            landmarks = self._landmark_predictor(rgb_elem, face)
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
        else:
            landmarks = None
        return bb, landmarks
