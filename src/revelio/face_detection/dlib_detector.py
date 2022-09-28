from pathlib import Path
from typing import Any, Optional

import dlib
import numpy as np
from PIL.Image import Image

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
        img_data = np.array(elem)
        faces, scores, idx = self._face_detector.run(img_data, 1)
        if len(faces) == 0:
            raise ValueError(
                f"Expected 1 face, got 0 in {elem.filename}"  # type: ignore
            )

        max_score_idx = np.argmax(scores)
        face = faces[max_score_idx]
        bb = (face.left(), face.top(), face.right(), face.bottom())

        if self._landmark_predictor is not None:
            landmarks = self._landmark_predictor(img_data, face)
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
        else:
            landmarks = None
        return bb, landmarks
