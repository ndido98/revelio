from typing import Any, Optional

import cv2 as cv
import numpy as np
from facenet_pytorch import MTCNN

from revelio.dataset.element import Image

from .detector import BoundingBox, FaceDetector, Landmarks


class MTCNNDetector(FaceDetector):  # pragma: no cover
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._mtcnn = MTCNN(select_largest=True)

    def process_element(self, elem: Image) -> tuple[BoundingBox, Optional[Landmarks]]:
        rgb_elem = cv.cvtColor(elem, cv.COLOR_BGR2RGB)
        boxes, _, landmarks = self._mtcnn.detect(rgb_elem, landmarks=True)
        boxes = tuple(np.squeeze(boxes).astype(int))
        landmarks = np.squeeze(landmarks).astype(int)
        return boxes, landmarks
