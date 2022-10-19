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
        if len(boxes) == 0:
            raise ValueError("Expected 1 face, got 0")
        biggest_box = np.argmax(np.prod(boxes[:, 2:] - boxes[:, :2], axis=1))
        box = boxes[biggest_box].astype(int)
        box = tuple(int(n) for n in box)
        landmarks = landmarks[biggest_box].astype(int)
        return box, landmarks
