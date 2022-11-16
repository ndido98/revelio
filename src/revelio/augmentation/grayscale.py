from typing import Optional

import cv2 as cv

from revelio.dataset.element import Image, Landmarks

from .step import AugmentationStep


class Grayscale(AugmentationStep):
    def process_element(
        self, image: Image, landmarks: Optional[Landmarks]
    ) -> tuple[Image, Optional[Landmarks]]:
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY), landmarks
