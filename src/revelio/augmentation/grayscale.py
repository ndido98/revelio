from typing import Any, Optional

import cv2 as cv
import numpy as np

from revelio.dataset.element import Image, Landmarks

from .step import AugmentationStep


class Grayscale(AugmentationStep):
    """
    Applies a grayscale filter to the image.
    """

    def __init__(self, *, stack: int = 1, **kwargs: Any):
        """
        Applies a grayscale filter to the image.

        Args:
            stack: The number of times to stack the grayscale image.
        """
        super().__init__(**kwargs)
        if stack < 1:
            raise ValueError("Stack must be positive")
        self._stack = stack

    def process_element(
        self, image: Image, landmarks: Optional[Landmarks]
    ) -> tuple[Image, Optional[Landmarks]]:
        grayscale = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        if self._stack > 1:
            grayscale = np.stack([grayscale] * self._stack, axis=-1)
        return grayscale, landmarks
