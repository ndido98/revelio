from typing import Any

import cv2 as cv
import numpy as np

from revelio.dataset.element import ElementImage

from .step import PreprocessingStep

_COLOR_SPACES = {
    "rgb": cv.COLOR_BGR2RGB,
    "gray": cv.COLOR_BGR2GRAY,
    "hsv": cv.COLOR_BGR2HSV,
    "hls": cv.COLOR_BGR2HLS,
    "ycrcb": cv.COLOR_BGR2YCR_CB,
}


class ColorSpace(PreprocessingStep):
    def __init__(self, *, target: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if target not in _COLOR_SPACES.keys():
            raise ValueError(f"Unknown color space: {target}")
        self._target = target

    def process_element(self, elem: ElementImage) -> ElementImage:
        converted = cv.cvtColor(elem.image, _COLOR_SPACES[self._target])
        if self._target == "gray":
            converted = converted[..., np.newaxis]
        return ElementImage(
            path=elem.path,
            image=converted,
            landmarks=elem.landmarks,
            features=elem.features,
        )
