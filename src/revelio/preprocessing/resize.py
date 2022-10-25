from typing import Any

import cv2 as cv

from revelio.dataset.element import ElementImage

from .step import PreprocessingStep

_VALID_ALGORITHMS = {
    "nearest": cv.INTER_NEAREST,
    "linear": cv.INTER_LINEAR,
    "cubic": cv.INTER_CUBIC,
    "area": cv.INTER_AREA,
    "lanczos4": cv.INTER_LANCZOS4,
}

_VALID_FILL_MODES = {
    "constant": cv.BORDER_CONSTANT,
    "reflect": cv.BORDER_REFLECT_101,
    "replicate": cv.BORDER_REPLICATE,
    "wrap": cv.BORDER_WRAP,
}


class Resize(PreprocessingStep):
    def __init__(
        self,
        *,
        width: int,
        height: int,
        algorithm: str = "cubic",
        keep_aspect_ratio: bool = True,
        fill_mode: str = "constant",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if algorithm not in _VALID_ALGORITHMS.keys():
            raise ValueError(f"Invalid algorithm: {algorithm}")
        if fill_mode not in _VALID_FILL_MODES.keys():
            raise ValueError(f"Invalid fill mode: {fill_mode}")
        self._width = width
        self._height = height
        self._algorithm = algorithm
        self._keep_aspect_ratio = keep_aspect_ratio
        self._fill_mode = fill_mode

    def process_element(self, elem: ElementImage) -> ElementImage:
        if not self._keep_aspect_ratio:
            new_img = cv.resize(
                elem.image,
                (self._height, self._width),
                interpolation=_VALID_ALGORITHMS[self._algorithm],
            )
        else:
            new_size = (self._height, self._width)
            old_size = elem.image.shape[:2]
            scale_factor = min(n / o for n, o in zip(new_size, old_size))
            rescaled = cv.resize(
                elem.image,
                None,
                fx=scale_factor,
                fy=scale_factor,
                interpolation=_VALID_ALGORITHMS[self._algorithm],
            )
            top_bottom, left_right = tuple(
                d - s for d, s in zip(new_size, rescaled.shape[:2])
            )
            top = top_bottom // 2
            bottom = top_bottom - top
            left = left_right // 2
            right = left_right - left
            new_img = cv.copyMakeBorder(
                rescaled,
                top,
                bottom,
                left,
                right,
                _VALID_FILL_MODES[self._fill_mode],
                (0, 0, 0),
            )
        return ElementImage(
            path=elem.path,
            image=new_img,
            landmarks=elem.landmarks,
            features=elem.features,
        )
