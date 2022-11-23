import random
from typing import Any, Optional

import cv2 as cv
import numpy as np

from revelio.dataset.element import Image, Landmarks

from .step import AugmentationStep

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


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_greater(name1: str, name2: str, value1: int, value2: int) -> None:
    if value1 <= value2:
        raise ValueError(f"{name1} must be greater than {name2}")


class Resize(AugmentationStep):
    def __init__(
        self,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        from_width: Optional[int] = None,
        to_width: Optional[int] = None,
        from_height: Optional[int] = None,
        to_height: Optional[int] = None,
        algorithms: Optional[list[str]] = None,
        fill_modes: Optional[list[str]] = None,
        keep_aspect_ratio_probability: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._parse_sizes(width, height, from_width, to_width, from_height, to_height)
        self._parse_algorithms(algorithms)
        self._parse_fill_modes(fill_modes)
        self._parse_keep_aspect_ratio_probability(keep_aspect_ratio_probability)

    def _parse_sizes(
        self,
        width: Optional[int],
        height: Optional[int],
        from_width: Optional[int],
        to_width: Optional[int],
        from_height: Optional[int],
        to_height: Optional[int],
    ) -> None:
        if width is not None and height is not None:
            # We have both width and height
            _validate_positive("Width", width)
            _validate_positive("Height", height)
        elif width is not None and from_height is not None and to_height is not None:
            # We have only width, we expect a random height
            _validate_positive("Width", width)
            _validate_positive("From height", from_height)
            _validate_positive("To height", to_height)
            _validate_greater("To height", "From height", to_height, from_height)
        elif height is not None and from_width is not None and to_width is not None:
            # We have only height, we expect a random width
            _validate_positive("Height", height)
            _validate_positive("From width", from_width)
            _validate_positive("To width", to_width)
            _validate_greater("To width", "From width", to_width, from_width)
        else:
            raise ValueError(
                "You must specify both width or height, "
                "either by specifying a fixed value or a range"
            )
        self._width = width
        self._height = height
        self._from_width = from_width
        self._to_width = to_width
        self._from_height = from_height
        self._to_height = to_height

    def _parse_algorithms(self, algorithms: Optional[list[str]]) -> None:
        if algorithms is None:
            self._algorithms = list(_VALID_ALGORITHMS.keys())
            return
        for algorithm in algorithms:
            if algorithm not in _VALID_ALGORITHMS:
                raise ValueError(f"Invalid algorithm: {algorithm}")
        self._algorithms = algorithms

    def _parse_fill_modes(self, fill_modes: Optional[list[str]]) -> None:
        if fill_modes is None:
            self._fill_modes = list(_VALID_FILL_MODES.keys())
            return
        for fill_mode in fill_modes:
            if fill_mode not in _VALID_FILL_MODES:
                raise ValueError(f"Invalid fill mode: {fill_mode}")
        self._fill_modes = fill_modes

    def _parse_keep_aspect_ratio_probability(
        self, keep_aspect_ratio_probability: float
    ) -> None:
        if keep_aspect_ratio_probability < 0 or keep_aspect_ratio_probability > 1:
            raise ValueError("Keep aspect ratio probability must be between 0 and 1")
        self._keep_aspect_ratio_probability = keep_aspect_ratio_probability

    def process_element(
        self, image: Image, landmarks: Optional[Landmarks]
    ) -> tuple[Image, Optional[Landmarks]]:
        algorithm = random.choice(self._algorithms)
        fill_mode = random.choice(self._fill_modes)
        keep_aspect_ratio = random.random() < self._keep_aspect_ratio_probability
        if self._width is not None:
            width = self._width
        elif self._from_width is not None and self._to_width is not None:
            width = random.randint(self._from_width, self._to_width)
        else:
            raise ValueError("Invalid width")
        if self._height is not None:
            height = self._height
        elif self._from_height is not None and self._to_height is not None:
            height = random.randint(self._from_height, self._to_height)
        else:
            raise ValueError("Invalid height")
        if not keep_aspect_ratio:
            scale_factors = (width / image.shape[1], height / image.shape[0])
            new_img = cv.resize(
                image,
                (height, width),
                interpolation=_VALID_ALGORITHMS[algorithm],
            )
            if landmarks is not None:
                new_landmarks = landmarks * np.array(scale_factors)
            else:
                new_landmarks = None
        else:
            new_size = (height, width)
            old_size = image.shape[:2]
            scale_factor = min(n / o for n, o in zip(new_size, old_size))
            rescaled = cv.resize(
                image,
                None,
                fx=scale_factor,
                fy=scale_factor,
                interpolation=_VALID_ALGORITHMS[algorithm],
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
                _VALID_FILL_MODES[fill_mode],
                (0, 0, 0),
            )
            if landmarks is not None:
                new_landmarks = landmarks * scale_factor + np.array([left, top])
            else:
                new_landmarks = None
        return new_img, new_landmarks
