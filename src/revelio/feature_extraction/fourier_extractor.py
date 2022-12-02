from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2 as cv
import numpy as np

from .same_size_extractor import SameSizeExtractor

if TYPE_CHECKING:
    from revelio.dataset.element import ElementImage


class FourierExtractor(SameSizeExtractor):  # pragma: no cover
    def __init__(
        self,
        *,
        include_phase: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._include_phase = include_phase

    def process_resized_element(self, elem: ElementImage) -> np.ndarray:
        gray = cv.cvtColor(elem.image, cv.COLOR_BGR2GRAY)
        dft = cv.dft(np.float32(gray), flags=cv.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude, phase = cv.cartToPolar(dft_shift[..., 0], dft_shift[..., 1])
        magnitude_spectrum: np.ndarray = 20 * np.log(magnitude)
        if self._include_phase:
            return np.stack([magnitude_spectrum, phase], axis=0)
        else:
            return magnitude_spectrum[np.newaxis, ...]
