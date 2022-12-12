from abc import abstractmethod
from typing import Any, Optional

import cv2 as cv
import numpy as np

from revelio.dataset.element import ElementImage

from .extractor import FeatureExtractor


class SameSizeExtractor(FeatureExtractor):  # pragma: no cover

    transparent: bool = True

    def __init__(
        self,
        resize_width: Optional[int] = None,
        resize_height: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._resize_width = resize_width
        self._resize_height = resize_height

    def process_element(self, elem: ElementImage) -> np.ndarray:
        if self._resize_width is not None and self._resize_height is not None:
            img = cv.resize(
                elem.image,
                (self._resize_width, self._resize_height),
                interpolation=cv.INTER_CUBIC,
            )
        else:
            img = elem.image.copy()
        new_elem = ElementImage(
            path=elem.path,
            image=img,
            landmarks=elem.landmarks,
            features=elem.features,
        )
        return self.process_resized_element(new_elem)

    @abstractmethod
    def process_resized_element(self, elem: ElementImage) -> np.ndarray:
        raise NotImplementedError  # pragma: no cover
