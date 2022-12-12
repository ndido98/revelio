from typing import Any

import numpy as np

from revelio.dataset.element import ElementImage

from .step import PreprocessingStep


class ToFloat(PreprocessingStep):  # pragma: no cover
    def __init__(self, *, max_per_channel: float | list[float] = 255.0, **kwargs: Any):
        super().__init__(**kwargs)
        if isinstance(max_per_channel, list):
            self._max_per_channel = np.array(max_per_channel, dtype=np.float32)
        else:
            self._max_per_channel = np.array([max_per_channel], dtype=np.float32)

    def process_element(self, elem: ElementImage) -> ElementImage:
        if elem.image.dtype == np.uint8:
            new_img = elem.image.astype(np.float32) / self._max_per_channel
            return ElementImage(
                path=elem.path,
                image=new_img,
                landmarks=elem.landmarks,
                features=elem.features,
            )
        else:
            return elem
