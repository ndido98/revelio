from typing import Any

import numpy as np

from revelio.dataset.element import ElementImage

from .step import PreprocessingStep


class SelectChannel(PreprocessingStep):
    def __init__(self, *, channel_index: int, stack: int = 1, **kwargs: Any):
        super().__init__(**kwargs)
        if channel_index < 0:
            raise ValueError("Channel index must be non-negative")
        self._channel_index = channel_index
        if stack < 1:
            raise ValueError("Stack must be positive")
        self._stack = stack

    def process_element(self, elem: ElementImage) -> ElementImage:
        selected = elem.image[..., self._channel_index]
        if self._stack > 1:
            selected = np.stack([selected] * self._stack, axis=-1)
        return ElementImage(
            path=elem.path,
            image=selected,
            landmarks=elem.landmarks,
            features=elem.features,
        )
