from typing import Any, Optional

import numpy as np

from revelio.dataset.element import ElementImage

from .step import PreprocessingStep

# Presets are always expressed for a BGR image in the range [0, 1]
_PRESETS = {
    "imagenet": ([0.406, 0.456, 0.485], [0.225, 0.224, 0.229]),
}


class Normalize(PreprocessingStep):  # pragma: no cover
    def __init__(
        self,
        *,
        preset: Optional[str] = None,
        mean: Optional[list[float]] = None,
        std: Optional[list[float]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if preset is not None:
            # If a preset is specified, we can't have a custom mean or std
            if mean is not None or std is not None:
                raise ValueError(
                    "Cannot specify a custom mean or std when a preset is specified"
                )
            if preset not in _PRESETS:
                raise ValueError(f"Unknown preset: {preset}")
            mean, std = _PRESETS[preset]
        else:
            if mean is None or std is None:
                raise ValueError("Must specify a preset or a custom mean and std")

        if len(mean) != 3 or len(std) != 3:
            raise ValueError("Mean and std must have length 3")
        self._preset = preset
        self._mean = np.array(mean, dtype=np.float32)
        self._std = np.array(std, dtype=np.float32)
        if np.all(self._std == 0):
            raise ValueError("Std cannot be zero")

    def process_element(self, elem: ElementImage) -> ElementImage:
        # Presets are always expressed when the image is in the range [0, 1],
        # so if the image is in the range [0, 255] we need to convert it
        # to the range [0, 1] and then convert it back to the range [0, 255]
        if self._preset is not None and elem.image.dtype == np.uint8:
            new_img = elem.image.astype(np.float32) / 255.0
            new_img = (new_img - self._mean) / self._std
            new_img = (new_img * 255.0).astype(np.uint8)
        else:
            new_img = (elem.image - self._mean) / self._std
        return ElementImage(
            path=elem.path,
            image=new_img,
            landmarks=elem.landmarks,
            features=elem.features,
        )
