import random
from typing import Any, Optional

import cv2 as cv

from revelio.dataset.element import Image, Landmarks

from .step import AugmentationStep


class JPEGCompression(AugmentationStep):
    """
    Applies JPEG compression artifacts to the image.
    This augmentation can be used to test the robustness of a model to JPEG compression.

    There are several modes of operation:
    - If `quality` is specified, the image will be compressed to the specified quality.
        If more qualities are specified, one is chosen randomly.
    - If `max_bytes` is specified, the image will be compressed to the highest quality
        that produces an image under the specified size.
    """

    def __init__(
        self,
        *,
        quality: Optional[int | list[int] | dict[str, int]] = None,
        max_bytes: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Applies JPEG compression artifacts to the image.

        Args:
            quality: The quality to compress the image to. If a list is provided, one
                quality is chosen randomly. If a dictionary with the `min` and `max`
                keys is provided, the quality is chosen randomly between the specified
                quality range.
            max_bytes: The maximum number of bytes the image can be compressed to.
        """
        super().__init__(**kwargs)
        if quality is not None:
            self._validate_quality(quality)
        elif max_bytes is not None:
            if max_bytes < 0:
                raise ValueError("Max bytes must be positive")
        else:
            raise ValueError("Either quality or max bytes must be specified")
        self._quality = quality
        self._max_bytes = max_bytes

    def _validate_quality(self, quality: int | list[int] | dict[str, int]) -> None:
        if isinstance(quality, int) and (quality < 0 or quality > 100):
            raise ValueError("Quality must be between 0 and 100")
        elif isinstance(quality, list):
            for q in quality:
                if q < 0 or q > 100:
                    raise ValueError("Quality must be between 0 and 100")
        elif isinstance(quality, dict):
            if quality.keys() != {"min", "max"}:
                raise ValueError("'min' and 'max' are required")
            if quality["min"] < 0 or quality["min"] > 100:
                raise ValueError("Min quality must be between 0 and 100")
            if quality["max"] < 0 or quality["max"] > 100:
                raise ValueError("Max quality must be between 0 and 100")
            if quality["min"] > quality["max"]:
                raise ValueError("Min quality must be less than max quality")
        else:
            raise TypeError("Quality must be an integer, list or dictionary")

    def process_element(
        self, image: Image, landmarks: Optional[Landmarks]
    ) -> tuple[Image, Optional[Landmarks]]:
        if self._quality is not None:
            if isinstance(self._quality, int):
                chosen_quality = self._quality
            elif isinstance(self._quality, list):
                chosen_quality = random.choice(self._quality)
            elif isinstance(self._quality, dict):
                chosen_quality = random.randint(
                    self._quality["min"], self._quality["max"]
                )
            else:
                raise TypeError("Quality must be an integer, list or dictionary")
            return self._convert_to_quality(image, chosen_quality), landmarks
        elif self._max_bytes is not None:
            return self._convert_to_max_bytes(image, self._max_bytes), landmarks
        else:
            raise ValueError("Either quality or max bytes must be specified")

    def _convert_to_quality(self, image: Image, quality: int) -> Image:
        _, encoded = cv.imencode(".jpg", image, [cv.IMWRITE_JPEG_QUALITY, quality])
        return cv.imdecode(encoded, cv.IMREAD_UNCHANGED)  # type: ignore

    def _convert_to_max_bytes(self, image: Image, max_bytes: int) -> Image:
        # Find the highest quality that produces an image under the desired size
        min_quality, max_quality = 0, 100
        while max_quality - min_quality > 1:
            quality = (min_quality + max_quality) // 2
            _, encoded = cv.imencode(".jpg", image, [cv.IMWRITE_JPEG_QUALITY, quality])
            if len(encoded) > max_bytes:
                max_quality = quality
            else:
                min_quality = quality
        _, encoded = cv.imencode(".jpg", image, [cv.IMWRITE_JPEG_QUALITY, min_quality])
        return cv.imdecode(encoded, cv.IMREAD_UNCHANGED)  # type: ignore
