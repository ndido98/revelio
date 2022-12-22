import random
from typing import Any, Callable, Optional

import cv2 as cv

from revelio.dataset.element import Image, Landmarks

from .step import AugmentationStep


def _encode_jpeg(image: Image, quality: int) -> Optional[bytes]:  # pragma: no cover
    success, encoded = cv.imencode(".jpg", image, [cv.IMWRITE_JPEG_QUALITY, quality])
    return encoded if success and encoded is not None else None


def _encode_jpeg2000(image: Image, quality: int) -> Optional[bytes]:  # pragma: no cover
    success, encoded = cv.imencode(
        ".jp2", image, [cv.IMWRITE_JPEG2000_COMPRESSION_X1000, quality]
    )
    return encoded if success and encoded is not None else None


def _try_encoder(
    encoder: Callable[[Image, int], Optional[bytes]],
    image: Image,
    quality: int,
) -> bytes:  # pragma: no cover
    try:
        encoded = encoder(image, quality)
    except cv.error as e:
        raise RuntimeError("Failed to encode image: OpenCV error") from e
    if encoded is None:
        raise RuntimeError("Failed to encode image: encoded is None")
    return encoded


def _try_decoder(encoded: bytes) -> Image:  # pragma: no cover
    try:
        decoded = cv.imdecode(encoded, cv.IMREAD_UNCHANGED)
    except cv.error as e:
        raise RuntimeError("Failed to decode image: OpenCV error") from e
    if decoded is None:
        raise RuntimeError("Failed to decode image: decoded is None")
    return decoded  # type: ignore


class JPEGCompression(AugmentationStep):  # pragma: no cover
    """
    Applies either JPEG or JPEG2000 compression artifacts to the image.
    This augmentation can be used to test the robustness of a model to JPEG compression.

    There are several modes of operation:
    - If `jpeg_quality` is specified, the image will be compressed to the specified
        quality using the JPEG encoding.
        If more qualities are specified, one is chosen randomly.
    - if `jpeg2000_quality` is specified, the image will be compressed to the specified
        quality using the JPEG2000 encoding.
        If more qualities are specified, one is chosen randomly.
    - If `max_bytes` is specified, the image will be compressed to the highest quality
        that produces an image under the specified size.
    """

    def __init__(
        self,
        *,
        jpeg_quality: Optional[int | list[int] | dict[str, int]] = None,
        jpeg2000_quality: Optional[int | list[int] | dict[str, int]] = None,
        max_bytes: Optional[int] = None,
        jpeg2000_probability: float = 0.0,
        **kwargs: Any,
    ):
        """
        Applies JPEG compression artifacts to the image.

        Args:
            jpeg_quality: The quality to compress the image to, using the JPEG encoding.
                If a list is provided, one quality is chosen randomly.
                If a dictionary with the `min` and `max` keys is provided, the quality
                is chosen randomly between the specified quality range.
            jpeg2000_quality: The quality to compress the image to, using the JPEG2000
                encoding.
                If a list is provided, one quality is chosen randomly.
                If a dictionary with the `min` and `max` keys is provided, the quality
                is chosen randomly between the specified quality range.
            max_bytes: The maximum number of bytes the image can be compressed to.
                The image will be compressed to the highest quality that produces an
                image under the specified size.
            jpeg2000_probability: The probability that the image will be compressed
                using the JPEG2000 encoding instead of the JPEG encoding.
        """
        super().__init__(**kwargs)
        if jpeg_quality is not None or jpeg2000_quality is not None:
            if jpeg_quality is not None:
                self._validate_quality(jpeg_quality, max_quality=100)
            if jpeg2000_quality is not None:
                self._validate_quality(jpeg2000_quality, max_quality=1000)
        elif max_bytes is not None:
            if max_bytes < 0:
                raise ValueError("Max bytes must be positive")
        else:
            raise ValueError(
                "Either JPEG/JPEG2000 quality or max bytes must be specified"
            )
        self._jpeg_quality = jpeg_quality
        self._jpeg2000_quality = jpeg2000_quality
        self._jpeg2000_probability = jpeg2000_probability
        self._max_bytes = max_bytes

    def _validate_quality(
        self, quality: int | list[int] | dict[str, int], max_quality: int
    ) -> None:
        if isinstance(quality, int) and (quality < 0 or quality > max_quality):
            raise ValueError(f"Quality must be between 0 and {max_quality}")
        elif isinstance(quality, list):
            for q in quality:
                if q < 0 or q > max_quality:
                    raise ValueError(f"Quality must be between 0 and {max_quality}")
        elif isinstance(quality, dict):
            if quality.keys() != {"min", "max"}:
                raise ValueError("'min' and 'max' are required")
            if quality["min"] < 0 or quality["min"] > max_quality:
                raise ValueError(f"Min quality must be between 0 and {max_quality}")
            if quality["max"] < 0 or quality["max"] > max_quality:
                raise ValueError(f"Max quality must be between 0 and {max_quality}")
            if quality["min"] > quality["max"]:
                raise ValueError("Min quality must be less than max quality")
        else:
            raise TypeError("Quality must be an integer, list or dictionary")

    def process_element(
        self, image: Image, landmarks: Optional[Landmarks]
    ) -> tuple[Image, Optional[Landmarks]]:
        if random.random() < self._jpeg2000_probability:
            algorithm = self._jpeg2000_quality
            encoder = _encode_jpeg2000
            max_allowed_quality = 1000
        else:
            algorithm = self._jpeg_quality
            encoder = _encode_jpeg
            max_allowed_quality = 100
        if algorithm is not None:
            if isinstance(algorithm, int):
                chosen_quality = algorithm
            elif isinstance(algorithm, list):
                chosen_quality = random.choice(algorithm)
            elif isinstance(algorithm, dict):
                chosen_quality = random.randint(algorithm["min"], algorithm["max"])
            else:
                raise TypeError("Quality must be an integer, list or dictionary")
            return self._convert_to_quality(image, chosen_quality, encoder), landmarks
        elif self._max_bytes is not None:
            return (
                self._convert_to_max_bytes(
                    image, self._max_bytes, max_allowed_quality, encoder
                ),
                landmarks,
            )
        else:
            raise ValueError("Either quality or max bytes must be specified")

    def _convert_to_quality(
        self,
        image: Image,
        quality: int,
        encoder: Callable[[Image, int], Optional[bytes]],
    ) -> Image:
        encoded = _try_encoder(encoder, image, quality)
        decoded = _try_decoder(encoded)
        return decoded

    def _convert_to_max_bytes(
        self,
        image: Image,
        max_bytes: int,
        max_allowed_quality: int,
        encoder: Callable[[Image, int], Optional[bytes]],
    ) -> Image:
        # Find the highest quality that produces an image under the desired size
        min_quality, max_quality = 0, max_allowed_quality
        while max_quality - min_quality > 1:
            quality = (min_quality + max_quality) // 2
            encoded = _try_encoder(encoder, image, quality)
            if len(encoded) > max_bytes:
                max_quality = quality
            else:
                min_quality = quality
        encoded = _try_encoder(encoder, image, min_quality)
        decoded = _try_decoder(encoded)
        return decoded
