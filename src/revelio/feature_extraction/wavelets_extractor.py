from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import cv2 as cv
import numpy as np
import pywt

from .same_size_extractor import SameSizeExtractor

if TYPE_CHECKING:
    from revelio.dataset.element import ElementImage


class WaveletsExtractor(SameSizeExtractor):  # pragma: no cover
    def __init__(
        self,
        *,
        wavelet: str,
        level: int,
        grayscale: bool = False,
        include_nodes: list[str] | None = None,
        exclude_nodes: list[str] | None = None,
        target_mean: float | None = None,
        target_std: float | None = None,
        resize_to_original: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if include_nodes is not None and exclude_nodes is not None:
            raise ValueError("Cannot specify both include_nodes and exclude_nodes")
        if target_mean is not None and target_std is None:
            raise ValueError("Cannot specify target_mean without target_std")
        if target_std is not None and target_mean is None:
            raise ValueError("Cannot specify target_std without target_mean")
        self._wavelet = wavelet
        self._level = level
        self._grayscale = grayscale
        self._include_nodes = (
            [re.compile(p) for p in include_nodes] if include_nodes else []
        )
        self._exclude_nodes = (
            [re.compile(p) for p in exclude_nodes] if exclude_nodes else []
        )
        self._target_mean = (
            np.array(target_mean, dtype=np.float32) if target_mean is not None else None
        )
        self._target_std = (
            np.array(target_std, dtype=np.float32) if target_std is not None else None
        )
        self._resize_to_original = resize_to_original

    def process_resized_element(self, elem: ElementImage) -> np.ndarray:
        if self._grayscale:
            img = cv.cvtColor(elem.image, cv.COLOR_BGR2GRAY)
        else:
            img = elem.image
        packet = pywt.WaveletPacket2D(data=img, wavelet=self._wavelet)
        nodes = packet.get_level(self._level)
        nodes = self._filter_nodes(nodes)
        features = np.array([n.data for n in nodes], dtype=np.float32)
        if self._target_mean is not None and self._target_std is not None:
            features = (
                self._target_std / np.std(features) * (features - np.mean(features))
                + self._target_mean
            )
        if self._resize_to_original:
            # Use OpenCV to resize the features to the original image size using
            # nearest neighbor interpolation; OpenCV uses a channel-last format,
            # so we need to transpose the features to that format and then back to
            # channel-first.
            features = np.transpose(features, (1, 2, 0))
            features = cv.resize(
                features,
                (img.shape[1], img.shape[0]),
                interpolation=cv.INTER_NEAREST,
            )
            features = np.transpose(features, (2, 0, 1))
        return features

    def _filter_nodes(self, nodes: list) -> list:
        def _matches_any(s: str, patterns: list[re.Pattern]) -> bool:
            return any(p.match(s) for p in patterns)

        if len(self._include_nodes) > 0:
            return [n for n in nodes if _matches_any(n.path, self._include_nodes)]
        elif len(self._exclude_nodes) > 0:
            return [n for n in nodes if not _matches_any(n.path, self._exclude_nodes)]
        else:
            return nodes
