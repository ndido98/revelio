from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal

import cv2 as cv
import numpy as np
import pywt

from .same_size_extractor import SameSizeExtractor
from .stationary_wavelet_packets import StationaryWaveletPacket2D  # type: ignore

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
        domain: Literal["spatial", "frequency", "uniform"] = "frequency",
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
        self._domain = domain

    def process_resized_element(self, elem: ElementImage) -> np.ndarray:
        if self._grayscale:
            img = cv.cvtColor(elem.image, cv.COLOR_BGR2GRAY)
        else:
            img = elem.image
        rows, cols = elem.image.shape[:2]
        if self._domain == "spatial":
            packet = pywt.WaveletPacket2D(data=img, wavelet=self._wavelet)
            nodes = packet.get_level(self._level)
            filtered = self._filter_nodes([n.path for n in nodes])
            nodes = [n for n in nodes if n.path in filtered]
            features = np.zeros((len(nodes), rows, cols), dtype=np.float32)
            for i, n in enumerate(nodes):
                new_packet = pywt.WaveletPacket2D(data=None, wavelet=self._wavelet)
                new_packet[n.path] = n.data
                reconstructed = new_packet.reconstruct(update=False)
                features[i, ...] = reconstructed[:rows, :cols]
        elif self._domain == "frequency":
            packet = pywt.WaveletPacket2D(data=img, wavelet=self._wavelet)
            nodes = packet.get_level(self._level)
            filtered = self._filter_nodes([n.path for n in nodes])
            nodes = [n for n in nodes if n.path in filtered]
            features = np.array([n.data for n in nodes], dtype=np.float32)
        elif self._domain == "uniform":
            # The image's shape must be a multiple of 2**level;
            # if it's not, we pad it to the nearest multiple
            padded = np.pad(
                img,
                (
                    (0, 2**self._level - img.shape[0] % 2**self._level),
                    (0, 2**self._level - img.shape[1] % 2**self._level),
                ),
                mode="constant",
            )
            packet = StationaryWaveletPacket2D(data=padded, wavelet=self._wavelet)
            nodes = packet.get_level(self._level)
            filtered = self._filter_nodes([n.path for n in nodes])
            nodes = [n for n in nodes if n.path in filtered]
            features = np.array(
                [n.data[:rows, :cols] for n in nodes],
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Unknown domain: {self._domain}")
        if self._target_mean is not None and self._target_std is not None:
            features = (
                self._target_std / np.std(features) * (features - np.mean(features))
                + self._target_mean
            )
        return features

    def _filter_nodes(self, nodes: list[str]) -> list[str]:
        def _matches_any(s: str, patterns: list[re.Pattern]) -> bool:
            return any(p.match(s) for p in patterns)

        if len(self._include_nodes) > 0:
            return [n for n in nodes if _matches_any(n, self._include_nodes)]
        elif len(self._exclude_nodes) > 0:
            return [n for n in nodes if not _matches_any(n, self._exclude_nodes)]
        else:
            return nodes
