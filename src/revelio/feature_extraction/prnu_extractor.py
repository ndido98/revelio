from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2 as cv
import numpy as np
import pywt

from .same_size_extractor import SameSizeExtractor

if TYPE_CHECKING:
    from revelio.dataset.element import ElementImage


# Taken from https://github.com/polimi-ispl/prnu-python


class PRNUExtractor(SameSizeExtractor):  # pragma: no cover
    def __init__(
        self,
        *,
        levels: int = 4,
        sigma: float = 5.0,
        wdft_sigma: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._levels = levels
        self._sigma = sigma
        self._wdft_sigma = wdft_sigma

    def process_resized_element(self, elem: ElementImage) -> np.ndarray:
        rgb = cv.cvtColor(elem.image, cv.COLOR_BGR2RGB)
        noise = self._extract_noise(rgb)
        noise = self._rgb2gray(noise)
        noise = self._zero_mean_total(noise)
        noise_std = (
            np.std(noise, ddof=1) if self._wdft_sigma == 0.0 else self._wdft_sigma
        )
        prnu = self._wiener_dft(noise, noise_std).astype(np.float32)
        return prnu[np.newaxis, ...]

    def _extract_noise(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)

        noise_var = self._sigma**2

        residual = np.zeros_like(img, dtype=np.float32)
        for ch in range(img.shape[2]):
            wavelet = None
            while wavelet is None and self._levels > 0:
                try:
                    wavelet = pywt.wavedec2(img[..., ch], "db4", level=self._levels)
                except ValueError:
                    self._levels -= 1
                    wavelet = None
            if wavelet is None:
                raise ValueError("Could not extract wavelet")

            wavelet_details = wavelet[1:]

            wavelet_details_filter = []
            for wavelet_level in wavelet_details:
                level_coeff_filt = [
                    self._wiener_adaptive(coeff, noise_var) for coeff in wavelet_level
                ]
                wavelet_details_filter.append(level_coeff_filt)

            wavelet[1:] = wavelet_details_filter
            wavelet[0][...] = 0

            reconstructed = pywt.waverec2(wavelet, "db4")
            img_height, img_width = img.shape
            residual[..., ch] = reconstructed[0:img_width, 0:img_height]
        assert residual.shape[:2] == img.shape[:2]
        return residual

    def _wiener_adaptive(self, img: np.ndarray, noise_var: float) -> np.ndarray:
        window_size_list = [3, 5, 7, 9]
        energy = img**2
        avg_win_energy = np.zeros(
            img.shape + (len(window_size_list),), dtype=np.float32
        )
        for i, window_size in enumerate(window_size_list):
            avg_win_energy[..., i] = cv.boxFilter(
                energy, -1, (window_size, window_size)
            )
        coef_var = np.maximum(0, avg_win_energy - noise_var)
        coef_var_min = np.min(coef_var, axis=2)
        img = img * noise_var / (coef_var_min + noise_var)
        return img

    def _rgb2gray(self, img: np.ndarray) -> np.ndarray:
        rgb2gray_vector = (
            np.array([0.29893602, 0.58704307, 0.11402090])
            .reshape(3, 1)
            .astype(np.float32)
        )

        w, h = img.shape[:2]
        linearized = img.reshape(w * h, 3)
        img_gray: np.ndarray = np.dot(linearized, rgb2gray_vector)
        img_gray = img_gray.reshape(w, h)

        return img_gray.astype(np.float32)

    def _zero_mean(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        i_zm = img - np.mean(img, dtype=np.float32)

        row_mean = np.mean(i_zm, axis=1, dtype=np.float32).reshape(h, 1)
        col_mean = np.mean(i_zm, axis=0, dtype=np.float32).reshape(1, w)
        return i_zm - row_mean - col_mean  # type: ignore

    def _zero_mean_total(self, img: np.ndarray) -> np.ndarray:
        img[0::2, 0::2] = self._zero_mean(img[0::2, 0::2])
        img[1::2, 0::2] = self._zero_mean(img[1::2, 0::2])
        img[0::2, 1::2] = self._zero_mean(img[0::2, 1::2])
        img[1::2, 1::2] = self._zero_mean(img[1::2, 1::2])
        return img

    def _wiener_dft(self, img: np.ndarray, sigma: float) -> np.ndarray:
        noise_var = sigma**2
        h, w = img.shape
        img_noise_fft = np.fft.fft2(img)
        img_noise_fft_mag = np.abs(img_noise_fft / (h * w) ** 0.5)
        img_noise_fft_mag_noise = self._wiener_adaptive(img_noise_fft_mag, noise_var)
        zeros_y, zeros_x = np.nonzero(img_noise_fft_mag == 0)
        img_noise_fft_mag[zeros_y, zeros_x] = 1
        img_noise_fft_mag_noise[zeros_y, zeros_x] = 0
        img_noise_fft_filt = img_noise_fft * img_noise_fft_mag_noise / img_noise_fft_mag
        img_noise_filt = np.fft.ifft2(img_noise_fft_filt).real
        return img_noise_filt.astype(np.float32)
