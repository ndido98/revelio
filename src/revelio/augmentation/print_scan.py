from typing import Any, Optional

import cv2 as cv
import numpy as np

from revelio.dataset.element import Image, Landmarks

from .step import AugmentationStep

_PREWITT_KERNEL_SEP1 = np.array([1, 1, 1])
_PREWITT_KERNEL_SEP2 = np.array([1, 0, -1])


def _get_sigma(image: Image) -> float:
    ref_diag1, ref_sigma1 = 195.6, 0.8
    ref_diag2, ref_sigma2 = 1799.3, 4.0
    diag = np.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2)
    # Linearly interpolate between the two reference sigmas based on the image diagonal
    # fmt: off
    result = (
        ref_sigma1 * (ref_diag2 - diag) / (ref_diag2 - ref_diag1)
        + ref_sigma2 * (diag - ref_diag1) / (ref_diag2 - ref_diag1)
    )
    # fmt: on
    # Clamp the resulting sigma to a minimum of 0.8,
    # which is the value OpenCV uses when using 3x3 filters
    return float(np.clip(result, 0.8, None))


def _gaussians_in_range(
    shape: tuple, mean: float, numbers_range: tuple[float, float]
) -> np.ndarray:
    from_range, to_range = numbers_range
    p = (np.random.random(shape) * 100).astype(np.int32)
    retval = np.where(
        p < np.abs(from_range - to_range),
        np.random.normal(loc=from_range, scale=mean - from_range, size=shape),
        np.random.normal(loc=mean, scale=to_range - mean, size=shape),
    )
    while True:
        before_from = retval < from_range
        after_to = retval > to_range
        if not np.any(before_from) and not np.any(after_to):
            break
        retval[before_from] = (from_range - retval[before_from]) + from_range
        retval[after_to] = to_range - (retval[after_to] - to_range)
    return retval


class PrintScan(AugmentationStep):
    """
    Apply a simulated print and scan process to the image,
    based on "Face morphing detection in the presence of printing/scanning
    and heterogeneous image sources" by Ferrara et al.
    """

    def __init__(
        self,
        *,
        apply_edge_noise: bool = True,
        apply_dark_area_noise: bool = True,
        color_correction_alpha: float = 8.3,
        color_correction_beta_k: float = 20.0,
        color_correction_beta_x: float = 35.0,
        color_correction_gamma: float = 0.6,
        cutoff_noise_threshold: float = 32.0,
        max_n2_noise_value: float = 20.0,
        **kwargs: Any,
    ):
        """
        Apply a simulated print and scan process to the image,
        based on "Face morphing detection in the presence of printing/scanning
        and heterogeneous image sources" by Ferrara et al.

        Args:
            apply_edge_noise: Whether to apply a simulated edge noise.
            apply_dark_area_noise: Whether to apply a simulated dark area noise.
            color_correction_alpha: Alpha parameter of the gamma correction.
            color_correction_beta_k: Beta_k parameter of the gamma correction.
            color_correction_beta_x: Beta_x parameter of the gamma correction.
            color_correction_gamma: Gamma parameter of the gamma correction.
            cutoff_noise_threshold: Threshold for the dark area noise.
            max_n2_noise_value: Maximum value for the dark area noise.
        """
        super().__init__(**kwargs)
        self._apply_edge_noise = apply_edge_noise
        self._apply_dark_area_noise = apply_dark_area_noise
        self._color_correction_alpha = color_correction_alpha
        self._color_correction_beta_k = color_correction_beta_k
        self._color_correction_beta_x = color_correction_beta_x
        self._color_correction_gamma = color_correction_gamma
        self._cutoff_noise_threshold = cutoff_noise_threshold
        self._max_n2_noise_value = max_n2_noise_value
        self._precompute_gradient_constants()
        self._precompute_gamma_lut()
        self._precompute_dark_area_lut()

    def _precompute_gradient_constants(self) -> None:
        self._gradient_x_weight = 0.4
        self._gradient_y_weight = 1.0 - self._gradient_x_weight
        self._max_gradient_magnitude = (
            self._gradient_x_weight * 255 * 255 + self._gradient_y_weight * 255 * 255
        )

    def _precompute_gamma_lut(self) -> None:
        lut = np.arange(256, dtype=np.int16)
        lut = np.round(
            self._color_correction_alpha
            * (
                np.maximum(lut - self._color_correction_beta_x, 0)
                ** self._color_correction_gamma
            )
            + self._color_correction_beta_k
        )
        self._gamma_lut = np.clip(lut, 0, 255).astype(np.uint8)

    def _precompute_dark_area_lut(self) -> None:
        lut = np.arange(256, dtype=np.float64)
        lut = 1 / (1 + np.exp(lut - self._cutoff_noise_threshold))
        lut = np.round(lut * self._max_n2_noise_value)
        self._dark_area_lut = np.clip(lut, 0, 255).astype(np.uint8)

    def process_element(
        self, image: Image, landmarks: Optional[Landmarks]
    ) -> tuple[Image, Optional[Landmarks]]:
        image = self._psf(image)
        if self._apply_edge_noise:
            image = self._edge_noise(image)
        image = self._gamma_correction(image)
        if self._apply_dark_area_noise:
            image = self._dark_area_noise(image)
        sigma = _get_sigma(image)
        image = cv.GaussianBlur(image, ksize=None, sigmaX=sigma, sigmaY=sigma)
        return image, landmarks

    def _compute_gradient_magnitude(self, image: Image) -> np.ndarray:
        gradient_x = cv.sepFilter2D(
            image, -1, _PREWITT_KERNEL_SEP2, _PREWITT_KERNEL_SEP1
        )
        gradient_y = cv.sepFilter2D(
            image, -1, _PREWITT_KERNEL_SEP1, _PREWITT_KERNEL_SEP2
        )
        return (  # type: ignore
            self._gradient_x_weight * gradient_x * gradient_x
            + self._gradient_y_weight * gradient_y * gradient_y
        ) / self._max_gradient_magnitude

    def _edge_noise(self, image: Image) -> Image:
        magnitude = self._compute_gradient_magnitude(image)
        noise = (
            _gaussians_in_range(image.shape, mean=128, numbers_range=(0, 255))
            * magnitude
        )
        summed = image.astype(np.int16) + noise
        return np.clip(np.round(summed), 0, 255).astype(np.uint8)  # type: ignore

    def _gamma_correction(self, image: Image) -> Image:
        return cv.LUT(image, self._gamma_lut)  # type: ignore

    def _dark_area_noise(self, image: Image) -> Image:
        noise = np.round(
            cv.LUT(image, self._dark_area_lut).astype(np.float32)
            * np.random.random(image.shape)
        ).astype(np.uint8)
        summed = image.astype(np.int16) + noise
        return np.clip(np.round(summed), 0, 255).astype(np.uint8)  # type: ignore

    def _psf(self, image: Image) -> Image:
        sigma = _get_sigma(image)
        return np.round(  # type: ignore
            cv.GaussianBlur(image, ksize=None, sigmaX=sigma, sigmaY=sigma)
        ).astype(np.uint8)
