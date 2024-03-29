from pathlib import Path
from typing import Optional
from unittest import mock

import numpy as np
import pytest

from revelio.augmentation.step import AugmentationStep
from revelio.dataset.element import (
    DatasetElement,
    ElementClass,
    ElementImage,
    Image,
    Landmarks,
)


class Invert(AugmentationStep):
    def process_element(
        self, image: Image, landmarks: Optional[Landmarks]
    ) -> tuple[Image, Optional[Landmarks]]:
        return 255 - image, landmarks


@pytest.fixture
def dataset_element() -> DatasetElement:
    black = np.array([[[0, 0, 0]]], dtype=np.uint8)
    img = ElementImage(path=Path("test"), image=black)
    return DatasetElement(
        dataset_root_path=Path("test"),
        original_dataset="test",
        x=(img, img),
        y=ElementClass.BONA_FIDE,
    )


def is_black(img: ElementImage) -> bool:
    return np.all(img.image[0, 0] == np.array([0, 0, 0]))  # type: ignore


def is_white(img: ElementImage) -> bool:
    return np.all(img.image[0, 0] == np.array([255, 255, 255]))  # type: ignore


def test_probability_0(dataset_element: DatasetElement) -> None:
    step = Invert(_probability=0.0, _applies_to="all")
    for _ in range(1000):
        processed = step.process(dataset_element)
        for img in processed.x:
            assert is_black(img)


def test_probability_1(dataset_element: DatasetElement) -> None:
    step = Invert(_probability=1.0, _applies_to="all")
    for _ in range(1000):
        processed = step.process(dataset_element)
        for img in processed.x:
            assert is_white(img)


def test_probability_0_5(dataset_element: DatasetElement) -> None:
    step = Invert(_probability=0.5, _applies_to="all")
    black_count = 0
    white_count = 0
    attempts = 1000
    # Chosen by fair dice roll. Guaranteed to be random.
    with mock.patch("random.random", return_value=0.4):
        for _ in range(attempts):
            processed = step.process(dataset_element)
            for img in processed.x:
                if is_black(img):
                    black_count += 1
                elif is_white(img):
                    white_count += 1
                else:
                    pytest.fail()
        assert black_count == 0
        assert white_count == attempts * 2


def test_applies_to(dataset_element: DatasetElement) -> None:
    step = Invert(_probability=1.0, _applies_to=[1])
    processed = step.process(dataset_element)
    assert is_black(processed.x[0])
    assert is_white(processed.x[1])
