from pathlib import Path
from unittest import mock

import pytest
from PIL import Image
from PIL.ImageOps import invert

from revelio.augmentation.step import AugmentationStep
from revelio.dataset.element import DatasetElement, ElementClass, ElementImage


class Invert(AugmentationStep):
    def process_element(self, elem: ElementImage) -> ElementImage:
        assert elem.image is not None
        return ElementImage(
            path=elem.path,
            image=invert(elem.image),
        )


@pytest.fixture
def dataset_element() -> DatasetElement:
    black = Image.new("RGB", (1, 1), "black")
    img = ElementImage(path=Path("test"), image=black)
    return DatasetElement(
        dataset_root_path=Path("test"),
        original_dataset="test",
        x=(img, img),
        y=ElementClass.BONA_FIDE,
    )


def is_black(img: ElementImage) -> bool:
    assert img.image is not None
    return img.image.getpixel((0, 0)) == (0, 0, 0)  # type: ignore


def is_white(img: ElementImage) -> bool:
    print(img.image)
    assert img.image is not None
    return img.image.getpixel((0, 0)) == (255, 255, 255)  # type: ignore


def test_probability_0(dataset_element: DatasetElement) -> None:
    step = Invert(probability=0.0)
    for _ in range(1000):
        processed = step.process(dataset_element)
        for img in processed.x:
            assert is_black(img)


def test_probability_1(dataset_element: DatasetElement) -> None:
    step = Invert(probability=1.0)
    for _ in range(1000):
        processed = step.process(dataset_element)
        for img in processed.x:
            assert is_white(img)


def test_probability_0_5(dataset_element: DatasetElement) -> None:
    step = Invert(probability=0.5)
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
