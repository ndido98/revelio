import unittest.mock as mock
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from revelio.config.config import Config
from revelio.config.model.face_detection import FaceDetection, FaceDetectionAlgorithm
from revelio.dataset.element import (
    DatasetElement,
    ElementClass,
    ElementImage,
    Image,
    Landmarks,
)
from revelio.face_detection.detector import BoundingBox, FaceDetector


class Dummy2(FaceDetector):
    def process_element(self, elem: Image) -> tuple[BoundingBox, Optional[Landmarks]]:
        return ((5, 5, 15, 15), np.array([1, 2, 3]))


@pytest.fixture
def config() -> Config:
    return Config.construct(
        face_detection=FaceDetection(
            enabled=True,
            output_path=Path("/path/to/fd"),
            algorithm=FaceDetectionAlgorithm(name="dummy2", args={}),
        ),
    )


@pytest.fixture
def dummy2(config: Config) -> FaceDetector:
    return Dummy2(_config=config)


@pytest.fixture
def dataset_element() -> DatasetElement:
    img = np.zeros(((30, 30, 3)), dtype=np.uint8)
    return DatasetElement(
        x=(
            ElementImage(
                path=Path("/path/to/ds1/image1.jpg"),
                image=img.copy(),
            ),
            ElementImage(
                path=Path("/path/to/ds1/image2.jpg"),
                image=img.copy(),
            ),
        ),
        y=ElementClass.BONA_FIDE,
        dataset_root_path=Path("/path/to/ds1"),
        original_dataset="ds1",
    )


def test_meta_path(dummy2: FaceDetector, dataset_element: DatasetElement) -> None:
    assert dummy2._get_meta_path(dataset_element, 0) == Path(
        "/path/to/fd/dummy2/ds1/image1.meta.xz"
    )
    assert dummy2._get_meta_path(dataset_element, 1) == Path(
        "/path/to/fd/dummy2/ds1/image2.meta.xz"
    )


def test_meta_file_write(dummy2: FaceDetector, dataset_element: DatasetElement) -> None:
    with (
        mock.patch("pathlib.Path.is_file", return_value=False),
        mock.patch("pathlib.Path.mkdir", return_value=None),
        mock.patch("revelio.utils.caching.zstd_cacher.ZstdCacher.save") as mock_savez,
    ):
        new_elem, cached = dummy2.process(dataset_element)
        assert not cached
        assert new_elem.x[0].image is not None
        assert new_elem.x[0].image.shape == (10, 10, 3)
        assert new_elem.x[1].image is not None
        assert new_elem.x[1].image.shape == (10, 10, 3)
        assert new_elem.x[0].landmarks is not None
        assert new_elem.x[0].landmarks.tolist() == [1, 2, 3]
        assert new_elem.x[1].landmarks is not None
        assert new_elem.x[1].landmarks.tolist() == [1, 2, 3]
        assert mock_savez.call_count == 2
        # We have to manually check the call args because Numpy arrays override __eq__
        for call in mock_savez.call_args_list:
            _, kwargs = call
            assert np.array_equal(kwargs["bb"], np.array([5, 5, 15, 15]))
            assert np.array_equal(kwargs["landmarks"], np.array([1, 2, 3]))


def test_meta_file_read(dummy2: FaceDetector, dataset_element: DatasetElement) -> None:
    with (
        mock.patch("pathlib.Path.is_file", return_value=True),
        mock.patch("pathlib.Path.mkdir", return_value=None),
        mock.patch(
            "revelio.utils.caching.zstd_cacher.ZstdCacher.load",
            return_value={
                "bb": np.array([5, 5, 15, 15]),
                "landmarks": np.array([1, 2, 3]),
            },
        ),
    ):
        new_elem, cached = dummy2.process(dataset_element)
        assert cached
        assert new_elem.x[0].image is not None
        assert new_elem.x[0].image.shape == (10, 10, 3)
        assert new_elem.x[1].image is not None
        assert new_elem.x[1].image.shape == (10, 10, 3)
        assert new_elem.x[0].landmarks is not None
        assert new_elem.x[0].landmarks.tolist() == [1, 2, 3]
        assert new_elem.x[1].landmarks is not None
        assert new_elem.x[1].landmarks.tolist() == [1, 2, 3]
