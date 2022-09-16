import unittest.mock as mock
from json import JSONDecodeError
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from PIL import Image as ImageModule
from PIL.Image import Image

from revelio.config.config import Config
from revelio.config.model.face_detection import FaceDetection, FaceDetectionAlgorithm
from revelio.dataset.element import DatasetElement, ElementClass, ElementImage
from revelio.face_detection.detector import BoundingBox, FaceDetector, Landmarks


class Dummy2(FaceDetector):
    def process_element(self, elem: Image) -> tuple[BoundingBox, Optional[Landmarks]]:
        return ((0, 0, 0, 0), np.array([1, 2, 3]))


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
    return Dummy2(config=config)


@pytest.fixture
def dataset_element() -> DatasetElement:
    return DatasetElement(
        x=(
            ElementImage(
                path=Path("/path/to/ds1/image1.jpg"),
                image=ImageModule.new("RGB", (1, 1), "black"),
            ),
            ElementImage(
                path=Path("/path/to/ds1/image2.jpg"),
                image=ImageModule.new("RGB", (1, 1), "black"),
            ),
        ),
        y=ElementClass.BONA_FIDE,
        original_dataset="ds1",
    )


def test_meta_path(dummy2: FaceDetector, dataset_element: DatasetElement) -> None:
    assert dummy2._get_meta_path(dataset_element, 0) == Path(
        "/path/to/fd/dummy2/ds1/image1.meta.json"
    )
    assert dummy2._get_meta_path(dataset_element, 1) == Path(
        "/path/to/fd/dummy2/ds1/image2.meta.json"
    )


def test_meta_file_write(dummy2: FaceDetector, dataset_element: DatasetElement) -> None:
    with (
        mock.patch("pathlib.Path.is_file", return_value=False),
        mock.patch("pathlib.Path.mkdir", return_value=None),
        mock.patch.object(Path, "write_text") as mock_write_text,
    ):
        dummy2.process(dataset_element)
        assert mock_write_text.call_count == 2
        mock_write_text.assert_called_with(
            '{"bb": [0, 0, 0, 0], "landmarks": [1, 2, 3]}'
        )


def test_meta_file_read(dummy2: FaceDetector, dataset_element: DatasetElement) -> None:
    with (
        mock.patch("pathlib.Path.is_file", return_value=True),
        mock.patch("pathlib.Path.mkdir", return_value=None),
        mock.patch.object(
            Path,
            "read_text",
            return_value='{"bb": [0, 0, 0, 0], "landmarks": [1, 2, 3]}',
        ),
    ):
        new_elem = dummy2.process(dataset_element)
        assert new_elem.x[0].landmarks is not None
        assert new_elem.x[0].landmarks.tolist() == [1, 2, 3]
        assert new_elem.x[1].landmarks is not None
        assert new_elem.x[1].landmarks.tolist() == [1, 2, 3]


def test_meta_file_read_error(
    dummy2: FaceDetector, dataset_element: DatasetElement
) -> None:
    with (
        mock.patch("pathlib.Path.is_file", return_value=True),
        mock.patch("pathlib.Path.mkdir", return_value=None),
        mock.patch.object(
            Path, "read_text", return_value='{"bb": [0, 0, 0, 0], "landmarks": [1, 2, 3'
        ),
        pytest.raises(JSONDecodeError),
    ):
        dummy2.process(dataset_element)


def test_meta_file_without_bounding_boxes(
    dummy2: FaceDetector, dataset_element: DatasetElement
) -> None:
    with (
        mock.patch("pathlib.Path.is_file", return_value=True),
        mock.patch("pathlib.Path.mkdir", return_value=None),
        mock.patch.object(Path, "read_text", return_value='{"landmarks": [1, 2, 3]}'),
        pytest.raises(ValueError),
    ):
        dummy2.process(dataset_element)
