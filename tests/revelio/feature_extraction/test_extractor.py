import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest

from revelio.config import Config
from revelio.config.model import FeatureExtraction, FeatureExtractionAlgorithm
from revelio.dataset import DatasetElement, ElementClass, ElementImage
from revelio.feature_extraction import FeatureExtractor


class Dummy1(FeatureExtractor):
    def process_element(self, elem: ElementImage) -> np.ndarray:
        return np.array([1, 2, 3])


class Dummy2(FeatureExtractor):
    def process_element(self, elem: ElementImage) -> np.ndarray:
        return np.array([4, 5, 6])


@pytest.fixture
def config() -> Config:
    return Config.construct(
        feature_extraction=FeatureExtraction(
            enabled=True,
            output_path=Path("/path/to/fe"),
            algorithms=[
                FeatureExtractionAlgorithm(name="dummy1", args={}),
                FeatureExtractionAlgorithm(name="dummy2", args={}),
            ],
        ),
    )


@pytest.fixture
def dummy1(config: Config) -> FeatureExtractor:
    return Dummy1(_config=config)


@pytest.fixture
def dummy2(config: Config) -> FeatureExtractor:
    return Dummy2(_config=config)


@pytest.fixture
def dataset_element() -> DatasetElement:
    return DatasetElement(
        x=(
            ElementImage(
                path=Path("/path/to/ds1/image1.jpg"),
                image=np.zeros((1, 1, 3), dtype=np.uint8),
            ),
            ElementImage(
                path=Path("/path/to/ds1/image2.jpg"),
                image=np.zeros((1, 1, 3), dtype=np.uint8),
            ),
        ),
        y=ElementClass.BONA_FIDE,
        dataset_root_path=Path("/path/to/ds1"),
        original_dataset="ds1",
    )


def test_features_path(
    dummy1: FeatureExtractor, dataset_element: DatasetElement
) -> None:
    assert dummy1._get_features_path(dataset_element, 0) == Path(
        "/path/to/fe/dummy1/ds1/image1.features.json"
    )
    assert dummy1._get_features_path(dataset_element, 1) == Path(
        "/path/to/fe/dummy1/ds1/image2.features.json"
    )


def test_features_present_non_forced(
    dummy1: Dummy1, dataset_element: DatasetElement
) -> None:
    with (
        mock.patch("pathlib.Path.is_file", return_value=True),
        mock.patch("pathlib.Path.mkdir", return_value=None),
        mock.patch.object(Path, "read_text", return_value="[1, 2, 3]"),
    ):
        new_elem = dummy1.process(dataset_element)
        for i in range(len(new_elem.x)):
            assert "dummy1" in new_elem.x[i].features
            assert np.all(new_elem.x[i].features["dummy1"] == np.array([1, 2, 3]))


def test_features_present_forced(
    dummy1: Dummy1, dataset_element: DatasetElement
) -> None:
    with (
        mock.patch("pathlib.Path.is_file", return_value=True),
        mock.patch("pathlib.Path.mkdir", return_value=None),
        mock.patch.object(Path, "write_text") as mock_write_text,
    ):
        new_elem = dummy1.process(dataset_element, force_online=True)
        for i in range(len(new_elem.x)):
            assert "dummy1" in new_elem.x[i].features
            assert np.all(new_elem.x[i].features["dummy1"] == np.array([1, 2, 3]))
        mock_write_text.assert_not_called()


def test_features_not_present_not_forced(
    dummy1: Dummy1, dataset_element: DatasetElement
) -> None:
    with (
        mock.patch("pathlib.Path.is_file", return_value=False),
        mock.patch("pathlib.Path.mkdir", return_value=None),
        mock.patch.object(Path, "write_text") as mock_write_text,
    ):
        new_elem = dummy1.process(dataset_element)
        for i in range(len(new_elem.x)):
            assert "dummy1" in new_elem.x[i].features
            assert np.all(new_elem.x[i].features["dummy1"] == np.array([1, 2, 3]))
        mock_write_text.assert_called()


def test_features_not_present_forced(
    dummy1: Dummy1, dataset_element: DatasetElement
) -> None:
    with (
        mock.patch("pathlib.Path.is_file", return_value=False),
        mock.patch("pathlib.Path.mkdir", return_value=None),
        mock.patch.object(Path, "write_text") as mock_write_text,
    ):
        new_elem = dummy1.process(dataset_element, force_online=True)
        for i in range(len(new_elem.x)):
            assert "dummy1" in new_elem.x[i].features
            assert np.all(new_elem.x[i].features["dummy1"] == np.array([1, 2, 3]))
        mock_write_text.assert_not_called()


def test_multiple_features(
    dummy1: Dummy1, dummy2: Dummy2, dataset_element: DatasetElement
) -> None:
    with (
        mock.patch("pathlib.Path.is_file", return_value=False),
        mock.patch("pathlib.Path.mkdir", return_value=None),
    ):
        elem = dummy1.process(dataset_element, force_online=True)
        elem = dummy2.process(elem, force_online=True)
        for i in range(len(elem.x)):
            assert "dummy1" in elem.x[i].features
            assert np.all(elem.x[i].features["dummy1"] == np.array([1, 2, 3]))
            assert "dummy2" in elem.x[i].features
            assert np.all(elem.x[i].features["dummy2"] == np.array([4, 5, 6]))
