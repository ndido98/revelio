from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pytest
from PIL.Image import Image

from revelio.augmentation.step import AugmentationStep
from revelio.config import Config
from revelio.config.model import Augmentation
from revelio.config.model import AugmentationStep as ConfigAugmentationStep
from revelio.config.model import (
    Dataset,
    DatasetSplit,
    FaceDetection,
    FaceDetectionAlgorithm,
    FeatureExtraction,
    FeatureExtractionAlgorithm,
)
from revelio.dataset import DatasetElement, DatasetFactory, ElementClass, ElementImage
from revelio.dataset.loaders import DatasetLoader
from revelio.face_detection.detector import BoundingBox, FaceDetector, Landmarks
from revelio.feature_extraction.extractor import FeatureExtractor


class DS1Loader(DatasetLoader):
    def load(self, path: Path) -> list[DatasetElement]:
        assert path == Path("/path/to/ds1")
        bona_fide = DatasetElement(
            x=(
                ElementImage(Path("/path/to/ds1/bf_a.png")),
                ElementImage(Path("/path/to/ds1/bf_b.png")),
            ),
            y=ElementClass.BONA_FIDE,
            original_dataset="ds1",
        )
        morphed = DatasetElement(
            x=(
                ElementImage(Path("/path/to/ds1/m_a.png")),
                ElementImage(Path("/path/to/ds1/m_b.png")),
            ),
            y=ElementClass.MORPHED,
            original_dataset="ds1",
        )
        return [bona_fide] * 50 + [morphed] * 500


class DS2Loader(DatasetLoader):
    def load(self, path: Path) -> list[DatasetElement]:
        assert path == Path("/path/to/ds2")
        bona_fide = DatasetElement(
            x=(
                ElementImage(Path("/path/to/ds2/bf_a.png")),
                ElementImage(Path("/path/to/ds2/bf_b.png")),
            ),
            y=ElementClass.BONA_FIDE,
            original_dataset="ds2",
        )
        morphed = DatasetElement(
            x=(
                ElementImage(Path("/path/to/ds2/m_a.png")),
                ElementImage(Path("/path/to/ds2/m_b.png")),
            ),
            y=ElementClass.MORPHED,
            original_dataset="ds2",
        )
        return [bona_fide] * 50 + [morphed] * 500


class Dummy(FaceDetector):
    def process_element(self, elem: Image) -> tuple[BoundingBox, Optional[Landmarks]]:
        return ((0, 0, 0, 0), None)


class Identity(AugmentationStep):
    def __init__(self, *, foo: str, **kwargs: Any) -> None:
        self.foo = foo
        super().__init__(**kwargs)

    def process_element(self, elem: ElementImage) -> ElementImage:
        return elem


class DummyFeatureExtractor(FeatureExtractor):
    def process_element(self, elem: ElementImage) -> np.ndarray:
        return np.array([1, 2, 3])


@pytest.fixture
def config() -> Config:
    return Config.construct(
        datasets=[
            Dataset.construct(
                name="ds1",
                path=Path("/path/to/ds1"),
                split=DatasetSplit(train=0.5, val=0.25, test=0.25),
            ),
            Dataset.construct(
                name="ds2",
                path=Path("/path/to/ds2"),
                split=DatasetSplit(train=0.8, val=0.1, test=0.1),
            ),
        ],
        face_detection=FaceDetection(
            enabled=True,
            output_path=Path("/path/to/face_detection"),
            algorithm=FaceDetectionAlgorithm(
                name="dummy",
                args={},
            ),
        ),
        augmentation=Augmentation(
            enabled=True,
            steps=[
                ConfigAugmentationStep(
                    uses="identity",
                    probability=1.0,
                    args={
                        "foo": "bar",
                    },
                )
            ],
        ),
        feature_extraction=FeatureExtraction(
            enabled=True,
            output_path=Path("/path/to/feature_extraction"),
            algorithms=[
                FeatureExtractionAlgorithm(
                    name="dummyfeatureextractor",
                    args={},
                ),
            ],
        ),
    )


@pytest.fixture
def bad_config_dataset() -> Config:
    return Config.construct(
        datasets=[
            Dataset.construct(
                name="nonexistent",
                path=Path("/path/to/nonexistent"),
                split=DatasetSplit(train=0.5, val=0.25, test=0.25),
            ),
        ],
        face_detection=None,
    )


def test_split(config: Config) -> None:
    factory = DatasetFactory(config)
    ds1_train = [elem for elem in factory._train if elem.original_dataset == "ds1"]
    ds2_train = [elem for elem in factory._train if elem.original_dataset == "ds2"]
    ds1_val = [elem for elem in factory._val if elem.original_dataset == "ds1"]
    ds2_val = [elem for elem in factory._val if elem.original_dataset == "ds2"]
    ds1_test = [elem for elem in factory._test if elem.original_dataset == "ds1"]
    ds2_test = [elem for elem in factory._test if elem.original_dataset == "ds2"]
    ds1_train_bf = [elem for elem in ds1_train if elem.y == ElementClass.BONA_FIDE]
    ds1_train_m = [elem for elem in ds1_train if elem.y == ElementClass.MORPHED]
    ds1_val_bf = [elem for elem in ds1_val if elem.y == ElementClass.BONA_FIDE]
    ds1_val_m = [elem for elem in ds1_val if elem.y == ElementClass.MORPHED]
    ds1_test_bf = [elem for elem in ds1_test if elem.y == ElementClass.BONA_FIDE]
    ds1_test_m = [elem for elem in ds1_test if elem.y == ElementClass.MORPHED]
    ds2_train_bf = [elem for elem in ds2_train if elem.y == ElementClass.BONA_FIDE]
    ds2_train_m = [elem for elem in ds2_train if elem.y == ElementClass.MORPHED]
    ds2_val_bf = [elem for elem in ds2_val if elem.y == ElementClass.BONA_FIDE]
    ds2_val_m = [elem for elem in ds2_val if elem.y == ElementClass.MORPHED]
    ds2_test_bf = [elem for elem in ds2_test if elem.y == ElementClass.BONA_FIDE]
    ds2_test_m = [elem for elem in ds2_test if elem.y == ElementClass.MORPHED]
    ds1_len = len(ds1_train) + len(ds1_val) + len(ds1_test)
    ds2_len = len(ds2_train) + len(ds2_val) + len(ds2_test)
    # Make sure each dataset has the correct number of elements
    assert ds1_len == 550
    assert ds2_len == 550
    # Make sure that the split of each dataset is correct, within a 1% tolerance
    assert len(ds1_train) / ds1_len == pytest.approx(0.5, abs=0.01)
    assert len(ds1_val) / ds1_len == pytest.approx(0.25, abs=0.01)
    assert len(ds1_test) / ds1_len == pytest.approx(0.25, abs=0.01)
    assert len(ds2_train) / ds2_len == pytest.approx(0.8, abs=0.01)
    assert len(ds2_val) / ds2_len == pytest.approx(0.1, abs=0.01)
    assert len(ds2_test) / ds2_len == pytest.approx(0.1, abs=0.01)
    # Make sure that the number of element in each class adds up
    assert len(ds1_train_bf) + len(ds1_val_bf) + len(ds1_test_bf) == 50
    assert len(ds1_train_m) + len(ds1_val_m) + len(ds1_test_m) == 500
    assert len(ds2_train_bf) + len(ds2_val_bf) + len(ds2_test_bf) == 50
    assert len(ds2_train_m) + len(ds2_val_m) + len(ds2_test_m) == 500


def test_nonexistent_dataset(bad_config_dataset: Config) -> None:
    with pytest.raises(ValueError):
        DatasetFactory(bad_config_dataset)


def test_augmentation_is_loaded(config: Config) -> None:
    factory = DatasetFactory(config)
    assert len(factory._augmentation_steps) == 1
    step: Identity = cast(Identity, factory._augmentation_steps[0])
    assert step.foo == "bar"
    assert step._probability == 1.0


def test_extraction_is_loaded(config: Config) -> None:
    factory = DatasetFactory(config)
    assert len(factory._feature_extractors) == 1


def test_train_dataset(config: Config) -> None:
    factory = DatasetFactory(config)
    ds = factory.get_train_dataset()
    assert ds._paths == factory._train


def test_val_dataset(config: Config) -> None:
    factory = DatasetFactory(config)
    ds = factory.get_val_dataset()
    assert ds._paths == factory._val
    assert len(ds._augmentation_steps) == 0


def test_test_dataset(config: Config) -> None:
    factory = DatasetFactory(config)
    ds = factory.get_test_dataset()
    assert ds._paths == factory._test
    assert len(ds._augmentation_steps) == 0