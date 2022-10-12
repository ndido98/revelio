from pathlib import Path
from typing import Any, Optional, cast

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

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
    Loader,
)
from revelio.config.model.preprocessing import Preprocessing
from revelio.config.model.preprocessing import (
    PreprocessingStep as ConfigPreprocessingStep,
)
from revelio.dataset import (
    DatasetElementDescriptor,
    DatasetFactory,
    ElementClass,
    ElementImage,
    Image,
)
from revelio.dataset.dataset_factory import _split_train_val_test
from revelio.dataset.loaders import DatasetLoader
from revelio.face_detection.detector import BoundingBox, FaceDetector, Landmarks
from revelio.feature_extraction.extractor import FeatureExtractor
from revelio.preprocessing.step import PreprocessingStep


class DS1Loader(DatasetLoader):
    def __init__(self, test: str) -> None:
        assert test == "me"

    def load(self, path: Path) -> list[DatasetElementDescriptor]:
        assert path == Path("/path/to/ds1")
        bona_fide = DatasetElementDescriptor(
            x=(
                Path("/path/to/ds1/bf_a.png"),
                Path("/path/to/ds1/bf_b.png"),
            ),
            y=ElementClass.BONA_FIDE,
        )
        morphed = DatasetElementDescriptor(
            x=(
                Path("/path/to/ds1/m_a.png"),
                Path("/path/to/ds1/m_b.png"),
            ),
            y=ElementClass.MORPHED,
        )
        return [bona_fide] * 50 + [morphed] * 500


class DS2Loader(DatasetLoader):
    def load(self, path: Path) -> list[DatasetElementDescriptor]:
        assert path == Path("/path/to/ds2")
        bona_fide = DatasetElementDescriptor(
            x=(
                Path("/path/to/ds2/bf_a.png"),
                Path("/path/to/ds2/bf_b.png"),
            ),
            y=ElementClass.BONA_FIDE,
        )
        morphed = DatasetElementDescriptor(
            x=(
                Path("/path/to/ds2/m_a.png"),
                Path("/path/to/ds2/m_b.png"),
            ),
            y=ElementClass.MORPHED,
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


class Pass(PreprocessingStep):
    def process_element(self, elem: ElementImage) -> ElementImage:
        return elem


@pytest.fixture
def config() -> Config:
    return Config.construct(
        datasets=[
            Dataset.construct(
                name="ds1",
                loader=Loader.construct(
                    name="ds1loader",
                    args={"test": "me"},
                ),
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
        preprocessing=Preprocessing(
            steps=[
                ConfigPreprocessingStep(
                    uses="pass",
                    args={},
                )
            ]
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


@given(
    to_split=st.lists(st.integers(), min_size=100, unique=True),
    split=(
        st.tuples(
            st.floats(
                0.0,
                1.0,
                width=16,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            ),
            st.floats(
                0.0,
                1.0,
                width=16,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            ),
            st.floats(
                0.0,
                1.0,
                width=16,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            ),
        ).filter(lambda t: 0 <= sum(t) <= 1.0)
    ),
)
def test_split_simple(to_split: list[int], split: tuple[float, float, float]) -> None:
    train_percentage, val_percentage, test_percentage = split
    train, val, test = _split_train_val_test(
        to_split, train_percentage, val_percentage, test_percentage
    )
    if test_percentage > 0:
        assert len(test) / len(to_split) == pytest.approx(test_percentage, abs=0.01)
    if val_percentage > 0:
        assert len(val) / len(to_split) == pytest.approx(val_percentage, abs=0.01)
    if train_percentage > 0:
        assert len(train) / len(to_split) == pytest.approx(train_percentage, abs=0.01)
    assert len(train) + len(val) + len(test) <= len(to_split)
    # Make sure we don't have any duplicate elements
    assert len(train) == len(set(train))
    assert len(val) == len(set(val))
    assert len(test) == len(set(test))
    # Make sure we don't have any duplicate elements between the three splits
    assert len(train) + len(val) + len(test) == len(set(train + val + test))
    if train_percentage == 0 and val_percentage == 0 and test_percentage == 0:
        # Special case: all percentages are 0, so we should have no elements
        assert len(train) == 0
        assert len(val) == 0
        assert len(test) == 0


def test_split(config: Config) -> None:
    factory = DatasetFactory(config)
    ds1_train = [elem for elem in factory._train if elem._dataset_name == "ds1"]
    ds2_train = [elem for elem in factory._train if elem._dataset_name == "ds2"]
    ds1_val = [elem for elem in factory._val if elem._dataset_name == "ds1"]
    ds2_val = [elem for elem in factory._val if elem._dataset_name == "ds2"]
    ds1_test = [elem for elem in factory._test if elem._dataset_name == "ds1"]
    ds2_test = [elem for elem in factory._test if elem._dataset_name == "ds2"]
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


def test_preprocessing_is_loaded(config: Config) -> None:
    factory = DatasetFactory(config)
    assert len(factory._preprocessing_steps) == 1


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
