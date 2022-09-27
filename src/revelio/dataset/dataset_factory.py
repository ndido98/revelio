import random
from typing import Optional

from revelio.augmentation import AugmentationStep
from revelio.config import Config
from revelio.face_detection import FaceDetector
from revelio.feature_extraction import FeatureExtractor
from revelio.registry import Registrable

from .dataset import Dataset
from .element import DatasetElement, ElementClass
from .loaders.loader import DatasetLoader

__all__ = ("DatasetFactory",)


def _split_dataset(
    dataset: list[DatasetElement],
    split: float,
) -> tuple[list[DatasetElement], list[DatasetElement]]:
    shuffled = random.sample(dataset, len(dataset))
    split_index = round(len(shuffled) * split)
    return dataset[:split_index], dataset[split_index:]


def _split_train_val_test(
    dataset: list[DatasetElement],
    train_percentage: float,
    val_percentage: float,
) -> tuple[list[DatasetElement], list[DatasetElement], list[DatasetElement]]:
    train_val_percentage = train_percentage + val_percentage
    train_val, test = _split_dataset(dataset, train_val_percentage)
    train, val = _split_dataset(train_val, train_percentage / train_val_percentage)
    return train, val, test


class DatasetFactory:

    _train: list[DatasetElement] = []
    _val: list[DatasetElement] = []
    _test: list[DatasetElement] = []

    _face_detector: Optional[FaceDetector]
    _augmentation_steps: list[AugmentationStep]
    _feature_extractors: list[FeatureExtractor]

    def __init__(self, config: Config) -> None:
        self._config = config
        loaders = self._get_loaders()
        # Merge the datasets with their respective train, val and test percentages
        for dataset, loader in zip(config.datasets, loaders):
            dataset_xy = loader.load(dataset.path)
            bona_fide_xy = [e for e in dataset_xy if e.y == ElementClass.BONA_FIDE]
            morphed_xy = [e for e in dataset_xy if e.y == ElementClass.MORPHED]

            bona_fide_train, bona_fide_val, bona_fide_test = _split_train_val_test(
                bona_fide_xy,
                dataset.split.train,
                dataset.split.val,
            )
            morphed_train, morphed_val, morphed_test = _split_train_val_test(
                morphed_xy,
                dataset.split.train,
                dataset.split.val,
            )

            self._train.extend(bona_fide_train + morphed_train)
            self._val.extend(bona_fide_val + morphed_val)
            self._test.extend(bona_fide_test + morphed_test)
        # Shuffle the three complete datasets
        self._train = random.sample(self._train, len(self._train))
        self._val = random.sample(self._val, len(self._val))
        self._test = random.sample(self._test, len(self._test))

        if self._config.face_detection.enabled:
            self._face_detector = self._get_face_detector()
        else:
            self._face_detector = None
        if self._config.augmentation.enabled:
            self._augmentation_steps = self._get_augmentation_steps()
        else:
            self._augmentation_steps = []
        if self._config.feature_extraction.enabled:
            self._feature_extractors = self._get_feature_extractors()
        else:
            self._feature_extractors = []

    def _get_loaders(self) -> list[DatasetLoader]:
        loaders: list[DatasetLoader] = []
        loader_errors = []
        for dataset in self._config.datasets:
            try:
                # HACK: Type[T] where T is abstract is disallowed, see find definition
                # for more details
                loader: DatasetLoader = Registrable.find(DatasetLoader, dataset.name)
                loaders.append(loader)
            except ValueError:
                loader_errors.append(dataset.name)

        # Report the datasets without loaders all at once
        if len(loader_errors) > 0:
            raise ValueError(
                "Could not find a loader for the following datasets: "
                f"{', '.join(loader_errors)}"
            )
        return loaders

    def _get_face_detector(self) -> FaceDetector:
        return Registrable.find(
            FaceDetector,
            self._config.face_detection.algorithm.name,
            _config=self._config,
            **self._config.face_detection.algorithm.args,
        )

    def _get_augmentation_steps(self) -> list[AugmentationStep]:
        return [
            Registrable.find(
                AugmentationStep,
                step.uses,
                probability=step.probability,
                **step.args,
            )
            for step in self._config.augmentation.steps
        ]

    def _get_feature_extractors(self) -> list[FeatureExtractor]:
        return [
            Registrable.find(
                FeatureExtractor,
                extractor.name,
                _config=self._config,
                **extractor.args,
            )
            for extractor in self._config.feature_extraction.algorithms
        ]

    def get_train_dataset(self) -> Dataset:
        return Dataset(
            self._train,
            self._face_detector,
            self._augmentation_steps,
            self._feature_extractors,
        )

    def get_val_dataset(self) -> Dataset:
        return Dataset(self._val, self._face_detector, [], self._feature_extractors)

    def get_test_dataset(self) -> Dataset:
        return Dataset(self._test, self._face_detector, [], self._feature_extractors)
