import random
from typing import Optional, TypeVar

from revelio.augmentation import AugmentationStep
from revelio.config import Config
from revelio.face_detection import FaceDetector
from revelio.feature_extraction import FeatureExtractor
from revelio.preprocessing.step import PreprocessingStep

from .dataset import Dataset
from .element import DatasetElementDescriptor, ElementClass
from .loaders.loader import DatasetLoader

__all__ = ("DatasetFactory",)


T = TypeVar("T")


def _split_dataset(
    dataset: list[T],
    split: float,
) -> tuple[list[T], list[T]]:
    shuffled = random.sample(dataset, len(dataset))
    split_index = round(len(shuffled) * split)
    return dataset[:split_index], dataset[split_index:]


def _split_train_val_test(
    dataset: list[T],
    train_percentage: float,
    val_percentage: float,
) -> tuple[list[T], list[T], list[T]]:
    train_val_percentage = train_percentage + val_percentage
    train_val, test = _split_dataset(dataset, train_val_percentage)
    train, val = _split_dataset(train_val, train_percentage / train_val_percentage)
    return train, val, test


class DatasetFactory:

    _train: list[DatasetElementDescriptor] = []
    _val: list[DatasetElementDescriptor] = []
    _test: list[DatasetElementDescriptor] = []

    _face_detector: Optional[FaceDetector]
    _augmentation_steps: list[AugmentationStep]
    _feature_extractors: list[FeatureExtractor]
    _preprocessing_steps: list[PreprocessingStep]

    def __init__(self, config: Config) -> None:
        self._config = config
        loaders = self._get_loaders()
        # Merge the datasets with their respective train, val and test percentages
        for dataset, loader in zip(config.datasets, loaders):
            dataset_xy = loader.load(dataset.path)
            for elem in dataset_xy:
                elem._dataset_name = dataset.name
                elem._root_path = dataset.path
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
        self._preprocessing_steps = self._get_preprocessing_steps()

    def _get_loaders(self) -> list[DatasetLoader]:
        loaders: list[DatasetLoader] = []
        loader_errors = []
        for dataset in self._config.datasets:
            try:
                if dataset.loader is None:
                    loader = DatasetLoader.find(dataset.name)
                elif dataset.loader.name is None:
                    loader = DatasetLoader.find(dataset.name, **dataset.loader.args)
                else:
                    loader = DatasetLoader.find(
                        dataset.loader.name,
                        add_affixes=False,
                        **dataset.loader.args,
                    )
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
        return FaceDetector.find(
            self._config.face_detection.algorithm.name,
            _config=self._config,
            **self._config.face_detection.algorithm.args,
        )

    def _get_augmentation_steps(self) -> list[AugmentationStep]:
        return [
            AugmentationStep.find(
                step.uses,
                _probability=step.probability,
                _applies_to=step.applies_to,
                **step.args,
            )
            for step in self._config.augmentation.steps
        ]

    def _get_feature_extractors(self) -> list[FeatureExtractor]:
        return [
            FeatureExtractor.find(
                extractor.name,
                _config=self._config,
                **extractor.args,
            )
            for extractor in self._config.feature_extraction.algorithms
        ]

    def _get_preprocessing_steps(self) -> list[PreprocessingStep]:
        return [
            PreprocessingStep.find(
                step.uses,
                **step.args,
            )
            for step in self._config.preprocessing.steps
        ]

    def get_train_dataset(self) -> Dataset:
        return Dataset(
            self._train,
            self._face_detector,
            self._augmentation_steps,
            self._feature_extractors,
            self._preprocessing_steps,
        )

    def get_val_dataset(self) -> Dataset:
        return Dataset(
            self._val,
            self._face_detector,
            [],
            self._feature_extractors,
            self._preprocessing_steps,
        )

    def get_test_dataset(self) -> Dataset:
        return Dataset(
            self._test,
            self._face_detector,
            [],
            self._feature_extractors,
            self._preprocessing_steps,
        )
