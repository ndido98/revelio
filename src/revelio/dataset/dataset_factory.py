from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypeVar

from revelio.dataset.descriptors_list import DatasetElementDescriptorsList
from revelio.utils.random import shuffled
from revelio.utils.rounding import round_half_up

from .dataset import Dataset
from .element import DatasetElementDescriptor, ElementClass
from .loaders.loader import DatasetLoader

if TYPE_CHECKING:
    from revelio.augmentation.step import AugmentationStep
    from revelio.config import Config
    from revelio.face_detection.detector import FaceDetector
    from revelio.feature_extraction.extractor import FeatureExtractor
    from revelio.preprocessing.step import PreprocessingStep

__all__ = ("DatasetFactory",)

log = logging.getLogger(__name__)

T = TypeVar("T")


def _split_dataset(
    dataset: list[T],
    split: float,
) -> tuple[list[T], list[T]]:
    shuffled_dataset = shuffled(dataset)
    split_index = round_half_up(len(shuffled_dataset) * split)
    return shuffled_dataset[:split_index], shuffled_dataset[split_index:]


def _split_train_val_test(
    dataset: list[T],
    train_percentage: float,
    val_percentage: float,
    test_percentage: float,
) -> tuple[list[T], list[T], list[T]]:
    if train_percentage == 0.0 and val_percentage == 0.0 and test_percentage == 0.0:
        return [], [], []
    if train_percentage + val_percentage + test_percentage > 1:
        raise ValueError("Train, val and test percentages must sum to 1 or less")
    if train_percentage == 0.0 and val_percentage == 0.0:
        test, _ = _split_dataset(dataset, test_percentage)
        return [], [], test
    if train_percentage == 0.0 and test_percentage == 0.0:
        val, _ = _split_dataset(dataset, val_percentage)
        return [], val, []
    if val_percentage == 0.0 and test_percentage == 0.0:
        train, _ = _split_dataset(dataset, train_percentage)
        return train, [], []
    if train_percentage == 0.0:
        test, rest = _split_dataset(dataset, test_percentage)
        val, _ = _split_dataset(rest, val_percentage / (1 - test_percentage))
        print(val_percentage, val_percentage / (1 - test_percentage), len(val))
        return [], val, test
    if val_percentage == 0.0:
        test, rest = _split_dataset(dataset, test_percentage)
        train, _ = _split_dataset(rest, train_percentage / (1 - test_percentage))
        return train, [], test
    if test_percentage == 0.0:
        val, rest = _split_dataset(dataset, val_percentage)
        train, _ = _split_dataset(rest, train_percentage / (1 - val_percentage))
        return train, val, []
    test, rest = _split_dataset(dataset, test_percentage)
    val, rest = _split_dataset(rest, val_percentage / (1 - test_percentage))
    train, _ = _split_dataset(
        rest, train_percentage / (1 - test_percentage - val_percentage)
    )
    return train, val, test


class DatasetFactory:

    _train: list[DatasetElementDescriptor] = []
    _val: list[DatasetElementDescriptor] = []
    _test: list[DatasetElementDescriptor] = []

    _face_detector: FaceDetector | None
    _augmentation_steps: list[AugmentationStep]
    _feature_extractors: list[FeatureExtractor]
    _preprocessing_steps: list[PreprocessingStep]

    def __init__(self, config: Config) -> None:
        self._config = config
        loaders = self._get_loaders()
        log.debug("Found %d loaders", len(loaders))
        # Merge the datasets with their respective train, val and test percentages
        current_x_count: int | None = None
        for dataset, loader in zip(config.datasets, loaders):
            dataset_xy = loader.load(dataset.path)
            for elem in dataset_xy:
                if current_x_count is None:
                    current_x_count = len(elem.x)
                elif current_x_count != len(elem.x):
                    raise ValueError(
                        "The number of images in the dataset is not consistent "
                        f"(expected {current_x_count}, got {len(elem.x)})"
                    )
                elem._dataset_name = dataset.name
                elem._root_path = dataset.path
            bona_fide_xy = [e for e in dataset_xy if e.y == ElementClass.BONA_FIDE]
            morphed_xy = [e for e in dataset_xy if e.y == ElementClass.MORPHED]

            log.debug(
                "First 10 bona fide elements of dataset %s:\n%s",
                dataset.name,
                "\n".join([", ".join(str(p) for p in e.x) for e in bona_fide_xy[:10]]),
            )
            log.debug(
                "Last 10 bona fide elements of dataset %s:\n%s",
                dataset.name,
                "\n".join([", ".join(str(p) for p in e.x) for e in bona_fide_xy[-10:]]),
            )
            log.debug(
                "First 10 morphed elements of dataset %s:\n%s",
                dataset.name,
                "\n".join([", ".join(str(p) for p in e.x) for e in morphed_xy[:10]]),
            )
            log.debug(
                "Last 10 morphed elements of dataset %s:\n%s",
                dataset.name,
                "\n".join([", ".join(str(p) for p in e.x) for e in morphed_xy[-10:]]),
            )

            bona_fide_train, bona_fide_val, bona_fide_test = _split_train_val_test(
                bona_fide_xy,
                dataset.split.train,
                dataset.split.val,
                dataset.split.test,
            )
            morphed_train, morphed_val, morphed_test = _split_train_val_test(
                morphed_xy,
                dataset.split.train,
                dataset.split.val,
                dataset.split.test,
            )

            self._train.extend(bona_fide_train + morphed_train)
            self._val.extend(bona_fide_val + morphed_val)
            self._test.extend(bona_fide_test + morphed_test)

            print(f"Loaded dataset: {dataset.name}")
            print("\tBona fide:")
            print(f"\t\tTraining: {len(bona_fide_train)}")
            print(f"\t\tValidation: {len(bona_fide_val)}")
            print(f"\t\tTest: {len(bona_fide_test)}")
            print("\tMorphed:")
            print(f"\t\tTraining: {len(morphed_train)}")
            print(f"\t\tValidation: {len(morphed_val)}")
            print(f"\t\tTest: {len(morphed_test)}")
        # Shuffle the three complete datasets
        self._train = shuffled(self._train)
        self._val = shuffled(self._val)
        self._test = shuffled(self._test)

        # Print some stats
        bf_train_size = sum(1 for e in self._train if e.y == ElementClass.BONA_FIDE)
        mr_train_size = sum(1 for e in self._train if e.y == ElementClass.MORPHED)
        bf_val_size = sum(1 for e in self._val if e.y == ElementClass.BONA_FIDE)
        mr_val_size = sum(1 for e in self._val if e.y == ElementClass.MORPHED)
        bf_test_size = sum(1 for e in self._test if e.y == ElementClass.BONA_FIDE)
        mr_test_size = sum(1 for e in self._test if e.y == ElementClass.MORPHED)
        print("Loaded datasets:")
        print(f"\tTotal training: {len(self._train)}")
        print(f"\tTotal validation: {len(self._val)}")
        print(f"\tTotal test: {len(self._test)}")
        print("\tBona fide:")
        print(f"\t\tTraining: {bf_train_size}")
        print(f"\t\tValidation: {bf_val_size}")
        print(f"\t\tTest: {bf_test_size}")
        print("\tMorphed:")
        print(f"\t\tTraining: {mr_train_size}")
        print(f"\t\tValidation: {mr_val_size}")
        print(f"\t\tTest: {mr_test_size}")

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
        self._train_preprocessing_steps = self._get_preprocessing_steps("train")
        self._val_preprocessing_steps = self._get_preprocessing_steps("val")
        self._test_preprocessing_steps = self._get_preprocessing_steps("test")

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
        from revelio.face_detection.detector import FaceDetector

        return FaceDetector.find(
            self._config.face_detection.algorithm.name,
            _config=self._config,
            **self._config.face_detection.algorithm.args,
        )

    def _get_augmentation_steps(self) -> list[AugmentationStep]:
        from revelio.augmentation.step import AugmentationStep

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
        from revelio.feature_extraction.extractor import FeatureExtractor

        return [
            FeatureExtractor.find(
                extractor.name,
                _config=self._config,
                **extractor.args,
            )
            for extractor in self._config.feature_extraction.algorithms
        ]

    def _get_preprocessing_steps(self, dataset: str) -> list[PreprocessingStep]:
        from revelio.preprocessing.step import PreprocessingStep

        return [
            PreprocessingStep.find(
                step.uses,
                **step.args,
            )
            for step in self._config.preprocessing.steps
            if step.datasets is None or dataset in step.datasets
        ]

    def get_train_dataset(self) -> Dataset:
        return Dataset(
            DatasetElementDescriptorsList(self._train),
            self._face_detector,
            self._augmentation_steps,
            self._feature_extractors,
            self._train_preprocessing_steps,
            True,
        )

    def get_val_dataset(self) -> Dataset:
        return Dataset(
            DatasetElementDescriptorsList(self._val),
            self._face_detector,
            [],
            self._feature_extractors,
            self._val_preprocessing_steps,
            True,
        )

    def get_test_dataset(self) -> Dataset:
        return Dataset(
            DatasetElementDescriptorsList(self._test),
            self._face_detector,
            [],
            self._feature_extractors,
            self._test_preprocessing_steps,
            False,
        )
