import random
from typing import Optional

from revelio.config import Config
from revelio.face_detection.detector import FaceDetector, _find_face_detector

from . import Dataset, DatasetElement, ElementClass
from .loaders.loader import DatasetLoader, _find_loader


def _split_dataset(
    dataset: list[DatasetElement],
    split: float,
) -> tuple[list[DatasetElement], list[DatasetElement]]:
    shuffled = random.sample(dataset, len(dataset))
    split_index = int(len(shuffled) * split)
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

    _train: list[DatasetElement]
    _val: list[DatasetElement]
    _test: list[DatasetElement]

    _face_detector: Optional[FaceDetector]

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

    def _get_loaders(self) -> list[DatasetLoader]:
        loaders: list[DatasetLoader] = []
        loader_errors = []
        for dataset in self._config.datasets:
            try:
                loader = _find_loader(dataset.name)
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
        return _find_face_detector(
            self._config.face_detection.algorithm.name,
            self._config,
            **self._config.face_detection.algorithm.args,
        )

    def get_train_dataset(self) -> Dataset:
        return Dataset(self._train, self._face_detector, [], None)

    def get_val_dataset(self) -> Dataset:
        return Dataset(self._val, self._face_detector, [], None)

    def get_test_dataset(self) -> Dataset:
        return Dataset(self._test, self._face_detector, [], None)
