import hashlib
from abc import abstractmethod
from pathlib import Path

import numpy as np

from revelio.config import Config
from revelio.dataset.element import DatasetElement, ElementImage
from revelio.registry import Registrable
from revelio.utils.caching import ZstdCacher


class FeatureExtractor(Registrable):
    """
    A feature extractor is responsible for extracting various types of features
    from images.

    A feature extractor must implement the `process_element` method, which takes
    an image and returns the features extracted from it.

    The features are stored in a dictionary, where the key is the name of the
    feature extractor (without any "extractor" suffixes; e.g. the features produced by
    a Wavelets Extractor are accessible via the `wavelets` key) and the value contains
    the features extracted by the feature extractor.

    If the `process_element` method raises an exception, the dataset element will
    be skipped and the exception will be logged.

    The `process` method is responsible for loading the extracted features from
    the disk, if they have been already computed, or else calling `process_element`
    and saving the results.
    The user should not override this method, but instead implement `process_element`.
    """

    def __init__(self, *, _config: Config) -> None:
        self._config = _config
        self._cacher = ZstdCacher()

    def _get_features_path(self, elem: DatasetElement, x_idx: int) -> Path:
        output_path = Path(self._config.feature_extraction.output_path)
        algorithm_name = type(self).__name__.lower()
        path_hash = hashlib.shake_256(
            str(elem.x[x_idx].path.parent).encode("utf-8")
        ).hexdigest(16)
        img_path = Path(path_hash[:2]) / path_hash[2:4] / path_hash[4:]
        img_name = elem.x[x_idx].path.stem
        return output_path / algorithm_name / img_path / f"{img_name}.features.xz"

    @abstractmethod
    def process_element(self, elem: ElementImage) -> np.ndarray:
        """
        Processes a single image and returns its features.

        If the `process_element` method raises an exception, the dataset element will
        be skipped and the exception will be logged.

        Args:
            elem: The image to process.

        Returns:
            A Numpy array containing the features extracted from the image.
        """
        raise NotImplementedError  # pragma: no cover

    def process(
        self, elem: DatasetElement, force_online: bool = False
    ) -> tuple[DatasetElement, bool]:
        """
        Processes a dataset element and returns an element with the same data, but
        with the extracted features added to the `features` dictionary of each image.

        This method saves the extracted features to the disk, so
        that they can be loaded later without having to recompute them.

        This method should not be overridden by the user, but instead the
        `process_element` method should be implemented.

        Args:
            elem: The dataset element to process.
            force_online: If True, the features are always computed online, even if
                they have been already computed and saved to the disk.

        Returns:
            A tuple containing the processed dataset element and a boolean indicating
            whether the feature extraction was loaded from cache.
        """
        new_xs = []
        cached = True
        algorithm_name = type(self).__name__.lower()
        algorithm_name = algorithm_name.replace("extractor", "")
        for i, x in enumerate(elem.x):
            features_path = self._get_features_path(elem, i)
            if features_path.is_file() and not force_online:
                try:
                    features = self._cacher.load(features_path)["features"]
                except ValueError as e:
                    raise RuntimeError(
                        f"Failed to load features: {features_path}"
                    ) from e
            else:
                cached = False
                try:
                    features = self.process_element(x)
                except Exception as e:
                    raise RuntimeError(f"Failed to process {x.path}: {e}") from e
                if features is None:
                    raise RuntimeError(f"Failed to process {x.path}: returned None")
                if not force_online:
                    # We don't need to save the features if we're forced to do it online
                    # (that means we have one or more augmentation steps)
                    features_path.parent.mkdir(parents=True, exist_ok=True)
                    self._cacher.save(features_path, features=features)
            new_x = ElementImage(
                path=x.path,
                image=x.image,
                features={**x.features, algorithm_name: features},
            )
            new_xs.append(new_x)
        return (
            DatasetElement(
                dataset_root_path=elem.dataset_root_path,
                original_dataset=elem.original_dataset,
                x=tuple(new_xs),
                y=elem.y,
            ),
            cached,
        )
