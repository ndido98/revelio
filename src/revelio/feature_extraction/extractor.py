from abc import abstractmethod
from pathlib import Path

import numpy as np

from revelio.config import Config
from revelio.dataset.element import DatasetElement, ElementImage
from revelio.registry import Registrable


class FeatureExtractor(Registrable):
    def __init__(self, *, _config: Config) -> None:
        self._config = _config

    def _get_features_path(self, elem: DatasetElement, x_idx: int) -> Path:
        output_path = Path(self._config.feature_extraction.output_path)
        algorithm_name = type(self).__name__.lower()
        relative_img_path = elem.x[x_idx].path.relative_to(elem.dataset_root_path)
        return (
            output_path
            / algorithm_name
            / elem.original_dataset
            / relative_img_path.parent
            / f"{relative_img_path.stem}.features.npz"
        )

    @abstractmethod
    def process_element(self, elem: ElementImage) -> np.ndarray:
        raise NotImplementedError  # pragma: no cover

    def process(
        self, elem: DatasetElement, force_online: bool = False
    ) -> DatasetElement:
        new_xs = []
        algorithm_name = type(self).__name__.lower()
        algorithm_name = algorithm_name.replace("extractor", "")
        for i, x in enumerate(elem.x):
            features_path = self._get_features_path(elem, i)
            if features_path.is_file() and not force_online:
                try:
                    features = np.load(features_path)["features"]
                except ValueError as e:
                    raise RuntimeError(
                        f"Failed to load features: {features_path}"
                    ) from e
                new_x = ElementImage(
                    path=x.path,
                    image=x.image,
                    features={**x.features, algorithm_name: features},
                )
                new_xs.append(new_x)
            else:
                try:
                    features = self.process_element(x)
                except Exception as e:
                    raise RuntimeError(f"Failed to process {x.path}: {e}") from e
                if not force_online:  # TODO: is it correct?
                    # We don't need to save the features if we're forced to do it online
                    # (that means we have one or more augmentation steps)
                    features_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(features_path, features=features)
                new_x = ElementImage(
                    path=x.path,
                    image=x.image,
                    features={**x.features, algorithm_name: features},
                )
                new_xs.append(new_x)
        return DatasetElement(
            dataset_root_path=elem.dataset_root_path,
            original_dataset=elem.original_dataset,
            x=tuple(new_xs),
            y=elem.y,
        )
