import json
from abc import abstractmethod
from pathlib import Path

import numpy as np

from revelio.config import Config
from revelio.dataset.element import DatasetElement, ElementImage
from revelio.registry import Registrable


class FeatureExtractor(Registrable):
    def __init__(self, *, config: Config) -> None:
        self._config = config
        super().__init__()

    def _get_features_path(self, elem: DatasetElement, x_idx: int) -> Path:
        output_path = Path(self._config.feature_extraction.output_path)
        algorithm_name = type(self).__name__.lower()
        return (
            output_path
            / algorithm_name
            / elem.original_dataset
            / (elem.x[x_idx].path.stem + ".features.json")
        )

    @abstractmethod
    def process_element(self, elem: ElementImage) -> np.ndarray:
        raise NotImplementedError  # pragma: no cover

    def process(
        self, elem: DatasetElement, force_online: bool = False
    ) -> DatasetElement:
        new_xs = []
        algorithm_name = type(self).__name__.lower()
        for i, x in enumerate(elem.x):
            features_path = self._get_features_path(elem, i)
            if features_path.is_file() and not force_online:
                features = np.array(json.loads(features_path.read_text()))
                new_x = ElementImage(
                    path=x.path,
                    image=x.image,
                    features={**x.features, algorithm_name: features},
                )
                new_xs.append(new_x)
            else:
                features = self.process_element(x)
                if not force_online:  # TODO: is it correct?
                    # We don't need to save the features if we're forced to do it online
                    # (that means we have one or more augmentation steps)
                    features_path.parent.mkdir(parents=True, exist_ok=True)
                    features_path.write_text(json.dumps(features.tolist()))
                new_x = ElementImage(
                    path=x.path,
                    image=x.image,
                    features={**x.features, algorithm_name: features},
                )
                new_xs.append(new_x)
        return DatasetElement(
            original_dataset=elem.original_dataset,
            x=tuple(new_xs),
            y=elem.y,
        )
