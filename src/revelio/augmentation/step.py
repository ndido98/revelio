import random
from abc import abstractmethod
from typing import Literal

from revelio.dataset.element import DatasetElement, ElementImage
from revelio.registry.registry import Registrable


class AugmentationStep(Registrable):
    def __init__(
        self, *, _applies_to: list[int] | Literal["all"], _probability: float
    ) -> None:
        self._applies_to = _applies_to
        self._probability = _probability

    @abstractmethod
    def process_element(self, elem: ElementImage) -> ElementImage:
        raise NotImplementedError  # pragma: no cover

    def process(self, elem: DatasetElement) -> DatasetElement:
        if random.random() < self._probability:
            new_xs = []
            for i, x in enumerate(elem.x):
                if self._applies_to == "all" or i in self._applies_to:
                    new_xs.append(self.process_element(x))
                else:
                    new_xs.append(x)
            return DatasetElement(
                dataset_root_path=elem.dataset_root_path,
                original_dataset=elem.original_dataset,
                x=tuple(new_xs),
                y=elem.y,
            )
        else:
            return elem
