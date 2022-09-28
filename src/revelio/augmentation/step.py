import random
from abc import abstractmethod

from revelio.dataset.element import DatasetElement, ElementImage
from revelio.registry.registry import Registrable


class AugmentationStep(Registrable):
    def __init__(self, *, probability: float) -> None:
        self._probability = probability
        super().__init__()

    @abstractmethod
    def process_element(self, elem: ElementImage) -> ElementImage:
        raise NotImplementedError  # pragma: no cover

    def process(self, elem: DatasetElement) -> DatasetElement:
        if random.random() < self._probability:
            new_xs = [self.process_element(x) for x in elem.x]
            return DatasetElement(
                dataset_root_path=elem.dataset_root_path,
                original_dataset=elem.original_dataset,
                x=tuple(new_xs),
                y=elem.y,
            )
        else:
            return elem
