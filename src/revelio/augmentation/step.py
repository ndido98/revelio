import random
from abc import abstractmethod

from revelio.dataset.element import DatasetElement
from revelio.registry.registry import Registrable


class AugmentationStep(Registrable):
    def __init__(self, *, probability: float) -> None:
        self._probability = probability
        super().__init__()

    @abstractmethod
    def process_element(self, elem: DatasetElement) -> DatasetElement:
        raise NotImplementedError

    def process(self, elem: DatasetElement) -> DatasetElement:
        if random.random() < self._probability:
            return self.process_element(elem)
        else:
            return elem
