from abc import abstractmethod

from revelio.dataset.element import DatasetElement, ElementImage
from revelio.registry.registry import Registrable


class PreprocessingStep(Registrable):
    @abstractmethod
    def process_element(self, elem: ElementImage) -> ElementImage:
        raise NotImplementedError  # pragma: no cover

    def process(self, elem: DatasetElement) -> DatasetElement:
        new_xs = [self.process_element(x) for x in elem.x]
        return DatasetElement(
            dataset_root_path=elem.dataset_root_path,
            original_dataset=elem.original_dataset,
            x=tuple(new_xs),
            y=elem.y,
        )
