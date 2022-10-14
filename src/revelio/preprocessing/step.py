from abc import abstractmethod

from revelio.dataset.element import DatasetElement, ElementImage
from revelio.registry.registry import Registrable


class PreprocessingStep(Registrable):
    """
    A preprocessing step is applied to all dataset elements before they are passed
    to the model, and can be used to perform any preprocessing that is not part of
    the model itself (e.g. resizing, cropping, normalization, etc.).

    The `process_element` method is called for each image of the dataset element.

    A preprocessing step must implement the `process_element` method, which takes
    an image and returns its preprocessed version.
    """

    @abstractmethod
    def process_element(self, elem: ElementImage) -> ElementImage:
        """
        Processes a single image of a dataset element,
        and returns its preprocessed version.

        Args:
            elem: The image to be preprocessed.

        Returns:
            The preprocessed image.
        """
        raise NotImplementedError  # pragma: no cover

    def process(self, elem: DatasetElement) -> DatasetElement:
        """
        Processes a dataset element, and returns its preprocessed version.

        This method should not be overridden by the user,
        but instead the `process_element` method should be implemented.

        Args:
            elem: The dataset element to be preprocessed.

        Returns:
            The preprocessed dataset element.
        """
        new_xs = [self.process_element(x) for x in elem.x]
        return DatasetElement(
            dataset_root_path=elem.dataset_root_path,
            original_dataset=elem.original_dataset,
            x=tuple(new_xs),
            y=elem.y,
        )
