"""
This module contains the class that defines an augmentation step, from which all
augmentation steps inherit.

A user can define their own augmentation steps by subclassing this class and
overriding the `process_element` method.
"""

import random
from abc import abstractmethod
from typing import Literal, Optional

from revelio.dataset.element import DatasetElement, ElementImage, Image, Landmarks
from revelio.registry.registry import Registrable


class AugmentationStep(Registrable):
    """
    An augmentation step is applied to a dataset element with a certain probability.
    The affected images of the dataset element can be specified with the `applies_to`
    parameter in the configuration file.
    If `applies_to` is set to "all", the augmentation step is applied to all images
    of the dataset element.
    If `applies_to` is set to a list of integers, the augmentation step is applied
    to the images with the specified indices.

    The probability parameter specifies the probability (between 0 and 1 inclusive)
    with which the augmentation step is applied to a dataset element.

    The process_element method is called for each image of the dataset element
    that is affected by the augmentation step.

    An augmentation step must implement the `process_element` method, which takes
    an image, its landmarks and returns its augmented version.

    Only dataset elements which are part of the training set are augmented;
    validation and test sets are not augmented.
    """

    def __init__(
        self, *, _applies_to: list[int] | Literal["all"], _probability: float
    ) -> None:
        self._applies_to = _applies_to
        self._probability = _probability

    @abstractmethod
    def process_element(
        self, image: Image, landmarks: Optional[Landmarks]
    ) -> tuple[Image, Optional[Landmarks]]:
        """
        Processes a single image of a dataset element,
        and returns its augmented version.

        Args:
            image: The image to be augmented.
            landmarks: The landmarks of the image to be augmented.

        Returns:
            A tuple containing the augmented image and its landmarks.
        """
        raise NotImplementedError  # pragma: no cover

    def process(self, elem: DatasetElement) -> DatasetElement:
        """
        Processes a dataset element, and returns its augmented version.

        This method should not be overridden by the user,
        but instead the `process_element` method should be implemented.

        Args:
            elem: The dataset element to be augmented.

        Returns:
            The augmented dataset element.
        """
        if random.random() < self._probability:
            new_xs = []
            for i, x in enumerate(elem.x):
                if self._applies_to == "all" or i in self._applies_to:
                    try:
                        new_img, new_landmarks = self.process_element(
                            x.image, x.landmarks
                        )
                    except Exception as e:
                        raise RuntimeError(f"Failed to process {x.path}: {e}") from e
                    if new_img is None:
                        raise RuntimeError(f"Failed to process {x.path}: image is None")
                    new_x = ElementImage(
                        path=x.path,
                        image=new_img,
                        landmarks=new_landmarks,
                        features=x.features,
                    )
                    new_xs.append(new_x)
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
