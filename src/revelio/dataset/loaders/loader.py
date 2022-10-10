from abc import abstractmethod
from pathlib import Path

from revelio.dataset.element import DatasetElementDescriptor
from revelio.registry.registry import Registrable

__all__ = ("DatasetLoader",)


class DatasetLoader(Registrable):
    """
    A dataset loader is responsible for loading a dataset from a given path.
    The configuration file specifies the dataset loader to use on the path given
    by the user, and its job is to correctly load the dataset from that path.

    To load a dataset element, the loader must return a list of descriptors, each
    containing the path of each image and the class of the element.

    In case of D-MAD, the convention that must be followed is that the first element
    in the tuple must be the probe image, and the second element must be the live
    capture image.

    To guarantee reproducibility of results, it is important that the list returned
    by the loader is the same for the same path, and its order must be deterministic.
    Also, please note that the glob functions in Python are not guaranteed to return
    the files in a deterministic order, so you should sort the list of files before
    returning it.
    """

    suffix = "Loader"

    @abstractmethod
    def load(self, path: Path) -> list[DatasetElementDescriptor]:
        """
        Loads a dataset from a given path.

        Args:
            path: The path to the dataset.

        Returns:
            A list of descriptors, each containing the path of each image and the
            class of the element.
        """
        raise NotImplementedError  # pragma: no cover
