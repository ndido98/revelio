from abc import abstractmethod
from pathlib import Path
from typing import Optional, TypeAlias

import numpy as np

from revelio.config.config import Config
from revelio.dataset.element import DatasetElement, ElementImage, Image, Landmarks
from revelio.registry.registry import Registrable

BoundingBox: TypeAlias = tuple[int, int, int, int]


class FaceDetector(Registrable):
    """
    A face detector is responsible for detecting faces in the dataset images.
    It is also responsible for cropping the images to only contain the face, and for
    detecting the facial landmarks (if the algorithm supports it).

    A face detector must implement the `process_element` method, which takes an image
    and returns the bounding box of the face and the facial landmarks, if the algorithm
    supports such extraction, or else None.

    The bounding box is a tuple of 4 integers, representing the top-left and
    bottom-right coordinates of the bounding box, while the landmarks is a NumPy array
    of variable length, where each row represents a landmark and each column represents
    the x and y integer coordinates of the landmark, with origin at the top-left corner
    of the image.
    If no landmarks can be computed from the image (e.g. because the chosen face
    detection algorithm does not support landmark extraction), the returned value for
    the landmarks should be None.

    If the `process_element` method raises an exception, the dataset element will be
    skipped and the exception will be logged.

    The `process` method is responsible for loading the bounding box and landmarks from
    the disk, if they have been already computed, or else calling `process_element`
    and saving the results.
    The user should not override this method, but instead implement `process_element`.
    """

    def __init__(self, *, _config: Config) -> None:
        self._config = _config

    def _get_meta_path(self, elem: DatasetElement, x_idx: int) -> Path:
        output_path = Path(self._config.face_detection.output_path)
        algorithm_name = type(self).__name__.lower()
        relative_img_path = elem.x[x_idx].path.relative_to(elem.dataset_root_path)
        return (
            output_path
            / algorithm_name
            / elem.original_dataset
            / relative_img_path.parent
            / f"{relative_img_path.stem}.meta.npz"
        )

    @abstractmethod
    def process_element(self, elem: Image) -> tuple[BoundingBox, Optional[Landmarks]]:
        """
        Processes a single image and returns the bounding box of the face and the
        facial landmarks, if the algorithm supports such extraction, or else None.

        The bounding box is a tuple of 4 integers, representing the top-left and
        bottom-right coordinates of the bounding box, while the landmarks is a NumPy
        array of variable length, where each row represents a landmark and each column
        represents the x and y integer coordinates of the landmark, with origin at the
        top-left corner of the image.
        If no landmarks can be computed from the image (e.g. because the chosen face
        detection algorithm does not support landmark extraction), the returned value
        for the landmarks should be None.

        If the `process_element` method raises an exception, the dataset element will
        be skipped and the exception will be logged.

        Args:
            elem: The image to process.

        Returns:
            A tuple containing the bounding box (required) and the facial landmarks
            (optional).
        """
        raise NotImplementedError  # pragma: no cover

    def process(self, elem: DatasetElement) -> tuple[DatasetElement, bool]:
        """
        Processes a dataset element and returns an element with the same data, but
        with each image cropped to only contain the face.
        Also, if the algorithm supports it, the facial landmarks are extracted and
        saved for each image of the element.

        This method saves the bounding box and the facial landmarks to the disk, so
        that they can be loaded later without having to recompute them.

        This method should not be overridden by the user, but instead the
        `process_element` method should be implemented.

        Args:
            elem: The dataset element to process.

        Returns:
            A tuple containing the processed dataset element and a boolean indicating
            whether the face detection was loaded from cache.
        """
        new_xs = []
        cached = True
        for i, x in enumerate(elem.x):
            meta_path = self._get_meta_path(elem, i)
            if meta_path.is_file():
                try:
                    meta = np.load(meta_path)
                except ValueError as e:
                    raise RuntimeError(f"Failed to load meta file: {meta_path}") from e
                landmarks = meta["landmarks"] if "landmarks" in meta else None
                if "bb" in meta:
                    # Crop the image using the precomputed bounding box
                    x1, y1, x2, y2 = meta["bb"]
                    image = x.image[y1:y2, x1:x2]
                    new_x = ElementImage(
                        path=x.path,
                        image=image,
                        landmarks=landmarks,
                    )
                    new_xs.append(new_x)
                else:
                    raise ValueError(f"No bounding box found in {meta_path}")
            else:
                cached = False
                try:
                    bb, landmarks = self.process_element(x.image)
                except Exception as e:
                    raise RuntimeError(f"Failed to process {x.path}: {e}") from e
                # Make sure that the bounding box is always inside the image
                clipped_bb = (
                    max(0, bb[0]),
                    max(0, bb[1]),
                    min(x.image.shape[1], bb[2]),
                    min(x.image.shape[0], bb[3]),
                )
                x1, y1, x2, y2 = clipped_bb
                new_x = ElementImage(
                    path=x.path,
                    image=x.image[y1:y2, x1:x2],
                    landmarks=landmarks,
                )
                # Create the meta file
                meta_path.parent.mkdir(parents=True, exist_ok=True)
                if landmarks is not None:
                    np.savez_compressed(meta_path, bb=clipped_bb, landmarks=landmarks)
                else:
                    np.savez_compressed(meta_path, bb=clipped_bb)
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
