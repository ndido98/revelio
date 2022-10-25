from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union, overload

import numpy as np

from .element import DatasetElementDescriptor, ElementClass

__all__ = ("DatasetElementDescriptorsList",)


def _encoded_len(s: str) -> int:
    return len(s.encode("utf-8"))


class DatasetElementDescriptorsList(Sequence[DatasetElementDescriptor]):
    def __init__(self, elems: list[DatasetElementDescriptor] | np.ndarray) -> None:
        if isinstance(elems, list):
            self._from_list(elems)
        elif isinstance(elems, np.ndarray):
            self._from_storage(elems)
        else:
            raise TypeError("elems must be a list or Numpy array.")

    def _from_list(self, elems: list[DatasetElementDescriptor]) -> None:
        # Make sure that the descriptors all have the same number of images
        current_x_count: Optional[int] = None
        for elem in elems:
            if current_x_count is None:
                current_x_count = len(elem.x)
            elif current_x_count != len(elem.x):
                raise ValueError("All descriptors must have the same number of images.")
        assert current_x_count is not None
        self._images_per_descriptor: int = current_x_count
        # Convert the list to Numpy arrays to avoid copy-on-access overhead when
        # using multiple workers
        longest_path_length: int = 0
        longest_dataset_root: int = 0
        longest_dataset_name: int = 0
        for elem in elems:
            for x in elem.x:
                longest_path_length = max(longest_path_length, _encoded_len(str(x)))
            longest_dataset_root = max(
                longest_dataset_root, _encoded_len(str(elem._root_path))
            )
            longest_dataset_name = max(
                longest_dataset_name, _encoded_len(str(elem._dataset_name))
            )
        if self._images_per_descriptor == 1:
            dt = np.dtype(
                [
                    ("x", f"|S{longest_path_length}"),
                    ("y", "u1"),
                    ("_root_path", f"|S{longest_dataset_root}"),
                    ("_dataset_name", f"|S{longest_dataset_name}"),
                ]
            )
            self._storage = np.array(
                [
                    (
                        str(elem.x[0]),
                        elem.y.value,
                        str(elem._root_path),
                        elem._dataset_name,
                    )
                    for elem in elems
                ],
                dtype=dt,
            )
        else:
            dt = np.dtype(
                [
                    ("x", f"|S{longest_path_length}", self._images_per_descriptor),
                    ("y", "u1"),
                    ("_root_path", f"|S{longest_dataset_root}"),
                    ("_dataset_name", f"|S{longest_dataset_name}"),
                ]
            )
            self._storage = np.array(
                [
                    (
                        tuple(str(x) for x in elem.x),
                        elem.y.value,
                        str(elem._root_path),
                        elem._dataset_name,
                    )
                    for elem in elems
                ],
                dtype=dt,
            )

    def _from_storage(self, elems: np.ndarray) -> None:
        # Make sure that the array is structured properly
        dt = elems.dtype
        if set(dt.names) != {"x", "y", "_root_path", "_dataset_name"}:
            raise TypeError(f"Unexpected dtype for elems: {dt}")
        if dt["x"].subdtype is not None:
            if not dt["x"].subdtype[0].kind.startswith("S"):
                raise TypeError(f"Unexpected subdtype for x: {dt}")
        elif not dt["x"].kind.startswith("S"):
            raise TypeError(f"Unexpected type for x: {dt}")
        if dt["y"].kind != "u" or dt["y"].itemsize != 1:
            raise TypeError(f"Unexpected dtype/size for y: {dt}")
        if not dt["_root_path"].kind.startswith("S"):
            raise TypeError(f"Unexpected dtype for _root_path: {dt}")
        if not dt["_dataset_name"].kind.startswith("S"):
            raise TypeError(f"Unexpected dtype for _dataset_name: {dt}")
        self._storage = elems
        self._images_per_descriptor = dt["x"].shape[0] if dt["x"].shape != () else 1

    def __len__(self) -> int:
        return len(self._storage)

    @overload
    def __getitem__(self, index: int) -> DatasetElementDescriptor:
        ...

    @overload
    def __getitem__(self, index: slice) -> "DatasetElementDescriptorsList":
        ...

    def __getitem__(
        self, index: int | slice
    ) -> Union[DatasetElementDescriptor, "DatasetElementDescriptorsList"]:
        if isinstance(index, int):
            elem = self._storage[index]
            if self._images_per_descriptor > 1:
                descriptor = DatasetElementDescriptor(
                    x=tuple(Path(x.decode("utf-8")) for x in elem["x"]),
                    y=ElementClass(elem["y"]),
                )
            else:
                descriptor = DatasetElementDescriptor(
                    x=(Path(elem["x"].decode("utf-8")),),
                    y=ElementClass(elem["y"]),
                )
            descriptor._root_path = Path(elem["_root_path"].decode("utf-8"))
            descriptor._dataset_name = elem["_dataset_name"].decode("utf-8")
            return descriptor
        elif isinstance(index, slice):
            sliced = self._storage[index]
            return DatasetElementDescriptorsList(sliced)
        else:
            raise TypeError("Index must be an int or slice.")

    def shuffled(self) -> "DatasetElementDescriptorsList":
        indices = np.arange(len(self))
        np.random.shuffle(indices)
        return DatasetElementDescriptorsList(self._storage[indices])
