import unittest.mock as mock
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch.utils.data import DataLoader

from revelio.dataset import Dataset, DatasetElementDescriptor, ElementClass


def black_img(*args: Any, **kwargs: Any) -> np.ndarray:
    return np.array([[[0, 0, 0]]], dtype=np.uint8)


def init_fn(worker_id: int) -> None:
    mock.patch("cv2.imread", side_effect=black_img).start()


@given(
    workers_count=st.integers(min_value=0, max_value=4),
    elems_count=st.integers(min_value=5, max_value=100),
    batch_size=st.integers(min_value=1, max_value=10),
)
@settings(deadline=timedelta(seconds=10))
def test_worker_sharding_correct(
    workers_count: int, elems_count: int, batch_size: int
) -> None:
    # Generate the elements
    dataset_elements = [
        DatasetElementDescriptor(x=(Path("x"), Path("y")), y=ElementClass(i % 2))
        for i in range(elems_count)
    ]
    for i, elem in enumerate(dataset_elements):
        elem._dataset_name = f"ds{i}"
        elem._root_path = Path(f"/path/to/ds{i}")
    with mock.patch("cv2.imread", side_effect=black_img):
        ds = Dataset(dataset_elements, None, [], [], [])
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=workers_count,
            worker_init_fn=init_fn,
        )
        got = list(dl)
    # Make sure we have the correct number of elements
    assert sum(batch["y"].shape[0] for batch in got) == elems_count
    bona_fide_count = sum(
        torch.sum(e["y"] == ElementClass.BONA_FIDE.value).item() for e in got
    )
    morphed_count = sum(
        torch.sum(e["y"] == ElementClass.MORPHED.value).item() for e in got
    )
    assert bona_fide_count == elems_count // 2 + (1 if elems_count % 2 == 1 else 0)
    assert morphed_count == elems_count // 2
