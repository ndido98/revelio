import unittest.mock as mock
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from torch.utils.data import DataLoader

from revelio.dataset import Dataset, DatasetElement, ElementClass, ElementImage


def black_img(*args: Any, **kwargs: Any) -> np.ndarray:
    return np.array([[[0, 0, 0]]])


def init_fn(worker_id: int) -> None:
    mock.patch("cv2.imread", side_effect=black_img).start()


@given(workers_count=st.integers(min_value=0, max_value=4))
@settings(deadline=timedelta(seconds=10))
def test_worker_sharding_correct(workers_count: int) -> None:
    dataset_elements = [
        DatasetElement(
            x=(
                ElementImage(path=Path("/path/to/ds1/image1.jpg")),
                ElementImage(path=Path("/path/to/ds1/image2.jpg")),
            ),
            y=ElementClass.BONA_FIDE,
            dataset_root_path=Path("/path/to/ds1"),
            original_dataset="ds1",
        ),
        DatasetElement(
            x=(
                ElementImage(path=Path("/path/to/ds1/image3.jpg")),
                ElementImage(path=Path("/path/to/ds1/image4.jpg")),
            ),
            y=ElementClass.BONA_FIDE,
            dataset_root_path=Path("/path/to/ds1"),
            original_dataset="ds1",
        ),
        DatasetElement(
            x=(
                ElementImage(
                    path=Path("/path/to/ds1/image5.jpg"),
                    image=black_img(),
                ),
                ElementImage(
                    path=Path("/path/to/ds1/image6.jpg"),
                    image=black_img(),
                ),
            ),
            y=ElementClass.MORPHED,
            dataset_root_path=Path("/path/to/ds1"),
            original_dataset="ds1",
        ),
        DatasetElement(
            x=(
                ElementImage(
                    path=Path("/path/to/ds1/image7.jpg"),
                    image=black_img(),
                ),
                ElementImage(
                    path=Path("/path/to/ds1/image8.jpg"),
                    image=black_img(),
                ),
            ),
            y=ElementClass.MORPHED,
            dataset_root_path=Path("/path/to/ds1"),
            original_dataset="ds1",
        ),
    ]
    with mock.patch("cv2.imread", side_effect=black_img):
        ds = Dataset(dataset_elements, None, [], [], [])
        dl = DataLoader(
            ds,
            batch_size=1,
            num_workers=workers_count,
            worker_init_fn=init_fn,
        )
        got = list(dl)
    assert len(got) == 4
    bona_fide_count = sum(1 for e in got if e["y"] == ElementClass.BONA_FIDE.value)
    morphed_count = sum(1 for e in got if e["y"] == ElementClass.MORPHED.value)
    assert bona_fide_count == 2
    assert morphed_count == 2
