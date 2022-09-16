import unittest.mock as mock
from datetime import timedelta
from pathlib import Path
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image as ImageModule
from PIL.Image import Image
from torch.utils.data import DataLoader

from revelio.dataset import Dataset, DatasetElement, ElementClass, ElementImage


def black_img(*args: Any, **kwargs: Any) -> Image:
    return ImageModule.new("RGB", (1, 1), "black")


def init_fn(worker_id: int) -> None:
    mock.patch("PIL.Image.open", side_effect=black_img).start()


@given(workers_count=st.integers(min_value=0, max_value=8))
@settings(deadline=timedelta(seconds=10))
def test_worker_sharding_correct(workers_count: int) -> None:
    dataset_elements = [
        DatasetElement(
            x=(
                ElementImage(path=Path("/path/to/ds1/image1.jpg")),
                ElementImage(path=Path("/path/to/ds1/image2.jpg")),
            ),
            y=ElementClass.BONA_FIDE,
            original_dataset="ds1",
        ),
        DatasetElement(
            x=(
                ElementImage(path=Path("/path/to/ds1/image3.jpg")),
                ElementImage(path=Path("/path/to/ds1/image4.jpg")),
            ),
            y=ElementClass.BONA_FIDE,
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
            original_dataset="ds1",
        ),
    ]
    with mock.patch("PIL.Image.open", side_effect=black_img):
        ds = Dataset(dataset_elements, None, [], [])
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
