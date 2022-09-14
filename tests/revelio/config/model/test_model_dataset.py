import os
from typing import Optional

import pytest
from pydantic import ValidationError

from revelio.config.model.dataset import Dataset, DatasetSplit


def generate_dataset(
    train: Optional[float] = None,
    val: Optional[float] = None,
    test: Optional[float] = None,
) -> Dataset:
    return Dataset(
        name="test", path=os.curdir, split=DatasetSplit(train=train, val=val, test=test)
    )


def test_check_sum_is_1() -> None:
    # All percentages are None or 0; should raise an error
    with pytest.raises(ValidationError):
        generate_dataset()
    with pytest.raises(ValidationError):
        generate_dataset(0, 0, 0)
    # Sum is 1; should not raise any error
    generate_dataset(0.5, 0.25, 0.25)
    generate_dataset(0.25, 0.5, 0.25)
    generate_dataset(0.25, 0.25, 0.5)
    # Sum is less than 1; should raise an error
    with pytest.raises(ValidationError):
        generate_dataset(0.5, 0.25, 0.24)
    # Sum is greater than 1; should raise an error
    with pytest.raises(ValidationError):
        generate_dataset(0.5, 0.25, 0.26)
