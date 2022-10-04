from typing import Any, Optional

from pydantic import BaseModel, DirectoryPath, FilePath, root_validator

from .utils import Percentage


class Loader(BaseModel):
    name: Optional[str] = None
    args: dict[str, Any] = {}


class DatasetSplit(BaseModel):
    train: Optional[Percentage]
    val: Optional[Percentage]
    test: Optional[Percentage]

    @root_validator
    def check_sum_is_1(cls, values: dict[str, Any]) -> dict[str, Any]:
        train, val, test = (
            values.get("train", 0),
            values.get("val", 0),
            values.get("test", 0),
        )
        if train + val + test != 1:
            raise ValueError("Train, val and test percentages must sum to 1")
        return values


class Dataset(BaseModel):
    name: str
    loader: Optional[Loader] = None
    path: FilePath | DirectoryPath
    split: DatasetSplit
