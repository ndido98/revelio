from typing import Any, Optional

from pydantic import BaseModel, DirectoryPath, FilePath, root_validator, validator

from .utils import Percentage


class Loader(BaseModel):
    name: Optional[str] = None
    args: dict[str, Any] = {}


class DatasetSplit(BaseModel):
    train: Optional[Percentage]
    val: Optional[Percentage]
    test: Optional[Percentage]

    @root_validator
    def check_sum_is_between_0_and_1(cls, values: dict[str, Any]) -> dict[str, Any]:
        train, val, test = (
            values.get("train", 0),
            values.get("val", 0),
            values.get("test", 0),
        )
        if train + val + test < 0:
            raise ValueError("Sum of train, val and test cannot be negative")
        if train + val + test > 1:
            raise ValueError("Train, val and test percentages must sum to 1 or less")
        return values


class Dataset(BaseModel):
    name: str
    loader: Optional[Loader] = None
    path: FilePath | DirectoryPath
    testing_groups: list[str] = []
    split: DatasetSplit

    @validator("testing_groups")
    def check_testing_groups_not_reserved(cls, values: list[str]) -> list[str]:
        reserved_groups = ("train", "val", "test", "all")
        if any(group in reserved_groups for group in values):
            raise ValueError(
                "Testing groups cannot be any of the following: "
                + ", ".join(reserved_groups)
            )
        return values
