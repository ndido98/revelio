from typing import Any, Generic, TypeVar

from pydantic import ConstrainedFloat, ConstrainedList


class Percentage(ConstrainedFloat):
    ge = 0
    le = 1


NonEmptyList_T = TypeVar("NonEmptyList_T")


class NonEmptyList(ConstrainedList, Generic[NonEmptyList_T]):
    min_items = 1


def args_cannot_start_with_underscores(value: dict[str, Any]) -> dict[str, Any]:
    if any(k.startswith("_") for k in value.keys()):
        raise ValueError("Argument names cannot start with underscores")
    return value
