from typing import Generic, TypeVar

from pydantic import ConstrainedFloat, ConstrainedList


class Percentage(ConstrainedFloat):
    ge = 0
    le = 1


NonEmptyList_T = TypeVar("NonEmptyList_T")


class NonEmptyList(ConstrainedList, Generic[NonEmptyList_T]):
    min_items = 1
