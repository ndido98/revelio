from typing import Any, Generic, Literal, TypeAlias, TypeVar, cast

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


AppliesTo: TypeAlias = NonEmptyList[int] | Literal["all"]


def convert_applies_to(value: str | int | NonEmptyList[str | int]) -> AppliesTo:
    def _convert_elem(elem: str | int) -> int | Literal["all"]:
        if isinstance(elem, int):
            return elem
        if elem == "all":
            return "all"
        elif elem == "probe":
            return 0
        elif elem == "live":
            return 1
        else:
            raise ValueError(f"Invalid applies_to value: {elem}")

    if isinstance(value, str) or isinstance(value, int):
        converted_elem = _convert_elem(value)
        return (
            cast(NonEmptyList[int], [converted_elem])
            if isinstance(converted_elem, int)
            else converted_elem
        )
    elif isinstance(value, list):
        converted_list = [_convert_elem(elem) for elem in value]
        # The converted list must not have "all" (the only string type possible;
        # the other ones are converted to int)
        if "all" in converted_list:
            raise ValueError(
                "Invalid applies_to value: 'all' "
                "cannot be combined with other values"
            )
        return cast(NonEmptyList[int], converted_list)


def check_applies_to_has_no_duplicates(value: AppliesTo) -> AppliesTo:
    if isinstance(value, list):
        elems: dict[int, int] = {}
        # Manually iterate over the list to have more precise error messages
        for i, elem in enumerate(value):
            if elem in elems:
                raise ValueError(
                    f"Invalid applies_to value: {elem} "
                    f"appears more than once (at index {i} and {elems[elem]})"
                )
            elems[elem] = i
    return value
