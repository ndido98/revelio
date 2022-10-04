from typing import Any, Literal, TypeAlias, cast

from pydantic import BaseModel, root_validator, validator

from .utils import NonEmptyList, Percentage, args_cannot_start_with_underscores

AppliesTo: TypeAlias = NonEmptyList[int] | Literal["all"]


class AugmentationStep(BaseModel):
    uses: str
    probability: Percentage = 1.0  # type: ignore
    applies_to: AppliesTo = "all"
    args: dict[str, Any] = {}

    _args_underscores = validator("args", allow_reuse=True)(
        args_cannot_start_with_underscores
    )

    @validator("applies_to", pre=True)
    def convert_applies_to(cls, v: str | int | NonEmptyList[str | int]) -> AppliesTo:
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

        if isinstance(v, str) or isinstance(v, int):
            converted_elem = _convert_elem(v)
            return (
                cast(NonEmptyList[int], [converted_elem])
                if isinstance(converted_elem, int)
                else converted_elem
            )
        elif isinstance(v, list):
            converted_list = [_convert_elem(elem) for elem in v]
            # The converted list must not have "all" (the only string type possible;
            # the other ones are converted to int)
            if "all" in converted_list:
                raise ValueError(
                    "Invalid applies_to value: 'all' "
                    "cannot be combined with other values"
                )
            return cast(NonEmptyList[int], converted_list)

    @validator("applies_to")
    def check_no_duplicates(cls, v: AppliesTo) -> AppliesTo:
        if isinstance(v, list):
            elems: dict[int, int] = {}
            # Manually iterate over the list to have more precise error messages
            for i, elem in enumerate(v):
                if elem in elems:
                    raise ValueError(
                        f"Invalid applies_to value: {elem} "
                        f"appears more than once (at index {i} and {elems[elem]})"
                    )
                elems[elem] = i
        return v


class Augmentation(BaseModel):
    enabled: bool = True
    steps: list[AugmentationStep]

    @root_validator
    def check_steps_not_empty_if_enabled(cls, values: dict[str, Any]) -> dict[str, Any]:
        is_enabled = values.get("enabled", True)
        steps = values.get("steps", [])
        if is_enabled and len(steps) == 0:
            raise ValueError("At least one augmentation step must be specified")
        return values
