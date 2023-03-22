from typing import Any

from pydantic import BaseModel, root_validator, validator

from .utils import (
    AppliesTo,
    Percentage,
    args_cannot_start_with_underscores,
    check_applies_to_has_no_duplicates,
    convert_applies_to,
)


class AugmentationStep(BaseModel):
    uses: str
    probability: Percentage = 1.0  # type: ignore
    applies_to: AppliesTo = "all"
    args: dict[str, Any] = {}

    _args_underscores = validator("args", allow_reuse=True)(
        args_cannot_start_with_underscores
    )

    _applies_to_convert = validator("applies_to", pre=True, allow_reuse=True)(
        convert_applies_to
    )

    _applies_to_duplicates = validator("applies_to", allow_reuse=True)(
        check_applies_to_has_no_duplicates
    )


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
