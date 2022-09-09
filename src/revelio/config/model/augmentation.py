from typing import Any

from pydantic import BaseModel, root_validator

from .utils import Percentage


class AugmentationStep(BaseModel):
    uses: str
    probability: Percentage = 1.0  # type: ignore
    args: dict[str, Any] = {}


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
