from typing import Any

from pydantic import BaseModel, validator

from .utils import args_cannot_start_with_underscores


class PreprocessingStep(BaseModel):
    uses: str
    args: dict[str, Any] = {}

    _args_underscores = validator("args", allow_reuse=True)(
        args_cannot_start_with_underscores
    )


class Preprocessing(BaseModel):
    steps: list[PreprocessingStep]
