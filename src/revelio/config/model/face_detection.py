from pathlib import Path
from typing import Any

from pydantic import BaseModel, validator

from .utils import args_cannot_start_with_underscores


class FaceDetectionAlgorithm(BaseModel):
    name: str
    args: dict[str, Any] = {}

    _args_underscores = validator("args", allow_reuse=True)(
        args_cannot_start_with_underscores
    )


class FaceDetection(BaseModel):
    enabled: bool = True
    output_path: Path
    algorithm: FaceDetectionAlgorithm
