from pathlib import Path
from typing import Any

from pydantic import BaseModel


class FaceDetectionAlgorithm(BaseModel):
    name: str
    args: dict[str, Any]


class FaceDetection(BaseModel):
    enabled: bool = True
    output_path: Path
    algorithm: FaceDetectionAlgorithm
