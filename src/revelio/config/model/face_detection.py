from typing import Any

from pydantic import BaseModel


class FaceDetectionAlgorithm(BaseModel):
    name: str
    params: dict[str, Any]


class FaceDetection(BaseModel):
    enabled: bool = True
    output_path: str
    algorithm: FaceDetectionAlgorithm
