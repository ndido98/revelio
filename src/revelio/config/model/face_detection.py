from typing import Any

from pydantic import BaseModel, DirectoryPath


class FaceDetectionAlgorithm(BaseModel):
    name: str
    params: dict[str, Any]


class FaceDetection(BaseModel):
    enabled: bool = True
    output_path: DirectoryPath
    algorithm: FaceDetectionAlgorithm
