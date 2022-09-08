from typing import Any

from pydantic import BaseModel

from .utils import NonEmptyList


class FeatureExtractionAlgorithm(BaseModel):
    name: str
    args: dict[str, Any]
    weight: float = 1.0


class FeatureExtraction(BaseModel):
    enabled: bool = True
    algorithms: NonEmptyList[FeatureExtractionAlgorithm]
