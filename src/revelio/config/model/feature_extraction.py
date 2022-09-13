from typing import Any

from pydantic import BaseModel, root_validator


class FeatureExtractionAlgorithm(BaseModel):
    name: str
    args: dict[str, Any]
    weight: float = 1.0


class FeatureExtraction(BaseModel):
    enabled: bool = True
    algorithms: list[FeatureExtractionAlgorithm]

    @root_validator
    def check_algorithms_is_not_empty_if_enabled(
        cls, values: dict[str, Any]
    ) -> dict[str, Any]:
        is_enabled = values.get("enabled", True)
        algorithms = values.get("algorithms", [])
        if is_enabled and len(algorithms) == 0:
            raise ValueError(
                "At least one feature extraction algorithm must be specified"
            )
        return values
