from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, NonNegativeInt, PositiveInt


class Model(BaseModel):
    name: str
    args: dict[str, Any] = {}


class Training(BaseModel):
    enabled: bool = True
    args: dict[str, Any] = {}


class Scores(BaseModel):
    bona_fide: Path
    morphed: Path


class Experiment(BaseModel):
    seed: Optional[int] = None
    workers_count: NonNegativeInt = 0
    batch_size: PositiveInt
    model: Model
    training: Training
    scores: Scores
    metrics: list[str] = []
