from typing import Any

from pydantic import BaseModel, DirectoryPath, NonNegativeInt, PositiveInt


class Model(BaseModel):
    name: str
    args: dict[str, Any] = {}


class Training(BaseModel):
    enabled: bool = True
    args: dict[str, Any] = {}


class Scores(BaseModel):
    bona_fide: DirectoryPath
    morphed: DirectoryPath


class Experiment(BaseModel):
    workers_count: NonNegativeInt = 0
    batch_size: PositiveInt
    model: Model
    training: Training
    scores: Scores
    metrics: list[str] = []
