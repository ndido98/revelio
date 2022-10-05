from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, FilePath, NonNegativeInt, PositiveInt, validator

from .utils import args_cannot_start_with_underscores


class Model(BaseModel):
    name: str
    checkpoint: Optional[FilePath]
    args: dict[str, Any] = {}

    _args_underscores = validator("args", allow_reuse=True)(
        args_cannot_start_with_underscores
    )


class Training(BaseModel):
    enabled: bool = True
    args: dict[str, Any] = {}

    _args_underscores = validator("args", allow_reuse=True)(
        args_cannot_start_with_underscores
    )


class Scores(BaseModel):
    bona_fide: Path
    morphed: Path

    @validator("bona_fide", "morphed", pre=True)
    def parse_file_path(cls, v: Path) -> Path:
        parsed = str(v).format(
            now=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            timestamp=datetime.now().timestamp(),
            today=datetime.now().strftime("%Y-%m-%d"),
        )
        return Path(parsed)


class Metric(BaseModel):
    name: str
    args: dict[str, Any] = {}

    _args_underscores = validator("args", allow_reuse=True)(
        args_cannot_start_with_underscores
    )


class Experiment(BaseModel):
    workers_count: NonNegativeInt = 0
    batch_size: PositiveInt
    model: Model
    training: Training
    scores: Scores
    metrics: list[Metric] = []
