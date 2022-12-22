from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .cacher import Cacher


class NPZCacher(Cacher):
    def save(self, filename: Path, **data: npt.ArrayLike) -> None:
        np.savez_compressed(filename, **data)

    def load(self, filename: Path) -> dict[str, Any]:
        with np.load(filename) as data:
            return dict(data)
