from typing import Any

import numpy as np
import numpy.typing as npt

from .model import Model


class RandomGuess(Model):
    def fit(self) -> None:
        pass

    def predict(self, batch: dict[str, Any]) -> npt.NDArray[np.float32]:
        pred = np.random.rand(len(batch["y"])).astype(np.float32)
        return pred
