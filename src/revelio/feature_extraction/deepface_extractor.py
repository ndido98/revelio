from typing import Any, Callable

import numpy as np
import tensorflow as tf
from deepface import DeepFace

from revelio.dataset.element import ElementImage

from .extractor import FeatureExtractor

_EXTRACTORS = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace",
]


def _make_process_element(extractor: str) -> Callable[[Any, ElementImage], np.ndarray]:
    def _(self: Any, elem: ElementImage) -> np.ndarray:
        with tf.device("/CPU:0"):
            features = DeepFace.represent(
                img_path=elem.image,
                model_name=extractor,
                detector_backend="skip",
            )
        return np.array(features, dtype=np.float32).squeeze()

    return _


for extractor in _EXTRACTORS:
    sanitized_name = extractor.replace("-", "")
    cls_name = f"{sanitized_name}Extractor"

    globals()[cls_name] = type(
        cls_name,
        (FeatureExtractor,),
        {
            "__module__": __name__,
            "process_element": _make_process_element(extractor),
        },
    )
