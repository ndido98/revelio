import yaml
from pydantic import BaseModel

from .model import Augmentation, Dataset, Experiment, FaceDetection, FeatureExtraction
from .model.utils import NonEmptyList


class Config(BaseModel):
    datasets: NonEmptyList[Dataset]
    face_detection: FaceDetection
    augmentation: Augmentation
    feature_extraction: FeatureExtraction
    experiment: Experiment

    @staticmethod
    def from_string(config_string: str) -> "Config":
        return Config(**yaml.safe_load(config_string))
