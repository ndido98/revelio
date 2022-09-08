from pydantic import BaseModel

from revelio.config.model.utils import NonEmptyList

from .model import Augmentation, Dataset, Experiment, FaceDetection, FeatureExtraction


class Config(BaseModel):
    datasets: NonEmptyList[Dataset]
    face_detection: FaceDetection
    augmentation: Augmentation
    feature_extraction: FeatureExtraction
    experiment: Experiment
