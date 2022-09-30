from typing import Any, Literal, TypeAlias

import torch
import torchvision.models as models

from .neuralnet import NeuralNetwork

ResNetSize: TypeAlias = Literal[18, 34, 50, 101, 152]


_constructors = {
    18: (models.resnet18, models.ResNet18_Weights.DEFAULT),
    34: (models.resnet34, models.ResNet34_Weights.DEFAULT),
    50: (models.resnet50, models.ResNet50_Weights.DEFAULT),
    101: (models.resnet101, models.ResNet101_Weights.DEFAULT),
    152: (models.resnet152, models.ResNet152_Weights.DEFAULT),
}


class ResNet(NeuralNetwork):  # pragma: no cover
    class Classifier(torch.nn.Module):
        def __init__(
            self,
            *,
            pretrained: bool = True,
            freeze: bool = True,
            size: ResNetSize = 18,
            **kwargs: Any,
        ) -> None:
            super().__init__()
            if freeze and not pretrained:
                raise ValueError("Cannot freeze a non-pretrained model")
            try:
                function, weights = _constructors[size]
            except KeyError as e:
                raise ValueError(f"There is no ResNet{size}") from e
            if pretrained:
                self.preprocess = weights.transforms()
                self.net = function(weights=weights, progress=False)
            else:
                self.preprocess = None
                self.net = function(weights=None, progress=False)
            # Replace the classification head
            self.net.fc = torch.nn.Linear(self.net.fc.in_features, 1)
            if freeze:
                for param in self.net.parameters():
                    param.requires_grad = False
                for param in self.net.fc.parameters():
                    param.requires_grad = True

        def forward(self, batch: list[dict[str, Any]]) -> torch.Tensor:
            if len(batch) != 1:
                raise ValueError(
                    "The ResNet classifier is only available for single images"
                )
            img = batch[0]["image"]
            preprocessed = self.preprocess(img) if self.preprocess is not None else img
            logits: torch.Tensor = self.net(preprocessed)
            return logits

    def get_classifier(self, **kwargs: Any) -> torch.nn.Module:
        return self.Classifier(**kwargs)
