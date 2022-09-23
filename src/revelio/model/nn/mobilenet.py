from typing import Any

import torch
import torchvision.models as models

from .neuralnet import NeuralNetwork


class MobileNet(NeuralNetwork):  # pragma: no cover
    class Classifier(torch.nn.Module):
        def __init__(
            self, *, pretrained: bool = True, freeze: bool = True, **kwargs: Any
        ) -> None:
            super().__init__()
            if freeze and not pretrained:
                raise ValueError("Cannot freeze a non-pretrained model")
            if pretrained:
                weights = models.MobileNet_V2_Weights.DEFAULT
                self.preprocess = weights.transforms()
            else:
                weights = None
                self.preprocess = None
            self.net = models.mobilenet_v2(weights=weights, progress=False)
            # Replace the classification head
            self.net.classifier[-1] = torch.nn.Linear(
                self.net.classifier[-1].in_features,
                1,
            )
            if freeze:
                for param in self.net.parameters():
                    param.requires_grad = False
                for param in self.net.classifier[-1].parameters():
                    param.requires_grad = True

        def forward(self, batch: list[dict[str, Any]]) -> torch.Tensor:
            if len(batch) != 1:
                raise ValueError(
                    "The MobileNet classifier is only available for single images"
                )
            img = batch[0]["image"]
            preprocessed = self.preprocess(img) if self.preprocess is not None else img
            logits: torch.Tensor = self.net(preprocessed)
            return logits

    def get_classifier(self, **kwargs: Any) -> torch.nn.Module:
        return self.Classifier(**kwargs)
