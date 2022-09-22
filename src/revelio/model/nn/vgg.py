from typing import Any, Literal, TypeAlias

import torch
import torchvision.models as models

from .neuralnet import NeuralNetwork

VGGSize: TypeAlias = Literal[11, 13, 16, 19]


_constructors = {
    11: (models.vgg11, models.VGG11_Weights.DEFAULT),
    13: (models.vgg13, models.VGG13_Weights.DEFAULT),
    16: (models.vgg16, models.VGG16_Weights.DEFAULT),
    19: (models.vgg19, models.VGG19_Weights.DEFAULT),
}


_bn_constructors = {
    11: (models.vgg11_bn, models.VGG11_BN_Weights.DEFAULT),
    13: (models.vgg13_bn, models.VGG13_BN_Weights.DEFAULT),
    16: (models.vgg16_bn, models.VGG16_BN_Weights.DEFAULT),
    19: (models.vgg19_bn, models.VGG19_BN_Weights.DEFAULT),
}


class VGG(NeuralNetwork):
    class Classifier(torch.nn.Module):
        def __init__(
            self,
            *,
            pretrained: bool = True,
            freeze: bool = True,
            size: VGGSize = 16,
            batch_norm: bool = False,
            **kwargs: Any,
        ) -> None:
            super().__init__()
            if freeze and not pretrained:
                raise ValueError("Cannot freeze a non-pretrained model")
            if pretrained:
                if not batch_norm:
                    function, weights = _constructors[size]
                else:
                    function, weights = _bn_constructors[size]
                self.preprocess = weights.transforms()
                self.net = function(weights=weights, progress=False)
            else:
                if not batch_norm:
                    function, _ = _constructors[size]
                else:
                    function, _ = _bn_constructors[size]
                self.preprocess = None
                self.net = function(weights=None, progress=False)
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
                    "The VGG classifier is only available for single images"
                )
            img = batch[0]["image"]
            preprocessed = self.preprocess(img) if self.preprocess is not None else img
            logits: torch.Tensor = self.net(preprocessed)
            return logits

    def get_classifier(self, **kwargs: Any) -> torch.nn.Module:
        return self.Classifier(**kwargs)
