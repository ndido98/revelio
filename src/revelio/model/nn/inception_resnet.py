from typing import Any

import facenet_pytorch as fnet
import torch

from .neuralnet import NeuralNetwork


class InceptionResNetV1(NeuralNetwork):  # pragma: no cover
    class Classifier(torch.nn.Module):
        def __init__(
            self,
            *,
            pretrained: bool = True,
            freeze: bool = True,
            dropout_prob: float = 0.6,
            **kwargs: Any,
        ) -> None:
            super().__init__()
            if freeze and not pretrained:
                raise ValueError("Cannot freeze a non-pretrained model")
            self.net = fnet.InceptionResnetV1(
                pretrained="vggface2" if pretrained else None,
                classify=True,
                num_classes=1,
                dropout_prob=dropout_prob,
            )
            # Freeze everything but the classification head
            if freeze:
                for param in self.net.parameters():
                    param.requires_grad = False
                for param in self.net.logits.parameters():
                    param.requires_grad = True

        def forward(self, batch: list[dict[str, Any]]) -> torch.Tensor:
            if len(batch) != 1:
                raise ValueError(
                    "The Inception-ResNetV1 classifier is only available "
                    "for single images"
                )
            img = batch[0]["image"]
            logits: torch.Tensor = self.net(img)
            return logits

    def get_classifier(self, **kwargs: Any) -> torch.nn.Module:
        return self.Classifier(**kwargs)
