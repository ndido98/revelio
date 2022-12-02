from typing import Any

import facenet_pytorch as fnet
import torch

from .neuralnet import NeuralNetwork


class FeatureInceptionResnet(NeuralNetwork):
    class BasicConv2d(torch.nn.Module):  # Taken from torchvision source
        def __init__(
            self,
            in_planes: int,
            out_planes: int,
            kernel_size: int | tuple[int, int],
            stride: int,
            padding: int = 0,
        ):
            super().__init__()
            self.conv = torch.nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )  # verify bias false
            self.bn = torch.nn.BatchNorm2d(
                out_planes,
                eps=0.001,  # value found in tensorflow
                momentum=0.1,  # default pytorch value
                affine=True,
            )
            self.relu = torch.nn.ReLU(inplace=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            return x

    class Classifier(torch.nn.Module):
        def __init__(
            self,
            *,
            feature_name: str,
            input_depth: int,
            pretrained: bool = True,
            dropout_prob: float = 0.6,
            **kwargs: Any,
        ):
            super().__init__()
            self.net = fnet.InceptionResnetV1(
                pretrained="vggface2" if pretrained else None,
                classify=True,
                num_classes=1,
                dropout_prob=dropout_prob,
            )
            self.net.conv2d_1a = FeatureInceptionResnet.BasicConv2d(
                input_depth, 32, kernel_size=3, stride=2
            )
            self._feature_name = feature_name

        def forward(self, batch: list[dict[str, Any]]) -> torch.Tensor:
            if len(batch) != 1:
                raise ValueError(
                    "The Feature Inception-ResNetV1 classifier is only available "
                    "for single images"
                )
            img = batch[0]["features"][self._feature_name]
            logits: torch.Tensor = self.net(img)
            return logits

    def get_classifier(self, **kwargs: Any) -> torch.nn.Module:
        return self.Classifier(**kwargs)
