from typing import Any, Literal

import torch
import torchvision.models as models

from .neuralnet import NeuralNetwork

_constructors = {
    ("base", 16): (models.vit_b_16, models.ViT_B_16_Weights.DEFAULT),
    ("base", 32): (models.vit_b_32, models.ViT_B_32_Weights.DEFAULT),
    ("large", 16): (models.vit_l_16, models.ViT_L_16_Weights.DEFAULT),
    ("large", 32): (models.vit_l_32, models.ViT_L_32_Weights.DEFAULT),
    ("huge", 14): (models.vit_h_14, models.ViT_H_14_Weights.DEFAULT),
}


class VisionTransformer(NeuralNetwork):  # pragma: no cover
    class Classifier(torch.nn.Module):
        def __init__(
            self,
            *,
            pretrained: bool = True,
            freeze: bool = True,
            variant: Literal["base", "large", "huge"] = "base",
            patch_size: int = 16,
            **kwargs: Any,
        ) -> None:
            super().__init__()
            if freeze and not pretrained:
                raise ValueError("Cannot freeze a non-pretrained model")
            try:
                function, weights = _constructors[(variant, patch_size)]
            except KeyError as e:
                raise ValueError(
                    f"There is no ViT-{variant[0].capitalize()}/{patch_size}"
                ) from e
            if pretrained:
                self.preprocess = weights.transforms()
                self.net = function(weights=weights, progress=False)
            else:
                self.preprocess = None
                self.net = function(weights=None, progress=False)
            # Replace the classification head
            self.net.heads[-1] = torch.nn.Linear(
                self.net.heads[-1].in_features,
                1,
            )
            if freeze:
                for param in self.net.parameters():
                    param.requires_grad = False
                for param in self.net.heads[-1].parameters():
                    param.requires_grad = True

        def forward(self, batch: list[dict[str, Any]]) -> torch.Tensor:
            if len(batch) != 1:
                raise ValueError(
                    "The ViT classifier is only available for single images"
                )
            img = batch[0]["image"]
            preprocessed = self.preprocess(img) if self.preprocess is not None else img
            logits: torch.Tensor = self.net(preprocessed)
            return logits

    def get_classifier(self, **kwargs: Any) -> torch.nn.Module:
        return self.Classifier(**kwargs)
