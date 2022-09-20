from typing import TYPE_CHECKING

import torch

from revelio.registry.registry import Registrable

if TYPE_CHECKING:
    from revelio.model.nn import NeuralNetwork


class Callback(Registrable):
    model: "NeuralNetwork"

    def before_training(self) -> None:
        pass

    def after_training(self, metrics: dict[str, torch.Tensor]) -> None:
        pass

    def before_training_epoch(self, epoch: int) -> None:
        pass

    def after_training_epoch(
        self, epoch: int, metrics: dict[str, torch.Tensor]
    ) -> None:
        pass

    def before_training_step(self, step: int) -> None:
        pass

    def after_training_step(self, step: int, metrics: dict[str, torch.Tensor]) -> None:
        pass

    def before_validation_epoch(self, epoch: int) -> None:
        pass

    def after_validation_epoch(
        self, epoch: int, metrics: dict[str, torch.Tensor]
    ) -> None:
        pass

    def before_validation_step(self, step: int) -> None:
        pass

    def after_validation_step(
        self, step: int, metrics: dict[str, torch.Tensor]
    ) -> None:
        pass
