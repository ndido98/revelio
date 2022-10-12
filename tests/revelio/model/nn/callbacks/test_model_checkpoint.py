import unittest.mock as mock
from pathlib import Path

import pytest
import torch

from revelio.model.nn.callbacks.model_checkpoint import ModelCheckpoint
from revelio.model.nn.neuralnet import NeuralNetwork


def test_metric_not_in_metrics() -> None:
    with pytest.raises(ValueError):
        checkpoint = ModelCheckpoint("model.pt", monitor="not_a_metric")
        checkpoint.after_validation_step(0, 0, {}, {"loss": torch.tensor(0.0)})


def test_direction_min_save_best_only() -> None:
    with (
        mock.patch("torch.save", return_value=None) as mock_save,
        mock.patch.object(NeuralNetwork, "get_state_dict", return_value={}) as mock_nn,
    ):
        checkpoint = ModelCheckpoint("model.pt", direction="min", save_best_only=True)
        checkpoint.model = mock_nn
        loss_values = [10.0, 5.0, 7.5, 6.5]
        for epoch, loss in enumerate(loss_values):
            checkpoint.after_validation_step(
                epoch, 0, {}, {"val_loss": torch.tensor(loss)}
            )
            checkpoint.after_validation_epoch(
                epoch, 1, {"val_loss": torch.tensor(loss)}
            )
        assert mock_save.call_count == 2


def test_direction_max_save_best_only() -> None:
    with (
        mock.patch("torch.save", return_value=None) as mock_save,
        mock.patch.object(NeuralNetwork, "get_state_dict", return_value={}) as mock_nn,
    ):
        checkpoint = ModelCheckpoint("model.pt", direction="max", save_best_only=True)
        checkpoint.model = mock_nn
        loss_values = [10.0, 5.0, 15.0, 12.5]
        for epoch, loss in enumerate(loss_values):
            checkpoint.after_validation_step(
                epoch, 0, {}, {"val_loss": torch.tensor(loss)}
            )
            checkpoint.after_validation_epoch(
                epoch, 1, {"val_loss": torch.tensor(loss)}
            )
        assert mock_save.call_count == 2


def test_file_name_formatting() -> None:
    with (
        mock.patch("torch.save", return_value=None) as mock_save,
        mock.patch.object(NeuralNetwork, "get_state_dict", return_value={}) as mock_nn,
    ):
        checkpoint = ModelCheckpoint("model_{epoch}_{val_loss}.pt")
        checkpoint.model = mock_nn
        loss_values = [10.0, 5.0, 15.0, 12.5]
        for epoch, loss in enumerate(loss_values):
            checkpoint.after_validation_step(
                epoch, 0, {}, {"val_loss": torch.tensor(loss)}
            )
            checkpoint.after_validation_epoch(
                epoch, 0, {"val_loss": torch.tensor(loss)}
            )
        assert mock_save.call_count == 4
        mock_save.assert_has_calls(
            [
                mock.call(mock_nn.get_state_dict(), Path("model_1_10.0.pt")),
                mock.call(mock_nn.get_state_dict(), Path("model_2_5.0.pt")),
                mock.call(mock_nn.get_state_dict(), Path("model_3_15.0.pt")),
                mock.call(mock_nn.get_state_dict(), Path("model_4_12.5.pt")),
            ]
        )
