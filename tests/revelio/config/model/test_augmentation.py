import pytest
from pydantic import ValidationError

from revelio.config.model.augmentation import Augmentation, AugmentationStep


def test_check_steps_not_empty_if_enabled() -> None:
    # No steps, but augmentation is disabled; should not raise any error
    Augmentation(enabled=False, steps=[])
    # No steps, but augmentation is enabled; should raise an error
    with pytest.raises(ValidationError):
        Augmentation(enabled=True, steps=[])
    # Steps are specified, augmentation is enabled; should not raise any error
    Augmentation(enabled=True, steps=[AugmentationStep(uses="test")])
    # Steps are specified, augmentation is disabled; should not raise any error
    Augmentation(enabled=False, steps=[AugmentationStep(uses="test")])
