import pytest
from pydantic import ValidationError

from revelio.config.model.augmentation import Augmentation, AugmentationStep


def test_applies_to_string_conversion() -> None:
    # "all" is converted to "all"
    str_all = AugmentationStep(uses="test", applies_to="all")
    assert str_all.applies_to == "all"
    # "probe" is converted to 0
    str_probe = AugmentationStep(uses="test", applies_to="probe")
    assert str_probe.applies_to == [0]
    # "live" is converted to 1
    str_live = AugmentationStep(uses="test", applies_to="live")
    assert str_live.applies_to == [1]


def test_applies_to_invalid_string() -> None:
    # Invalid string
    with pytest.raises(ValidationError):
        AugmentationStep(uses="test", applies_to="invalid")


def test_applies_to_list_conversion() -> None:
    step = AugmentationStep(uses="test", applies_to=["probe", "live", 2])
    assert step.applies_to == [0, 1, 2]


def test_applies_to_list_conversion_with_all() -> None:
    # "all" is not allowed in a list
    with pytest.raises(ValidationError):
        AugmentationStep(uses="test", applies_to=["all", "probe", "live", 2])
    # "all" is not allowed in a list, even if it's the only element
    with pytest.raises(ValidationError):
        AugmentationStep(uses="test", applies_to=["all"])


def test_applies_to_list_conversion_with_duplicates() -> None:
    # Duplicates are not allowed in a list
    with pytest.raises(ValidationError):
        AugmentationStep(uses="test", applies_to=["probe", "probe", "live", 2])
    # Duplicates are not allowed in a list, even if they're the same value
    with pytest.raises(ValidationError):
        AugmentationStep(uses="test", applies_to=["probe", "probe", "probe", 0])
    # Duplicates are not allowed in a list, even if they're expressed as string aliases
    with pytest.raises(ValidationError):
        AugmentationStep(uses="test", applies_to=["probe", 0])


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
