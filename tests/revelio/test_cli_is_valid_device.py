from argparse import ArgumentTypeError

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from revelio.cli import _is_valid_device


@given(st.booleans(), st.integers(min_value=0))
def test_cpu_is_always_valid(is_cuda_available: bool, cuda_device_count: int) -> None:
    assert _is_valid_device("cpu", is_cuda_available, cuda_device_count) == "cpu"


def test_cuda_is_valid_if_available() -> None:
    assert _is_valid_device("cuda", True, 1) == "cuda"


def test_cuda_is_not_valid_if_not_available() -> None:
    with pytest.raises(ArgumentTypeError):
        _is_valid_device("cuda", False, 0)


@given(
    # fmt: off
    st.tuples(st.integers(min_value=0), st.integers(min_value=1))
    .filter(lambda t: t[0] < t[1])
)
def test_cuda_is_valid_with_colon_integer(config: tuple[int, int]) -> None:
    device_index, cuda_device_count = config
    assert (
        _is_valid_device(f"cuda:{device_index}", True, cuda_device_count)
        == f"cuda:{device_index}"
    )


@given(st.characters(blacklist_categories=("Nd",)), st.integers(min_value=0))
def test_cuda_is_not_valid_with_colon_non_integer(
    device_index: str, cuda_device_count: int
) -> None:
    with pytest.raises(ArgumentTypeError):
        _is_valid_device(f"cuda:{device_index}", True, cuda_device_count)


@given(
    # fmt: off
    st.tuples(st.integers(min_value=0), st.integers(min_value=0))
    .filter(lambda t: t[0] >= t[1])
)
@example((..., 1))
def test_cuda_is_not_valid_with_colon_integer_out_of_range(
    config: tuple[int, int]
) -> None:
    device_index, cuda_device_count = config
    with pytest.raises(ArgumentTypeError):
        _is_valid_device(f"cuda:{device_index}", True, cuda_device_count)


@given(st.characters().filter(lambda s: len(s) > 0 and s[0] != ":"))
def test_cuda_is_not_valid_with_appended_string(extra: str) -> None:
    with pytest.raises(ArgumentTypeError):
        _is_valid_device(f"cuda{extra}", True, 1)


@given(st.characters().filter(lambda s: s not in ("cpu", "cuda")))
def test_invalid_device(device_name: str) -> None:
    with pytest.raises(ArgumentTypeError):
        _is_valid_device(device_name, True, 1)
