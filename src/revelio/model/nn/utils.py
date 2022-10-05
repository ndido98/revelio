from typing import Any

import torch


def _dict_to_device(data: dict[str, Any], device: str) -> dict[str, Any]:
    def _to_device(elem: Any) -> Any:
        if isinstance(elem, torch.Tensor):
            return elem.to(device)
        elif isinstance(elem, dict):
            return _dict_to_device(elem, device)
        elif isinstance(elem, list):
            return [_to_device(e) for e in elem]
        elif isinstance(elem, str):
            return elem
        else:
            raise TypeError(
                f"Unexpected type {type(elem)} when trying to move to device"
            )

    result: dict[str, Any] = {}
    for k, v in data.items():
        result[k] = _to_device(v)
    return result
