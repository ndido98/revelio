from typing import Any

import torch


def _dict_to_device(data: dict[str, Any], device: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            result[k] = _dict_to_device(v, device)
        elif isinstance(v, list):
            result[k] = [_dict_to_device(x, device) for x in v]
        else:
            raise TypeError(f"Unexpected type {type(v)} when trying to move to device")
    return result
