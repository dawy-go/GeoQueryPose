from contextlib import nullcontext
from typing import Union

import torch


def autocast_disabled(device: Union[torch.device, str]):
    """Return a device-appropriate context that keeps operations in FP32."""
    device_type = (
        device.type
        if isinstance(device, torch.device)
        else str(device).split(":", maxsplit=1)[0]
    )
    try:
        return torch.amp.autocast(device_type=device_type, enabled=False)
    except (AttributeError, TypeError):
        if device_type == "cuda":
            from torch.cuda.amp import autocast as cuda_autocast

            return cuda_autocast(enabled=False)
        return nullcontext()
