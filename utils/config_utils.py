from __future__ import annotations

import os
from pathlib import Path

from omegaconf import OmegaConf


def load_config(path) :
    """Load an OmegaConf YAML file with an optional recursive ``extends`` key."""
    config_path = Path(path).resolve()
    cfg = OmegaConf.load(config_path)
    base_value = OmegaConf.select(cfg, "extends", default=None)
    if not base_value:
        return cfg

    del cfg["extends"]
    base_path = Path(os.path.expandvars(os.path.expanduser(str(base_value))))
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path
    base_cfg = load_config(base_path)
    return OmegaConf.merge(base_cfg, cfg)
