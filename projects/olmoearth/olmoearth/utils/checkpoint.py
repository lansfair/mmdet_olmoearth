from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_olmoearth_model(model_config_path: str | Path) -> Any:
    """Build an OLMoEarth model from the released config.json."""

    model_config_path = Path(model_config_path)
    if not model_config_path.exists():
        raise FileNotFoundError(
            f"model_config_path does not exist: {model_config_path}"
        )

    from olmoearth_pretrain.config import Config
    from olmoearth_pretrain.model_loader import patch_legacy_encoder_config

    with open(model_config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config_dict = patch_legacy_encoder_config(config_dict)
    model_config = Config.from_dict(config_dict["model"])
    return model_config.build()
