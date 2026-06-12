from __future__ import annotations

from typing import Any

import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RGBToCopernicusFM(BaseTransform):
    """Prepare ordinary RGB imagery for Copernicus-FM spectral input."""

    def __init__(
        self,
        rgb_channel_order: str = "RGB",
        input_value_range: str = "auto",
        lon: float = np.nan,
        lat: float = np.nan,
        time: float = np.nan,
        patch_area: float = np.nan,
    ) -> None:
        rgb_channel_order = rgb_channel_order.upper()
        if sorted(rgb_channel_order) != ["B", "G", "R"]:
            raise ValueError("rgb_channel_order must be a permutation of RGB")
        if input_value_range not in {"auto", "0_255", "0_1"}:
            raise ValueError("input_value_range must be auto, 0_255, or 0_1")
        self.rgb_channel_order = rgb_channel_order
        self.input_value_range = input_value_range
        self.meta = np.array([lon, lat, time, patch_area], dtype=np.float32)

    def _to_unit_scale(self, image: np.ndarray) -> np.ndarray:
        mode = self.input_value_range
        if mode == "auto":
            max_value = float(np.nanmax(image)) if image.size else 0.0
            mode = "0_1" if max_value <= 1.5 else "0_255"
        if mode == "0_255":
            return image / 255.0
        return image

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        image = results["img"].astype(np.float32, copy=False)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected 3-channel RGB image, got {image.shape}")
        image = self._to_unit_scale(image)
        channel_to_index = {
            name: idx for idx, name in enumerate(self.rgb_channel_order)
        }
        results["img"] = np.stack(
            [
                image[..., channel_to_index["R"]],
                image[..., channel_to_index["G"]],
                image[..., channel_to_index["B"]],
            ],
            axis=-1,
        ).astype(np.float32, copy=False)
        results["copernicus_meta"] = self.meta.copy()
        results["copernicus_rgb_adapter"] = {
            "rgb_channel_order": self.rgb_channel_order,
            "input_value_range": self.input_value_range,
            "band_order": ["R", "G", "B"],
        }
        return results
