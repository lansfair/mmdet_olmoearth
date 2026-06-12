from __future__ import annotations

from typing import Any

import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS

from ..utils import RGB_TO_SENTINEL2_L2A, get_modality_bands


def _load_computed_norm(modality: str) -> dict[str, dict[str, float]]:
    from olmoearth_pretrain.data.normalize import load_computed_config

    return load_computed_config()[modality]


@TRANSFORMS.register_module()
class OlmoEarthNormalize(BaseTransform):
    """Apply OLMoEarth computed 2-std normalization to flattened imagery."""

    def __init__(
        self,
        modality: str,
        num_timesteps: int = 1,
        band_names: list[str] | None = None,
        std_multiplier: float = 2.0,
    ) -> None:
        self.modality = modality
        self.num_timesteps = num_timesteps
        self.band_names = band_names or list(get_modality_bands(modality))
        self.std_multiplier = std_multiplier
        self.norm_config = _load_computed_norm(modality)

    def _normalize_band(
        self,
        values: np.ndarray,
        band_name: str,
    ) -> np.ndarray:
        stats = self.norm_config[band_name]
        min_val = stats["mean"] - self.std_multiplier * stats["std"]
        max_val = stats["mean"] + self.std_multiplier * stats["std"]
        return (values - min_val) / (max_val - min_val)

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        image = results["img"].astype(np.float32, copy=True)
        expected = len(self.band_names) * self.num_timesteps
        if image.shape[-1] != expected:
            raise ValueError(
                f"Expected {expected} channels, got {image.shape[-1]}"
            )
        for band_idx, band_name in enumerate(self.band_names):
            for t in range(self.num_timesteps):
                channel_idx = band_idx * self.num_timesteps + t
                image[..., channel_idx] = self._normalize_band(
                    image[..., channel_idx],
                    band_name,
                )
        results["img"] = image
        results["olmoearth_modality"] = self.modality
        results["olmoearth_num_timesteps"] = self.num_timesteps
        results["olmoearth_band_names"] = self.band_names
        results.setdefault("present_bands", self.band_names)
        return results


@TRANSFORMS.register_module()
class RGBToOlmoEarthS2(BaseTransform):
    """Map RGB imagery into normalized Sentinel-2 slots for OLMoEarth."""

    def __init__(
        self,
        num_timesteps: int = 1,
        rgb_channel_order: str = "RGB",
        input_value_range: str = "auto",
        std_multiplier: float = 2.0,
    ) -> None:
        rgb_channel_order = rgb_channel_order.upper()
        if sorted(rgb_channel_order) != ["B", "G", "R"]:
            raise ValueError("rgb_channel_order must be a permutation of RGB")
        if input_value_range not in {"auto", "0_255", "0_1", "s2"}:
            raise ValueError(
                "input_value_range must be auto, 0_255, 0_1, or s2"
            )
        self.num_timesteps = num_timesteps
        self.rgb_channel_order = rgb_channel_order
        self.input_value_range = input_value_range
        self.band_names = list(get_modality_bands("sentinel2_l2a"))
        self.norm_config = _load_computed_norm("sentinel2_l2a")
        self.std_multiplier = std_multiplier

    def _to_s2_scale(self, image: np.ndarray) -> np.ndarray:
        if self.input_value_range == "s2":
            return image
        mode = self.input_value_range
        if mode == "auto":
            max_value = float(np.nanmax(image)) if image.size else 0.0
            mode = "0_1" if max_value <= 1.5 else "0_255"
        if mode == "0_1":
            return image * 10000.0
        if mode == "0_255":
            return image * (10000.0 / 255.0)
        return image

    def _normalize_band(
        self,
        values: np.ndarray,
        band_name: str,
    ) -> np.ndarray:
        stats = self.norm_config[band_name]
        min_val = stats["mean"] - self.std_multiplier * stats["std"]
        max_val = stats["mean"] + self.std_multiplier * stats["std"]
        return (values - min_val) / (max_val - min_val)

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        image = results["img"].astype(np.float32, copy=False)
        expected = 3 * self.num_timesteps
        if image.ndim != 3 or image.shape[-1] != expected:
            raise ValueError(
                f"Expected RGB image with {expected} channels, "
                f"got {image.shape}"
            )
        image = self._to_s2_scale(image)
        height, width = image.shape[:2]
        out = np.zeros(
            (height, width, len(self.band_names) * self.num_timesteps),
            dtype=np.float32,
        )
        channel_to_index = {
            name: idx for idx, name in enumerate(self.rgb_channel_order)
        }
        for t in range(self.num_timesteps):
            rgb_base = t * 3
            for rgb_name, s2_band in RGB_TO_SENTINEL2_L2A.items():
                rgb_idx = rgb_base + channel_to_index[rgb_name]
                band_idx = self.band_names.index(s2_band)
                out_idx = band_idx * self.num_timesteps + t
                out[..., out_idx] = self._normalize_band(
                    image[..., rgb_idx],
                    s2_band,
                )
        results["img"] = out
        results["olmoearth_modality"] = "sentinel2_l2a"
        results["olmoearth_num_timesteps"] = self.num_timesteps
        results["olmoearth_band_names"] = self.band_names
        results["present_bands"] = list(RGB_TO_SENTINEL2_L2A.values())
        results["olmoearth_rgb_adapter"] = {
            "rgb_channel_order": self.rgb_channel_order,
            "input_value_range": self.input_value_range,
            "mapped_bands": RGB_TO_SENTINEL2_L2A,
        }
        return results
