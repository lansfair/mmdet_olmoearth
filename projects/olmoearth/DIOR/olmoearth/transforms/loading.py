from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


def _load_geotiff(path: str | Path) -> np.ndarray:
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "Reading OLMoEarth GeoTIFF inputs requires rasterio."
        ) from exc

    with rasterio.open(path) as src:
        array = src.read()
    if array.shape[0] == 1:
        return array[0]
    return array


def _tchw_to_hw_flat(image: np.ndarray) -> np.ndarray:
    if image.ndim != 4:
        raise ValueError(f"Expected TCHW image, got {image.shape}")
    t, c, h, w = image.shape
    return image.transpose(2, 3, 1, 0).reshape(h, w, c * t)


def _chw_to_hwc(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError(f"Expected CHW image, got {image.shape}")
    return image.transpose(1, 2, 0)


@TRANSFORMS.register_module()
class LoadOlmoEarthTifFromFile(BaseTransform):
    """Load one or more GeoTIFFs as an OLMoEarth flattened HWC image."""

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        if "img_paths" in results:
            images = []
            for path in results["img_paths"]:
                image = _load_geotiff(path)
                if image.ndim == 2:
                    image = image[None, ...]
                images.append(image)
            shape_set = {image.shape for image in images}
            if len(shape_set) != 1:
                raise ValueError(
                    "All img_paths must have the same CHW shape, got "
                    f"{sorted(shape_set)}"
                )
            image = _tchw_to_hw_flat(np.stack(images, axis=0))
        else:
            image = _load_geotiff(results["img_path"])
            if image.ndim == 2:
                image = image[None, ...]
            image = _chw_to_hwc(image)

        results["img"] = image.astype(np.float32, copy=False)
        results["img_shape"] = results["img"].shape[:2]
        results["ori_shape"] = results["img"].shape[:2]
        return results
