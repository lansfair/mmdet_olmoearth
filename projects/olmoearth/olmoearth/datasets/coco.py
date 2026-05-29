from __future__ import annotations

import os.path as osp
from typing import List, Union

from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class OlmoEarthCocoDataset(CocoDataset):
    """COCO bbox dataset that preserves OLMoEarth image metadata."""

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        data_info = super().parse_data_info(raw_data_info)
        img_info = raw_data_info["raw_img_info"]
        if not isinstance(data_info, dict):
            return data_info
        for key in (
            "img_paths",
            "timestamps",
            "present_bands",
            "olmoearth_modality",
            "olmoearth_num_timesteps",
            "olmoearth_band_names",
            "rslearn",
        ):
            if key not in img_info:
                continue
            value = img_info[key]
            if key == "img_paths":
                value = [
                    path
                    if osp.isabs(path)
                    else osp.join(self.data_prefix.get("img", ""), path)
                    for path in value
                ]
            data_info[key] = value
        return data_info
