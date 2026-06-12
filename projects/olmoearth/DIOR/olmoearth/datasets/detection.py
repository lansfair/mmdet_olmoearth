from __future__ import annotations

import os.path as osp
from typing import Any

from mmengine.fileio import load
from mmengine.utils import is_abs
from mmdet.datasets import BaseDetDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class OlmoEarthDetDataset(BaseDetDataset):
    """Detection dataset for OLMoEarth rslearn manifests.

    The manifest keeps rslearn/OLMoEarth metadata directly instead of forcing
    multi-timestep GeoTIFF samples into COCO fields.
    """

    METAINFO = {"classes": ("object",)}

    def _resolve_path(self, path: str) -> str:
        if is_abs(path):
            return path
        img_prefix = self.data_prefix.get("img", "")
        return osp.join(self.data_root, img_prefix, path)

    def _update_metainfo(self, manifest: dict[str, Any]) -> None:
        manifest_meta = manifest.get("metainfo", {})
        classes = manifest_meta.get("classes")
        if classes is not None and (
            "classes" not in self._metainfo
            or tuple(self._metainfo["classes"]) == self.METAINFO["classes"]
        ):
            self._metainfo["classes"] = tuple(classes)
        self._metainfo.setdefault(
            "dataset_type",
            manifest_meta.get("dataset_type", "olmoearth_rslearn_detection"),
        )

    def load_data_list(self) -> list[dict[str, Any]]:
        manifest = load(self.ann_file, backend_args=self.backend_args)
        if "samples" not in manifest:
            raise KeyError(
                f"OLMoEarth detection manifest {self.ann_file} must contain "
                "a 'samples' list."
            )
        self._update_metainfo(manifest)

        data_list = []
        for index, sample in enumerate(manifest["samples"]):
            img_paths = sample.get("img_paths")
            if not img_paths:
                raise KeyError(
                    f"Sample {index} in {self.ann_file} has no img_paths."
                )
            resolved_paths = [self._resolve_path(path) for path in img_paths]
            bboxes = sample.get("bboxes", [])
            labels = sample.get("labels", [])
            if len(bboxes) != len(labels):
                raise ValueError(
                    f"Sample {sample.get('sample_id', index)} has "
                    f"{len(bboxes)} boxes but {len(labels)} labels."
                )

            instances = []
            for bbox, label in zip(bboxes, labels):
                instances.append(
                    {
                        "bbox": [float(coord) for coord in bbox],
                        "bbox_label": int(label),
                        "ignore_flag": 0,
                    }
                )

            data_info = {
                "img_id": sample.get("img_id", sample.get("sample_id", index)),
                "sample_id": sample.get("sample_id", str(index)),
                "img_path": resolved_paths[0],
                "img_paths": resolved_paths,
                "height": int(sample["height"]),
                "width": int(sample["width"]),
                "instances": instances,
                "valid": int(sample.get("valid", 1)),
                "timestamps": sample.get("timestamps"),
                "present_bands": sample.get("present_bands"),
                "olmoearth_modality": sample.get("olmoearth_modality"),
                "olmoearth_num_timesteps": sample.get(
                    "olmoearth_num_timesteps"
                ),
                "olmoearth_band_names": sample.get("olmoearth_band_names"),
                "rslearn": sample.get("rslearn", {}),
            }
            data_list.append(data_info)
        return data_list

    def filter_data(self) -> list[dict[str, Any]]:
        if self.test_mode:
            return self.data_list

        filter_cfg = self.filter_cfg or {}
        filter_empty_gt = filter_cfg.get("filter_empty_gt", False)
        filter_invalid = filter_cfg.get("filter_invalid", True)
        min_size = filter_cfg.get("min_size", 0)

        valid_data_infos = []
        for data_info in self.data_list:
            if filter_invalid and not data_info.get("valid", 1):
                continue
            if filter_empty_gt and len(data_info["instances"]) == 0:
                continue
            if min(data_info["width"], data_info["height"]) < min_size:
                continue
            valid_data_infos.append(data_info)
        return valid_data_infos
