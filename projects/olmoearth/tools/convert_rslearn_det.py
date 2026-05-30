from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from common import progress_iter, save_json, save_timesteps_as_geotiffs


S2_BANDS = [
    "B02",
    "B03",
    "B04",
    "B08",
    "B05",
    "B06",
    "B07",
    "B8A",
    "B11",
    "B12",
    "B01",
    "B09",
]


@dataclass(frozen=True)
class DetectionSpec:
    dataset_name: str
    classes: tuple[str, ...]
    property_name: str
    split_tags: dict[str, str]
    patch_size: int
    time_year: int


DATASET_SPECS = {
    "generic": DetectionSpec(
        dataset_name="generic",
        classes=("object",),
        property_name="category",
        split_tags={"train": "train", "val": "val", "test": "val"},
        patch_size=8,
        time_year=2024,
    )
}


def _parse_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _import_rslearn():
    from rslearn.config import DType
    from rslearn.dataset import Dataset
    from rslearn.train.dataset import DataInput, ModelDataset, SplitConfig
    from rslearn.train.tasks.detection import DetectionTask
    from rslearn.train.tasks.multi_task import MultiTask
    from upath import UPath

    return (
        DType,
        Dataset,
        DataInput,
        DetectionTask,
        ModelDataset,
        MultiTask,
        SplitConfig,
        UPath,
    )


def _to_numpy(value: Any) -> np.ndarray:
    import torch

    if hasattr(value, "image"):
        value = value.image
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.detach().cpu().numpy()


def _time_ranges(value: Any):
    return getattr(value, "timestamps", None)


def _legacy_timestamps(num_timesteps: int, year: int) -> np.ndarray:
    return np.asarray(
        [[1, month, year] for month in range(num_timesteps)],
        dtype=np.int64,
    )


def _actual_timestamps(time_ranges) -> np.ndarray | None:
    if not time_ranges:
        return None
    values = []
    for start, end in time_ranges:
        midpoint = start + (end - start) / 2
        if not isinstance(midpoint, datetime):
            return None
        values.append([midpoint.day, midpoint.month - 1, midpoint.year])
    return np.asarray(values, dtype=np.int64)


def _metadata_to_dict(metadata: Any) -> dict[str, Any]:
    if metadata is None:
        return {}
    out: dict[str, Any] = {}
    for name in (
        "window_group",
        "window_name",
        "window_bounds",
        "crop_bounds",
        "patch_idx",
        "num_patches_in_window",
        "num_crops_in_window",
        "time_range",
        "dataset_source",
    ):
        if not hasattr(metadata, name):
            continue
        value = getattr(metadata, name)
        if value is None:
            out[name] = None
        elif name == "time_range":
            out[name] = [value[0].isoformat(), value[1].isoformat()]
        elif isinstance(value, tuple):
            out[name] = list(value)
        else:
            out[name] = value
    return out


def _build_split_config(split_tag: str, split_group: str):
    SplitConfig = _import_rslearn()[6]
    return SplitConfig(groups=[split_group], tags={"split": split_tag})


def _build_rslearn_dataset(
    dataset_path: Path,
    split_tag: str,
    split_group: str,
    image_layers: list[str],
    target_layers: list[str],
    classes: list[str],
    property_name: str,
    workers: int,
    box_size: int | None,
    clip_boxes: bool,
    exclude_by_center: bool,
    read_class_id: bool,
    skip_unknown_categories: bool,
    skip_empty_examples: bool,
):
    (
        DType,
        Dataset,
        DataInput,
        DetectionTask,
        ModelDataset,
        MultiTask,
        _,
        UPath,
    ) = _import_rslearn()
    return ModelDataset(
        dataset=Dataset(path=UPath(dataset_path)),
        split_config=_build_split_config(split_tag, split_group),
        inputs={
            "sentinel2_l2a": DataInput(
                data_type="raster",
                layers=image_layers,
                bands=S2_BANDS,
                passthrough=True,
                dtype=DType.FLOAT32,
                load_all_item_groups=True,
                load_all_layers=True,
            ),
            "label": DataInput(
                data_type="vector",
                layers=target_layers,
                is_target=True,
            ),
        },
        task=MultiTask(
            tasks={
                "detect": DetectionTask(
                    property_name=property_name,
                    classes=classes,
                    read_class_id=read_class_id,
                    skip_unknown_categories=skip_unknown_categories,
                    skip_empty_examples=skip_empty_examples,
                    box_size=box_size,
                    clip_boxes=clip_boxes,
                    exclude_by_center=exclude_by_center,
                    enable_map_metric=True,
                )
            },
            input_mapping={"detect": {"label": "targets"}},
        ),
        workers=workers,
        fix_crop_pick=True,
    )


def _resolve_timestamps(
    raster: Any,
    num_timesteps: int,
    time_year: int,
    timestamp_mode: str,
) -> np.ndarray:
    if timestamp_mode == "actual":
        timestamps = _actual_timestamps(_time_ranges(raster))
        if timestamps is not None:
            return timestamps
    if timestamp_mode in {"legacy", "actual"}:
        return _legacy_timestamps(num_timesteps, time_year)
    raise ValueError("timestamp_mode must be 'legacy' or 'actual'")


def _valid_xyxy_box(box: np.ndarray) -> bool:
    x1, y1, x2, y2 = [float(v) for v in box]
    return x2 > x1 and y2 > y1


def _convert_split(
    rslearn_dataset,
    output_root: Path,
    manifest_name: str,
    spec: DetectionSpec,
    classes: list[str],
    property_name: str,
    timestamp_mode: str,
) -> dict[str, Any]:
    samples = []

    total = len(rslearn_dataset)
    for idx in progress_iter(
        range(total),
        total=total,
        desc=f"{manifest_name}: converting rslearn detections",
    ):
        input_dict, target_dict, metadata = rslearn_dataset[idx]
        raster = input_dict["sentinel2_l2a"]
        image = _to_numpy(raster).astype(np.float32)
        if image.ndim != 4:
            raise ValueError(
                f"Expected sentinel2_l2a CTHW image, got {image.shape}"
            )
        targets = target_dict["detect"]
        boxes = _to_numpy(targets["boxes"]).astype(np.float32)
        labels = _to_numpy(targets["labels"]).astype(np.int64)
        valid = int(_to_numpy(targets["valid"]).reshape(()))
        height = int(float(_to_numpy(targets["height"]).reshape(())))
        width = int(float(_to_numpy(targets["width"]).reshape(())))

        timestamps = _resolve_timestamps(
            raster,
            image.shape[1],
            spec.time_year,
            timestamp_mode,
        )

        sample_id = f"{manifest_name}_{idx:06d}"
        sample_dir = output_root / "samples" / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        image_paths = save_timesteps_as_geotiffs(
            sample_dir,
            "sentinel2_l2a",
            image.transpose(1, 0, 2, 3),
            S2_BANDS,
        )
        rel_paths = [f"samples/{sample_id}/{path}" for path in image_paths]

        out_boxes = []
        out_labels = []
        for box, label in zip(boxes, labels):
            if not valid or not _valid_xyxy_box(box):
                continue
            out_boxes.append([float(v) for v in box])
            out_labels.append(int(label))

        samples.append(
            {
                "sample_id": sample_id,
                "img_id": idx + 1,
                "img_paths": rel_paths,
                "height": height,
                "width": width,
                "bboxes": out_boxes,
                "labels": out_labels,
                "valid": valid,
                "timestamps": timestamps.tolist(),
                "present_bands": S2_BANDS,
                "olmoearth_modality": "sentinel2_l2a",
                "olmoearth_num_timesteps": int(image.shape[1]),
                "olmoearth_band_names": S2_BANDS,
                "rslearn": {
                    **_metadata_to_dict(metadata),
                    "source_index": idx,
                    "raw_shape": list(image.shape),
                    "timestamp_mode": timestamp_mode,
                },
            }
        )

    return {
        "metainfo": {
            "dataset": spec.dataset_name,
            "split": manifest_name,
            "format": "olmoearth_rslearn_detection_manifest",
            "classes": classes,
            "property_name": property_name,
            "box_format": "xyxy",
            "label_offset": 0,
        },
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an rslearn DetectionTask dataset into OLMoEarth "
            "detection manifests plus multi-timestep GeoTIFF imagery for "
            "MMDetection."
        )
    )
    parser.add_argument("--dataset", default="generic", choices=DATASET_SPECS)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--classes", default=None, help="Comma-separated names.")
    parser.add_argument("--property-name", default=None)
    parser.add_argument("--image-layers", required=True)
    parser.add_argument("--target-layers", required=True)
    parser.add_argument("--split-group", default="spatial_split")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--box-size", type=int, default=None)
    parser.add_argument("--no-clip-boxes", action="store_true")
    parser.add_argument("--exclude-by-center", action="store_true")
    parser.add_argument("--read-class-id", action="store_true")
    parser.add_argument("--skip-unknown-categories", action="store_true")
    parser.add_argument("--skip-empty-examples", action="store_true")
    parser.add_argument(
        "--timestamp-mode",
        choices=["legacy", "actual"],
        default="legacy",
    )
    args = parser.parse_args()

    spec = DATASET_SPECS[args.dataset]
    classes = _parse_list(args.classes) or list(spec.classes)
    property_name = args.property_name or spec.property_name
    image_layers = _parse_list(args.image_layers)
    target_layers = _parse_list(args.target_layers)
    if not image_layers or not target_layers:
        raise ValueError("image-layers and target-layers cannot be empty")

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    split_summaries = {}
    for manifest_name, split_tag in spec.split_tags.items():
        rslearn_dataset = _build_rslearn_dataset(
            dataset_path=input_root,
            split_tag=split_tag,
            split_group=args.split_group,
            image_layers=image_layers,
            target_layers=target_layers,
            classes=classes,
            property_name=property_name,
            workers=args.workers,
            box_size=args.box_size,
            clip_boxes=not args.no_clip_boxes,
            exclude_by_center=args.exclude_by_center,
            read_class_id=args.read_class_id,
            skip_unknown_categories=args.skip_unknown_categories,
            skip_empty_examples=args.skip_empty_examples,
        )
        manifest = _convert_split(
            rslearn_dataset=rslearn_dataset,
            output_root=output_root,
            manifest_name=manifest_name,
            spec=spec,
            classes=classes,
            property_name=property_name,
            timestamp_mode=args.timestamp_mode,
        )
        save_json(output_root / f"{manifest_name}.json", manifest)
        split_summaries[manifest_name] = {
            "samples": len(manifest["samples"]),
            "boxes": sum(
                len(sample["bboxes"]) for sample in manifest["samples"]
            ),
        }

    save_json(
        output_root / "metainfo.json",
        {
            "dataset": spec.dataset_name,
            "classes": classes,
            "property_name": property_name,
            "image_layout": "img_paths_tif_tchw",
            "modalities": ["sentinel2_l2a"],
            "bands": S2_BANDS,
            "patch_size": spec.patch_size,
            "splits": split_summaries,
        },
    )


if __name__ == "__main__":
    main()
