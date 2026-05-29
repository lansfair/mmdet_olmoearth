from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_manifest(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("samples"),
        list,
    ):
        raise TypeError(
            f"Manifest must be {{'metainfo': ..., 'samples': list}}: {path}"
        )
    return payload


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _read_tif_shape(path: Path) -> tuple[int, int, int] | None:
    try:
        import rasterio
    except ImportError:
        return None
    with rasterio.open(path) as src:
        return src.count, src.height, src.width


def _check_box(
    bbox: list[Any],
    width: int,
    height: int,
    sample_id: str,
) -> None:
    if len(bbox) != 4:
        raise ValueError(f"{sample_id}: bbox must have 4 values, got {bbox}")
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{sample_id}: invalid xyxy bbox {bbox}")
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError(
            f"{sample_id}: bbox {bbox} is outside image size "
            f"{(width, height)}"
        )


def check_manifest(
    data_root: Path,
    ann_file: Path,
    max_samples: int | None,
    check_tif: bool,
) -> dict[str, Any]:
    manifest = _load_manifest(ann_file)
    classes = manifest.get("metainfo", {}).get("classes", ["object"])
    num_classes = len(classes)
    samples = manifest["samples"]
    if max_samples is not None:
        samples = samples[:max_samples]

    summary = {
        "ann_file": str(ann_file),
        "checked_samples": len(samples),
        "num_classes": num_classes,
        "valid_samples": 0,
        "boxes": 0,
        "tif_checked": 0,
    }

    for index, sample in enumerate(samples):
        sample_id = str(sample.get("sample_id", index))
        width = int(sample["width"])
        height = int(sample["height"])
        img_paths = sample.get("img_paths")
        if not img_paths:
            raise KeyError(f"{sample_id}: missing img_paths")

        if int(sample.get("valid", 1)):
            summary["valid_samples"] += 1

        for img_path in img_paths:
            resolved = _resolve_path(data_root, img_path)
            if not resolved.exists():
                raise FileNotFoundError(f"{sample_id}: missing {resolved}")
            if check_tif:
                shape = _read_tif_shape(resolved)
                if shape is not None:
                    _, tif_h, tif_w = shape
                    if (tif_w, tif_h) != (width, height):
                        raise ValueError(
                            f"{sample_id}: {resolved} has shape "
                            f"{(tif_w, tif_h)}, manifest has "
                            f"{(width, height)}"
                        )
                    summary["tif_checked"] += 1

        bboxes = sample.get("bboxes", [])
        labels = sample.get("labels", [])
        if len(bboxes) != len(labels):
            raise ValueError(
                f"{sample_id}: {len(bboxes)} bboxes but {len(labels)} labels"
            )
        for bbox, label in zip(bboxes, labels):
            label = int(label)
            if label < 0 or label >= num_classes:
                raise ValueError(
                    f"{sample_id}: label {label} outside "
                    f"[0, {num_classes})"
                )
            _check_box(bbox, width, height, sample_id)
        summary["boxes"] += len(bboxes)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-check OLMoEarth MMDetection manifests."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--no-check-tif",
        action="store_true",
        help="Only check paths and annotations, without opening GeoTIFFs.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    ann_file = Path(args.ann_file)
    if not ann_file.is_absolute():
        ann_file = data_root / ann_file

    summary = check_manifest(
        data_root=data_root,
        ann_file=ann_file,
        max_samples=args.max_samples,
        check_tif=not args.no_check_tif,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
