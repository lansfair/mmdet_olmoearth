from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmdet.registry import METRICS


def _bbox_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(
        a[:, 3] - a[:, 1],
        0.0,
        None,
    )
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(
        b[:, 3] - b[:, 1],
        0.0,
        None,
    )
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-6, None)


def _linear_assignment(iou: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.optimize import linear_sum_assignment

        return linear_sum_assignment(-iou)
    except ImportError:
        pairs = []
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        flat_indices = np.argsort(iou.reshape(-1))[::-1]
        for flat_index in flat_indices:
            row = int(flat_index // iou.shape[1])
            col = int(flat_index % iou.shape[1])
            if row in used_rows or col in used_cols:
                continue
            used_rows.add(row)
            used_cols.add(col)
            pairs.append((row, col))
        if not pairs:
            return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
        rows, cols = zip(*pairs)
        return np.asarray(rows), np.asarray(cols)


def _match_counts(
    pred_bboxes: np.ndarray,
    gt_bboxes: np.ndarray,
    iou_thr: float,
) -> tuple[int, int, int]:
    if len(pred_bboxes) == 0:
        return 0, 0, len(gt_bboxes)
    if len(gt_bboxes) == 0:
        return 0, len(pred_bboxes), 0

    ious = _bbox_iou_matrix(pred_bboxes, gt_bboxes)
    pred_indices, gt_indices = _linear_assignment(ious)
    matched = 0
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        if ious[pred_idx, gt_idx] >= iou_thr:
            matched += 1
    fp = len(pred_bboxes) - matched
    fn = len(gt_bboxes) - matched
    return matched, fp, fn


@METRICS.register_module()
class OlmoEarthDetMetric(BaseMetric):
    """rslearn-style detection F1 metric for OLMoEarth manifests."""

    default_prefix = "olmoearth_det"

    def __init__(
        self,
        num_classes: int | None = None,
        iou_thr: float = 0.5,
        score_thresholds: Sequence[float] = (0.05,),
        collect_device: str = "cpu",
        prefix: str | None = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.num_classes = num_classes
        self.iou_thr = iou_thr
        self.score_thresholds = tuple(float(v) for v in score_thresholds)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            metainfo = getattr(data_sample, "metainfo", {})
            gt_instances = data_sample["gt_instances"]
            pred_instances = data_sample["pred_instances"]
            self.results.append(
                {
                    "valid": int(metainfo.get("valid", 1)),
                    "gt_bboxes": gt_instances["bboxes"].cpu().numpy(),
                    "gt_labels": gt_instances["labels"].cpu().numpy(),
                    "pred_bboxes": pred_instances["bboxes"].cpu().numpy(),
                    "pred_labels": pred_instances["labels"].cpu().numpy(),
                    "pred_scores": pred_instances["scores"].cpu().numpy(),
                }
            )

    def _num_classes(self, results: list[dict]) -> int:
        if self.num_classes is not None:
            return self.num_classes
        classes = self.dataset_meta.get("classes", ())
        if classes:
            return len(classes)
        max_label = -1
        for result in results:
            for key in ("gt_labels", "pred_labels"):
                labels = result[key]
                if len(labels):
                    max_label = max(max_label, int(labels.max()))
        return max_label + 1

    def _counts_at_threshold(
        self,
        results: list[dict],
        score_thr: float,
        num_classes: int,
    ) -> tuple[int, int, int]:
        tp = fp = fn = 0
        for result in results:
            if not result["valid"]:
                continue
            for label in range(num_classes):
                gt_mask = result["gt_labels"] == label
                pred_mask = (
                    (result["pred_labels"] == label)
                    & (result["pred_scores"] >= score_thr)
                )
                matched, false_pos, false_neg = _match_counts(
                    result["pred_bboxes"][pred_mask],
                    result["gt_bboxes"][gt_mask],
                    self.iou_thr,
                )
                tp += matched
                fp += false_pos
                fn += false_neg
        return tp, fp, fn

    def compute_metrics(self, results: list[dict]) -> dict:
        eval_results = OrderedDict()
        num_classes = self._num_classes(results)
        if num_classes <= 0:
            eval_results["best_f1"] = 0.0
            eval_results["best_precision"] = 0.0
            eval_results["best_recall"] = 0.0
            eval_results["best_score_thr"] = 0.0
            eval_results["iou_thr"] = self.iou_thr
            return eval_results

        best = {
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "score_thr": self.score_thresholds[0],
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }

        for score_thr in self.score_thresholds:
            tp, fp, fn = self._counts_at_threshold(
                results,
                score_thr,
                num_classes,
            )
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            suffix = f"{score_thr:.2f}"
            eval_results[f"f1@{suffix}"] = f1
            eval_results[f"precision@{suffix}"] = precision
            eval_results[f"recall@{suffix}"] = recall
            if f1 > best["f1"]:
                best = {
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                    "score_thr": score_thr,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                }

        eval_results["best_f1"] = best["f1"]
        eval_results["best_precision"] = best["precision"]
        eval_results["best_recall"] = best["recall"]
        eval_results["best_score_thr"] = best["score_thr"]
        eval_results["best_tp"] = best["tp"]
        eval_results["best_fp"] = best["fp"]
        eval_results["best_fn"] = best["fn"]
        eval_results["iou_thr"] = self.iou_thr
        return eval_results
