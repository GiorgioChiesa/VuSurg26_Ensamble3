#!/usr/bin/env python3
"""Benchmark clevis pseudo-label boxes against EndoVis ground-truth boxes.

The default paths compare COCO exports from the EndoVis pipeline run:

    uv run python scripts/benchmark_endovis_boxes.py

Ground truth may be a COCO file or a directory containing ``*_coco.json`` files.
Predictions may be one or more COCO files, a directory containing ``coco_*.json``
files, or a raw ``clevis_boxes.json`` list from phase 5.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))


@dataclass(frozen=True)
class BoxRecord:
    image_key: tuple[str, int]
    bbox_xywh: tuple[float, float, float, float]
    width: int | None
    height: int | None
    score: float
    category_id: int | str | None
    source: str
    annotation_id: int | str | None = None
    category_name: str | None = None


@dataclass(frozen=True)
class MatchRecord:
    video_id: str
    frame_idx: int
    pred_annotation_id: int | str | None
    gt_annotation_id: int | str | None
    pred_score: float
    iou: float
    pred_bbox_xywh: tuple[float, float, float, float]
    gt_bbox_xywh: tuple[float, float, float, float]
    pred_category_id: int | str | None = None
    gt_category_id: int | str | None = None
    pred_category_name: str | None = None
    gt_category_name: str | None = None
    pred_area: float = 0.0
    gt_area: float = 0.0
    area_ratio: float = 0.0
    containment: float = 0.0


def _read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _video_base(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    text = Path(text).stem if "." in Path(text).name else text
    return text.split("__src", 1)[0]


def _parse_frame_from_text(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value)
    patterns = (
        r"frame[_-]?(\d+)",
        r"frame(\d+)",
        r"slice_nr_(\d+)",
        r"(\d+)(?=\.[A-Za-z0-9]+$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    stem = Path(text).stem
    if stem.isdigit():
        return int(stem)
    return None


def _image_key(image: dict[str, Any], fallback_video_id: str | None = None) -> tuple[str, int]:
    video = (
        _video_base(image.get("original_filename"))
        or _video_base(image.get("video_id"))
        or (None if "/" not in str(image.get("file_name", "")) else _video_base(str(image.get("file_name", "")).split("/", 1)[0]))
        or _video_base(fallback_video_id)
        or _video_base(image.get("id"))
    )
    frame = image.get("frame_id")
    if frame is None:
        frame = _parse_frame_from_text(image.get("file_name"))
    if frame is None:
        frame = _parse_frame_from_text(image.get("id"))
    if video is None or frame is None:
        raise ValueError(f"Cannot derive video/frame key from image record: {image}")
    return video, int(frame)


def _raw_box_key(record: dict[str, Any]) -> tuple[str, int]:
    """Key a raw pipeline box record by (video, absolute frame index).

    ``frame_idx`` is sample-relative -- it restarts at 0 for every roster clip --
    while COCO ``image_id`` is the absolute frame index in the source video.
    Prefer ``absolute_frame_idx`` when the producing stage emitted it; falling
    back keeps older artifacts (which only carry the relative index) readable.
    """
    video = _video_base(record.get("video_id"))
    frame = record.get("absolute_frame_idx")
    if frame is None:
        frame = record.get("frame_idx", record.get("frame_id"))
    if video is None or frame is None:
        raise ValueError(f"Cannot derive video/frame key from raw box record: {record}")
    return video, int(frame)


def _bbox_xywh(values: Iterable[Any]) -> tuple[float, float, float, float]:
    bbox = tuple(float(v) for v in values)
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox with 4 values, got {bbox}")
    return bbox


def _category_ids_by_name(coco: dict[str, Any], names: set[str]) -> set[int | str]:
    ids: set[int | str] = set()
    lowered = {name.lower() for name in names}
    for cat in coco.get("categories", []):
        if str(cat.get("name", "")).lower() in lowered:
            ids.add(cat.get("id"))
    return ids


def _load_coco_records(
    path: Path,
    *,
    category_ids: set[int | str] | None = None,
    category_names: set[str] | None = None,
    default_score: float = 1.0,
    fallback_video_id: str | None = None,
) -> list[BoxRecord]:
    coco = _read_json(path)
    if not isinstance(coco, dict) or "images" not in coco or "annotations" not in coco:
        raise ValueError(f"{path} is not a COCO annotation file")

    allowed_ids = set(category_ids) if category_ids is not None else None
    if category_names:
        name_ids = _category_ids_by_name(coco, category_names)
        allowed_ids = name_ids if allowed_ids is None else allowed_ids | name_ids

    images_by_id = {image["id"]: image for image in coco.get("images", [])}
    names_by_category = {
        cat.get("id"): str(cat.get("name")) for cat in coco.get("categories", []) if cat.get("name")
    }
    records: list[BoxRecord] = []
    for ann in coco.get("annotations", []):
        category_id = ann.get("category_id")
        if allowed_ids is not None and category_id not in allowed_ids:
            continue
        image = images_by_id.get(ann.get("image_id"))
        if image is None:
            continue
        bbox = _bbox_xywh(ann.get("bbox", ()))
        if bbox[2] <= 0 or bbox[3] <= 0:
            continue
        records.append(
            BoxRecord(
                image_key=_image_key(image, fallback_video_id=fallback_video_id),
                bbox_xywh=bbox,
                width=int(image["width"]) if image.get("width") is not None else None,
                height=int(image["height"]) if image.get("height") is not None else None,
                score=float(ann.get("score", default_score)),
                category_id=category_id,
                source=str(path),
                annotation_id=ann.get("id"),
                category_name=names_by_category.get(category_id),
            )
        )
    return records


def _raw_record_class_name(item: dict[str, Any]) -> str | None:
    for key in ("category_name", "instrument_type", "tool_type"):
        value = item.get(key)
        if value:
            return str(value)
    return None


def _load_raw_prediction_records(
    path: Path,
    *,
    image_width: int | None,
    image_height: int | None,
    category_map: dict[str, int] | None = None,
    unmapped_classes: Counter | None = None,
) -> list[BoxRecord]:
    """Read a raw pipeline box list (e.g. clevis_boxes.json).

    ``category_map`` maps a pipeline class name onto a GT category id. It exists
    because the two taxonomies are not string-identical: the classifier says
    "monopolar curved scissors" while cat1 category 7 is "monopolar curved
    scissor", and that class alone is 25% of the GT boxes. Names that resolve to
    nothing are tallied in ``unmapped_classes`` rather than silently defaulted,
    so a broken map shows up as a count instead of as a depressed mAP.

    Precedence, and why it is not simply "the map wins":
      1. an explicit ``category_map`` entry for the record's class name — this is
         what a ``--pred-category-map`` override is for, so it must still bite;
      2. otherwise the record's own ``category_id``, which stages like
         ``segment-clevis`` already resolved through *this same* map, so asking
         the map again with the already-translated cat1 name finds nothing;
      3. otherwise the fallback id below.

    Only a record that fails 1 *and* 2 is genuinely unmapped. Tallying case 2
    reported 935 correctly-scored cat1 predictions as unmapped and sent a reader
    hunting a bug that was not there.
    """
    data = _read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} is neither COCO nor a raw list of box records")

    records: list[BoxRecord] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict) or "bbox_xywh" not in item:
            continue
        bbox = _bbox_xywh(item["bbox_xywh"])
        if bbox[2] <= 0 or bbox[3] <= 0:
            continue
        class_name = _raw_record_class_name(item)
        record_category_id = item.get("category_id")
        category_id = record_category_id
        if category_map is not None and class_name is not None:
            mapped = category_map.get(class_name)
            if mapped is not None:
                category_id = mapped
            elif record_category_id is None and unmapped_classes is not None:
                unmapped_classes[class_name] += 1
        if category_id is None:
            category_id = 1
        # A record that states its own frame size is authoritative: the CLI
        # default (640x480) silently rescales cat1's 640x512 boxes by 6.7% in y.
        record_width = item.get("frame_width")
        record_height = item.get("frame_height")
        records.append(
            BoxRecord(
                image_key=_raw_box_key(item),
                bbox_xywh=bbox,
                width=int(record_width) if record_width else image_width,
                height=int(record_height) if record_height else image_height,
                score=float(item.get("score", item.get("confidence", 1.0))),
                category_id=category_id,
                source=str(path),
                annotation_id=item.get("id", idx),
                category_name=class_name,
            )
        )
    return records


def _expand_ground_truth_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    coco_paths = sorted(path.glob("*_coco.json"))
    if coco_paths:
        return coco_paths
    merged = path / "gt_merged.json"
    if merged.exists():
        return [merged]
    raise FileNotFoundError(f"No *_coco.json or gt_merged.json found in {path}")


def _expand_prediction_paths(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_file():
            expanded.append(path)
        elif path.is_dir():
            coco_paths = sorted(path.glob("coco_*.json"))
            if coco_paths:
                expanded.extend(coco_paths)
            else:
                raw_path = path / "clevis_boxes.json"
                if raw_path.exists():
                    expanded.append(raw_path)
                else:
                    raise FileNotFoundError(f"No coco_*.json or clevis_boxes.json found in {path}")
        else:
            raise FileNotFoundError(path)
    return expanded


def load_ground_truth(
    path: Path,
    *,
    category_ids: set[int | str] | None = None,
    category_names: set[str] | None = None,
) -> list[BoxRecord]:
    records: list[BoxRecord] = []
    for gt_path in _expand_ground_truth_paths(path):
        fallback_video_id = re.sub(r"_coco$", "", gt_path.stem)
        records.extend(
            _load_coco_records(
                gt_path,
                category_ids=category_ids,
                category_names=category_names,
                default_score=1.0,
                fallback_video_id=fallback_video_id,
            )
        )
    return records


def load_predictions(
    paths: list[Path],
    *,
    category_ids: set[int | str] | None = None,
    category_names: set[str] | None = None,
    image_width: int | None = 640,
    image_height: int | None = 480,
    category_map: dict[str, int] | None = None,
    unmapped_classes: Counter | None = None,
) -> list[BoxRecord]:
    records: list[BoxRecord] = []
    for pred_path in _expand_prediction_paths(paths):
        data = _read_json(pred_path)
        if isinstance(data, dict) and "images" in data and "annotations" in data:
            records.extend(
                _load_coco_records(
                    pred_path,
                    category_ids=category_ids,
                    category_names=category_names,
                    default_score=1.0,
                    fallback_video_id=re.sub(r"_coco$", "", pred_path.stem),
                )
            )
        else:
            records.extend(
                _load_raw_prediction_records(
                    pred_path,
                    image_width=image_width,
                    image_height=image_height,
                    category_map=category_map,
                    unmapped_classes=unmapped_classes,
                )
            )
    return records


def load_category_map(path: Path) -> dict[str, int]:
    """Read a {pipeline class name: cat1 category} mapping.

    Accepts either ``{"needle driver": 11}`` or the richer
    ``{"needle driver": {"id": 11, "name": "needle driver"}}`` written by
    clevis_eval.category_map. A null value means "this class has no cat1
    counterpart" and is dropped, so its detections stay unmapped and get counted.
    """
    raw = _read_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} is not a category map object")
    mapping: dict[str, int] = {}
    for name, value in raw.items():
        if value is None:
            continue
        if isinstance(value, dict):
            if value.get("id") is None:
                continue
            mapping[str(name)] = int(value["id"])
        else:
            mapping[str(name)] = int(value)
    return mapping


def _group_by_image(records: Iterable[BoxRecord]) -> dict[tuple[str, int], list[BoxRecord]]:
    grouped: dict[tuple[str, int], list[BoxRecord]] = defaultdict(list)
    for record in records:
        grouped[record.image_key].append(record)
    return dict(grouped)


def _scale_bbox(
    bbox: tuple[float, float, float, float],
    *,
    from_width: int | None,
    from_height: int | None,
    to_width: int | None,
    to_height: int | None,
) -> tuple[float, float, float, float]:
    if not from_width or not from_height or not to_width or not to_height:
        return bbox
    sx = to_width / from_width
    sy = to_height / from_height
    return bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy


def _iou_xywh(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    union = aw * ah + bw * bh - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _intersection_xywh(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _containment(
    pred: tuple[float, float, float, float],
    gt: tuple[float, float, float, float],
) -> float:
    """Fraction of the GT box covered by the prediction.

    Read together with IoU this separates the two failure modes mAP conflates:
    containment near 1.0 with low IoU means the box is in the right place but
    far too large, while containment near 0.0 means it is in the wrong place.
    The fixes for those are opposite, so the diagnostic has to distinguish them.
    """
    gt_area = gt[2] * gt[3]
    if gt_area <= 0:
        return 0.0
    return _intersection_xywh(pred, gt) / gt_area


def _precision_recall_ap(tp_flags: list[int], fp_flags: list[int], gt_count: int) -> tuple[float, float, float]:
    if gt_count == 0:
        return 0.0, 0.0, 0.0

    tp_total = 0
    fp_total = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for tp, fp in zip(tp_flags, fp_flags):
        tp_total += tp
        fp_total += fp
        precisions.append(tp_total / max(tp_total + fp_total, 1))
        recalls.append(tp_total / gt_count)

    ap = 0.0
    for recall_target in [i / 100 for i in range(101)]:
        eligible = [p for p, r in zip(precisions, recalls) if r >= recall_target]
        ap += max(eligible) if eligible else 0.0
    ap /= 101

    precision = tp_total / max(tp_total + fp_total, 1)
    recall = tp_total / gt_count
    return precision, recall, ap


def _evaluate_at_threshold(
    gt_by_image: dict[tuple[str, int], list[BoxRecord]],
    pred_by_image: dict[tuple[str, int], list[BoxRecord]],
    *,
    threshold: float,
    scale_predictions: bool,
    class_aware: bool = False,
) -> tuple[dict[str, float], list[MatchRecord]]:
    gt_images = set(gt_by_image)
    eval_predictions: list[tuple[BoxRecord, BoxRecord | None, float]] = []
    matches: list[MatchRecord] = []

    for image_key in sorted(gt_images):
        gts = gt_by_image[image_key]
        preds = sorted(pred_by_image.get(image_key, []), key=lambda p: p.score, reverse=True)
        used_gt: set[int] = set()

        for pred in preds:
            best_idx: int | None = None
            best_iou = 0.0
            best_gt: BoxRecord | None = None
            for gt_idx, gt in enumerate(gts):
                if gt_idx in used_gt:
                    continue
                if class_aware and pred.category_id != gt.category_id:
                    continue
                pred_bbox = pred.bbox_xywh
                if scale_predictions:
                    pred_bbox = _scale_bbox(
                        pred_bbox,
                        from_width=pred.width,
                        from_height=pred.height,
                        to_width=gt.width,
                        to_height=gt.height,
                    )
                iou = _iou_xywh(pred_bbox, gt.bbox_xywh)
                if iou > best_iou:
                    best_idx = gt_idx
                    best_iou = iou
                    best_gt = gt

            if best_idx is not None and best_gt is not None and best_iou >= threshold:
                used_gt.add(best_idx)
                pred_bbox = pred.bbox_xywh
                if scale_predictions:
                    pred_bbox = _scale_bbox(
                        pred_bbox,
                        from_width=pred.width,
                        from_height=pred.height,
                        to_width=best_gt.width,
                        to_height=best_gt.height,
                    )
                eval_predictions.append((pred, best_gt, best_iou))
                pred_area = pred_bbox[2] * pred_bbox[3]
                gt_area = best_gt.bbox_xywh[2] * best_gt.bbox_xywh[3]
                matches.append(
                    MatchRecord(
                        video_id=image_key[0],
                        frame_idx=image_key[1],
                        pred_annotation_id=pred.annotation_id,
                        gt_annotation_id=best_gt.annotation_id,
                        pred_score=pred.score,
                        iou=best_iou,
                        pred_bbox_xywh=pred_bbox,
                        gt_bbox_xywh=best_gt.bbox_xywh,
                        pred_category_id=pred.category_id,
                        gt_category_id=best_gt.category_id,
                        pred_category_name=pred.category_name,
                        gt_category_name=best_gt.category_name,
                        pred_area=pred_area,
                        gt_area=gt_area,
                        area_ratio=pred_area / gt_area if gt_area > 0 else 0.0,
                        containment=_containment(pred_bbox, best_gt.bbox_xywh),
                    )
                )
            else:
                eval_predictions.append((pred, None, best_iou))

    eval_predictions.sort(key=lambda item: item[0].score, reverse=True)
    tp_flags = [1 if gt is not None else 0 for _, gt, _ in eval_predictions]
    fp_flags = [0 if gt is not None else 1 for _, gt, _ in eval_predictions]
    gt_count = sum(len(v) for v in gt_by_image.values())
    tp = sum(tp_flags)
    fp = sum(fp_flags)
    fn = gt_count - tp
    precision, recall, ap = _precision_recall_ap(tp_flags, fp_flags, gt_count)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    matched_ious = [m.iou for m in matches]

    return (
        {
            "threshold": threshold,
            "true_positive": float(tp),
            "false_positive": float(fp),
            "false_negative": float(fn),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap": ap,
            "mean_matched_iou": statistics.fmean(matched_ious) if matched_ious else 0.0,
            "median_matched_iou": statistics.median(matched_ious) if matched_ious else 0.0,
        },
        matches,
    )


def evaluate(
    ground_truth: list[BoxRecord],
    predictions: list[BoxRecord],
    *,
    iou_thresholds: tuple[float, ...] = DEFAULT_IOU_THRESHOLDS,
    match_threshold: float = 0.5,
    scale_predictions: bool = True,
    eval_scope: str = "overlap",
    class_aware: bool = False,
    eval_image_keys: set[tuple[str, int]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, float]], list[MatchRecord]]:
    gt_by_image = _group_by_image(ground_truth)
    pred_by_image = _group_by_image(predictions)
    gt_images = set(gt_by_image) if eval_image_keys is None else set(eval_image_keys)
    pred_images = set(pred_by_image)
    overlap_images = gt_images & pred_images
    if eval_scope == "ground-truth":
        eval_images = gt_images
    elif eval_scope in ("overlap", "pseudo-labeled"):
        eval_images = overlap_images
    else:
        raise ValueError(f"Unknown eval_scope: {eval_scope}")
    # An image with no GT box of the class under evaluation still belongs in the
    # loop: detections there are false positives, and dropping the image would
    # silently forgive them. Hence the explicit empty list rather than a filter.
    eval_gt_by_image = {key: gt_by_image.get(key, []) for key in eval_images}

    threshold_rows: list[dict[str, float]] = []
    matches_at_requested_threshold: list[MatchRecord] = []
    for threshold in iou_thresholds:
        row, matches = _evaluate_at_threshold(
            eval_gt_by_image,
            pred_by_image,
            threshold=threshold,
            scale_predictions=scale_predictions,
            class_aware=class_aware,
        )
        threshold_rows.append(row)
        if math.isclose(threshold, match_threshold):
            matches_at_requested_threshold = matches

    if not matches_at_requested_threshold and match_threshold not in iou_thresholds:
        _, matches_at_requested_threshold = _evaluate_at_threshold(
            eval_gt_by_image,
            pred_by_image,
            threshold=match_threshold,
            scale_predictions=scale_predictions,
            class_aware=class_aware,
        )

    ap_values = [row["ap"] for row in threshold_rows]
    ap50 = next((row["ap"] for row in threshold_rows if math.isclose(row["threshold"], 0.5)), 0.0)
    summary = {
        "gt_images": len(gt_images),
        "pred_images": len(pred_images),
        "overlap_images": len(overlap_images),
        "pseudo_labeled_images_with_gt": len(overlap_images),
        "pseudo_labeled_images_without_gt": len(pred_images - gt_images),
        "evaluated_gt_images": len(eval_images),
        "gt_annotations": len(ground_truth),
        "evaluated_gt_annotations": sum(len(v) for v in eval_gt_by_image.values()),
        "pred_annotations": len(predictions),
        "evaluated_pred_annotations": sum(len(pred_by_image.get(k, [])) for k in eval_images),
        "ignored_gt_annotations_out_of_scope": sum(len(v) for k, v in gt_by_image.items() if k not in eval_images),
        "ignored_pred_annotations_out_of_scope": sum(len(v) for k, v in pred_by_image.items() if k not in eval_images),
        "map_50_95": statistics.fmean(ap_values) if ap_values else 0.0,
        "ap50": ap50,
        "match_threshold": match_threshold,
        "scale_predictions": scale_predictions,
        "eval_scope": eval_scope,
        "class_aware": class_aware,
    }

    match_row = next(
        (row for row in threshold_rows if math.isclose(row["threshold"], match_threshold)),
        None,
    )
    if match_row is not None:
        summary.update(
            {
                "precision": match_row["precision"],
                "recall": match_row["recall"],
                "f1": match_row["f1"],
                "mean_matched_iou": match_row["mean_matched_iou"],
                "median_matched_iou": match_row["median_matched_iou"],
            }
        )
    return summary, threshold_rows, matches_at_requested_threshold


def _center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0


def diagnostic_pairs(
    ground_truth: list[BoxRecord],
    predictions: list[BoxRecord],
    *,
    scale_predictions: bool = True,
    class_aware: bool = False,
) -> list[MatchRecord]:
    """Pair every prediction with its nearest GT box, however badly it scores.

    Deliberately not ``_evaluate_at_threshold(threshold=0.0)``: that requires
    ``iou > 0`` before it records a pair, so a prediction that misses the GT
    entirely produces no record at all -- and "the box is in the wrong place" is
    exactly the failure the containment diagnostic exists to name. Here a
    zero-overlap prediction still yields a record, with containment 0.0.

    Pairing is many-to-one (no ``used_gt`` exclusion) because the goal is to
    characterize every prediction, not to score a detection set.
    """
    gt_by_image = _group_by_image(ground_truth)
    pred_by_image = _group_by_image(predictions)
    records: list[MatchRecord] = []

    for image_key in sorted(set(gt_by_image) & set(pred_by_image)):
        gts = gt_by_image[image_key]
        for pred in pred_by_image[image_key]:
            candidates = [
                gt
                for gt in gts
                if not class_aware or pred.category_id == gt.category_id
            ]
            if not candidates:
                continue
            scored = []
            for gt in candidates:
                pred_bbox = pred.bbox_xywh
                if scale_predictions:
                    pred_bbox = _scale_bbox(
                        pred_bbox,
                        from_width=pred.width,
                        from_height=pred.height,
                        to_width=gt.width,
                        to_height=gt.height,
                    )
                iou = _iou_xywh(pred_bbox, gt.bbox_xywh)
                px, py = _center(pred_bbox)
                gx, gy = _center(gt.bbox_xywh)
                distance = math.hypot(px - gx, py - gy)
                scored.append((iou, -distance, gt, pred_bbox))
            iou, _neg_distance, gt, pred_bbox = max(scored, key=lambda item: (item[0], item[1]))
            pred_area = pred_bbox[2] * pred_bbox[3]
            gt_area = gt.bbox_xywh[2] * gt.bbox_xywh[3]
            records.append(
                MatchRecord(
                    video_id=image_key[0],
                    frame_idx=image_key[1],
                    pred_annotation_id=pred.annotation_id,
                    gt_annotation_id=gt.annotation_id,
                    pred_score=pred.score,
                    iou=iou,
                    pred_bbox_xywh=pred_bbox,
                    gt_bbox_xywh=gt.bbox_xywh,
                    pred_category_id=pred.category_id,
                    gt_category_id=gt.category_id,
                    pred_category_name=pred.category_name,
                    gt_category_name=gt.category_name,
                    pred_area=pred_area,
                    gt_area=gt_area,
                    area_ratio=pred_area / gt_area if gt_area > 0 else 0.0,
                    containment=_containment(pred_bbox, gt.bbox_xywh),
                )
            )
    return records


def evaluate_per_class(
    ground_truth: list[BoxRecord],
    predictions: list[BoxRecord],
    *,
    iou_thresholds: tuple[float, ...] = DEFAULT_IOU_THRESHOLDS,
    match_threshold: float = 0.5,
    scale_predictions: bool = True,
    eval_scope: str = "ground-truth",
) -> dict[str, Any]:
    """COCO-style per-class AP, averaged over classes with non-zero GT support.

    This is the leaderboard reduction. The class-agnostic ``evaluate`` above
    happily matches a needle driver prediction to a scissors GT box, which
    flatters every number it reports; here each class is scored against its own
    GT only, over the full evaluated image set so that cross-class detections
    still count as false positives.
    """
    all_image_keys = {record.image_key for record in ground_truth}
    gt_by_category: dict[Any, list[BoxRecord]] = defaultdict(list)
    for record in ground_truth:
        gt_by_category[record.category_id].append(record)
    pred_by_category: dict[Any, list[BoxRecord]] = defaultdict(list)
    for record in predictions:
        pred_by_category[record.category_id].append(record)

    per_class: list[dict[str, Any]] = []
    for category_id in sorted(gt_by_category, key=lambda c: (c is None, str(c))):
        class_gt = gt_by_category[category_id]
        class_pred = pred_by_category.get(category_id, [])
        summary, _rows, _matches = evaluate(
            class_gt,
            class_pred,
            iou_thresholds=iou_thresholds,
            match_threshold=match_threshold,
            scale_predictions=scale_predictions,
            eval_scope=eval_scope,
            class_aware=False,
            eval_image_keys=all_image_keys,
        )
        per_class.append(
            {
                "category_id": category_id,
                "category_name": next(
                    (r.category_name for r in class_gt if r.category_name), None
                ),
                "gt_boxes": len(class_gt),
                "pred_boxes": len(class_pred),
                "ap_50_95": summary["map_50_95"],
                "ap50": summary["ap50"],
                "precision": summary.get("precision", 0.0),
                "recall": summary.get("recall", 0.0),
                "f1": summary.get("f1", 0.0),
                "median_matched_iou": summary.get("median_matched_iou", 0.0),
            }
        )

    supported = [row for row in per_class if row["gt_boxes"] > 0]
    return {
        "map_50_95": statistics.fmean([r["ap_50_95"] for r in supported]) if supported else 0.0,
        "ap50": statistics.fmean([r["ap50"] for r in supported]) if supported else 0.0,
        "supported_classes": len(supported),
        "per_class": per_class,
    }


def area_report(matches: list[MatchRecord]) -> list[dict[str, Any]]:
    """Per (video, class) box-size diagnostics from an already-paired match list.

    Pass the matches from a threshold-0 pass: at IoU 0.5 the only surviving pairs
    are the ones already good enough to match, which is precisely the population
    that cannot tell you why the rest failed.
    """
    grouped: dict[tuple[str, Any], list[MatchRecord]] = defaultdict(list)
    for match in matches:
        grouped[(match.video_id, match.gt_category_id)].append(match)

    rows: list[dict[str, Any]] = []
    for (video_id, category_id), group in sorted(
        grouped.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))
    ):
        rows.append(
            {
                "video_id": video_id,
                "category_id": category_id,
                "category_name": next(
                    (m.gt_category_name for m in group if m.gt_category_name), None
                ),
                "pairs": len(group),
                "median_gt_area": statistics.median([m.gt_area for m in group]),
                "median_pred_area": statistics.median([m.pred_area for m in group]),
                "median_area_ratio": statistics.median([m.area_ratio for m in group]),
                "median_iou": statistics.median([m.iou for m in group]),
                "median_containment": statistics.median([m.containment for m in group]),
            }
        )
    return rows


def classification_confusion(matches: list[MatchRecord]) -> dict[str, Any]:
    """Predicted class vs GT category over box pairs that already localized well.

    Restricting to matched pairs is the point: a class error only costs mAP when
    the box was otherwise good, so mixing in unlocalized detections would
    conflate a classification problem with a localization one.

    Agreement is decided on ``category_id``, never on the name: the pipeline
    carries the classifier's taxonomy ("monopolar curved scissors") while GT
    carries cat1's ("monopolar curved scissor"), so comparing names would report
    a quarter of the boxes as class errors that the category map already
    resolved correctly. Names are display only, resolved through GT where the
    ids agree.
    """
    label_by_id: dict[Any, str] = {}
    for match in matches:
        if match.gt_category_id is not None and match.gt_category_name:
            label_by_id.setdefault(match.gt_category_id, match.gt_category_name)

    def _label(category_id: Any, fallback: str | None) -> str:
        return label_by_id.get(category_id) or fallback or str(category_id)

    counts: Counter = Counter()
    for match in matches:
        gt_label = _label(match.gt_category_id, match.gt_category_name)
        pred_label = (
            gt_label
            if match.pred_category_id == match.gt_category_id
            else _label(match.pred_category_id, match.pred_category_name)
        )
        counts[(gt_label, pred_label)] += 1
    total = sum(counts.values())
    correct = sum(n for (gt, pred), n in counts.items() if gt == pred)
    per_class_totals: Counter = Counter()
    per_class_correct: Counter = Counter()
    for (gt, pred), n in counts.items():
        per_class_totals[gt] += n
        if gt == pred:
            per_class_correct[gt] += n
    f1_terms = []
    for gt_name, support in per_class_totals.items():
        tp = per_class_correct[gt_name]
        predicted = sum(n for (_g, p), n in counts.items() if p == gt_name)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1_terms.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "pairs": total,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": statistics.fmean(f1_terms) if f1_terms else 0.0,
        "confusion": [
            {"gt": gt, "pred": pred, "count": n}
            for (gt, pred), n in sorted(counts.items(), key=lambda kv: -kv[1])
        ],
    }


def _parse_id_set(values: list[str]) -> set[int | str] | None:
    if not values:
        return None
    parsed: set[int | str] = set()
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            parsed.add(int(part) if part.lstrip("-").isdigit() else part)
    return parsed


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare clevis pseudo-label boxes against EndoVis ground truth."
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/endovis"),
        help="COCO ground-truth file or directory. Directories prefer *_coco.json over gt_merged.json.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        default=[Path("data/cache_endovis/7_export")],
        help="Prediction COCO file(s), directory with coco_*.json, or raw clevis_boxes.json.",
    )
    parser.add_argument("--gt-category-id", action="append", default=[], help="GT category id(s) to keep.")
    parser.add_argument("--gt-category-name", action="append", default=[], help="GT category name(s) to keep.")
    parser.add_argument("--pred-category-id", action="append", default=[], help="Prediction category id(s) to keep.")
    parser.add_argument("--pred-category-name", action="append", default=[], help="Prediction category name(s) to keep.")
    parser.add_argument("--match-iou", type=float, default=0.5, help="IoU threshold used for headline P/R/F1.")
    parser.add_argument(
        "--eval-scope",
        choices=("overlap", "ground-truth", "pseudo-labeled"),
        default="overlap",
        help="Use only images with both GT and predictions, all GT images, or explicitly pseudo-labeled frames.",
    )
    parser.add_argument(
        "--iou-thresholds",
        default="0.50:0.95:0.05",
        help="Comma list or start:stop:step range for AP thresholds.",
    )
    parser.add_argument(
        "--no-scale-predictions",
        action="store_true",
        help="Do not scale prediction boxes to the matching GT image dimensions.",
    )
    parser.add_argument("--raw-pred-width", type=int, default=640, help="Image width for raw clevis_boxes.json predictions.")
    parser.add_argument(
        "--raw-pred-height",
        type=int,
        default=480,
        help="Image height for raw clevis_boxes.json predictions. NOTE: cat1 frames are 512 high, not 480 -- pass it explicitly or every box is stretched by 6.7%%.",
    )
    parser.add_argument(
        "--class-aware",
        action="store_true",
        help="Only match a prediction to a GT box of the same category. Off by default so class-less heuristics stay comparable.",
    )
    parser.add_argument(
        "--per-class",
        action="store_true",
        help="Also report COCO-style per-class AP averaged over classes with GT support (the leaderboard reduction).",
    )
    parser.add_argument(
        "--pred-category-map",
        type=Path,
        default=None,
        help="JSON mapping pipeline class name -> cat1 category id (or {id,name}). Unmapped names are counted, not defaulted.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path for summary JSON.")
    parser.add_argument("--matches-csv", type=Path, default=None, help="Optional path for matched-box CSV at --match-iou.")
    parser.add_argument("--thresholds-csv", type=Path, default=None, help="Optional path for metrics by IoU threshold.")
    parser.add_argument("--area-report", type=Path, default=None, help="Optional CSV path for per (video, class) box-size diagnostics.")
    return parser


def _parse_thresholds(value: str) -> tuple[float, ...]:
    if ":" in value:
        start_s, stop_s, step_s = value.split(":")
        start, stop, step = float(start_s), float(stop_s), float(step_s)
        thresholds = []
        current = start
        while current <= stop + 1e-9:
            thresholds.append(round(current, 4))
            current += step
        return tuple(thresholds)
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _print_summary(summary: dict[str, Any], threshold_rows: list[dict[str, float]], *, title: str = "EndoVis clevis box benchmark") -> None:
    print(title)
    print(f"  GT images / boxes:      {summary['gt_images']} / {summary['gt_annotations']}")
    print(f"  Pred images / boxes:    {summary['pred_images']} / {summary['pred_annotations']}")
    print(f"  Overlap images:         {summary['overlap_images']}")
    print(f"  Pred images without GT: {summary['pseudo_labeled_images_without_gt']}")
    print(f"  Eval scope:             {summary['eval_scope']}")
    print(f"  Evaluated GT boxes:     {summary['evaluated_gt_annotations']} on {summary['evaluated_gt_images']} images")
    print(f"  Evaluated pred boxes:   {summary['evaluated_pred_annotations']}")
    print(f"  Ignored GT boxes:       {summary['ignored_gt_annotations_out_of_scope']} (out of scope)")
    print(f"  Ignored pred boxes:     {summary['ignored_pred_annotations_out_of_scope']} (out of scope)")
    print(f"  Scale predictions:      {summary['scale_predictions']}")
    print(f"  mAP@0.50:0.95:         {summary['map_50_95']:.4f}")
    print(f"  AP@0.50:               {summary['ap50']:.4f}")
    print(f"  P/R/F1@{summary['match_threshold']:.2f}:       {summary.get('precision', 0):.4f} / {summary.get('recall', 0):.4f} / {summary.get('f1', 0):.4f}")
    print(f"  Mean matched IoU:       {summary.get('mean_matched_iou', 0):.4f}")
    print()
    print("By IoU threshold:")
    for row in threshold_rows:
        print(
            f"  {row['threshold']:.2f}: "
            f"AP={row['ap']:.4f} P={row['precision']:.4f} R={row['recall']:.4f} "
            f"TP={int(row['true_positive'])} FP={int(row['false_positive'])} FN={int(row['false_negative'])}"
        )


def _print_per_class(per_class: dict[str, Any]) -> None:
    print()
    print(
        f"Per-class AP (mAP@0.50:0.95 = {per_class['map_50_95']:.4f} over "
        f"{per_class['supported_classes']} supported classes, AP@0.50 = {per_class['ap50']:.4f})"
    )
    print(f"  {'class':34s} {'GT':>6s} {'pred':>7s} {'AP':>7s} {'AP50':>7s} {'R':>6s} {'medIoU':>7s}")
    for row in per_class["per_class"]:
        name = row["category_name"] or str(row["category_id"])
        print(
            f"  {name:34s} {row['gt_boxes']:6d} {row['pred_boxes']:7d} "
            f"{row['ap_50_95']:7.4f} {row['ap50']:7.4f} {row['recall']:6.3f} "
            f"{row['median_matched_iou']:7.3f}"
        )


def _print_area_report(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    print()
    print("Box-size diagnostics (threshold-0 pairing; ratio>1 means predictions are too big)")
    print(f"  {'video':12s} {'class':30s} {'pairs':>6s} {'area_ratio':>11s} {'IoU':>6s} {'contain':>8s}")
    for row in rows:
        name = row["category_name"] or str(row["category_id"])
        print(
            f"  {row['video_id']:12s} {name:30s} {row['pairs']:6d} "
            f"{row['median_area_ratio']:11.2f} {row['median_iou']:6.3f} "
            f"{row['median_containment']:8.3f}"
        )


def _print_confusion(confusion: dict[str, Any], *, top: int = 12) -> None:
    if not confusion["pairs"]:
        return
    print()
    print(
        f"Classification on matched boxes: accuracy {confusion['accuracy']:.4f}, "
        f"macro-F1 {confusion['macro_f1']:.4f} over {confusion['pairs']} pairs"
    )
    for row in confusion["confusion"][:top]:
        flag = "  " if row["gt"] == row["pred"] else "->"
        print(f"  {flag} {row['gt']:30s} predicted as {row['pred']:30s} {row['count']:6d}")


def main() -> None:
    args = build_arg_parser().parse_args()
    gt_records = load_ground_truth(
        args.ground_truth,
        category_ids=_parse_id_set(args.gt_category_id),
        category_names=set(args.gt_category_name),
    )
    category_map = load_category_map(args.pred_category_map) if args.pred_category_map else None
    unmapped_classes: Counter = Counter()
    pred_records = load_predictions(
        args.predictions,
        category_ids=_parse_id_set(args.pred_category_id),
        category_names=set(args.pred_category_name),
        image_width=args.raw_pred_width,
        image_height=args.raw_pred_height,
        category_map=category_map,
        unmapped_classes=unmapped_classes,
    )
    if unmapped_classes:
        print("WARNING: prediction classes with no cat1 category (counted, not scored):")
        for name, count in unmapped_classes.most_common():
            print(f"  {name}: {count}")
        print()
    thresholds = _parse_thresholds(args.iou_thresholds)
    summary, threshold_rows, matches = evaluate(
        gt_records,
        pred_records,
        iou_thresholds=thresholds,
        match_threshold=args.match_iou,
        scale_predictions=not args.no_scale_predictions,
        eval_scope=args.eval_scope,
        class_aware=args.class_aware,
    )
    pseudo_summary, pseudo_threshold_rows, pseudo_matches = evaluate(
        gt_records,
        pred_records,
        iou_thresholds=thresholds,
        match_threshold=args.match_iou,
        scale_predictions=not args.no_scale_predictions,
        eval_scope="pseudo-labeled",
    )

    _print_summary(summary, threshold_rows)
    if args.eval_scope != "pseudo-labeled":
        print()
        _print_summary(
            pseudo_summary,
            pseudo_threshold_rows,
            title="Pseudo-labeled-frame metrics",
        )

    per_class = None
    if args.per_class:
        per_class = evaluate_per_class(
            gt_records,
            pred_records,
            iou_thresholds=thresholds,
            match_threshold=args.match_iou,
            scale_predictions=not args.no_scale_predictions,
            eval_scope=args.eval_scope,
        )
        _print_per_class(per_class)

    # Nearest-GT pairing keeps the boxes that failed to match, which are the only
    # ones that can explain *why* they failed.
    diagnostic_matches = diagnostic_pairs(
        gt_records,
        pred_records,
        scale_predictions=not args.no_scale_predictions,
        class_aware=args.class_aware,
    )
    area_rows = area_report(diagnostic_matches)
    confusion = classification_confusion(matches)
    _print_area_report(area_rows)
    _print_confusion(confusion)

    if args.output_json:
        _write_json(
            args.output_json,
            {
                "summary": summary,
                "thresholds": threshold_rows,
                "matches": [asdict(match) for match in matches],
                "per_class": per_class,
                "area_report": area_rows,
                "classification": confusion,
                "unmapped_prediction_classes": dict(unmapped_classes),
                "pseudo_labeled_frame_metrics": {
                    "summary": pseudo_summary,
                    "thresholds": pseudo_threshold_rows,
                    "matches": [asdict(match) for match in pseudo_matches],
                },
            },
        )
    if args.thresholds_csv:
        _write_csv(args.thresholds_csv, threshold_rows)
    if args.matches_csv:
        _write_csv(args.matches_csv, [asdict(match) for match in matches])
    if args.area_report:
        _write_csv(args.area_report, area_rows)


if __name__ == "__main__":
    main()
