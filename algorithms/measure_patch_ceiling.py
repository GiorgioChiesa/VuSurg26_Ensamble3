#!/usr/bin/env python3
"""Measure the ceiling of the detect -> classify -> localise-in-patch cascade.

WHY THIS RUNS BEFORE ANY LOCATOR IS BUILT

The patch branch can only predict a clevis that falls inside some full-instrument
detection. That makes detection coverage a HARD CAP on recall, and therefore on
AP, no matter how good the patch locator is. Building a locator first and reading
a low mAP as "the locator is bad" is the mistake this script exists to prevent:
the number may have been unreachable from the start.

WHAT IT REPORTS

  coverage        fraction of GT clevis boxes that fall inside a detection, both
                  for the raw detection box and for the PADDED SQUARE CROP the
                  locator actually sees (instrument_type/datasets.py::
                  _expanded_square_crop). The padded number is the real cap.

  oracle variants scored on the official metric through benchmark_endovis_boxes,
                  so they are directly comparable to every other cat1 number:
                    oracle_box_oracle_class  perfect locator AND perfect class,
                                             limited only by detection coverage
                                             -> the ceiling of the whole branch
                    oracle_box_pred_class    perfect locator, real classifier
                                             -> isolates the classification cost
                    detection_box_as_clevis  the detection box submitted as-is
                                             -> the floor: what you score with no
                                                patch locator at all

  patch prior     where the clevis sits inside its instrument box, as a
                  distribution. Tight -> a fixed geometric prior is already a
                  strong locator; broad -> the segmentation models must work.

Usage:
    python scripts/measure_patch_ceiling.py \\
        --config configs/cat1_patch.yaml \\
        --detections data/cache_cat1_gt/10_rfdetr_annotations/track_annotations.json \\
        [--classification data/cache_cat1_gt/11_instrument_classification/rfdetr_annotations.json] \\
        --output-root runs/clevis_patch --run-id 00_ceiling
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_benchmark_module():
    path = Path(__file__).resolve().parent / "benchmark_endovis_boxes.py"
    spec = importlib.util.spec_from_file_location("benchmark_endovis_boxes", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_benchmark_module()


# ── geometry ────────────────────────────────────────────────────────────────


def _xyxy_from_detection(det: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """RF-DETR records carry both bbox_xyxy and a CENTER-based bbox_xywh.

    Preferring xyxy is not cosmetic: reading that xywh as top-left (the
    convention everywhere else in this repo) shifts every box by half its size.
    """
    box = det.get("bbox_xyxy")
    if box and len(box) == 4:
        return tuple(float(v) for v in box)  # type: ignore[return-value]
    box = det.get("bbox_xywh")
    if box and len(box) == 4:
        cx, cy, w, h = (float(v) for v in box)
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    return None


def _padded_square_crop_bounds(
    bbox_xyxy: tuple[float, float, float, float],
    *,
    frame_width: int,
    frame_height: int,
    padding_fraction: float,
) -> tuple[float, float, float, float]:
    """The region _expanded_square_crop actually shows the locator.

    Mirrors instrument_type/datasets.py:436-461 -- proportional padding per side,
    then clamp to the frame. The square canvas that follows only adds black
    padding, so it cannot bring more of the frame into view; the clamped
    rectangle is the true field of view.
    """
    x1, y1, x2, y2 = bbox_xyxy
    w, h = x2 - x1, y2 - y1
    x1 -= w * padding_fraction
    x2 += w * padding_fraction
    y1 -= h * padding_fraction
    y2 += h * padding_fraction
    return (
        max(0.0, min(float(frame_width), x1)),
        max(0.0, min(float(frame_height), y1)),
        max(0.0, min(float(frame_width), x2)),
        max(0.0, min(float(frame_height), y2)),
    )


def _containment(inner_xyxy, outer_xyxy) -> float:
    """|inner ∩ outer| / |inner| -- how much of the GT box the crop would show."""
    ax1, ay1, ax2, ay2 = inner_xyxy
    bx1, by1, bx2, by2 = outer_xyxy
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    area = (ax2 - ax1) * (ay2 - ay1)
    if area <= 0:
        return 0.0
    return (inter_w * inter_h) / area


def _gt_xyxy(record) -> tuple[float, float, float, float]:
    x, y, w, h = record.bbox_xywh
    return (x, y, x + w, y + h)


# ── loading ─────────────────────────────────────────────────────────────────


def load_detections(path: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """Flat list of frame records -> {(video_id, absolute_frame_idx): [detection]}.

    Keyed on absolute_frame_idx, never frame_idx: the latter is clip-relative and
    restarts at 0 for every sample, so it collides across the 178 roster clips.
    """
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise SystemExit(f"{path} is not a list of frame records")
    by_frame: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for frame in data:
        if not isinstance(frame, dict):
            continue
        video_id = frame.get("video_id")
        frame_idx = frame.get("absolute_frame_idx", frame.get("frame_idx"))
        if video_id is None or frame_idx is None:
            continue
        for det in frame.get("detections") or []:
            box = _xyxy_from_detection(det)
            if box is None or box[2] <= box[0] or box[3] <= box[1]:
                continue
            by_frame[(str(video_id), int(frame_idx))].append(
                {
                    "bbox_xyxy": box,
                    "confidence": float(det.get("confidence", 1.0)),
                    "track_id": det.get("track_id"),
                }
            )
    return by_frame


def aggregate_classes_per_track(records: list[dict[str, Any]]) -> dict[Any, str]:
    """One class per track, from the MEAN of calibrated per-class probabilities.

    Not a majority vote over hard labels: ``top_k`` already carries the
    distribution, and averaging it preserves the calibration rather than
    discarding it at the argmax. This is ADR 0011's rule, applied here to
    ByteTrack tracks instead of to mask tracks.

    The reason to try it at all: a per-frame classifier makes largely independent
    errors along a track, so averaging over the ~30 frames of a clip should
    recover accuracy that a single frame cannot. Track ids restart per sample
    (ADR 0006), so the key must include sample_id or two clips' track 0 merge.
    """
    sums: dict[Any, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: Counter = Counter()
    for rec in records:
        key = (rec.get("video_id"), rec.get("sample_id"), rec.get("track_id"))
        if key[2] is None:
            continue
        counts[key] += 1
        top_k = rec.get("top_k") or [
            {
                "instrument_type": rec.get("instrument_type"),
                "confidence": rec.get("instrument_type_confidence", 0.0),
            }
        ]
        for entry in top_k:
            name = entry.get("instrument_type")
            if name:
                sums[key][name] += float(entry.get("confidence", 0.0))
    return {
        key: max(scores.items(), key=lambda kv: kv[1])[0]
        for key, scores in sums.items()
        if scores
    }


def load_classification(
    path: Path, *, aggregation: str = "per_frame"
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """classify-instrument output -> per-frame detections carrying a real class."""
    payload = json.loads(Path(path).read_text())
    records = payload.get("records", payload) if isinstance(payload, dict) else payload
    track_classes = (
        aggregate_classes_per_track(records) if aggregation == "per_track" else {}
    )
    by_frame: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        if track_classes:
            key = (rec.get("video_id"), rec.get("sample_id"), rec.get("track_id"))
            if key in track_classes:
                rec = {**rec, "instrument_type": track_classes[key]}
        video_id = rec.get("video_id")
        frame_idx = rec.get("absolute_frame_idx", rec.get("frame_idx"))
        box = rec.get("bbox_xyxy")
        if video_id is None or frame_idx is None or not box:
            continue
        by_frame[(str(video_id), int(frame_idx))].append(
            {
                "bbox_xyxy": tuple(float(v) for v in box),
                "confidence": float(rec.get("detection_confidence", 1.0)),
                "instrument_type": rec.get("instrument_type"),
                "instrument_type_confidence": float(
                    rec.get("instrument_type_confidence", 0.0)
                ),
                "track_id": rec.get("track_id"),
            }
        )
    return by_frame


# ── coverage ────────────────────────────────────────────────────────────────


def measure_coverage(
    gt_records: list,
    detections: dict[tuple[str, int], list[dict[str, Any]]],
    *,
    padding_fraction: float,
    containment_threshold: float,
) -> dict[str, Any]:
    covered_raw = 0
    covered_padded = 0
    total = 0
    best_containment: list[float] = []
    per_video_total: Counter = Counter()
    per_video_covered: Counter = Counter()
    frames_with_gt = set()
    frames_with_detection = set()
    # (offset_x, offset_y, size_ratio) of the GT clevis inside its instrument box
    prior_dx: list[float] = []
    prior_dy: list[float] = []
    prior_scale: list[float] = []

    for record in gt_records:
        total += 1
        video_id, frame_idx = record.image_key
        per_video_total[video_id] += 1
        frames_with_gt.add(record.image_key)
        gt_box = _gt_xyxy(record)
        frame_dets = detections.get((str(video_id), int(frame_idx)), [])
        if frame_dets:
            frames_with_detection.add(record.image_key)

        best_raw = 0.0
        best_padded = 0.0
        best_det = None
        for det in frame_dets:
            raw = _containment(gt_box, det["bbox_xyxy"])
            padded = _containment(
                gt_box,
                _padded_square_crop_bounds(
                    det["bbox_xyxy"],
                    frame_width=record.width or 640,
                    frame_height=record.height or 512,
                    padding_fraction=padding_fraction,
                ),
            )
            if padded > best_padded:
                best_padded = padded
                best_det = det
            best_raw = max(best_raw, raw)

        best_containment.append(best_padded)
        if best_raw >= containment_threshold:
            covered_raw += 1
        if best_padded >= containment_threshold:
            covered_padded += 1
            per_video_covered[video_id] += 1
            if best_det is not None:
                dx1, dy1, dx2, dy2 = best_det["bbox_xyxy"]
                dw, dh = dx2 - dx1, dy2 - dy1
                if dw > 0 and dh > 0:
                    gx1, gy1, gx2, gy2 = gt_box
                    prior_dx.append((((gx1 + gx2) / 2) - dx1) / dw)
                    prior_dy.append((((gy1 + gy2) / 2) - dy1) / dh)
                    prior_scale.append(
                        (((gx2 - gx1) * (gy2 - gy1)) / (dw * dh)) ** 0.5
                    )

    def _dist(values: list[float]) -> dict[str, float]:
        if not values:
            return {}
        ordered = sorted(values)
        return {
            "n": len(ordered),
            "p10": ordered[int(0.10 * (len(ordered) - 1))],
            "median": statistics.median(ordered),
            "p90": ordered[int(0.90 * (len(ordered) - 1))],
            "stdev": statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
        }

    return {
        "gt_boxes": total,
        "containment_threshold": containment_threshold,
        "padding_fraction": padding_fraction,
        "covered_by_detection_box": covered_raw,
        "covered_by_padded_crop": covered_padded,
        "recall_ceiling_detection_box": covered_raw / total if total else 0.0,
        "recall_ceiling_padded_crop": covered_padded / total if total else 0.0,
        "median_best_containment": statistics.median(best_containment)
        if best_containment
        else 0.0,
        "gt_frames": len(frames_with_gt),
        "gt_frames_with_any_detection": len(frames_with_detection),
        "per_video": {
            video: {
                "gt_boxes": per_video_total[video],
                "covered": per_video_covered[video],
                "recall_ceiling": per_video_covered[video] / per_video_total[video]
                if per_video_total[video]
                else 0.0,
            }
            for video in sorted(per_video_total)
        },
        "patch_prior": {
            "clevis_center_x_in_detection": _dist(prior_dx),
            "clevis_center_y_in_detection": _dist(prior_dy),
            "clevis_size_over_detection_size": _dist(prior_scale),
        },
    }


# ── oracle prediction sets ──────────────────────────────────────────────────


def build_oracle_predictions(
    gt_records: list,
    detections: dict[tuple[str, int], list[dict[str, Any]]],
    *,
    padding_fraction: float,
    containment_threshold: float,
    classification: dict[tuple[str, int], list[dict[str, Any]]] | None,
    category_map: dict[str, int],
) -> dict[str, list[dict[str, Any]]]:
    """Three prediction sets, all in the raw clevis_boxes.json record schema.

    Emitting them in that schema means benchmark_endovis_boxes scores them with
    exactly the same code path as a real variant -- no separate metric to trust.
    """
    oracle_oracle: list[dict[str, Any]] = []
    oracle_pred: list[dict[str, Any]] = []
    detection_as_clevis: list[dict[str, Any]] = []
    class_hits = 0
    class_total = 0
    class_confusion: Counter = Counter()

    emitted_detection_boxes: set[tuple[str, int, int]] = set()

    for record in gt_records:
        video_id, frame_idx = record.image_key
        key = (str(video_id), int(frame_idx))
        gt_box = _gt_xyxy(record)
        frame_dets = detections.get(key, [])
        frame_classified = (classification or {}).get(key, [])

        best_padded = 0.0
        best_det = None
        for det in frame_dets:
            padded = _containment(
                gt_box,
                _padded_square_crop_bounds(
                    det["bbox_xyxy"],
                    frame_width=record.width or 640,
                    frame_height=record.height or 512,
                    padding_fraction=padding_fraction,
                ),
            )
            if padded > best_padded:
                best_padded = padded
                best_det = det
        if best_det is None or best_padded < containment_threshold:
            continue

        base = {
            "video_id": str(video_id),
            "absolute_frame_idx": int(frame_idx),
            "frame_width": record.width or 640,
            "frame_height": record.height or 512,
            "bbox_xywh": [
                gt_box[0],
                gt_box[1],
                gt_box[2] - gt_box[0],
                gt_box[3] - gt_box[1],
            ],
            "confidence": best_det["confidence"],
        }
        oracle_oracle.append(
            {
                **base,
                "category_id": record.category_id,
                "category_name": record.category_name,
            }
        )

        # Same box, but the class the real classifier assigned to that detection.
        predicted_type = None
        if frame_classified:
            nearest = max(
                frame_classified,
                key=lambda c: _containment(c["bbox_xyxy"], best_det["bbox_xyxy"])
                * _containment(best_det["bbox_xyxy"], c["bbox_xyxy"]),
            )
            predicted_type = nearest.get("instrument_type")
        if predicted_type is not None:
            mapped = category_map.get(predicted_type)
            oracle_pred.append(
                {
                    **base,
                    "category_id": mapped if mapped is not None else -1,
                    "category_name": predicted_type,
                }
            )
            class_total += 1
            correct = mapped == record.category_id
            if correct:
                class_hits += 1
            # Display the GT name on a hit, never the classifier's spelling of it.
            # "monopolar curved scissors" and cat1's "monopolar curved scissor"
            # are one class with one id; printing the raw pair marked 2379
            # CORRECT boxes as errors and made a 0.71-accuracy classifier look
            # like a 0.49 one. Agreement is an id question, exactly as
            # benchmark_endovis_boxes.classification_confusion says.
            shown = record.category_name if correct else predicted_type
            class_confusion[(record.category_name, shown)] += 1

        # The detection box submitted unchanged: the no-locator floor. One row
        # per detection per frame, deduplicated so two GT boxes in one frame do
        # not emit the same detection twice.
        for det_idx, det in enumerate(frame_dets):
            marker = (str(video_id), int(frame_idx), det_idx)
            if marker in emitted_detection_boxes:
                continue
            emitted_detection_boxes.add(marker)
            dx1, dy1, dx2, dy2 = det["bbox_xyxy"]
            detection_as_clevis.append(
                {
                    "video_id": str(video_id),
                    "absolute_frame_idx": int(frame_idx),
                    "frame_width": record.width or 640,
                    "frame_height": record.height or 512,
                    "bbox_xywh": [dx1, dy1, dx2 - dx1, dy2 - dy1],
                    "confidence": det["confidence"],
                    "category_id": record.category_id,
                    "category_name": record.category_name,
                }
            )

    return {
        "oracle_box_oracle_class": oracle_oracle,
        "oracle_box_pred_class": oracle_pred,
        "detection_box_as_clevis": detection_as_clevis,
        "_classification": {
            "pairs": class_total,
            "accuracy": class_hits / class_total if class_total else 0.0,
            "confusion": [
                {"gt": gt, "pred": pred, "count": n}
                for (gt, pred), n in sorted(class_confusion.items(), key=lambda kv: -kv[1])
            ],
        },
    }


# ── driver ──────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gt", type=Path, default=None)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--classification", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("runs/clevis_patch"))
    parser.add_argument("--run-id", default="00_ceiling")
    parser.add_argument(
        "--containment-threshold",
        type=float,
        default=0.9,
        help="Fraction of the GT box that must fall inside the crop to count as covered.",
    )
    parser.add_argument("--iou-thresholds", default="0.50:0.95:0.05")
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument(
        "--class-aggregation",
        choices=("per_frame", "per_track"),
        default="per_frame",
        help="per_track averages calibrated probabilities along a ByteTrack track (ADR 0011's rule).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    from clevis_pseudo_labeler.config import load_config

    cfg = load_config(args.config)
    ev = cfg.clevis_eval
    gt_dir = args.gt or Path(ev.gt_dir)
    padding_fraction = cfg.instrument_type.predict.padding_fraction
    if padding_fraction is None:
        padding_fraction = 0.10
    category_map = {
        str(name): int(value.id if hasattr(value, "id") else value["id"])
        for name, value in (ev.category_map or {}).items()
        if value is not None
    }

    gt_records = bench.load_ground_truth(Path(gt_dir))
    if not gt_records:
        raise SystemExit(f"No ground-truth boxes found under {gt_dir}")
    detections = load_detections(args.detections)
    classification = (
        load_classification(args.classification, aggregation=args.class_aggregation)
        if args.classification
        else None
    )

    n_dets = sum(len(v) for v in detections.values())
    print(f"Ground truth: {len(gt_records)} boxes over {len({r.image_key for r in gt_records})} frames")
    print(f"Detections:   {n_dets} boxes over {len(detections)} frames from {args.detections}")
    if classification is not None:
        print(
            f"Classified:   {sum(len(v) for v in classification.values())} boxes "
            f"from {args.classification}"
        )
    print(
        f"Settings:     padding_fraction={padding_fraction} "
        f"containment>={args.containment_threshold} "
        f"class_aggregation={args.class_aggregation}"
    )

    coverage = measure_coverage(
        gt_records,
        detections,
        padding_fraction=padding_fraction,
        containment_threshold=args.containment_threshold,
    )

    print()
    print("=== COVERAGE: the hard cap on everything downstream ===")
    print(
        f"  GT boxes inside a raw detection box : "
        f"{coverage['covered_by_detection_box']:6d} / {coverage['gt_boxes']}  "
        f"= {coverage['recall_ceiling_detection_box']:.4f}"
    )
    print(
        f"  GT boxes inside the PADDED crop     : "
        f"{coverage['covered_by_padded_crop']:6d} / {coverage['gt_boxes']}  "
        f"= {coverage['recall_ceiling_padded_crop']:.4f}   <-- the real ceiling"
    )
    print(
        f"  GT frames holding any detection     : "
        f"{coverage['gt_frames_with_any_detection']} / {coverage['gt_frames']}"
    )
    print()
    print("  per video:")
    for video, row in coverage["per_video"].items():
        print(
            f"    {video:12s} {row['covered']:5d} / {row['gt_boxes']:5d}  "
            f"= {row['recall_ceiling']:.4f}"
        )

    prior = coverage["patch_prior"]
    print()
    print("=== PATCH PRIOR: where the clevis sits inside its instrument box ===")
    for label, key in (
        ("centre x (0=left edge, 1=right edge)", "clevis_center_x_in_detection"),
        ("centre y (0=top edge,  1=bottom edge)", "clevis_center_y_in_detection"),
        ("sqrt(area ratio) clevis/detection    ", "clevis_size_over_detection_size"),
    ):
        d = prior.get(key) or {}
        if d:
            print(
                f"  {label}  p10={d['p10']:.3f}  median={d['median']:.3f}  "
                f"p90={d['p90']:.3f}  sd={d['stdev']:.3f}"
            )
    print("  A wide p10-p90 band means a fixed prior cannot localise well and the")
    print("  segmentation/VLM locators have to earn their place.")

    oracles = build_oracle_predictions(
        gt_records,
        detections,
        padding_fraction=padding_fraction,
        containment_threshold=args.containment_threshold,
        classification=classification,
        category_map=category_map,
    )
    class_report = oracles.pop("_classification")

    thresholds = bench._parse_thresholds(args.iou_thresholds)
    out_dir = Path(args.output_root) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    scored: list[dict[str, Any]] = []
    print()
    print("=== ORACLE VARIANTS (official metric, same scorer as every cat1 run) ===")
    for name, records in oracles.items():
        if not records:
            print(f"  {name:26s} SKIPPED (no records)")
            continue
        path = out_dir / f"pred_{name}.json"
        path.write_text(json.dumps(records))
        pred_records = bench.load_predictions(
            [path], image_width=ev.pred_width, image_height=ev.pred_height
        )
        summary, _, _ = bench.evaluate(
            gt_records,
            pred_records,
            iou_thresholds=thresholds,
            match_threshold=args.match_iou,
            scale_predictions=True,
            eval_scope=ev.eval_scope,
            class_aware=ev.class_aware,
        )
        per_class = bench.evaluate_per_class(
            gt_records,
            pred_records,
            iou_thresholds=thresholds,
            match_threshold=args.match_iou,
            scale_predictions=True,
            eval_scope=ev.eval_scope,
        )
        scored.append({"variant": name, "summary": summary, "per_class": per_class})
        print(
            f"  {name:26s} mAP={per_class['map_50_95']:.4f}  "
            f"AP50={per_class['ap50']:.4f}  P={summary['precision']:.3f}  "
            f"R={summary['recall']:.3f}  #pred={summary['pred_annotations']}"
        )

    if class_report["pairs"]:
        print()
        print(
            f"=== CLASSIFIER on covered detections: accuracy "
            f"{class_report['accuracy']:.4f} over {class_report['pairs']} boxes ==="
        )
        for row in class_report["confusion"][:12]:
            mark = "   " if row["gt"] == row["pred"] else "-> "
            print(f"  {mark}{str(row['gt']):32s} as {str(row['pred']):32s} {row['count']}")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ground_truth_dir": str(gt_dir),
        "detections": str(args.detections),
        "classification": str(args.classification) if args.classification else None,
        "settings": {
            "padding_fraction": padding_fraction,
            "containment_threshold": args.containment_threshold,
            "eval_scope": ev.eval_scope,
            "class_aware": ev.class_aware,
            "pred_width": ev.pred_width,
            "pred_height": ev.pred_height,
        },
        "coverage": coverage,
        "oracles": scored,
        "classification_on_covered_detections": class_report,
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print()
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
