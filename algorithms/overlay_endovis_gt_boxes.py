#!/usr/bin/env python3
"""Overlay EndoVis COCO ground-truth boxes onto EndoVis videos.

Default usage renders one annotated MP4 per ``*_fps1_coco.json`` file:

    uv run python scripts/overlay_endovis_gt_boxes.py
"""

from __future__ import annotations

import argparse
import json
import re
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2


COLOR_PALETTE_BGR = (
    (0, 255, 255),
    (0, 165, 255),
    (255, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 255, 0),
    (80, 180, 255),
    (180, 80, 255),
    (80, 255, 180),
    (255, 180, 80),
)


@dataclass(frozen=True)
class BoxAnnotation:
    frame_idx: int
    bbox_xywh: tuple[float, float, float, float]
    category_id: int | str | None
    category_name: str | None
    annotation_id: int | str | None = None
    image_width: int | None = None
    image_height: int | None = None
    score: float | None = None


@dataclass
class VideoAnnotations:
    video_id: str
    boxes_by_frame: dict[int, list[BoxAnnotation]] = field(default_factory=lambda: defaultdict(list))

    def add(self, box: BoxAnnotation) -> None:
        self.boxes_by_frame[box.frame_idx].append(box)


@dataclass(frozen=True)
class OverlayLayer:
    """One set of boxes drawn in one colour.

    Ground truth and predictions are separate layers rather than one palette-keyed
    set: when the question is "does the prediction sit where the GT sits", colour
    has to encode provenance, not class. Class goes in the text label.
    """

    annotations: VideoAnnotations
    color: tuple[int, int, int] | None = None
    label_prefix: str = ""


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
    for pattern in (r"frame[_-]?(\d+)", r"frame(\d+)", r"slice_nr_(\d+)", r"(\d+)(?=\.[A-Za-z0-9]+$)"):
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    stem = Path(text).stem
    if stem.isdigit():
        return int(stem)
    return None


def _image_key(image: dict[str, Any], fallback_video_id: str | None = None) -> tuple[str, int]:
    file_name = str(image.get("file_name", ""))
    video_id = (
        _video_base(image.get("original_filename"))
        or _video_base(image.get("video_id"))
        or (None if "/" not in file_name else _video_base(file_name.split("/", 1)[0]))
        or _video_base(fallback_video_id)
        or _video_base(image.get("id"))
    )
    frame_idx = image.get("frame_id")
    if frame_idx is None:
        frame_idx = _parse_frame_from_text(image.get("file_name"))
    if frame_idx is None:
        frame_idx = _parse_frame_from_text(image.get("id"))
    if video_id is None or frame_idx is None:
        raise ValueError(f"Cannot derive video/frame key from image record: {image}")
    return video_id, int(frame_idx)


def _bbox_xywh(values: Iterable[Any]) -> tuple[float, float, float, float]:
    bbox = tuple(float(v) for v in values)
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox with 4 values, got {bbox}")
    return bbox


def _parse_id_set(values: list[str]) -> set[int | str] | None:
    if not values:
        return None
    parsed: set[int | str] = set()
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                parsed.add(int(part) if part.lstrip("-").isdigit() else part)
    return parsed


def _category_ids_by_name(categories: dict[int | str, str], names: set[str]) -> set[int | str]:
    lowered = {name.lower() for name in names}
    return {category_id for category_id, name in categories.items() if name.lower() in lowered}


def _annotation_paths(path: Path) -> list[Path]:
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


def load_annotations(
    annotation_path: Path,
    *,
    category_ids: set[int | str] | None = None,
    category_names: set[str] | None = None,
) -> dict[str, VideoAnnotations]:
    videos: dict[str, VideoAnnotations] = {}
    for path in _annotation_paths(annotation_path):
        coco = _read_json(path)
        if not isinstance(coco, dict) or "images" not in coco or "annotations" not in coco:
            raise ValueError(f"{path} is not a COCO annotation file")

        categories = {cat.get("id"): str(cat.get("name", cat.get("id"))) for cat in coco.get("categories", [])}
        allowed_ids = set(category_ids) if category_ids is not None else None
        if category_names:
            name_ids = _category_ids_by_name(categories, category_names)
            allowed_ids = name_ids if allowed_ids is None else allowed_ids | name_ids

        fallback_video_id = re.sub(r"_coco$", "", path.stem)
        images_by_id = {image["id"]: image for image in coco["images"]}
        for ann in coco["annotations"]:
            category_id = ann.get("category_id")
            if allowed_ids is not None and category_id not in allowed_ids:
                continue
            image = images_by_id.get(ann.get("image_id"))
            if image is None:
                continue
            bbox = _bbox_xywh(ann.get("bbox", ()))
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue
            video_id, frame_idx = _image_key(image, fallback_video_id=fallback_video_id)
            video_annotations = videos.setdefault(video_id, VideoAnnotations(video_id=video_id))
            video_annotations.add(
                BoxAnnotation(
                    frame_idx=frame_idx,
                    bbox_xywh=bbox,
                    category_id=category_id,
                    category_name=categories.get(category_id),
                    annotation_id=ann.get("id"),
                    image_width=int(image["width"]) if image.get("width") is not None else None,
                    image_height=int(image["height"]) if image.get("height") is not None else None,
                )
            )
    return videos


def load_prediction_annotations(
    prediction_path: Path,
    *,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> dict[str, VideoAnnotations]:
    """Read a raw pipeline box list (clevis_boxes.json / clevis_masks.json).

    Keys frames by ``absolute_frame_idx`` when present: ``frame_idx`` is relative
    to a roster clip and restarts at 0, so drawing by it would paint every clip's
    boxes onto the opening frames of the video.
    """
    data = _read_json(prediction_path)
    if not isinstance(data, list):
        raise ValueError(f"{prediction_path} is not a raw list of box records")

    videos: dict[str, VideoAnnotations] = {}
    for item in data:
        if not isinstance(item, dict) or "bbox_xywh" not in item:
            continue
        bbox = _bbox_xywh(item["bbox_xywh"])
        if bbox[2] <= 0 or bbox[3] <= 0:
            continue
        video_id = _video_base(item.get("video_id"))
        frame_idx = item.get("absolute_frame_idx")
        if frame_idx is None:
            frame_idx = item.get("frame_idx")
        if video_id is None or frame_idx is None:
            continue
        name = (
            item.get("category_name")
            or item.get("instrument_type")
            or item.get("tool_type")
        )
        score = item.get("instrument_type_confidence", item.get("confidence"))
        video_annotations = videos.setdefault(video_id, VideoAnnotations(video_id=video_id))
        video_annotations.add(
            BoxAnnotation(
                frame_idx=int(frame_idx),
                bbox_xywh=bbox,
                category_id=item.get("category_id"),
                category_name=str(name) if name else None,
                annotation_id=item.get("instrument_track_id"),
                image_width=item.get("frame_width", frame_width),
                image_height=item.get("frame_height", frame_height),
                score=float(score) if score is not None else None,
            )
        )
    return videos


def load_coco_prediction_annotations(
    prediction_path: Path,
    *,
    min_score: float = 0.0,
) -> dict[str, VideoAnnotations]:
    """Read a COCO *prediction* document (``coco_<video_id>.json``).

    `load_annotations` already parses COCO, but it drops the per-box ``score``
    that every prediction carries, and predictions are exactly where a score is
    worth drawing. This is the missing adapter between the k-fold predictors
    (`scripts/predict_folds_*.py`, which write `cascade.coco_predictions.coco_document`)
    and the renderer: without it, a fold prediction cannot be put on a video at all.

    A directory is expanded to its ``coco_*.json`` files — note the *prefix*,
    whereas ground truth uses the ``*_coco.json`` suffix.
    """
    if prediction_path.is_dir():
        paths = sorted(prediction_path.glob("coco_*.json"))
        if not paths:
            raise FileNotFoundError(f"No coco_*.json found in {prediction_path}")
    else:
        paths = [prediction_path]

    videos: dict[str, VideoAnnotations] = {}
    for path in paths:
        coco = _read_json(path)
        if not isinstance(coco, dict) or "images" not in coco or "annotations" not in coco:
            raise ValueError(f"{path} is not a COCO prediction document")

        categories = {
            cat.get("id"): str(cat.get("name", cat.get("id")))
            for cat in coco.get("categories", [])
        }
        fallback_video_id = re.sub(r"^coco_", "", path.stem)
        images_by_id = {image["id"]: image for image in coco["images"]}

        for ann in coco["annotations"]:
            score = ann.get("score")
            if score is not None and float(score) < min_score:
                continue
            image = images_by_id.get(ann.get("image_id"))
            if image is None:
                continue
            bbox = _bbox_xywh(ann.get("bbox", ()))
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue
            category_id = ann.get("category_id")
            video_id, frame_idx = _image_key(image, fallback_video_id=fallback_video_id)
            video_annotations = videos.setdefault(
                video_id, VideoAnnotations(video_id=video_id)
            )
            video_annotations.add(
                BoxAnnotation(
                    frame_idx=frame_idx,
                    bbox_xywh=bbox,
                    category_id=category_id,
                    category_name=categories.get(category_id),
                    annotation_id=ann.get("id"),
                    image_width=int(image["width"]) if image.get("width") is not None else None,
                    image_height=int(image["height"]) if image.get("height") is not None else None,
                    score=float(score) if score is not None else None,
                )
            )
    return videos


def _find_video_path(data_dir: Path, video_id: str) -> Path:
    candidates = [
        data_dir / f"{video_id}.mp4",
        data_dir / video_id / f"{video_id}.mp4",
        data_dir / "sample_videos" / video_id / f"{video_id}.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No video found for {video_id}; tried: {', '.join(str(p) for p in candidates)}")


def _scale_bbox_to_frame(
    bbox: tuple[float, float, float, float],
    *,
    image_width: int | None,
    image_height: int | None,
    frame_width: int,
    frame_height: int,
    scale_boxes: bool,
) -> tuple[float, float, float, float]:
    if not scale_boxes or not image_width or not image_height:
        return bbox
    sx = frame_width / image_width
    sy = frame_height / image_height
    return bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy


def _clip_box(
    bbox: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int] | None:
    x, y, w, h = bbox
    x1 = max(0, min(frame_width - 1, int(round(x))))
    y1 = max(0, min(frame_height - 1, int(round(y))))
    x2 = max(0, min(frame_width - 1, int(round(x + w))))
    y2 = max(0, min(frame_height - 1, int(round(y + h))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _category_color(category_id: int | str | None) -> tuple[int, int, int]:
    if category_id is None:
        return (0, 255, 255)
    try:
        idx = int(category_id)
    except (TypeError, ValueError):
        idx = zlib.crc32(str(category_id).encode())
    return COLOR_PALETTE_BGR[idx % len(COLOR_PALETTE_BGR)]


def _draw_label(frame: Any, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y - th - baseline - 4)
    bottom = max(th + baseline + 4, y)
    cv2.rectangle(frame, (x, top), (min(frame.shape[1] - 1, x + tw + 6), bottom), (0, 0, 0), -1)
    cv2.putText(frame, text, (x + 3, bottom - baseline - 2), font, scale, color, thickness, cv2.LINE_AA)


def draw_boxes_on_frame(
    frame: Any,
    boxes: list[BoxAnnotation],
    *,
    scale_boxes: bool = True,
    draw_labels: bool = True,
    thickness: int = 2,
    color: tuple[int, int, int] | None = None,
    label_prefix: str = "",
    label_below: bool = False,
) -> int:
    """Draw one layer of boxes. A fixed ``color`` overrides the per-category palette."""
    frame_height, frame_width = frame.shape[:2]
    drawn = 0
    for box in boxes:
        scaled = _scale_bbox_to_frame(
            box.bbox_xywh,
            image_width=box.image_width,
            image_height=box.image_height,
            frame_width=frame_width,
            frame_height=frame_height,
            scale_boxes=scale_boxes,
        )
        clipped = _clip_box(scaled, frame_width, frame_height)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        box_color = color if color is not None else _category_color(box.category_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
        if draw_labels:
            label = f"{label_prefix}{box.category_name or box.category_id}"
            if box.score is not None:
                label = f"{label} {box.score:.2f}"
            # GT and prediction labels would otherwise land on the same pixels
            # whenever the boxes agree -- which is exactly the case worth reading.
            anchor_y = y2 + 16 if label_below else y1
            _draw_label(frame, label, (x1, min(frame_height - 1, anchor_y)), box_color)
        drawn += 1
    return drawn


def render_overlay_video(
    video_path: Path,
    annotations: VideoAnnotations | list[OverlayLayer],
    output_path: Path,
    *,
    frame_stride: int = 1,
    start_frame: int = 0,
    end_frame: int | None = None,
    max_frames: int | None = None,
    scale_boxes: bool = True,
    draw_labels: bool = True,
    thickness: int = 2,
) -> int:
    """Render one video with every layer drawn on each frame.

    Accepts a bare VideoAnnotations (the original single-layer behaviour) or a
    list of OverlayLayer, drawn in order so later layers sit on top.
    """
    layers = (
        annotations
        if isinstance(annotations, list)
        else [OverlayLayer(annotations=annotations)]
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = source_fps if source_fps > 0 else 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stop_frame = frame_count if end_frame is None else min(end_frame, frame_count)
    output_fps = max(1.0, fps / max(1, frame_stride))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video writer: {output_path}")

    written = 0
    frame_idx = max(0, start_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    try:
        while frame_idx < stop_frame:
            ok, frame = cap.read()
            if not ok:
                break
            if (frame_idx - start_frame) % frame_stride == 0:
                for layer_idx, layer in enumerate(layers):
                    draw_boxes_on_frame(
                        frame,
                        layer.annotations.boxes_by_frame.get(frame_idx, []),
                        scale_boxes=scale_boxes,
                        draw_labels=draw_labels,
                        thickness=thickness,
                        color=layer.color,
                        label_prefix=layer.label_prefix,
                        label_below=layer_idx > 0,
                    )
                writer.write(frame)
                written += 1
                if max_frames is not None and written >= max_frames:
                    break
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    return written


def _parse_bgr(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Expected B,G,R triple, got {value!r}")
    channels = tuple(int(part) for part in parts)
    if any(not 0 <= channel <= 255 for channel in channels):
        raise argparse.ArgumentTypeError(f"BGR channels must be in [0, 255], got {value!r}")
    return channels  # type: ignore[return-value]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Overlay COCO ground-truth boxes, and optionally pipeline predictions, onto videos.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/endovis"), help="Directory containing EndoVis MP4 videos.")
    parser.add_argument("--annotations", type=Path, default=Path("data/endovis"), help="COCO annotation file or directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/endovis/gt_overlays"), help="Directory for annotated MP4 files.")
    parser.add_argument("--video-id", action="append", default=[], help="Video id(s) to render, e.g. 4_fps1. Default: all annotated videos.")
    parser.add_argument("--category-id", action="append", default=[], help="Category id(s) to draw.")
    parser.add_argument("--category-name", action="append", default=[], help="Category name(s) to draw.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Render every Nth frame.")
    parser.add_argument("--start-frame", type=int, default=0, help="First frame index to render.")
    parser.add_argument("--end-frame", type=int, default=None, help="Exclusive end frame index.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of output frames per video.")
    parser.add_argument("--thickness", type=int, default=2, help="Box line thickness in pixels.")
    parser.add_argument("--no-labels", action="store_true", help="Draw boxes without category text labels.")
    parser.add_argument("--no-scale-boxes", action="store_true", help="Do not scale annotation boxes to video dimensions.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output videos.")
    parser.add_argument(
        "--pred",
        action="append",
        type=Path,
        default=[],
        help="Raw pipeline box JSON (clevis_boxes.json / clevis_masks.json) to overlay alongside the GT. Repeatable.",
    )
    parser.add_argument(
        "--pred-coco",
        action="append",
        type=Path,
        default=[],
        help="COCO prediction document or a directory of coco_<video>.json (what "
        "scripts/predict_folds_*.py write) to overlay alongside the GT. Repeatable.",
    )
    parser.add_argument(
        "--pred-min-score",
        type=float,
        default=0.0,
        help="Drop --pred-coco boxes below this score. Fold predictions are written "
        "at conf 0.05, so an unfiltered overlay is unreadable.",
    )
    parser.add_argument(
        "--gt-color",
        type=_parse_bgr,
        default=None,
        help="BGR colour for ground-truth boxes, e.g. 0,255,0 (green). Default: per-category palette.",
    )
    parser.add_argument(
        "--pred-color",
        type=_parse_bgr,
        default=(255, 0, 0),
        help="BGR colour for prediction boxes, e.g. 255,0,0 (blue).",
    )
    parser.add_argument("--output-suffix", default=None, help="Output file suffix (default: gt_boxes, or gt_vs_pred with --pred).")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be >= 1")

    annotations_by_video = load_annotations(
        args.annotations,
        category_ids=_parse_id_set(args.category_id),
        category_names=set(args.category_name),
    )
    predictions_by_video: dict[str, VideoAnnotations] = {}

    def _merge(loaded: dict[str, VideoAnnotations]) -> None:
        for video_id, video_annotations in loaded.items():
            existing = predictions_by_video.setdefault(
                video_id, VideoAnnotations(video_id=video_id)
            )
            for boxes in video_annotations.boxes_by_frame.values():
                for box in boxes:
                    existing.add(box)

    for pred_path in args.pred:
        _merge(load_prediction_annotations(pred_path))
    for pred_path in args.pred_coco:
        _merge(load_coco_prediction_annotations(pred_path, min_score=args.pred_min_score))

    selected_video_ids = args.video_id or sorted(set(annotations_by_video) | set(predictions_by_video))
    if not selected_video_ids:
        print("No annotated videos found.")
        return

    has_predictions = bool(args.pred or args.pred_coco)
    suffix = args.output_suffix or ("gt_vs_pred" if has_predictions else "gt_boxes")
    rendered = 0
    for video_id in selected_video_ids:
        annotations = annotations_by_video.get(video_id, VideoAnnotations(video_id=video_id))
        video_path = _find_video_path(args.data_dir, video_id)
        output_path = args.output_dir / f"{video_id}_{suffix}.mp4"
        if output_path.exists() and not args.force:
            print(f"Skipping existing output: {output_path} (use --force to overwrite)")
            continue
        layers = [OverlayLayer(annotations=annotations, color=args.gt_color, label_prefix="GT ")]
        predictions = predictions_by_video.get(video_id)
        if predictions is not None:
            layers.append(
                OverlayLayer(annotations=predictions, color=args.pred_color, label_prefix="")
            )
        frame_total = render_overlay_video(
            video_path,
            layers,
            output_path,
            frame_stride=args.frame_stride,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            max_frames=args.max_frames,
            scale_boxes=not args.no_scale_boxes,
            draw_labels=not args.no_labels,
            thickness=args.thickness,
        )
        box_total = sum(len(boxes) for boxes in annotations.boxes_by_frame.values())
        pred_total = (
            sum(len(boxes) for boxes in predictions.boxes_by_frame.values())
            if predictions is not None
            else 0
        )
        print(
            f"Wrote {output_path} ({frame_total} frames, {box_total} GT boxes, {pred_total} predicted)"
        )
        rendered += 1

    print(f"Rendered {rendered} video(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
