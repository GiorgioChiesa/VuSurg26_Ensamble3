"""Fold predictions in the shape `benchmark_endovis_boxes` reads.

WHY THIS EXISTS

Every k-fold figure produced so far — 0.3381 for the plain detector, 0.3331 for
the 28-class control, 0.3441 with the EndoVis init — comes from RF-DETR's own
validation loop. The incumbent 0.194 came from the patch localiser's own
evaluation. Two different scorers, so the comparison rests on the assumption
that they agree. This module removes the assumption: it writes plain COCO that
`scripts/aggregate_cat1_folds.py` can score, so every number is produced by the
one scorer.

The `image_id` is the frame index, because that is how the cat1 ground truth
keys its images (`{"id": 44, "file_name": "0000000044.jpg"}`). Get it wrong and
every prediction lands on the wrong frame while still scoring *something*.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence


def frame_index_from_stem(stem: str) -> int:
    """`"0000000044"` -> `44`. Raises on anything that is not a frame number."""
    if not stem.isdigit():
        raise ValueError(f"not a frame-index stem: {stem!r}")
    return int(stem)


def coco_document(
    *,
    video_id: str,
    width: int,
    height: int,
    frames: Iterable[tuple[int, Sequence[dict[str, Any]]]],
    categories: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble one video's predictions into a COCO document.

    `frames` yields `(frame_index, detections)`. A frame with no detection still
    contributes its image entry: dropping it would shrink the evaluated frame
    set and quietly flatter recall.
    """
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1

    for frame_index, detections in frames:
        images.append(
            {
                "id": frame_index,
                # `benchmark_endovis_boxes._image_key` prefers this over anything
                # derived from the filename, so the (video, frame) key stays
                # correct no matter what the file ends up being called.
                "video_id": video_id,
                "width": width,
                "height": height,
                "file_name": f"{frame_index:010d}.jpg",
            }
        )
        for detection in detections:
            x1, y1, x2, y2 = (float(v) for v in detection["bbox_xyxy"])
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": frame_index,
                    "category_id": int(detection["class_id"]),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                    "score": float(detection.get("confidence", 1.0)),
                }
            )
            annotation_id += 1

    document: dict[str, Any] = {
        "video_id": video_id,
        "images": images,
        "annotations": annotations,
    }
    if categories is not None:
        document["categories"] = list(categories)
    return document
