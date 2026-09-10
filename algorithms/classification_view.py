"""Present a detector's own class head as if it were the crop classifier.

`scripts/measure_patch_ceiling.py --classification` consumes the per-detection
record shape that `instrument-type predict` emits: one record per box, carrying
`video_id`, `absolute_frame_idx`, `bbox_xyxy`, `detection_confidence` and
`instrument_type`. `rfdetr-annotate-roster` instead writes one record per
*frame*, with the boxes nested under `detections`.

A 14-class detector already answers "what tool is this", so this module
flattens the one shape into the other. That is what lets the same ceiling script
score the detector-native class against the 0.349 the crop classifier scored,
with no change to the script and therefore no doubt that the two numbers were
produced the same way.
"""

from __future__ import annotations

from typing import Any, Iterable

FORMAT = "clevis.instrument-box-labels"
VERSION = 2
#: Marks records whose class came from the detector rather than from a
#: calibrated crop classifier, so a later reader cannot confuse the two.
DETECTOR_SOURCE = "detector"

_CARRIED_FRAME_FIELDS = (
    "video_id",
    "sample_id",
    "absolute_frame_idx",
    "frame_idx",
    "split",
    "source_video",
)


def detections_as_classification(
    frame_records: Iterable[dict[str, Any]],
    *,
    source_checkpoint: str | None = None,
) -> dict[str, Any]:
    """Flatten `track_annotations.json` frames into classification records.

    The detector emits a single score per (box, class) — there is no separate
    class score to report — so that one value is written under both
    `detection_confidence` and `instrument_type_confidence`. It is *not* a
    calibrated probability the way ADR 0010's temperature-scaled classifier
    output is, which is why `instrument_type_source` says `detector`.
    """
    records: list[dict[str, Any]] = []
    classes: set[str] = set()

    for frame in frame_records:
        carried = {key: frame.get(key) for key in _CARRIED_FRAME_FIELDS}
        for index, detection in enumerate(frame.get("detections") or []):
            box = detection.get("bbox_xyxy")
            if not box:
                continue
            class_name = detection.get("class_name")
            confidence = float(detection.get("confidence", 0.0))
            if class_name is not None:
                classes.add(str(class_name))
            records.append(
                {
                    **carried,
                    "source_detection_idx": index,
                    "box_source": "rfdetr",
                    "bbox_xyxy": [float(value) for value in box],
                    "class_id": detection.get("class_id"),
                    "class_name": class_name,
                    "detection_confidence": confidence,
                    "instrument_type": class_name,
                    "instrument_type_confidence": confidence,
                    "instrument_type_source": DETECTOR_SOURCE,
                    "track_id": detection.get("track_id"),
                }
            )

    return {
        "format": FORMAT,
        "version": VERSION,
        "box_source": "rfdetr",
        "source_checkpoint": source_checkpoint,
        "classes": sorted(classes),
        "records": records,
    }
