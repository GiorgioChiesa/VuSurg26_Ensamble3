"""Per-frame filtering applied AFTER weighted box fusion.

Three problems measured on the first Grand Challenge submission, which emitted
56,445 boxes over 1,800 frames (31.4 per frame) where the cat1 ground truth
never holds more than 3:

1. **The score floor was applied in the wrong place.** It gated each model's
   predictions *before* fusion, but WBF then multiplies a cluster's mean score
   by the fraction of views that found it — so a box seen by 1 of 8 views leaves
   fusion at ~0.006 and nothing filters it. The floor has to be applied again,
   after fusion, to the score that is actually reported.
2. **The same object was reported under several class labels.** WBF fuses within
   a class only, so a detector hedging between "needle driver" and "force
   bipolar" emits both; 26,905 cross-class pairs overlapped at IoU > 0.9.
3. **No physical bound.** A da Vinci has three instrument arms plus the
   endoscope, and the cat1 ground truth confirms it: box counts per frame are
   1 (798 frames), 2 (2,615) and 3 (1,765) — never more.
"""

from __future__ import annotations

from typing import Any, Sequence

#: The cat1 ground truth never exceeds this in any frame.
MAX_DETECTIONS_PER_FRAME = 3


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    union = (
        max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        - inter
    )
    return inter / union if union > 0 else 0.0


def filter_frame_detections(
    detections: Sequence[dict[str, Any]],
    *,
    min_score: float,
    max_detections: int = MAX_DETECTIONS_PER_FRAME,
    cross_class_iou: float = 0.55,
) -> list[dict[str, Any]]:
    """Score floor, cross-class suppression and a physical cap, in that order.

    The order matters: suppressing first would let a low-scoring box win its
    position against a better one that the floor was about to remove anyway.
    """
    survivors = sorted(
        (d for d in detections if float(d.get("confidence", 0.0)) >= min_score),
        key=lambda d: float(d["confidence"]),
        reverse=True,
    )

    kept: list[dict[str, Any]] = []
    for detection in survivors:
        box = detection["bbox_xyxy"]
        if any(_iou(box, k["bbox_xyxy"]) >= cross_class_iou for k in kept):
            continue
        kept.append(detection)
        if len(kept) >= max_detections:
            break
    return kept
