"""Test-time augmentation helpers.

Only the horizontal flip is offered. A surgical scene has no canonical
left/right — instruments enter from either side — so mirroring is label
preserving. A VERTICAL flip is not: the endoscope has a fixed up/down and
instruments enter from the bottom, so flipping vertically would ask the model
about a scene it will never see.
"""

from __future__ import annotations

from typing import Any, Sequence


def unflip_horizontal(
    detections: Sequence[dict[str, Any]], *, width: float
) -> list[dict[str, Any]]:
    """Map boxes predicted on a mirrored frame back to the original.

    Mirroring swaps which edge is left, so x1 and x2 exchange roles; returning
    them in the original order is what keeps the box valid rather than inverted.
    """
    restored = []
    for detection in detections:
        x1, y1, x2, y2 = (float(v) for v in detection["bbox_xyxy"])
        restored.append({**detection, "bbox_xyxy": [width - x2, y1, width - x1, y2]})
    return restored
