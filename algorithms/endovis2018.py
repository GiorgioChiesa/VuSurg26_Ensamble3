"""EndoVis 2018 part masks -> clevis (wrist) boxes.

WHY THIS DATASET

The cat1 ground truth is the only source of *real* clevis geometry the pipeline
has, and it is 11,323 boxes over 7 videos that the k-fold protocol also has to
validate on. EndoVis 2018 Robotic Scene Segmentation is the one additional
source already sitting on this machine: it encodes each part as a grayscale
class index, and value 3 is `instrument-wrist` — the same anatomy cat1 calls the
clevis. Measured over the 1,800 training labels: present in 672 frames,
20.3M pixels, the largest instrument-part class by pixel count.

WHAT IT CANNOT DO

EndoVis carries no da Vinci tool taxonomy. Every box here is class-agnostic, so
this data can pretrain a clevis localiser's *geometry* and never its class head.
Since the class ceiling moved from 0.349 to 0.851 once classification went into
the detector, geometry is the binding constraint again — which is what makes
this worth converting now and not before.

The domain is also not SurgVU: different console, different optics, different
procedures. Treat any gain as a transfer result to be measured per fold, never
as free extra training data.

Class indices (EndoVis 2018): 0 background-tissue, 1 instrument-shaft,
2 instrument-clasper, **3 instrument-wrist**, 4 kidney-parenchyma,
5 covered-kidney, 6 thread, 7 clamps, 8 suturing-needle, 9 suction-instrument,
10 small-intestine, 11 ultrasound-probe.
"""

from __future__ import annotations

import cv2
import numpy as np

#: `instrument-wrist` in the EndoVis 2018 label encoding.
WRIST_VALUE = 3

#: One YOLO class. The taxonomy is deliberately absent — see the module docstring.
CLASS_NAME = "tool_wrist"


def wrist_boxes(
    label: np.ndarray,
    *,
    part_value: int = WRIST_VALUE,
    min_area: int = 64,
    min_width: int = 2,
    min_height: int = 2,
) -> list[tuple[float, float, float, float]]:
    """Normalised ``(cx, cy, w, h)`` boxes for every wrist blob in one label.

    One box per connected component, not one per frame: a surgical frame
    routinely holds two instruments, and a single bounding box over both would
    span the gap between them and teach the locator the opposite of what the
    clevis is.

    Returns an empty list when the frame has no wrist. That is a valid frame —
    whether to export it as a negative example is the caller's decision.
    """
    mask = (label == part_value).astype(np.uint8)
    if not mask.any():
        return []

    height, width = mask.shape[:2]
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    boxes: list[tuple[float, float, float, float]] = []
    for index in range(1, count):  # 0 is the background component
        x, y, w, h, area = (
            stats[index, cv2.CC_STAT_LEFT],
            stats[index, cv2.CC_STAT_TOP],
            stats[index, cv2.CC_STAT_WIDTH],
            stats[index, cv2.CC_STAT_HEIGHT],
            stats[index, cv2.CC_STAT_AREA],
        )
        if area < min_area or w < min_width or h < min_height:
            continue
        boxes.append(
            (
                (x + w / 2) / width,
                (y + h / 2) / height,
                w / width,
                h / height,
            )
        )
    return boxes
