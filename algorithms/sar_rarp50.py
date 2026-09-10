"""SAR-RARP50 segmentation -> clevis (wrist) boxes.

WHY THIS DATASET AND NOT JUST ENDOVIS

EndoVis 2018 gave 1,197 usable wrist frames and its pretrain moved the cat1
k-fold by +0.006, i.e. nothing measurable. SAR-RARP50 is the same idea at
twenty times the scale: 44 training videos, ~563 annotated frames each, roughly
24,000 frames carrying a `tool wrist` mask. If "more real clevis geometry" is
the lever, this is the dataset that has to show it.

LAYOUT
    <root>/train/video_NN/segmentation/<frame>.png   1 Hz, grayscale class index
    <root>/train/video_NN/video_left.avi             60 Hz RGB
The mask filename IS the frame index into the AVI — that is the whole join, and
`segmentation_frame_index` refuses anything that does not parse rather than
defaulting to frame 0.

CLASS INDICES (verified against 150 sampled masks, not assumed)
    0 background, 1 tool clasper, **2 tool wrist**, 3 tool shaft,
    4 suturing needle, 5 thread, 6 suction tool, 7 needle holder, 8 clamps,
    9 catheter.
Note this differs from EndoVis 2018, where the wrist is 3.

LICENCE
    CC BY-NC-SA 4.0 — non-commercial. That constrains what a model trained on it
    may be used for; the download manifest records it alongside the data.

Like EndoVis, these boxes are CLASS-AGNOSTIC: no da Vinci taxonomy, so they can
pretrain geometry and never a class head.
"""

from __future__ import annotations

import re
from pathlib import Path

#: `tool wrist` in the SAR-RARP50 label encoding. NOT 3 — that is EndoVis 2018.
WRIST_VALUE = 2

#: One YOLO class; the taxonomy is deliberately absent.
CLASS_NAME = "tool_wrist"

_FRAME_RE = re.compile(r"^(\d+)$")


def segmentation_frame_index(path: str | Path) -> int:
    """Frame index encoded in a segmentation mask's filename.

    Raises `ValueError` on anything that is not a bare digit stem. Falling back
    to 0 would pair a box with the wrong picture, and nothing downstream could
    detect it.
    """
    stem = Path(path).stem
    match = _FRAME_RE.match(stem)
    if not match:
        raise ValueError(f"not a SAR-RARP50 segmentation frame name: {stem!r}")
    return int(match.group(1))
