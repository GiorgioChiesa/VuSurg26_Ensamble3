#!/usr/bin/env python3
"""Build the "duplicated classes" control dataset for the two-decoder cascade.

WHAT THE CONTROL CONTROLS FOR

The cascade's decoder B is conditioned on decoder A's whole-instrument box. If
the cascade wins, that could be the conditioning -- or it could be that *any*
model given whole-instrument boxes as extra supervision learns the clevis
better. This dataset gives a single-decoder detector the same supervision with
none of the conditioning: 2N classes, where class ``c`` is instrument ``c``'s
whole box and class ``c + N`` is that instrument's clevis. It is a stock YOLO
directory on purpose, so it runs through the unchanged ``train_rfdetr_detector``
with no code changes anywhere in the training path.

The whole-instrument boxes take their class from the GT clevis they contain,
never from the detector's own prediction; see
``clevis_pseudo_labeler.cascade.dual_class`` for why, and for the containment
rule (shared with ``build_patch_localizer_dataset.py``).

Usage:
    python scripts/build_dual_class_dataset.py \\
        --source-dir exports/cascade/cat1_clevis_yolo \\
        --detections runs/rfdetr_cascade/cat1_annotations/track_annotations.json \\
        --output-dir exports/cascade/cat1_dual28

    # k-fold: restrict to a subset of videos
    python scripts/build_dual_class_dataset.py --video 1_fps1 --video 2_fps1 ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from clevis_pseudo_labeler.cascade.dual_class import (
    DEFAULT_CONTAINMENT_THRESHOLD,
    DEFAULT_PADDING_FRACTION,
    build_dual_class_dataset,
)

DEFAULT_SOURCE = Path("exports/cascade/cat1_clevis_yolo")
DEFAULT_DETECTIONS = Path("runs/rfdetr_cascade/cat1_annotations/track_annotations.json")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE,
        help="YOLO dataset holding the clevis GT (becomes classes N..2N-1).",
    )
    parser.add_argument(
        "--detections",
        type=Path,
        default=DEFAULT_DETECTIONS,
        help="track_annotations.json with the whole-instrument boxes.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--video",
        action="append",
        dest="videos",
        default=None,
        help="Restrict the build to this video id; repeatable (k-fold driver).",
    )
    parser.add_argument(
        "--padding-fraction",
        type=float,
        default=DEFAULT_PADDING_FRACTION,
        help="Detection box is grown by this fraction before the containment test.",
    )
    parser.add_argument(
        "--containment-threshold",
        type=float,
        default=DEFAULT_CONTAINMENT_THRESHOLD,
        help="Fraction of the GT clevis box that must fall inside the padded detection.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Drop detections below this score before matching (default: keep all).",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=None,
        help="Override the probed frame width (must be paired with --frame-height).",
    )
    parser.add_argument("--frame-height", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if (args.frame_width is None) != (args.frame_height is None):
        raise SystemExit("pass --frame-width and --frame-height together")
    frame_size = (
        (args.frame_width, args.frame_height) if args.frame_width is not None else None
    )

    summary = build_dual_class_dataset(
        source_dir=args.source_dir,
        detections_path=args.detections,
        output_dir=args.output_dir,
        videos=args.videos,
        padding_fraction=args.padding_fraction,
        containment_threshold=args.containment_threshold,
        min_confidence=args.min_confidence,
        frame_size=frame_size,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
