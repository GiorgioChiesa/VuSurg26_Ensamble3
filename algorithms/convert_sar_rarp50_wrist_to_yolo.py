#!/usr/bin/env python3
"""Build a YOLO clevis dataset from SAR-RARP50 segmentation masks.

Box extraction is `cascade.endovis2018.wrist_boxes` — the same connected-component
rule, parameterised by class index — and the frame join is
`cascade.sar_rarp50.segmentation_frame_index`. Both are unit tested; this is the
I/O around them.

DECODING STRATEGY
    Masks are 1 Hz, the AVI is 60 Hz, so only ~1 frame in 60 is ever needed. The
    videos are ~1.2 GB each and seeking per frame on a GlusterFS mount is far
    slower than a single sequential pass, so this reads each video once and keeps
    the frames whose index has a mask. That is the difference between minutes and
    hours over 44 videos.

Usage:
    python scripts/convert_sar_rarp50_wrist_to_yolo.py \\
        --source-root data/sar_rarp50/train \\
        --output-dir exports/cascade/sar_rarp50_wrist \\
        --val-videos video_01,video_02
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import cv2
import yaml

from clevis_pseudo_labeler.cascade.endovis2018 import wrist_boxes
from clevis_pseudo_labeler.cascade.sar_rarp50 import (
    CLASS_NAME,
    WRIST_VALUE,
    segmentation_frame_index,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source-root", type=Path, default=Path("data/sar_rarp50/train"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--part-value", type=int, default=WRIST_VALUE)
    parser.add_argument("--min-area", type=int, default=64)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument(
        "--val-videos",
        default="",
        help="Comma-separated video ids held out for the pretrain's own validation.",
    )
    parser.add_argument("--limit-videos", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    val_videos = {v.strip() for v in args.val_videos.split(",") if v.strip()}
    videos = sorted(d for d in args.source_root.iterdir() if (d / "segmentation").is_dir())
    if args.limit_videos:
        videos = videos[: args.limit_videos]

    counters: collections.Counter[str] = collections.Counter()
    per_video: dict[str, int] = {}
    splits_seen: set[str] = set()

    for video_dir in videos:
        video_id = video_dir.name
        split = "val" if video_id in val_videos else "train"
        avi = video_dir / "video_left.avi"
        masks = sorted(video_dir.glob("segmentation/*.png"))
        if not avi.exists() or not masks:
            counters["videos_skipped"] += 1
            continue

        wanted: dict[int, Path] = {}
        for mask in masks:
            try:
                wanted[segmentation_frame_index(mask)] = mask
            except ValueError:
                counters["masks_unparseable"] += 1

        capture = cv2.VideoCapture(str(avi))
        written = 0
        index = 0
        # One sequential pass; see the module docstring on why not seeking.
        while wanted:
            ok, frame = capture.read()
            if not ok:
                break
            mask_path = wanted.pop(index, None)
            index += 1
            if mask_path is None:
                continue
            label = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if label is None:
                counters["masks_unreadable"] += 1
                continue
            if label.shape[:2] != frame.shape[:2]:
                frame = cv2.resize(frame, (label.shape[1], label.shape[0]))
                counters["frames_resized_to_mask"] += 1

            boxes = wrist_boxes(label, part_value=args.part_value, min_area=args.min_area)
            counters["frames_paired"] += 1
            if not boxes:
                counters["frames_without_wrist_skipped"] += 1
                continue

            stem = f"{index - 1:09d}"
            out_image = args.output_dir / "images" / split / video_id / f"{stem}.jpg"
            out_label = args.output_dir / "labels" / split / video_id / f"{stem}.txt"
            out_image.parent.mkdir(parents=True, exist_ok=True)
            out_label.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_image), frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            out_label.write_text(
                "".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for cx, cy, w, h in boxes)
            )
            counters["frames_written"] += 1
            counters["boxes"] += len(boxes)
            written += 1
        capture.release()

        if wanted:
            counters["masks_past_end_of_video"] += len(wanted)
        if written:
            splits_seen.add(split)
            per_video[f"{split}/{video_id}"] = written
        print(f"  {video_id} [{split}]: {written} frame", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                **{split: f"images/{split}" for split in sorted(splits_seen)},
                "nc": 1,
                "names": [CLASS_NAME],
            },
            sort_keys=False,
        )
    )
    summary = {
        "format": "clevis.sar-rarp50-wrist-dataset",
        "version": 1,
        "source_root": str(args.source_root),
        "licence": "CC BY-NC-SA 4.0",
        "part_value": args.part_value,
        "min_area": args.min_area,
        "videos": len(videos),
        "val_videos": sorted(val_videos),
        "counters": dict(counters),
        "per_video": dict(sorted(per_video.items())),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_video"}, indent=2))


if __name__ == "__main__":
    main()
